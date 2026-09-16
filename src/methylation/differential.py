"""Per-probe differential methylation testing.

Every probe is tested independently for a difference in beta value between the
two insulin-resistance groups, and the resulting p-values are corrected for
multiple testing across all probes with the Benjamini-Hochberg procedure. Both
a parametric and a rank-based test are run, together with a normality test per
group; the volcano plot uses the rank-based one.
"""

import logging

import numpy as np
import polars as pl
from scipy.stats import kstest, mannwhitneyu, ttest_ind
from statsmodels.stats.multitest import multipletests

logger = logging.getLogger(__name__)

P_VALUE_COLUMNS = [
    "ks_test_p_value_IR-",
    "ks_test_p_value_IR+",
    "u_test_p_value",
    "t_test_p_value",
]
"""Raw p-value columns that receive a multiple-testing correction."""

P_VALUE_THRESHOLD = 0.05
"""Adjusted p-value below which a probe counts as differentially methylated."""

LOG2FC_THRESHOLD = 0.2
"""Minimum absolute log2 fold change for a probe to count as differential.

0.2 on a log2 scale is a 14.9% difference in median methylation level.
"""


def probe_statistics(
    matrix: pl.DataFrame,
    negative_ids: list[str],
    positive_ids: list[str],
    batch_size: int = 10_000,
) -> pl.DataFrame:
    """Summarise and test every probe across the two insulin-resistance groups.

    Four tests are run per probe: a Kolmogorov-Smirnov test of each group against
    a normal distribution fitted to that group, the Mann-Whitney U test, and
    Welch's t-test. The KS tests estimate the normal's parameters from the sample
    being tested, which makes them anti-conservative. They are reported for
    information only: the significance thresholding uses the rank-based test.

    Note:
        The standard deviation reported per group uses the sample estimator
        (``ddof=1``); the KS tests are given the population estimator
        (``ddof=0``), which is what fitting a normal to the sample calls for.

    Args:
        matrix: ``TargetID`` plus one column of beta values per participant, as
            returned by :func:`src.methylation.extract.assemble_beta_matrix`.
        negative_ids: Column names of the insulin-resistance negative group.
        positive_ids: Column names of the insulin-resistance positive group.
        batch_size: Probes materialised as a dense array at a time. Affects
            memory use only.

    Returns:
        One row per probe: ``TargetID``, then the sum, standard deviation, count
        and median of each group, then the four raw p-values.
    """
    logger.info(
        "Testing %d probes, %d IR- against %d IR+",
        matrix.height,
        len(negative_ids),
        len(positive_ids),
    )

    batches = []
    for start in range(0, matrix.height, batch_size):
        block = matrix.slice(start, batch_size)
        negative = block.select(negative_ids).to_numpy()
        positive = block.select(positive_ids).to_numpy()

        ks_negative, ks_positive, mann_whitney, welch = [], [], [], []
        for row in range(block.height):
            a, b = negative[row], positive[row]
            ks_negative.append(kstest(a, "norm", args=(a.mean(), a.std()))[1])
            ks_positive.append(kstest(b, "norm", args=(b.mean(), b.std()))[1])
            mann_whitney.append(mannwhitneyu(a, b)[1])
            welch.append(ttest_ind(a, b, equal_var=False)[1])

        batches.append(
            pl.DataFrame(
                {
                    "TargetID": block["TargetID"],
                    "IR-sum": negative.sum(axis=1),
                    "IR-std": negative.std(axis=1, ddof=1),
                    "IR-n": np.full(block.height, len(negative_ids), dtype=np.int32),
                    "IR-med": np.median(negative, axis=1),
                    "IR+sum": positive.sum(axis=1),
                    "IR+std": positive.std(axis=1, ddof=1),
                    "IR+n": np.full(block.height, len(positive_ids), dtype=np.int32),
                    "IR+med": np.median(positive, axis=1),
                    "ks_test_p_value_IR-": np.array(ks_negative),
                    "ks_test_p_value_IR+": np.array(ks_positive),
                    "u_test_p_value": np.array(mann_whitney),
                    "t_test_p_value": np.array(welch),
                }
            )
        )
        logger.info("Tested %d/%d probes", min(start + batch_size, matrix.height), matrix.height)

    return pl.concat(batches)


def adjust_pvalues(
    stats: pl.DataFrame,
    columns: list[str] | None = None,
    method: str = "fdr_bh",
) -> pl.DataFrame:
    """Correct each p-value column for multiple testing across all probes.

    Args:
        stats: Output of :func:`probe_statistics`.
        columns: Raw p-value columns to correct. Defaults to
            :data:`P_VALUE_COLUMNS`.
        method: Correction passed to ``statsmodels.stats.multitest.multipletests``.
            The default is Benjamini-Hochberg false discovery rate control.

    Returns:
        The frame with one ``Adj. <column>`` per corrected column appended.
    """
    for column in columns or P_VALUE_COLUMNS:
        adjusted = multipletests(stats[column], method=method)[1]
        stats = stats.with_columns(pl.Series(adjusted).alias(f"Adj. {column}"))
    logger.info("Applied %s correction across %d probes", method, stats.height)
    return stats


def volcano_frame(
    stats: pl.DataFrame,
    p_threshold: float = P_VALUE_THRESHOLD,
    log2fc_threshold: float = LOG2FC_THRESHOLD,
) -> pl.DataFrame:
    """Add the volcano-plot coordinates and the significance flag.

    The y coordinate is the negative log10 of the Benjamini-Hochberg adjusted
    Mann-Whitney p-value -- a q-value, not a p-value. The x coordinate is the
    log2 ratio of the two group *medians*, which is robust to the long tail of
    beta values a few participants can carry.

    Args:
        stats: Output of :func:`adjust_pvalues`.
        p_threshold: Adjusted p-value below which a probe is significant.
        log2fc_threshold: Minimum absolute log2 fold change.

    Returns:
        The frame with ``IR-mean``, ``IR+mean``, ``neg_log10_p``, ``log2_fc``
        and a boolean ``significant``.
    """
    frame = (
        stats.with_columns(
            (pl.col("IR-sum") / pl.col("IR-n")).alias("IR-mean"),
            (pl.col("IR+sum") / pl.col("IR+n")).alias("IR+mean"),
        )
        .with_columns(
            (-np.log10(pl.col("Adj. u_test_p_value"))).alias("neg_log10_p"),
            np.log2(pl.col("IR+med") / pl.col("IR-med")).alias("log2_fc"),
        )
        .with_columns(
            (
                (pl.col("neg_log10_p") > -np.log10(p_threshold))
                & (pl.col("log2_fc").abs() > log2fc_threshold)
            ).alias("significant")
        )
    )
    logger.info(
        "Significant probes at q < %s and |log2FC| > %s: %d",
        p_threshold,
        log2fc_threshold,
        frame["significant"].sum(),
    )
    return frame
