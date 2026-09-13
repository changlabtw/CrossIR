"""Descriptive statistics tables.

Reimplements the `stats_table` function of the legacy project's
`5_analysis.ipynb` (cell [9]) as a module-level function. It produces one sheet
of `stats.xlsx`, corresponding to thesis Tables `stats1`, `stats2` and `stats3`.
"""

import logging
from functools import reduce

import pandas as pd
import polars as pl
from scipy.stats import kstest, mannwhitneyu, ttest_ind

logger = logging.getLogger(__name__)

ALPHA = 0.01
"""Significance level below which a p-value is reported as ``"< 0.01"``."""

NON_FEATURE_COLUMNS = ["Release_No", "DIABETES", "SEX", "MET_ID"]
"""Columns summarised separately or not at all: identifiers and the sex block."""


def _sex_counts(df: pl.DataFrame) -> pl.DataFrame:
    """Build the participant-count block that heads each sheet.

    Produces a total row followed by one row per sex, each cell showing the
    count and, on the sex rows, its share of the column total.

    Args:
        df: Cohort table containing ``Release_No``, ``SEX`` and ``IR``.

    Returns:
        Four columns -- ``column``, ``All``, ``IR-``, ``IR+`` -- with rows
        ``N``, ``Male`` and ``Female``.
    """
    counts = (
        df.select(pl.col("Release_No", "SEX", "IR"))
        .group_by(["SEX", "IR"], maintain_order=True)
        .agg(pl.col("Release_No").count())
        .pivot(index="SEX", on="IR", values="Release_No")
        .rename({"SEX": "column", "false": "IR-", "true": "IR+"})
        .with_columns((pl.col("IR-") + pl.col("IR+")).alias("All"))
    )

    totals = counts.sum().with_columns(
        pl.all().exclude("column").map_elements(lambda value: f"{value:,.0f}", return_dtype=pl.Utf8)
    )
    shares = counts.with_columns(
        pl.all().exclude("column").map_elements(lambda value: f"{value:,.0f}", return_dtype=pl.Utf8)
        + (pl.all().exclude("column") / pl.all().exclude("column").sum()).map_elements(
            lambda value: f"({value:.1%})", return_dtype=pl.Utf8
        )
    )

    return (
        pl.concat([totals, shares])
        .with_columns(
            pl.col("column").map_elements(
                lambda code: ["Male", "Female", "N"][int(code - 1)], return_dtype=pl.Utf8
            )
        )
        .select(["column", "All", "IR-", "IR+"])
    )


def _group_p_values(df: pl.DataFrame) -> pl.DataFrame:
    """Test every variable for normality and for a difference between IR groups.

    Three tests are run per variable: a Kolmogorov-Smirnov test of each IR group
    against a normal distribution, Welch's t-test, and the Mann-Whitney U test.
    Each result is reported as ``"< 0.01"`` when significant and as the p-value
    itself otherwise.

    Note:
        Each KS test compares a sample against a normal whose mean and standard
        deviation are estimated from that same sample, which makes the test
        anti-conservative (the Lilliefors situation). This is reproduced from the
        legacy code; replacing the test is out of scope for a reproduction phase.

    Args:
        df: Cohort table with an ``IR`` column and numeric feature columns.

    Returns:
        One row per column of ``df`` with the four p-value columns.
    """
    rows = []
    for column in df.columns:
        positive = df.filter(pl.col("IR") == True)[column].to_numpy()  # noqa: E712
        negative = df.filter(pl.col("IR") == False)[column].to_numpy()  # noqa: E712

        ks_positive = kstest(positive, "norm", args=(positive.mean(), positive.std()))[1]
        ks_negative = kstest(negative, "norm", args=(negative.mean(), negative.std()))[1]
        welch = ttest_ind(positive, negative, equal_var=False)[1]
        mann_whitney = mannwhitneyu(positive, negative)[1]

        rows.append(
            [
                column,
                f"< {ALPHA}" if ks_negative < ALPHA else ks_negative,
                f"< {ALPHA}" if ks_positive < ALPHA else ks_positive,
                f"< {ALPHA}" if welch < ALPHA else welch,
                f"< {ALPHA}" if mann_whitney < ALPHA else mann_whitney,
            ]
        )

    return pl.DataFrame(
        rows,
        schema=["column", "ks_p_value-", "ks_p_value+", "t_test_p_value", "u_test_p_value"],
        orient="row",
    )


def descriptive_statistics(df: pl.DataFrame) -> pd.DataFrame:
    """Summarise one cohort as a publication table.

    Each variable is reported as ``mean±SD`` overall and within each insulin
    resistance group, followed by the p-values from :func:`_group_p_values`.
    A participant-count block is prepended when the cohort carries an ``IR``
    label.

    Args:
        df: Cleaned cohort table, as written by stage 01.

    Returns:
        The finished sheet, ready for :meth:`pandas.DataFrame.to_excel`.
    """
    labelled = "IR" in df.columns
    if labelled:
        df = df.sort("IR")
        sex_counts = _sex_counts(df)

    df = df.select(pl.all().exclude(NON_FEATURE_COLUMNS))

    parts = [
        df.mean().transpose(include_header=True, column_names=["All"]),
        df.std().transpose(include_header=True, column_names=["std_All"]),
    ]
    if labelled:
        parts.append(
            df.group_by("IR", maintain_order=True)
            .agg(pl.all().mean())
            .transpose(include_header=True, column_names=["IR-", "IR+"])
        )
        parts.append(
            df.group_by("IR", maintain_order=True)
            .agg(pl.all().std())
            .transpose(include_header=True, column_names=["std_IR-", "std_IR+"])
        )
        parts.append(_group_p_values(df))

    table = reduce(lambda left, right: left.join(right, on="column"), parts)
    table = table.filter(pl.col("column") != "IR")
    table = table.with_columns(
        pl.struct(["All", "std_All"]).map_elements(
            lambda row: f"{row['All']:.1f}±{row['std_All']:.1f}", return_dtype=pl.Utf8
        )
    )
    if not table.select("^.*IR.*$").is_empty():
        table = table.with_columns(
            pl.struct(["IR-", "std_IR-"]).map_elements(
                lambda row: f"{row['IR-']:.1f}±{row['std_IR-']:.1f}", return_dtype=pl.Utf8
            )
        ).with_columns(
            pl.struct(["IR+", "std_IR+"]).map_elements(
                lambda row: f"{row['IR+']:.1f}±{row['std_IR+']:.1f}", return_dtype=pl.Utf8
            )
        )

    table = table.select(pl.all().exclude("^std.*$")).sort("column")
    if "IR+" in table.columns:
        table = pl.concat([sex_counts, table], how="diagonal_relaxed")

    logger.info("Descriptive statistics table: %d rows", table.height)
    return table.to_pandas()
