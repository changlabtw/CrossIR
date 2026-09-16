"""Descriptive statistics tables.

One function per table. :func:`descriptive_statistics` produces one sheet of
``stats.xlsx``: per-variable means and standard deviations split by
insulin-resistance status, with the tests behind the comparison.
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


def _group_p_values(df: pl.DataFrame, alpha: float | None = ALPHA) -> pl.DataFrame:
    """Test every variable for normality and for a difference between IR groups.

    Three tests are run per variable: a Kolmogorov-Smirnov test of each IR group
    against a normal distribution, Welch's t-test, and the Mann-Whitney U test.
    Each result is reported as ``"< 0.01"`` when significant and as the p-value
    itself otherwise.

    Note:
        Each KS test compares a sample against a normal whose mean and standard
        deviation are estimated from that same sample, which makes the test
        anti-conservative (the Lilliefors situation). Read it as a rough
        indication of non-normality, not as a calibrated test.

    Args:
        df: Cohort table with an ``IR`` column and numeric feature columns.
        alpha: Significance level below which a p-value is reported as
            ``"< alpha"`` rather than as a number. Pass ``None`` to report every
            p-value as its own value, which is what the reader needs to judge how
            far below the threshold a result actually falls.

    Returns:
        One row per column of ``df`` with the four p-value columns.
    """
    def report(value: float):
        """Render one p-value, thresholded or not."""
        return f"< {alpha}" if alpha is not None and value < alpha else value

    rows = []
    for column in df.columns:
        if column == "IR":
            # The grouping column itself is constant within each group, so every
            # test on it is degenerate. Its row is dropped downstream anyway.
            continue
        positive = df.filter(pl.col("IR") == True)[column].to_numpy()  # noqa: E712
        negative = df.filter(pl.col("IR") == False)[column].to_numpy()  # noqa: E712

        ks_positive = kstest(positive, "norm", args=(positive.mean(), positive.std()))[1]
        ks_negative = kstest(negative, "norm", args=(negative.mean(), negative.std()))[1]
        welch = ttest_ind(positive, negative, equal_var=False)[1]
        mann_whitney = mannwhitneyu(positive, negative)[1]

        rows.append(
            [
                column,
                report(ks_negative),
                report(ks_positive),
                report(welch),
                report(mann_whitney),
            ]
        )

    return pl.DataFrame(
        rows,
        schema=["column", "ks_p_value-", "ks_p_value+", "t_test_p_value", "u_test_p_value"],
        orient="row",
        strict=False,
    )


def descriptive_statistics(df: pl.DataFrame, alpha: float | None = ALPHA) -> pd.DataFrame:
    """Summarise one cohort as a publication table.

    Each variable is reported as ``mean±SD`` overall and within each insulin
    resistance group, followed by the p-values from :func:`_group_p_values`.
    A participant-count block is prepended when the cohort carries an ``IR``
    label.

    Args:
        df: Cleaned cohort table.
        alpha: Significance level for the p-value columns. ``None`` reports every
            p-value as its own value instead of ``"< alpha"``.

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
        parts.append(_group_p_values(df, alpha))

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


MODEL_LABELS = {
    "Polynomial_Regression": (
        r"\parbox[t]{5cm}{\linespread{1}\selectfont{Second-degree}\\{Polynomial Regression}}"
    )
}
"""LaTeX labels for models whose name does not typeset directly."""


def _metric_cell(row: dict, bold: bool) -> str:
    """Render one model's three metrics as a LaTeX ``\\parbox``.

    Args:
        row: Mapping with ``r2``, ``mae`` and ``rmse``.
        bold: Whether to wrap each metric in ``\\boldsymbol``.

    Returns:
        The ``\\parbox`` source for one table cell.
    """
    metrics = [
        f"R^{{2}} = {row['r2']:.3f}",
        f"MAE={row['mae']:.3f}",
        f"RMSE={row['rmse']:.3f}",
    ]
    rendered = [f"{{${rf'\boldsymbol{{{m}}}' if bold else m}$}}" for m in metrics]
    return r"\parbox[t]{3cm}{\linespread{1}\selectfont" + r"\\".join(rendered) + "}"


def regression_latex_table(results: pl.DataFrame, highlight: str = "CatBoost") -> str:
    """Render the regression results as LaTeX table rows.

    Generating the markup rather than typing it keeps the typeset table and the
    spreadsheet from drifting apart.

    Args:
        results: Tidy results from
            :func:`src.models.regression.evaluate_feature_sets`, with columns
            ``model``, ``feature_set``, ``r2``, ``mae`` and ``rmse``.
        highlight: Model whose row is set in bold, normally the best performer.

    Returns:
        One ``&``-separated LaTeX row per model, newline separated, each ending
        in ``\\\\``.
    """
    feature_sets = list(dict.fromkeys(results["feature_set"].to_list()))
    lines = []
    for model in dict.fromkeys(results["model"].to_list()):
        bold = model == highlight
        label = MODEL_LABELS.get(model, rf"\textbf{{{model}}}" if bold else model)
        cells = [
            _metric_cell(
                results.filter(
                    (pl.col("model") == model) & (pl.col("feature_set") == feature_set)
                ).to_dicts()[0],
                bold,
            )
            for feature_set in feature_sets
        ]
        lines.append(" & ".join([label, *cells]) + r" \\")

    logger.info("Rendered LaTeX table with %d rows", len(lines))
    return "\n".join(lines)
