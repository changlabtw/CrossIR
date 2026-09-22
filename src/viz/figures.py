"""Descriptive figures.

Every figure the pipeline produces, one function each. Each takes the data and a
destination path, draws, saves and closes, so that a notebook cell is a single
call and no figure state leaks between them.
"""

import logging
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score, roc_curve

from src.data.io import HOMAIR_CUTOFF, display_path
from src.viz.palettes import NEUTRAL_GREY, OKABE_ITO

logger = logging.getLogger(__name__)

LOG_SCALED_FEATURES = ["SGOT", "SGPT", "TG", "CREATININE", "FASTING_GLUCOSE"]
"""Right-skewed variables plotted on a natural-log scale."""

CORRELATION_EXCLUDED = [
    "Release_No",
    "DIABETES",
    "SEX",
    "MET_ID",
    "FASTING_INSULIN",
    "IR",
]
"""Identifiers, the target, and the quantity the target is derived from."""



def plot_feature_boxplots(df: pl.DataFrame, title: str, path: str | Path) -> Path:
    """Draw one boxplot per variable, split by cohort and insulin resistance status.

    Lays the variables out on a 4x4 grid with the unused last panel removed, and
    places a single shared legend above the grid.

    Args:
        df: Long table with one column per variable plus ``RACE`` (the cohort
            label used on the x axis) and a boolean ``IR``.
        title: Figure suptitle.
        path: Destination PNG path.

    Returns:
        The path written to.
    """
    df = df.with_columns(
        pl.col("IR").map_elements(
            lambda value: "IR+" if value else "IR-", return_dtype=pl.Utf8
        )
    )
    variables = [column for column in df.columns if column not in ["IR", "RACE"]]

    rows, columns = 4, 4
    figure, axes = plt.subplots(rows, columns, figsize=(15, 15), sharey=False, sharex=False)
    for row, column in product(range(rows), range(columns)):
        index = int(columns * row + column)
        if index >= len(variables):
            break

        variable = variables[index]
        # Rebuilt each iteration so the log transform below never compounds.
        panel_data = df.to_pandas()

        prefix, suffix = "", ""
        if variable in LOG_SCALED_FEATURES:
            panel_data.loc[:, variable] = np.log(panel_data.loc[:, variable])
            prefix, suffix = "Log(", ")"

        sns.boxplot(
            data=panel_data,
            x="RACE",
            y=variable,
            hue="IR",
            ax=axes[row, column],
            width=0.5,
            gap=0.1,
            notch=True,
        )
        axes[row, column].set_title(prefix + variable + suffix)
        axes[row, column].set_ylabel(prefix + variable + suffix)
        axes[row, column].legend_.remove()

    figure.delaxes(axes[3, 3])

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.97), frameon=False
    )
    plt.suptitle(title, fontsize=20, y=0.99)
    return _save(path)


def plot_correlation_matrix(
    frames: dict[str, pl.DataFrame],
    path: str | Path,
    cmap: str = "coolwarm",
) -> Path:
    """Draw a lower-triangular correlation heatmap for each cohort.

    Args:
        frames: Mapping of panel title to cohort table, laid out on a 2x2 grid
            in insertion order.
        path: Destination PNG path.
        cmap: Diverging colour map for the correlation scale. The default
            ``coolwarm`` runs red to blue, the hardest pair to separate under the
            common forms of colour vision deficiency; pass
            :data:`src.viz.palettes.COLOR_BLIND_DIVERGING` for the accessible
            variant.

    Returns:
        The path written to.
    """
    rows, columns = 2, 2
    figure, axes = plt.subplots(rows, columns, figsize=(15, 15), sharey=False, sharex=False)
    for row, column in product(range(rows), range(columns)):
        index = int(columns * row + column)
        frame = list(frames.values())[index]
        frame = frame.select(pl.all().exclude(CORRELATION_EXCLUDED))
        frame = frame.select(sorted(frame.columns))

        matrix = frame.to_pandas().corr()
        mask = np.triu(np.ones_like(matrix, dtype=bool))

        sns.heatmap(
            matrix.mask(mask),
            annot=True,
            fmt=".2f",
            annot_kws={"size": 6},
            cmap=cmap,
            vmax=1,
            vmin=-1,
            cbar=False,
            mask=mask,
            ax=axes[row, column],
        )
        axes[row, column].set_title(list(frames.keys())[index])

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.suptitle("Correlation matrix of datasets", fontsize=20, y=0.99)
    return _save(path)


def plot_homair_distributions(
    frames: dict[str, pl.DataFrame],
    path: str | Path,
    cutoff_color: str = "red",
) -> Path:
    """Draw the log HOMA-IR distribution of each cohort with the cut-off marked.

    Args:
        frames: Mapping of panel title to cohort table, each carrying a
            ``HOMA-IR`` column. Laid out on a 2x2 grid with the unused panel
            removed.
        path: Destination PNG path.
        cutoff_color: Colour of the dashed HOMA-IR cut-off line. Line style
            already distinguishes it from the histogram, so this only affects
            how easily it is picked out; ``"#D55E00"`` is the accessible
            choice.

    Returns:
        The path written to.
    """
    titles = list(frames)
    rows, columns = 2, 2
    figure, axes = plt.subplots(rows, columns, figsize=(15, 15), sharey=False, sharex=False)
    for row, column in product(range(rows), range(columns)):
        index = int(columns * row + column)
        if index >= len(titles):
            break

        title = titles[index]
        values = np.log(frames[title].to_pandas()["HOMA-IR"])

        sns.histplot(data=values, kde=True, ax=axes[row, column])
        axes[row, column].set_title(title)
        axes[row, column].set_ylabel("Count")
        axes[row, column].set_xlabel("Log(HOMA-IR index)")
        axes[row, column].set_ylim(top=axes[row, column].get_ylim()[1])
        axes[row, column].vlines(
            x=np.log(HOMAIR_CUTOFF),
            ymin=0,
            ymax=axes[row, column].get_ylim()[1],
            color=cutoff_color,
            linestyle="dashed",
        )

    figure.delaxes(axes[1, 1])

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.97), frameon=False
    )
    plt.suptitle("Distribution of HOMA-IR index across datasets", fontsize=20, y=0.99)
    return _save(path)


def _save(path: str | Path) -> Path:
    """Write the current figure and close it.

    Args:
        path: Destination PNG path. The parent directory is created if needed.

    Returns:
        The path written to.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path)
    plt.close()
    logger.info("Wrote %s", display_path(path))
    return path


def plot_confusion_matrix(confusion: dict, title: str, path: str | Path) -> Path:
    """Draw a confusion matrix as an annotated heatmap.

    Args:
        confusion: Mapping with ``TP``, ``TN``, ``FP`` and ``FN``, as returned by
            :func:`src.models.evaluate.compute_metrics`.
        title: Name of the model, used in the title.
        path: Destination PNG path.

    Returns:
        The path written to.
    """
    matrix = np.array(
        [[confusion["TN"], confusion["FP"]], [confusion["FN"], confusion["TP"]]]
    )
    labels = ["Negative", "Positive"]

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=",.0f",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar=False,
        annot_kws={"size": 16},
    )
    plt.title(f"Confusion Matrix: {title}", fontsize=16)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.tight_layout()
    return _save(path)



def plot_calibration(curves: dict, predictions: dict, title: str, path: str | Path) -> Path:
    """Draw a reliability diagram, with the distribution of predictions beneath it.

    The upper panel plots each bin's mean predicted probability against the
    fraction of that bin who were actually insulin resistant. A model whose
    probabilities mean what they say lies on the diagonal. The lower panel shows
    where the predictions themselves fall, because a curve is only as trustworthy
    as the number of participants behind each of its points.

    Note:
        Drawn from the Okabe-Ito palette, so it needs no colour-blind sibling.
        The two models are additionally distinguished by marker shape, so the
        figure survives being printed in greyscale.

    Args:
        curves: Model name mapped to the frame returned by
            :func:`src.models.evaluate.calibration_table`.
        predictions: The same model names mapped to their predicted
            probabilities, which the lower panel histograms.
        title: Name used in the title.
        path: Destination PNG path.

    Returns:
        The path written to.
    """
    colours = [OKABE_ITO["blue"], OKABE_ITO["orange"]]
    markers = ["o", "s"]

    figure, axes = plt.subplots(
        2, 1, figsize=(7, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )

    axes[0].plot([0, 1], [0, 1], linestyle="--", linewidth=1, color=NEUTRAL_GREY, label="Perfect calibration")
    for index, (name, table) in enumerate(curves.items()):
        axes[0].plot(
            table["mean_predicted"],
            table["observed"],
            marker=markers[index % len(markers)],
            color=colours[index % len(colours)],
            label=name,
        )
        axes[1].hist(
            predictions[name],
            bins=50,
            range=(0, 1),
            histtype="step",
            color=colours[index % len(colours)],
        )

    axes[0].set_ylabel("Observed frequency", fontsize=12)
    axes[0].set_title(f"Calibration: {title}", fontsize=16)
    axes[0].legend(loc="upper left")
    axes[1].set_xlabel("Predicted probability of insulin resistance", fontsize=12)
    axes[1].set_ylabel("Participants", fontsize=12)

    figure.tight_layout()
    return _save(path)


def plot_roc_pr_curves(y_true, preds, title: str, path: str | Path) -> Path:
    """Draw the ROC and precision-recall curves, stacked.

    Both are shown because they answer different questions under class
    imbalance: the ROC curve is insensitive to the number of negatives, the
    precision-recall curve is not.

    Args:
        y_true: Observed binary labels.
        preds: Predicted probability of the positive class.
        title: Name of the model, used in the titles.
        path: Destination PNG path.

    Returns:
        The path written to.
    """
    false_positive_rate, true_positive_rate, _ = roc_curve(y_true, preds)
    roc_auc = roc_auc_score(y_true, preds)
    precision, recall, _ = precision_recall_curve(y_true, preds)
    pr_auc = auc(recall, precision)

    _, axes = plt.subplots(2, 1, figsize=(7, 12), sharey=False, sharex=False)

    axes[0].plot(
        false_positive_rate,
        true_positive_rate,
        color="darkorange",
        lw=2,
        label=f"ROC curve (AUC = {roc_auc:.3f})",
    )
    axes[0].plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title(f"ROC Curve: {title}")
    axes[0].legend(loc="lower right")

    axes[1].plot(
        recall, precision, color="darkorange", lw=2, label=f"PR curve (AUC = {pr_auc:.3f})"
    )
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title(f"Precision-Recall Curve: {title}")
    axes[1].legend(loc="lower left")

    plt.tight_layout()
    return _save(path)


TG_HDL_PANEL_LABELS = {
    "NHANES + KNHANES": "NHANES\n+KNHANES",
    "TWB (IR was predicted)": "TWB\n(IR was predicted)",
}
"""Panel titles that are wrapped onto two lines to fit the left margin."""


def plot_tg_hdl_boxplots(frames: dict[str, pl.DataFrame], path: str | Path) -> Path:
    """Compare the distribution of log(TG/HDL-C) by IR status across cohorts.

    The triglyceride to HDL-cholesterol ratio is a widely used surrogate marker
    of insulin resistance, so agreement between the labelled cohorts and the
    Taiwan Biobank *predictions* is evidence that the predicted labels behave
    like real ones. One stacked panel per cohort, sharing both axes.

    Note:
        Each panel takes its own frame from ``frames``, keyed by the title it is
        drawn with, so a panel cannot end up showing a different cohort from the
        one its title names.

    Args:
        frames: Ordered mapping of panel title to a cohort table carrying ``TG``,
            ``HDL_C`` and ``IR``. ``IR`` may be boolean or 0/1.
        path: Destination PNG path.

    Returns:
        The path written to.
    """
    _, axes = plt.subplots(len(frames), 1, figsize=(8, 6), sharey=True, sharex=True)

    for axis, (title, frame) in zip(axes, frames.items()):
        data = frame.with_columns(
            pl.when(pl.col("IR").cast(pl.Int64) == 1)
            .then(pl.lit("IR+"))
            .otherwise(pl.lit("IR-"))
            .alias("IR"),
            np.log(pl.col("TG") / pl.col("HDL_C")).alias("TG/HDL_C"),
        )
        sns.boxplot(
            data=data.to_pandas(), y="IR", x="TG/HDL_C", hue="IR", ax=axis, notch=True
        )
        axis.set_ylabel(
            TG_HDL_PANEL_LABELS.get(title, title),
            rotation=0,
            labelpad=2,
            ha="right",
            va="center",
        )
        axis.set_xlabel("log(TG/HDL-C)")

    plt.tight_layout(rect=[0, 0, 1, 0.9])
    plt.suptitle("Boxplot of log(TG/HDL-C) by IR and datasets", y=0.95)
    return _save(path)


def plot_quantile_comparison(
    reference: np.ndarray,
    comparison: np.ndarray,
    labels: tuple[str, str],
    title: str,
    path: str | Path,
    point_color: str = "blue",
    line_color: str = "red",
) -> Path:
    """Draw a two-sample Q-Q plot against the line of equality.

    Both samples are evaluated at the same evenly spaced percentiles, as many of
    them as there are observations in the smaller sample. Points on the dashed
    ``y = x`` line mean the two distributions agree at that quantile.

    Args:
        reference: Sample plotted on the x axis.
        comparison: Sample plotted on the y axis.
        labels: Axis names for ``reference`` and ``comparison``; ``" Quantiles"``
            is appended to each.
        title: Figure title.
        path: Destination PNG path.
        point_color: Colour of the quantile scatter.
        line_color: Colour of the dashed ``y = x`` reference line. Red on blue
            is the pair worth avoiding; ``"#D55E00"`` reads as distinct from the
            points under every form of colour vision deficiency.

    Returns:
        The path written to.
    """
    percentiles = np.linspace(0, 100, min(len(reference), len(comparison)))
    reference_quantiles = np.percentile(reference, percentiles)
    comparison_quantiles = np.percentile(comparison, percentiles)

    plt.figure(figsize=(8, 6))
    plt.scatter(
        reference_quantiles,
        comparison_quantiles,
        color=point_color,
        s=10,
        label="Quantile points",
    )

    low = min(reference_quantiles.min(), comparison_quantiles.min())
    high = max(reference_quantiles.max(), comparison_quantiles.max())
    plt.plot([low, high], [low, high], color=line_color, linestyle="--", label="y = x")

    plt.xlabel(f"{labels[0]} Quantiles")
    plt.ylabel(f"{labels[1]} Quantiles")
    plt.title(title)
    plt.legend()
    return _save(path)


def plot_volcano(
    df,
    path: str | Path,
    p_threshold: float = 0.05,
    log2fc_threshold: float = 0.2,
    significant_color: str = "red",
    background_color: str = "gray",
    fold_change_color: str = "blue",
    p_value_color: str = "green",
) -> Path:
    """Draw a volcano plot of the per-probe differential methylation results.

    Effect size on the x axis against statistical confidence on the y axis, so
    that probes which are both large and reliable separate into the upper
    corners. The dashed lines mark the two thresholds; points meeting both are
    red.

    Args:
        df: Frame with ``log2_fc``, ``neg_log10_p`` and a boolean ``significant``,
            as returned by :func:`src.methylation.differential.volcano_frame`.
        path: Destination PNG path.
        p_threshold: Adjusted p-value threshold, drawn as a horizontal line.
        log2fc_threshold: Fold-change threshold, drawn as two vertical lines.
        significant_color: Colour of the probes clearing both thresholds.
        background_color: Colour of every other probe.
        fold_change_color: Colour of the two vertical threshold lines.
        p_value_color: Colour of the horizontal threshold line. The defaults put
            red points against a green line, which is the classic pair to avoid;
            the accessible variant uses vermillion points and draws both sets of
            threshold lines in blue, since their orientation already tells them
            apart.

    Returns:
        The path written to.
    """
    data = df.to_pandas() if hasattr(df, "to_pandas") else df

    plt.figure(figsize=(8, 6))
    plt.scatter(
        data["log2_fc"],
        data["neg_log10_p"],
        c=data["significant"].map({True: significant_color, False: background_color}),
        alpha=0.5,
        s=40,
        edgecolor=None,
    )
    plt.axvline(x=-log2fc_threshold, color=fold_change_color, linestyle="--", linewidth=1)
    plt.axvline(x=log2fc_threshold, color=fold_change_color, linestyle="--", linewidth=1)
    plt.axhline(y=-np.log10(p_threshold), color=p_value_color, linestyle="--", linewidth=1)

    plt.title("Volcano plot of DNA methylation analysis", fontsize=16)
    plt.xlabel("Log2 Fold Change", fontsize=14)
    plt.ylabel("-Log10(q-value)", fontsize=14)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    return _save(path)


WRAPPED_TERM_MARKERS = ["Homo", "during"]
"""Words before which a long pathway name is broken onto a second line."""


def _wrap_term(term: str) -> str:
    """Break a long pathway name onto separate lines at known marker words.

    Each marker is applied in turn to the result of the previous one, so a name
    containing several of them gains several line breaks. One term in this study
    -- "Regulation of p27 Phosphorylation during Cell Cycle Progression Homo
    sapiens h p27Pathway" -- carries both markers and wraps onto three lines.
    """
    for marker in WRAPPED_TERM_MARKERS:
        if marker in term:
            cut = term.rfind(" ", 0, term.find(marker))
            if cut > 0:
                term = term[:cut] + "\n" + term[cut + 1:]
    return term


def plot_enrichment_bars(
    results,
    path: str | Path,
    top_n: int = 3,
    colors: list[str] | None = None,
) -> Path:
    """Draw the top enriched pathways of each gene-set library.

    Args:
        results: Enrichment table with ``Gene_set``, ``Term``, ``Combined Score``
            and ``Adjusted P-value``, already filtered to the significant rows
            and sorted by combined score.
        path: Destination PNG path.
        top_n: Pathways shown per library.
        colors: One colour per gene-set library, assigned in the order the
            libraries first appear. ``None`` uses ``tab10``, which pairs red
            with green within its first four entries; pass
            :data:`src.viz.palettes.COLOR_BLIND_CATEGORICAL` instead.

    Returns:
        The path written to.
    """
    import pandas as pd

    top = pd.concat(
        [
            results[results["Gene_set"] == library].head(top_n)
            for library in results["Gene_set"].unique()
        ]
    )
    top = top.assign(Term=top["Term"].map(_wrap_term))

    plt.figure(figsize=(18, 12))
    libraries = top["Gene_set"].unique()
    hues = colors[: len(libraries)] if colors else sns.color_palette("tab10", len(libraries))
    palette = dict(zip(libraries, hues))
    axis = sns.barplot(x="Combined Score", y="Term", hue="Gene_set", data=top, palette=palette)

    plt.title(
        "Top3 significant enrichment pathway for each database (Adjust p-value < 0.05)",
        fontweight="bold",
        size=20,
    )
    plt.xlabel("Combined Score", fontweight="bold", size=15)
    plt.ylabel("Pathway", fontweight="bold", size=15)
    plt.legend(bbox_to_anchor=(0, -0.1), loc="center left", ncol=4, fontsize="x-large")
    plt.xlim(right=2000)
    plt.xticks(size=15)
    plt.yticks(size=15)

    terms = list(top["Term"])
    for _, row in top.iterrows():
        axis.text(
            row["Combined Score"] + 10,
            terms.index(row["Term"]),
            f"{row['Combined Score']:.1f} (Adj. p: {row['Adjusted P-value']:.4g})",
            va="center",
            ha="left",
            fontsize=12,
        )

    plt.tight_layout()
    return _save(path)


def add_panel_label(
    source: str | Path,
    label: str,
    path: str | Path,
    strip_fraction: float = 0.06,
) -> Path:
    """Write a copy of a saved figure with a panel letter above it.

    The label is placed in a strip added to the top of the image rather than
    drawn over the plot, so no pixel of the original figure is covered. The
    strip takes its colour from the source image's top-left pixel, which makes
    the join invisible on the white backgrounds matplotlib writes.

    Args:
        source: PNG written by one of the plotting functions.
        label: Panel letter as it should appear, for example ``"(a)"``.
        path: Destination PNG path.
        strip_fraction: Strip height as a fraction of the image width. The font
            is sized to fill it, so this sets the letter size too.

    Returns:
        The path written to.
    """
    from matplotlib import font_manager
    from PIL import Image, ImageDraw, ImageFont

    source_image = Image.open(source).convert("RGB")
    strip = max(24, round(source_image.width * strip_fraction))
    font = ImageFont.truetype(
        font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans", weight="bold")),
        size=round(strip * 0.7),
    )

    labelled = Image.new(
        "RGB", (source_image.width, source_image.height + strip), source_image.getpixel((0, 0))
    )
    labelled.paste(source_image, (0, strip))
    ImageDraw.Draw(labelled).text((strip // 2, strip // 2), label, font=font, fill=(0, 0, 0), anchor="lm")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    labelled.save(path)
    logger.info("Wrote %s", display_path(path))
    return path
