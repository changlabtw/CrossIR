"""Descriptive figures.

Reimplements the plotting cells of the legacy project's `5_analysis.ipynb`
(cells [17-18], [21-22] and [55]) as module-level functions, producing thesis
Fig. `box-stats`, Fig. `correlation` and Fig. `homa`.

Styling is reproduced from the legacy code so that the output can be compared
against the published figures. A publication style sheet is a Phase 3 task.
"""

import logging
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score, roc_curve

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

HOMAIR_CUTOFF = 2.5
"""Cut-off drawn as a dashed line on the HOMA-IR distributions."""


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


def plot_correlation_matrix(frames: dict[str, pl.DataFrame], path: str | Path) -> Path:
    """Draw a lower-triangular correlation heatmap for each cohort.

    Args:
        frames: Mapping of panel title to cohort table, laid out on a 2x2 grid
            in insertion order.
        path: Destination PNG path.

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
            cmap="coolwarm",
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


def plot_homair_distributions(frames: dict[str, pl.DataFrame], path: str | Path) -> Path:
    """Draw the log HOMA-IR distribution of each cohort with the cut-off marked.

    Args:
        frames: Mapping of panel title to cohort table, each carrying a
            ``HOMA-IR`` column. Laid out on a 2x2 grid with the unused panel
            removed.
        path: Destination PNG path.

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
            color="red",
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
    logger.info("Wrote %s", path)
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
