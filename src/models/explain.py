"""SHAP-based model interpretation.

Reimplements `InsulinResistancePredictor.feature_importance` and
`plot_shap_graph` from the legacy project's `scripts/model_training.py`, together
with the feature-name prettifier from `5_analysis.ipynb` cell [53].

SHAP attributes each prediction to the features that produced it, so the ranking
here is over *contributions to predictions*, not over the model's internal split
statistics. The two disagree: the twenty features the final model was retrained
on come from this ranking, not from ``feature_importances_``
(``docs/audit.md`` F30).
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import shap

from src.data.io import display_path

logger = logging.getLogger(__name__)

TRANSFORM_SUFFIXES = ["log", "sqrt", "square"]
"""Unary transforms whose suffix is rewritten as a function call for display."""

PLOT_RANDOM_STATE = 30
"""Seed for the SHAP summary plot's point shuffling. See :func:`plot_shap_summary`."""


def shap_values(model, X) -> np.ndarray:
    """Compute SHAP values for a fitted tree model.

    Prefers `fasttreeshap`, which is substantially quicker on wide matrices, and
    falls back to `shap` when it cannot read the model. A model whose
    contributions do not sum exactly to its output is retried with the
    additivity check disabled, as the legacy code does.

    Args:
        model: Fitted tree-based classifier.
        X: Feature matrix to explain. Pass the features only -- see F28.

    Returns:
        SHAP values with shape ``(n_samples, n_features)``.
    """
    try:
        import fasttreeshap

        explainer = fasttreeshap.TreeExplainer(model)
        logger.info("Explaining with fasttreeshap")
    except (ImportError, UnicodeDecodeError):
        explainer = shap.TreeExplainer(model)
        logger.info("Explaining with shap")

    try:
        explanation = explainer(X)
    except Exception:
        explanation = explainer(X, check_additivity=False)

    logger.info("SHAP values: %s", explanation.values.shape)
    return explanation.values


def rank_features(values: np.ndarray, feature_names: list[str], top_n: int | None = None) -> list[str]:
    """Order features by mean absolute SHAP value, most influential first.

    Args:
        values: SHAP values, shape ``(n_samples, n_features)``.
        feature_names: Column names in the order they were explained.
        top_n: Keep only this many, or all of them when ``None``.

    Returns:
        Feature names, most influential first.
    """
    mean_absolute = np.abs(values).mean(axis=0)
    order = np.argsort(-mean_absolute)
    ranked = [feature_names[index] for index in order]
    return ranked[:top_n] if top_n else ranked


def prettify_feature_names(names: list[str]) -> list[str]:
    """Rewrite generated feature names into readable mathematical notation.

    ``BMI_mul_TG`` becomes ``BMI×TG``, ``TG_div_T_CHO`` becomes ``TG/T_CHO``, and
    ``FASTING_GLUCOSE_log`` becomes ``log(FASTING_GLUCOSE)``.

    Args:
        names: Raw column names.

    Returns:
        Display names, in the same order.
    """
    pretty = [name.replace("_mul_", "×").replace("_div_", "/") for name in names]
    for suffix in TRANSFORM_SUFFIXES:
        pretty = [
            f"{suffix}(" + name.replace(f"_{suffix}", "") + ")" if suffix in name else name
            for name in pretty
        ]
    return pretty


def plot_shap_summary(
    values: np.ndarray,
    X,
    path: str | Path,
    pretty: bool = True,
    max_display: int = 20,
    random_state: int | None = PLOT_RANDOM_STATE,
) -> Path:
    """Draw the SHAP summary (beeswarm) plot.

    Note:
        `shap.summary_plot` shuffles the scatter points through the global numpy
        random number generator, so that overlapping points of different colours
        do not stack in a systematic order. Without a fixed seed the same data
        therefore renders differently every time -- two renders of one array
        differ in roughly 2% of their pixels. Seeding here makes the figure
        reproducible; it does not affect any value, only which point is drawn on
        top of which (``docs/audit.md`` F31).

    Args:
        values: SHAP values, shape ``(n_samples, n_features)``.
        X: The feature matrix those values explain.
        path: Destination PNG path.
        pretty: Rewrite feature names with :func:`prettify_feature_names`.
        max_display: How many features to show.
        random_state: Seed for the point shuffling. ``None`` leaves the global
            generator alone and the figure becomes irreproducible.

    Returns:
        The path written to.
    """
    if random_state is not None:
        np.random.seed(random_state)
    names = list(X.columns)
    shap.summary_plot(
        values,
        X,
        feature_names=prettify_feature_names(names) if pretty else names,
        max_display=max_display,
        show=False,
    )
    plt.tight_layout()

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path)
    plt.close()
    logger.info("Wrote %s", display_path(path))
    return path
