"""SHAP-based model interpretation.

SHAP attributes each prediction to the features that produced it, so the ranking
here is over *contributions to predictions*, not over the model's internal split
statistics. The two disagree from the third feature onwards, so which of them a
shortlist came from is worth stating: the twenty features the final model is
retrained on come from this ranking, not from ``feature_importances_``.
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
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
    additivity check disabled.

    Args:
        model: Fitted tree-based classifier.
        X: Feature matrix to explain. Pass the features only: columns the model
            never split on receive zero attribution, so extra columns are
            harmless but meaningless.

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


COLOR_BLIND_CMAP = LinearSegmentedColormap.from_list(
    "okabe_ito_blue_orange", ["#0072B2", "#E69F00"]
)
"""Colour-blind-friendly replacement for the default red-to-blue SHAP colour map.

Built from two Okabe-Ito hues, blue for low feature values and orange for high.
Red and blue are hard to separate under the common forms of colour vision
deficiency; blue and orange stay distinguishable under all three, and the pair
keeps the two-ended low-to-high reading the default map has.
"""


def plot_shap_summary(
    values: np.ndarray,
    X,
    path: str | Path,
    pretty: bool = True,
    max_display: int = 20,
    random_state: int | None = PLOT_RANDOM_STATE,
    cmap=None,
) -> Path:
    """Draw the SHAP summary (beeswarm) plot.

    Note:
        `shap.summary_plot` shuffles the scatter points through the global numpy
        random number generator, so that overlapping points of different colours
        do not stack in a systematic order. Without a fixed seed the same data
        therefore renders differently every time -- two renders of one array
        differ in roughly 2% of their pixels. Seeding here makes the figure
        reproducible; it does not affect any value, only which point is drawn on
        top of which.

    Args:
        values: SHAP values, shape ``(n_samples, n_features)``.
        X: The feature matrix those values explain.
        path: Destination PNG path.
        pretty: Rewrite feature names with :func:`prettify_feature_names`.
        max_display: How many features to show.
        random_state: Seed for the point shuffling. ``None`` leaves the global
            generator alone and the figure becomes irreproducible.
        cmap: Colour map for the feature-value scale. ``None`` keeps the library
            default; pass :data:`COLOR_BLIND_CMAP` for the accessible version.

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
        **({} if cmap is None else {"cmap": cmap}),
    )
    plt.tight_layout()

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path)
    plt.close()
    logger.info("Wrote %s", display_path(path))
    return path
