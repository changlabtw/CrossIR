"""Model evaluation.

Classification metrics, together with the two statistics used to judge
predictions on a cohort that carries no observed label.

Specificity, NPV and Youden's index are derived from the confusion matrix
directly rather than through a library helper, so that every reported quantity
traces back to the same four counts. Values are rounded to three decimals, which
is the precision at which they are reported.
"""

import logging

import numpy as np
import polars as pl
import statsmodels.api as sm
from sklearn.metrics import (
    auc,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
)

logger = logging.getLogger(__name__)

DECISION_THRESHOLD = 0.5
"""Probability above which a participant is predicted insulin resistant."""

METRIC_DECIMALS = 3
"""Decimal places every reported metric is rounded to."""

BOOTSTRAP_RESAMPLES = 1000
"""Resamples drawn for a confidence interval."""

BOOTSTRAP_RANDOM_STATE = 30
"""Seed for the resampling, so a reported interval is reproducible.

The same value as :data:`src.models.train.RANDOM_STATE`, restated here rather
than imported so that scoring a saved model does not pull in the training
module and the three boosting libraries with it.
"""

CALIBRATION_BINS = 10
"""Bins of predicted probability in a reliability diagram.

Quantile bins, so each holds the same number of participants and none comes back
empty at the sparse upper end of the predicted range.
"""


def compute_metrics(
    y_true,
    preds: np.ndarray,
    threshold: float = DECISION_THRESHOLD,
) -> dict:
    """Score predicted probabilities against the observed labels.

    Args:
        y_true: Observed binary labels.
        preds: Predicted probability of the positive class.
        threshold: Probability above which a prediction counts as positive.

    Returns:
        Mapping with ``confusion_matrix`` (``TP``, ``TN``, ``FP``, ``FN``) and
        the nine reported metrics: ``roc_auc``, ``accuracy``,
        ``sensitivity(recall)``, ``specificity``, ``PPV(precision)``, ``NPV``,
        ``f1_score``, ``Youdens_Index`` and ``pr_auc``.
    """
    roc_auc = roc_auc_score(y_true, preds)

    y_pred = (preds > threshold).astype(int)
    matrix = confusion_matrix(y_true, y_pred)
    true_negative, false_positive = matrix[0, 0], matrix[0, 1]
    false_negative, true_positive = matrix[1, 0], matrix[1, 1]

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary"
    )
    specificity = true_negative / (true_negative + false_positive)

    precision_curve, recall_curve, _ = precision_recall_curve(y_true, preds)
    pr_auc = auc(recall_curve, precision_curve)

    return {
        "optimal_threshold": threshold,
        "confusion_matrix": {
            "TP": int(true_positive),
            "TN": int(true_negative),
            "FP": int(false_positive),
            "FN": int(false_negative),
        },
        "roc_auc": round(roc_auc, METRIC_DECIMALS),
        "accuracy": round((true_positive + true_negative) / np.sum(matrix), METRIC_DECIMALS),
        "sensitivity(recall)": round(recall, METRIC_DECIMALS),
        "specificity": round(specificity, METRIC_DECIMALS),
        "PPV(precision)": round(precision, METRIC_DECIMALS),
        "NPV": round(true_negative / (true_negative + false_negative), METRIC_DECIMALS),
        "f1_score": round(f1, METRIC_DECIMALS),
        "Youdens_Index": round(recall + specificity - 1, METRIC_DECIMALS),
        "pr_auc": round(pr_auc, METRIC_DECIMALS),
    }


def model_feature_names(model) -> list[str]:
    """Return the feature names a fitted model expects, in its own order.

    The boosting libraries disagree on the attribute name: scikit-learn's
    convention is ``feature_names_in_``, CatBoost uses ``feature_names_``.

    Args:
        model: Fitted estimator.

    Returns:
        Feature names in the order the model was trained on.

    Raises:
        AttributeError: If the model records no feature names.
    """
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    if hasattr(model, "feature_names_"):
        return list(model.feature_names_)
    raise AttributeError(f"{type(model).__name__} records no feature names")


def score_external(model, df, target: str = "IR", threshold: float = DECISION_THRESHOLD) -> dict:
    """Score a model against a cohort it was not trained on.

    Columns are selected by the model's own feature names, so the external
    cohort's table may hold them in a different order or alongside extra
    columns.

    Args:
        model: Fitted classifier.
        df: Feature table for the external cohort, including the target column.
        target: Name of the target column.
        threshold: Probability above which a prediction counts as positive.

    Returns:
        The output of :func:`compute_metrics`, with the predicted probabilities
        added under ``preds``.
    """
    X = df.select(model_feature_names(model)).to_pandas()
    preds = model.predict_proba(X)[:, 1]
    y_true = df[target].to_numpy().astype(int)

    metrics = compute_metrics(y_true, preds, threshold)
    metrics["preds"] = preds
    metrics["n_samples"] = len(y_true)
    logger.info(
        "External scoring on %d samples: AUC=%.3f", len(y_true), metrics["roc_auc"]
    )
    return metrics


def predict_labels(model, df, threshold: float = DECISION_THRESHOLD) -> np.ndarray:
    """Predict binary labels for a cohort that carries no observed label.

    The unlabelled counterpart of :func:`score_external`. Taiwan Biobank
    measures no fasting insulin, so HOMA-IR and therefore the ``IR`` label
    cannot be computed for it; the model supplies the label instead and no
    performance metric is available.

    Args:
        model: Fitted classifier.
        df: Feature table containing at least the model's own features. Extra
            columns and a different column order are both fine, since columns
            are selected by the model's feature names.
        threshold: Probability above which a participant is labelled insulin
            resistant.

    Returns:
        Array of ``0``/``1`` labels, one per row of ``df``, in row order.
    """
    X = df.select(model_feature_names(model)).to_pandas()
    preds = model.predict_proba(X)[:, 1]
    labels = (preds > threshold).astype(int)
    logger.info(
        "Predicted %d of %d rows positive (%.1f%%)",
        int(labels.sum()),
        len(labels),
        100 * labels.mean(),
    )
    return labels


def contributing_feature_ratio(model, feature_names: list[str], X=None) -> dict:
    """Fraction of the input features the model actually uses.

    The number of features with importance above zero over the number of input
    features, which weighs model performance against how efficiently the feature
    set is used.

    Note:
        Features that are constant across the cohort are excluded from both
        counts. A constant column cannot contribute by construction, so counting
        it as a non-contributing feature penalises the model for a column that
        carries no information. In this project ``RACE`` is constant within any
        single cohort and is the only such column.

    Args:
        model: Fitted estimator exposing ``feature_importances_``.
        feature_names: Column names in the order the model was trained on.
        X: Training matrix used to detect constant columns. When ``None``, no
            column is treated as constant.

    Returns:
        Mapping with ``ratio`` (rounded to three decimals), ``contributing``,
        ``total``, and ``constant_features`` -- the names that were excluded.
    """
    importances = np.asarray(model.feature_importances_)

    constant = []
    if X is not None:
        constant = [name for name in feature_names if X[name].nunique(dropna=False) <= 1]

    considered = [
        index for index, name in enumerate(feature_names) if name not in constant
    ]
    contributing = int((importances[considered] > 0).sum())
    total = len(considered)

    logger.info(
        "CFR: %d/%d contributing (%d constant column(s) excluded)",
        contributing,
        total,
        len(constant),
    )
    return {
        "ratio": round(contributing / total, METRIC_DECIMALS),
        "contributing": contributing,
        "total": total,
        "constant_features": constant,
    }


def cohens_d(x, y) -> float:
    """Standardised difference between the means of two independent samples.

    Used to size the gap between a variable's distribution in the labelled
    cohorts and in the Taiwan Biobank predicted-positive group, where no
    classification metric is available. The pooled standard deviation uses the
    sample estimator (``ddof=1``) in both groups.

    Args:
        x: First sample.
        y: Second sample.

    Returns:
        Cohen's *d*. Positive when ``x`` has the larger mean. By the usual
        convention, magnitudes near 0.2, 0.5 and 0.8 are small, medium and large.
    """
    x, y = np.asarray(x), np.asarray(y)
    pooled_variance = (
        (len(x) - 1) * x.std(ddof=1) ** 2 + (len(y) - 1) * y.std(ddof=1) ** 2
    ) / (len(x) + len(y) - 2)
    return (x.mean() - y.mean()) / np.sqrt(pooled_variance)


def bootstrap_metrics(
    y_true,
    preds: np.ndarray,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    random_state: int = BOOTSTRAP_RANDOM_STATE,
    threshold: float = DECISION_THRESHOLD,
) -> pl.DataFrame:
    """Percentile confidence intervals for every metric, by resampling the test set.

    Participants are drawn with replacement and :func:`compute_metrics` is
    recomputed on each resample, so an interval is built from exactly the
    quantities the study reports. Nothing is refitted: the model's predicted
    probabilities are resampled alongside the labels, which is what makes this
    cheap enough to run on every reported row.

    Note:
        The interval covers sampling variation in the *test set* only. It says
        nothing about variation from a different train/test split, a different
        hyperparameter search, or a different cohort.

    Args:
        y_true: Observed binary labels.
        preds: Predicted probability of the positive class.
        n_resamples: Resamples to draw.
        random_state: Seed for the resampling, so the interval is reproducible.
        threshold: Probability above which a prediction counts as positive.

    Returns:
        One row per metric with ``metric``, ``estimate`` on the full test set,
        and the 2.5th and 97.5th percentiles as ``ci_lower`` and ``ci_upper``.
    """
    y_true = np.asarray(y_true)
    preds = np.asarray(preds)

    point = compute_metrics(y_true, preds, threshold)
    names = [key for key, value in point.items() if isinstance(value, float) and key != "optimal_threshold"]

    generator = np.random.default_rng(random_state)
    draws = {name: [] for name in names}
    for _ in range(n_resamples):
        index = generator.integers(0, len(y_true), len(y_true))
        resampled = compute_metrics(y_true[index], preds[index], threshold)
        for name in names:
            draws[name].append(resampled[name])

    logger.info("Bootstrapped %d metrics over %d resamples", len(names), n_resamples)
    return pl.DataFrame(
        {
            "metric": names,
            "estimate": [point[name] for name in names],
            "ci_lower": [round(float(np.percentile(draws[name], 2.5)), METRIC_DECIMALS) for name in names],
            "ci_upper": [round(float(np.percentile(draws[name], 97.5)), METRIC_DECIMALS) for name in names],
        }
    )


def calibration_table(y_true, preds: np.ndarray, n_bins: int = CALIBRATION_BINS) -> pl.DataFrame:
    """Group predictions into bins and compare each bin's mean to what happened.

    A reliability diagram plots the two columns against each other: a
    well-calibrated model puts them on the diagonal, so that a predicted
    probability of 0.3 is borne out in about 30% of those participants.

    Args:
        y_true: Observed binary labels.
        preds: Predicted probability of the positive class.
        n_bins: Number of quantile bins.

    Returns:
        One row per bin with ``bin``, ``mean_predicted``, ``observed``, ``n``.
    """
    y_true = np.asarray(y_true).astype(float)
    preds = np.asarray(preds)

    edges = np.quantile(preds, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    assignment = np.digitize(preds, edges[1:-1])

    rows = []
    for index in range(n_bins):
        selected = assignment == index
        if not selected.any():
            continue
        rows.append(
            {
                "bin": index + 1,
                "mean_predicted": float(preds[selected].mean()),
                "observed": float(y_true[selected].mean()),
                "n": int(selected.sum()),
            }
        )
    return pl.DataFrame(rows)


def calibration_summary(y_true, preds: np.ndarray) -> dict:
    """Brier score with the calibration intercept and slope.

    The intercept and slope come from a logistic regression of the outcome on
    the logit of the prediction. A model whose probabilities mean what they say
    has intercept 0 and slope 1; a slope below 1 marks predictions that are too
    extreme in both directions, and a non-zero intercept marks a systematic
    over- or under-estimate of risk.

    Args:
        y_true: Observed binary labels.
        preds: Predicted probability of the positive class.

    Returns:
        Mapping with ``brier_score``, ``calibration_intercept`` and
        ``calibration_slope``.
    """
    y_true = np.asarray(y_true).astype(float)
    preds = np.clip(np.asarray(preds), 1e-12, 1 - 1e-12)

    logit = np.log(preds / (1 - preds))
    fitted = sm.Logit(y_true, sm.add_constant(logit)).fit(disp=0)

    return {
        "brier_score": round(float(np.mean((preds - y_true) ** 2)), METRIC_DECIMALS),
        "calibration_intercept": round(float(fitted.params[0]), METRIC_DECIMALS),
        "calibration_slope": round(float(fitted.params[1]), METRIC_DECIMALS),
    }
