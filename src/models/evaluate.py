"""Classification metrics.

Reimplements the evaluation half of the legacy project's
`scripts/model_training.py` (`InsulinResistancePredictor.evaluate_model`) as
functions.

Specificity, NPV and Youden's index are derived from the confusion matrix
directly rather than through a library helper, matching the legacy arithmetic
exactly. Every value is rounded to three decimals, as the legacy code rounds
before writing its JSON.
"""

import logging

import numpy as np
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
"""Rounding applied to every reported metric, matching the legacy output."""


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
