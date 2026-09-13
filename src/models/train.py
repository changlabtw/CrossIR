"""Classifier construction and training.

Reimplements the training half of the legacy project's
`scripts/model_training.py` (`InsulinResistancePredictor`) as functions. The
evaluation half lives in :mod:`src.models.evaluate`.

Class imbalance is handled by weighting rather than resampling: each library is
given its own form of the same positive-to-negative ratio, computed on the
training split.
"""

import logging

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

TARGET = "IR"
"""Binary target column: ``True`` when HOMA-IR exceeds the cut-off."""

TEST_SIZE = 0.3
"""Share of participants held out for testing."""

RANDOM_STATE = 30
"""Seed for the train/test split and the cross-validation folds.

Note:
    This seeds the *splitting*, not the estimators. The boosters are constructed
    without an explicit seed, exactly as the legacy code constructs them, because
    their library defaults are deterministic and the published metrics are only
    reachable that way. See ``docs/validation-log.md``.
"""

DECISION_THRESHOLD = 0.5
"""Probability above which a participant is predicted insulin resistant."""


def split_features_target(df: pl.DataFrame, target: str = TARGET) -> tuple[pd.DataFrame, pd.Series]:
    """Separate the feature matrix from the target column.

    Args:
        df: Table containing the features and the target.
        target: Name of the target column.

    Returns:
        The feature matrix and the target as an integer series.
    """
    frame = df.to_pandas()
    return frame.drop(columns=[target]), frame[target].astype(int)


def stratified_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple:
    """Split into training and test sets, preserving the class balance.

    Args:
        X: Feature matrix.
        y: Target vector.
        test_size: Share held out for testing.
        random_state: Seed controlling which participants land in which split.

    Returns:
        ``(X_train, X_test, y_train, y_test)``.
    """
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)


def class_weights(y_train: pd.Series, algorithm: str):
    """Compute the class-imbalance correction in the form the library expects.

    Args:
        y_train: Training target vector.
        algorithm: ``"XGBoost"``, ``"lightGBM"`` or ``"CatBoost"``.

    Returns:
        A scalar ratio for XGBoost, a mapping for LightGBM, or a two-element
        list for CatBoost.

    Raises:
        ValueError: If the algorithm is not recognised.
    """
    negative = int((y_train == 0).sum())
    positive = int((y_train == 1).sum())

    if algorithm == "XGBoost":
        return negative / positive
    if algorithm == "lightGBM":
        return {0: 1, 1: negative / positive}
    if algorithm == "CatBoost":
        total = negative + positive
        return [total / (2 * negative), total / (2 * positive)]
    raise ValueError(f"Unknown algorithm: {algorithm}")


def build_classifier(algorithm: str, y_train: pd.Series, params: dict | None = None):
    """Create an unfitted classifier with class weighting applied.

    Note:
        No ``random_state`` is passed. The boosting libraries have deterministic
        default seeds, and supplying one explicitly is not necessarily a no-op --
        LightGBM derives ``feature_fraction_seed``, ``bagging_seed`` and
        ``data_random_seed`` from ``seed`` -- so the estimators are constructed
        exactly as the legacy code constructs them.

    Args:
        algorithm: ``"XGBoost"``, ``"lightGBM"`` or ``"CatBoost"``.
        y_train: Training target, used for the class weights.
        params: Tuned hyperparameters, or ``None`` for library defaults.

    Returns:
        An unfitted estimator.

    Raises:
        ValueError: If the algorithm is not recognised.
    """
    params = params or {}
    weights = class_weights(y_train, algorithm)

    if algorithm == "XGBoost":
        # The legacy code also passed `use_label_encoder=False` and
        # `verbose=False`; xgboost reports both as unused and ignores them.
        return xgb.XGBClassifier(
            **params,
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=weights,
        )
    if algorithm == "lightGBM":
        return lgb.LGBMClassifier(
            **params,
            verbose=-1,
            force_row_wise=True,
            eval_metric="logloss",
            class_weight=weights,
        )
    if algorithm == "CatBoost":
        return CatBoostClassifier(
            **params,
            verbose=False,
            eval_metric="Logloss",
            class_weights=weights,
            allow_writing_files=False,
        )
    raise ValueError(f"Unknown algorithm: {algorithm}")


def train_classifier(
    df: pl.DataFrame,
    algorithm: str = "XGBoost",
    target: str = TARGET,
    params: dict | None = None,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> dict:
    """Split, fit and score one classifier on one feature set.

    Args:
        df: Feature table including the target column.
        algorithm: Which boosting library to use.
        target: Name of the target column.
        params: Tuned hyperparameters, or ``None`` for library defaults.
        test_size: Share held out for testing.
        random_state: Seed for the split.

    Returns:
        Mapping with the fitted ``model``, the four split frames
        (``X_train``, ``X_test``, ``y_train``, ``y_test``) and ``preds``, the
        predicted probability of insulin resistance on the test set.
    """
    X, y = split_features_target(df, target)
    X_train, X_test, y_train, y_test = stratified_split(X, y, test_size, random_state)
    logger.info(
        "%s on %d features: %d training / %d test rows",
        algorithm,
        X.shape[1],
        len(X_train),
        len(X_test),
    )

    model = build_classifier(algorithm, y_train, params)
    model.fit(X_train, y_train)
    preds = model.predict_proba(X_test)[:, 1]

    return {
        "model": model,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "preds": preds,
    }


def split_frames(run: dict, target: str = TARGET) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Rebuild the training and test matrices the legacy runs exported.

    Args:
        run: Result of :func:`train_classifier`.
        target: Name of the target column.

    Returns:
        The training matrix, and the test matrix with a ``preds`` column
        appended -- the same layout as the legacy ``training_data.csv`` and
        ``testing_data.csv``.
    """
    training = pl.from_pandas(pd.concat([run["X_train"], run["y_train"]], axis=1))
    testing = pl.from_pandas(pd.concat([run["X_test"], run["y_test"]], axis=1))
    testing = testing.with_columns(preds=pl.Series(np.asarray(run["preds"])))
    return training, testing
