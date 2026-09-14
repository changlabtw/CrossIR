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
from bayes_opt import BayesianOptimization
from catboost import CatBoostClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

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


def build_voting_classifier(estimators: list[tuple[str, object]]) -> VotingClassifier:
    """Combine fitted or unfitted estimators into a soft-voting ensemble.

    Note:
        ``VotingClassifier.fit`` clones every estimator and refits it, so the
        result depends only on the estimators' hyperparameters and the training
        data. The legacy ``EnsembleModel`` loaded already-fitted pickles from the
        run folder and passed them here, which had the same effect; building the
        estimators directly from configuration is equivalent and avoids the
        pickle round-trip.

    Args:
        estimators: ``(name, estimator)`` pairs.

    Returns:
        An unfitted soft-voting classifier.
    """
    return VotingClassifier(estimators=estimators, voting="soft")


def _parameter_bounds(algorithm: str) -> dict[str, tuple[float, float]]:
    """Search space for the Bayesian optimisation, per algorithm.

    Reproduced from `scripts/model_training.py` ``run()``. Only the parameters a
    given library supports are included.

    Args:
        algorithm: ``"XGBoost"``, ``"lightGBM"`` or ``"CatBoost"``.

    Returns:
        Mapping of parameter name to ``(low, high)``.
    """
    bounds = {
        "max_depth": (3, 12),
        "learning_rate": (0.01, 0.2),
        "reg_lambda": (0.001, 10),
    }
    if algorithm in ("XGBoost", "lightGBM"):
        bounds["colsample_bytree"] = (0.5, 1)
        bounds["min_child_weight"] = (1, 20)
        bounds["subsample"] = (0.5, 1)
        bounds["reg_alpha"] = (0.001, 10)
    if algorithm in ("XGBoost", "CatBoost"):
        bounds["colsample_bylevel"] = (0.5, 1)
    if algorithm == "XGBoost":
        bounds["gamma"] = (0, 5)
    return bounds


def _best_iteration(model, algorithm: str) -> int:
    """Read the early-stopping iteration from a fitted model.

    Args:
        model: Fitted estimator.
        algorithm: Which library it came from.

    Returns:
        The iteration at which training stopped improving.
    """
    if algorithm == "XGBoost":
        return model.best_iteration
    if algorithm == "lightGBM":
        return model.best_iteration_
    return model.tree_count_


def tune_hyperparameters(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    algorithm: str,
    init_points: int = 15,
    n_iter: int = 25,
    n_splits: int = 5,
    max_estimators: int = 1000,
    early_stopping_rounds: int = 50,
    random_state: int = RANDOM_STATE,
) -> dict:
    """Search for hyperparameters that maximise cross-validated AUC.

    Each candidate is scored by 5-fold stratified cross-validation with early
    stopping, and the number of trees reported for the winning candidate is the
    mean iteration at which its folds stopped improving.

    This is **not** on the default path. The published models were tuned once and
    their parameters committed to ``configs/hyperparameters.yaml``; re-tuning is
    provided so the search can be repeated, not so it runs on every execution.

    Args:
        X_train: Training feature matrix.
        y_train: Training target.
        algorithm: Which boosting library to tune.
        init_points: Random probes before the surrogate model takes over.
        n_iter: Guided iterations after the random probes.
        n_splits: Cross-validation folds.
        max_estimators: Tree cap during the search.
        early_stopping_rounds: Patience within each fold.
        random_state: Seed for the optimiser and the folds.

    Returns:
        The best parameters found, with ``max_depth`` and ``n_estimators`` as
        integers.
    """
    integer_params = ("max_depth", "n_estimators")
    history: list[list] = []

    def objective(**candidate):
        candidate = {
            key: int(value) if isinstance(value, float) and key in integer_params else value
            for key, value in candidate.items()
        }
        model = build_classifier(algorithm, y_train, candidate)
        model.set_params(n_estimators=max_estimators, early_stopping_rounds=early_stopping_rounds)
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

        scores, iterations = [], []
        for train_index, validation_index in cv.split(X_train, y_train):
            X_fold, X_validation = X_train.iloc[train_index], X_train.iloc[validation_index]
            y_fold, y_validation = y_train.iloc[train_index], y_train.iloc[validation_index]

            if algorithm == "XGBoost":
                model.fit(X_fold, y_fold, eval_set=[(X_validation, y_validation)], verbose=False)
            else:
                model.fit(X_fold, y_fold, eval_set=[(X_validation, y_validation)])

            iterations.append(_best_iteration(model, algorithm))
            scores.append(roc_auc_score(y_validation, model.predict_proba(X_validation)[:, 1]))

        mean_score = float(np.mean(scores))
        history.append([mean_score, int(np.mean(iterations)) if iterations else max_estimators])
        return mean_score

    optimizer = BayesianOptimization(
        f=objective, pbounds=_parameter_bounds(algorithm), random_state=random_state
    )
    optimizer.maximize(init_points=init_points, n_iter=n_iter)

    best = {
        key: int(value) if key in integer_params else value
        for key, value in optimizer.max["params"].items()
    }
    # The tree count comes from the best-scoring candidate's mean stopping point.
    best["n_estimators"] = max(history, key=lambda row: row[0])[1]
    logger.info("%s tuned: %s", algorithm, best)
    return best
