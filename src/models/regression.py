"""Continuous HOMA-IR regression.

Reimplements the legacy project's `6_regression.ipynb` as functions. This is the
preliminary analysis behind thesis Table `regression2`: before HOMA-IR is
thresholded into a binary label, it asks how well the index itself can be
predicted from routine clinical variables.

The target is the natural logarithm of HOMA-IR, so every metric reported here is
on the log scale. Exponentiating an error gives a multiplicative factor on the
HOMA-IR scale: an MAE of 0.337 means a typical prediction is out by a factor of
about 1.40.
"""

import logging

import numpy as np
import polars as pl
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from xgboost import XGBRegressor

logger = logging.getLogger(__name__)

TARGET = "HOMA-IR"
"""Column the regressors predict, on a natural-log scale."""

CORE_FEATURES = [
    "AGE",
    "BMI",
    "FASTING_GLUCOSE",
    "HBA1C",
    "HDL_C",
    "SEX",
    "TG",
    "T_CHO",
]
"""Routinely measured variables shared by both feature sets."""

ADDITIONAL_FEATURES = [
    "BUN",
    "CREATININE",
    "URIC_ACID",
    "LDL_C",
    "SGOT",
    "SGPT",
    "MAP",
    "BODY_WAISTLINE",
]
"""Further blood chemistry and anthropometry, added only to the extended set."""

BASELINE_FEATURES = CORE_FEATURES + ["RACE"]
"""Nine variables available in every cohort."""

EXTENDED_FEATURES = CORE_FEATURES + ADDITIONAL_FEATURES + ["RACE"]
"""Seventeen variables.

Note:
    ``RACE`` is kept in last position in both sets, matching the legacy column
    order. Column order is not cosmetic for tree ensembles -- it decides how ties
    between equally good splits are broken -- so it is preserved exactly.
"""

FEATURE_SETS = {"KNHANES_9": BASELINE_FEATURES, "KNHANES_17": EXTENDED_FEATURES}
"""The two feature sets compared in the thesis table, named by their size."""


def build_regressors(random_state: int = 30) -> dict:
    """Create the six regressors compared in the thesis, with default hyperparameters.

    No tuning is performed at this stage; the point of the comparison is the
    difference between model families and between feature sets, not the best
    attainable fit.

    Args:
        random_state: Seed for the four estimators that accept one. The two
            linear models are deterministic.

    Returns:
        Ordered mapping of model name to an unfitted estimator.
    """
    return {
        "LinearRegression": LinearRegression(),
        "Polynomial_Regression": make_pipeline(PolynomialFeatures(2), LinearRegression()),
        "RandomForest": RandomForestRegressor(random_state=random_state),
        # The legacy code also passed `verbose=False` here; XGBoost reports it as an
        # unused parameter and ignores it, so it is dropped. Gate G3 confirms the
        # metrics are unchanged.
        "XGBoost": XGBRegressor(random_state=random_state),
        "LightGBM": LGBMRegressor(verbose=-1, random_state=random_state),
        "CatBoost": CatBoostRegressor(verbose=0, random_state=random_state),
    }


def cross_validate_regressor(model, X, y: np.ndarray, cv: KFold) -> dict:
    """Fit and score one regressor across the folds of a cross-validation split.

    Args:
        model: Unfitted estimator. It is refitted on each fold.
        X: Feature matrix as a pandas DataFrame.
        y: Target vector.
        cv: Cross-validation splitter.

    Returns:
        Mapping with the mean ``r2``, ``mae`` and ``rmse`` across folds.
    """
    r2, mae, rmse = [], [], []
    for train_index, validation_index in cv.split(X):
        X_train, X_validation = X.iloc[train_index], X.iloc[validation_index]
        y_train, y_validation = y[train_index], y[validation_index]

        model.fit(X_train, y_train)
        predictions = model.predict(X_validation)

        r2.append(r2_score(y_validation, predictions))
        mae.append(mean_absolute_error(y_validation, predictions))
        rmse.append(root_mean_squared_error(y_validation, predictions))

    return {"r2": float(np.mean(r2)), "mae": float(np.mean(mae)), "rmse": float(np.mean(rmse))}


def evaluate_feature_sets(
    df: pl.DataFrame,
    feature_sets: dict[str, list[str]] | None = None,
    models: dict | None = None,
    n_splits: int = 5,
    random_state: int = 30,
) -> pl.DataFrame:
    """Cross-validate every regressor on every feature set.

    The same splitter is reused across feature sets, so each model sees the same
    folds throughout.

    Args:
        df: Cohort table containing ``HOMA-IR``, the feature columns and a
            ``RACE`` column.
        feature_sets: Mapping of set name to column list. Defaults to
            :data:`FEATURE_SETS`.
        models: Mapping of model name to estimator. Defaults to
            :func:`build_regressors`.
        n_splits: Number of cross-validation folds.
        random_state: Seed for the splitter and the estimators.

    Returns:
        One row per model and feature set, with columns ``model``,
        ``feature_set``, ``n_features``, ``r2``, ``mae`` and ``rmse``.
    """
    feature_sets = feature_sets or FEATURE_SETS
    models = models or build_regressors(random_state)
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    y = np.log(df[TARGET]).to_numpy().ravel()

    rows = []
    for set_name, columns in feature_sets.items():
        X = df.select(columns).to_pandas()
        for model_name, model in models.items():
            scores = cross_validate_regressor(model, X, y, cv)
            logger.info(
                "%s | %s: R2=%.3f MAE=%.3f RMSE=%.3f",
                set_name,
                model_name,
                scores["r2"],
                scores["mae"],
                scores["rmse"],
            )
            rows.append(
                {
                    "model": model_name,
                    "feature_set": set_name,
                    "n_features": len(columns),
                    **scores,
                }
            )

    return pl.DataFrame(rows)
