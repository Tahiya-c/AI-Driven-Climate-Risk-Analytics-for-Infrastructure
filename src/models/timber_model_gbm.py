"""LightGBM (falling back to XGBoost) regressor builders and Optuna search space.

Tree ensembles need no feature scaling, so these are returned as bare
estimators rather than wrapped in a StandardScaler pipeline like the MLP
in ``src/models/timber_model.py`` -- they still satisfy the same fit/predict
interface, so nothing else in the codebase needs to know the difference.
"""

from __future__ import annotations

from typing import Any

try:
    import lightgbm as lgb

    GBM_BACKEND: str | None = "lightgbm"
except ImportError:  # pragma: no cover - exercised only when lightgbm is absent
    lgb = None
    try:
        import xgboost as xgb

        GBM_BACKEND = "xgboost"
    except ImportError:
        xgb = None
        GBM_BACKEND = None


# Best hyperparameters found by ``python -m src.models.timber_tuning --model gbm``
# (Optuna TPE, blocked time-series CV on the training years only -- see
# docs/timber_model_tuning.md). Used by build_pipeline(..., tuned=True) in
# src/models/timber_model.py. Re-run the tuner and paste its output here
# after any change to the data or FEATURE_COLUMNS.
TUNED_GBM_PARAMS: dict[str, Any] = {
    "n_estimators": 1414,
    "learning_rate": 0.06940770580816218,
    "max_depth": 6,
    "min_child_samples": 19,
    "subsample": 0.9312056899958331,
    "colsample_bytree": 0.952467072361228,
    "reg_alpha": 0.0029959017791319023,
    "reg_lambda": 0.05052144275934393,
    "num_leaves": 19,
    "subsample_freq": 1,
    "linear_tree": True,
}


def suggest_gbm_params(trial) -> dict[str, Any]:
    """Optuna search space for the LightGBM surrogate.

    ``linear_tree`` fits a linear model in every leaf instead of a constant
    (Shi et al., 2019, "Gradient Boosting with Piece-Wise Linear Regression
    Trees"). Plain constant-leaf trees cannot extrapolate past the largest
    age seen in training, which is exactly what the time-based test split
    asks for, so it is offered as a searchable option rather than assumed.
    """
    require_gbm_backend()
    params: dict[str, Any] = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 2000, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 5e-3, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 10),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 60),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }
    if GBM_BACKEND == "lightgbm":
        params["num_leaves"] = trial.suggest_int("num_leaves", 4, 64, log=True)
        params["subsample_freq"] = 1
        params["linear_tree"] = trial.suggest_categorical("linear_tree", [False, True])
    else:
        params["min_child_weight"] = params.pop("min_child_samples")
    return params


def require_gbm_backend() -> str:
    if GBM_BACKEND is None:
        raise ImportError(
            "Neither lightgbm nor xgboost is installed. Install one with "
            "`pip install lightgbm` (preferred) or `pip install xgboost` "
            "to use --model gbm."
        )
    return GBM_BACKEND


def build_gbm_regressor(random_state: int = 42, **overrides: Any):
    """Point-estimate GBM regressor (mean squared error objective)."""
    backend = require_gbm_backend()
    if backend == "lightgbm":
        params = {
            "n_estimators": 400,
            "learning_rate": 0.05,
            "max_depth": 5,
            "num_leaves": 31,
            "random_state": random_state,
            "verbose": -1,
        }
        params.update(overrides)
        return lgb.LGBMRegressor(**params)

    params = {
        "n_estimators": 400,
        "learning_rate": 0.05,
        "max_depth": 5,
        "random_state": random_state,
    }
    params.update(overrides)
    return xgb.XGBRegressor(**params)


def build_gbm_quantile_regressor(alpha: float, random_state: int = 42):
    """GBM regressor for one quantile (e.g. alpha=0.1 for the 10th percentile)."""
    backend = require_gbm_backend()
    if backend == "lightgbm":
        return lgb.LGBMRegressor(
            objective="quantile",
            alpha=alpha,
            n_estimators=400,
            learning_rate=0.05,
            max_depth=5,
            num_leaves=31,
            random_state=random_state,
            verbose=-1,
        )
    return xgb.XGBRegressor(
        objective="reg:quantileerror",
        quantile_alpha=alpha,
        n_estimators=400,
        learning_rate=0.05,
        max_depth=5,
        random_state=random_state,
    )
