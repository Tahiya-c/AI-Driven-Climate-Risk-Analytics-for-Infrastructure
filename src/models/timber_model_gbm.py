"""LightGBM (falling back to XGBoost) regressor builders.

Tree ensembles need no feature scaling, so these are returned as bare
estimators rather than wrapped in a StandardScaler pipeline like the MLP
in ``src/timber/model.py`` -- they still satisfy the same fit/predict
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
