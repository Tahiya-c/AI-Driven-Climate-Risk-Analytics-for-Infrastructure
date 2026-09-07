"""Quantile regression for replacement-year uncertainty bands.

Only implemented for the GBM backend: LightGBM/XGBoost support a native
quantile loss, giving a genuine conditional-quantile estimate per tree
split. scikit-learn's MLPRegressor has no equivalent built in, so it is
not offered here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.timber.features import FEATURE_COLUMNS, TARGET_COLUMN, build_feature_matrix
from src.timber.model import pile_based_split, time_based_split
from src.timber.model_gbm import build_gbm_quantile_regressor
from src.timber.scenarios import build_pile_trajectories

QUANTILES: tuple[float, ...] = (0.1, 0.5, 0.9)


@dataclass
class QuantileModels:
    models: dict[float, object]
    split_strategy: str
    feature_columns: list[str]


def train_quantile_models(
    frame: pd.DataFrame,
    quantiles: tuple[float, ...] = QUANTILES,
    split_strategy: str = "time",
    feature_columns: list[str] | None = None,
    random_state: int = 42,
) -> QuantileModels:
    """Fit one GBM quantile regressor per quantile against remaining_diameter_mm."""
    feature_columns = feature_columns if feature_columns is not None else FEATURE_COLUMNS
    features = build_feature_matrix(frame, feature_columns=feature_columns)
    target = frame[TARGET_COLUMN]

    if split_strategy == "time":
        train_idx, _test_idx = time_based_split(frame)
    elif split_strategy == "pile":
        train_idx, _test_idx = pile_based_split(frame)
    else:
        raise ValueError(f"Unknown split_strategy: {split_strategy!r}")

    models = {}
    for alpha in quantiles:
        model = build_gbm_quantile_regressor(alpha=alpha, random_state=random_state)
        model.fit(features.loc[train_idx], target.loc[train_idx])
        models[alpha] = model

    return QuantileModels(
        models=models, split_strategy=split_strategy, feature_columns=feature_columns
    )


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, alpha: float) -> float:
    """Quantile ("pinball") loss for one quantile's predictions."""
    diff = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    return float(np.mean(np.maximum(alpha * diff, (alpha - 1) * diff)))


def replacement_year_bands(
    frame: pd.DataFrame, quantile_models: QuantileModels, asset_code: str
) -> pd.DataFrame:
    """Per-scenario replacement-year estimate at each fitted quantile.

    Reuses ``build_pile_trajectories`` unchanged (including its
    monotonicity enforcement) once per quantile model, so a low diameter
    quantile (a pile deteriorating faster than typical) naturally yields
    the *early* end of the replacement-year band and a high diameter
    quantile yields the *late* end.
    """
    rows = []
    for alpha, model in sorted(quantile_models.models.items()):
        trajectories, _split_year = build_pile_trajectories(frame, model, asset_code)
        for scenario, trajectory in trajectories.items():
            rows.append(
                {
                    "scenario": scenario,
                    "quantile": alpha,
                    "replacement_year": trajectory.replacement_year,
                }
            )
    table = pd.DataFrame(rows)
    return table.pivot(index="scenario", columns="quantile", values="replacement_year")
