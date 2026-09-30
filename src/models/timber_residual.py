"""Residual-correction surrogate.

Instead of predicting the deterministic model's raw remaining_diameter_mm
directly, this trains a model to predict *the correction needed* to bring
that raw output into a physically valid range (see
``add_residual_correction_target`` in ``src/features/timber_targets.py``), then
reconstructs a prediction as ``raw_value + predicted_correction``.

Both approaches are compared only on test rows where the deterministic
model's raw output was already >= 0 -- i.e. rows the correction was never
meant to fix -- to answer: does adding a correction layer hurt accuracy
where it isn't needed, compared to predicting the diameter directly?
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.features.timber_features import FEATURE_COLUMNS, build_feature_matrix
from src.models.timber_model import build_pipeline, pile_based_split, time_based_split
from src.features.timber_targets import (
    CLIPPED_DIAMETER_COLUMN,
    DIAMETER_TARGET,
    RESIDUAL_TARGET,
    add_residual_correction_target,
)


@dataclass
class ResidualCompareResult:
    residual_model: object
    direct_model: object
    model_kind: str
    split_strategy: str
    n_valid_test: int
    residual_test_mae: float
    residual_test_rmse: float
    residual_test_r2: float
    direct_test_mae: float
    direct_test_rmse: float
    direct_test_r2: float


def train_residual_vs_direct(
    frame: pd.DataFrame,
    model_kind: str = "gbm",
    split_strategy: str = "time",
    feature_columns: list[str] | None = None,
    random_state: int = 42,
) -> ResidualCompareResult:
    """Train both approaches and compare them on the physically-valid subset.

    Both models see the same engineered feature columns -- never the
    deterministic model's own ``remaining_diameter_mm``, which is the
    thing being predicted/corrected, so using it as a feature would be
    leakage. The residual model's raw-value input at reconstruction time
    (``frame[DIAMETER_TARGET]``) is the deterministic model's own output
    for that row, which in real use would already be available (it's the
    thing being corrected), not a held-out label.
    """
    feature_columns = feature_columns if feature_columns is not None else FEATURE_COLUMNS
    frame = add_residual_correction_target(frame)
    features = build_feature_matrix(frame, feature_columns=feature_columns)

    if split_strategy == "time":
        train_idx, test_idx = time_based_split(frame)
    elif split_strategy == "pile":
        train_idx, test_idx = pile_based_split(frame)
    else:
        raise ValueError(f"Unknown split_strategy: {split_strategy!r}")

    residual_model = build_pipeline(model_kind=model_kind, random_state=random_state)
    residual_model.fit(features.loc[train_idx], frame.loc[train_idx, RESIDUAL_TARGET])

    direct_model = build_pipeline(model_kind=model_kind, random_state=random_state)
    direct_model.fit(features.loc[train_idx], frame.loc[train_idx, DIAMETER_TARGET])

    valid_mask = frame.loc[test_idx, DIAMETER_TARGET] >= 0
    valid_test_idx = test_idx[valid_mask]

    raw_valid = frame.loc[valid_test_idx, DIAMETER_TARGET]
    # remaining_diameter_clipped_mm equals raw_valid exactly on this
    # subset (raw already >= 0), kept explicit for readability.
    truth_valid = frame.loc[valid_test_idx, CLIPPED_DIAMETER_COLUMN]

    residual_prediction = residual_model.predict(features.loc[valid_test_idx])
    reconstructed_prediction = raw_valid.to_numpy() + residual_prediction

    direct_prediction = direct_model.predict(features.loc[valid_test_idx])

    def _metrics(prediction) -> tuple[float, float, float]:
        return (
            mean_absolute_error(truth_valid, prediction),
            mean_squared_error(truth_valid, prediction) ** 0.5,
            r2_score(truth_valid, prediction),
        )

    residual_mae, residual_rmse, residual_r2 = _metrics(reconstructed_prediction)
    direct_mae, direct_rmse, direct_r2 = _metrics(direct_prediction)

    return ResidualCompareResult(
        residual_model=residual_model,
        direct_model=direct_model,
        model_kind=model_kind,
        split_strategy=split_strategy,
        n_valid_test=len(valid_test_idx),
        residual_test_mae=residual_mae,
        residual_test_rmse=residual_rmse,
        residual_test_r2=residual_r2,
        direct_test_mae=direct_mae,
        direct_test_rmse=direct_rmse,
        direct_test_r2=direct_r2,
    )
