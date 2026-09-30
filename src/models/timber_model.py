"""Train an MLP or GBM surrogate for the timber pile deterioration model.

Run directly to train, print metrics, and cache the model:

    python -m src.models.timber_model

By default (``tuned=True``) both model kinds are wrapped in
``DeteriorationRateRegressor``: same inputs (``FEATURE_COLUMNS``), same output
(``remaining_diameter_mm``), but internally they add ``thermal_dose`` and learn
the average deterioration rate instead of the raw diameter -- see that class
and docs/timber_model_tuning.md for why. ``tuned=False`` restores the original
plain MLP/GBM estimators.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import MODEL_DIR
from src.data.timber_data_prep import DATA_PATH, load_timber_data
from src.features.timber_features import TARGET_COLUMN, add_thermal_dose, build_feature_matrix

MODEL_PATH = MODEL_DIR / "timber_condition_model.joblib"
MODEL_KINDS = ("mlp", "gbm")

# Best hyperparameters found by ``python -m src.models.timber_tuning --model mlp``
# (Optuna TPE, blocked time-series CV on the training years only -- see
# docs/timber_model_tuning.md). Re-run the tuner and paste its output here
# after any change to the data or FEATURE_COLUMNS.
TUNED_MLP_PARAMS: dict[str, Any] = {
    "hidden_layer_sizes": (136, 136, 136),
    "activation": "relu",
    "alpha": 0.000976112544311046,
    "learning_rate_init": 0.0069782812651260325,
    "batch_size": 128,
}

# Columns DeteriorationRateRegressor needs in X to convert between diameter and rate.
RATE_REQUIRED_COLUMNS = ("age_years", "initial_diameter_mm", "water_factor")


class DeteriorationRateRegressor(RegressorMixin, BaseEstimator):
    """Fit ``estimator`` on the average deterioration rate, predict remaining diameter.

    The deterministic engine's output is roughly
    ``D0 - 2 * (sapwood_layer + (age - lag) * r_heart)``, i.e. remaining
    diameter falls about linearly with age at a rate set by the pile's
    factors and the sea temperature. The time-based test split asks a model
    to forecast ages (51-65) it never saw in training (-6..50). Tree
    ensembles predict a flat constant beyond their training range and a small
    MLP extrapolates poorly, so both struggle with the raw diameter target.

    The average rate ``(D0 - D) / age`` is close to stationary in age, so
    this wrapper trains the inner estimator on that instead and converts
    back with ``D0 - rate * age``. Pre-installation rows (age <= 0) have rate
    0 and are returned as exactly ``D0``. Both conversions use only
    ``initial_diameter_mm`` and ``age_years``, which are already model inputs,
    so no target information leaks in. ``thermal_dose`` (see
    ``add_thermal_dose``) is appended to X inside fit/predict, so callers keep
    passing the plain ``FEATURE_COLUMNS`` matrix and get diameters back --
    the same interface as the unwrapped estimators.
    """

    def __init__(self, estimator: Any):
        self.estimator = estimator

    @staticmethod
    def _age(X: pd.DataFrame) -> np.ndarray:
        return X["age_years"].to_numpy(dtype=float)

    def fit(self, X: pd.DataFrame, y) -> DeteriorationRateRegressor:
        age = self._age(X)
        lost = X["initial_diameter_mm"].to_numpy(dtype=float) - np.asarray(y, dtype=float)
        rate = np.where(age > 0, lost / np.clip(age, 1, None), 0.0)
        self.estimator_ = clone(self.estimator).fit(add_thermal_dose(X), rate)
        self.feature_names_in_ = np.asarray(X.columns)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        rate = np.asarray(self.estimator_.predict(add_thermal_dose(X)), dtype=float)
        age = np.clip(self._age(X), 0, None)
        return X["initial_diameter_mm"].to_numpy(dtype=float) - rate * age


def suggest_mlp_params(trial) -> dict[str, Any]:
    """Optuna search space for the MLP surrogate."""
    n_layers = trial.suggest_int("n_layers", 1, 3)
    width = trial.suggest_int("width", 16, 256, log=True)
    return {
        "hidden_layer_sizes": tuple([width] * n_layers),
        "activation": trial.suggest_categorical("activation", ["relu", "tanh"]),
        "alpha": trial.suggest_float("alpha", 1e-6, 1e-1, log=True),
        "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64, 128]),
    }


def build_estimator(model_kind: str, params: dict[str, Any] | None = None, random_state: int = 42):
    """Inner estimator for the rate-target surrogate (MLP scales X and y; GBM needs neither)."""
    if model_kind == "mlp":
        mlp = MLPRegressor(
            max_iter=4000,
            early_stopping=True,
            n_iter_no_change=25,
            random_state=random_state,
            **(params or {}),
        )
        return TransformedTargetRegressor(
            regressor=Pipeline([("scaler", StandardScaler()), ("mlp", mlp)]),
            transformer=StandardScaler(),
        )
    if model_kind == "gbm":
        from src.models.timber_model_gbm import build_gbm_regressor

        return build_gbm_regressor(random_state=random_state, **(params or {}))
    raise ValueError(f"Unknown model_kind: {model_kind!r} (expected one of {MODEL_KINDS})")


def build_tuned_model(
    model_kind: str, params: dict[str, Any] | None = None, random_state: int = 42
) -> DeteriorationRateRegressor:
    """Rate-target surrogate using ``params`` (default: the stored TUNED_*_PARAMS)."""
    if params is None:
        if model_kind == "gbm":
            from src.models.timber_model_gbm import TUNED_GBM_PARAMS

            params = TUNED_GBM_PARAMS
        else:
            params = TUNED_MLP_PARAMS
    return DeteriorationRateRegressor(build_estimator(model_kind, params, random_state))


@dataclass
class TrainResult:
    """Fitted estimator plus the metrics used to report and sanity-check it."""

    pipeline: object
    train_mae: float
    train_r2: float
    test_mae: float
    test_r2: float
    split_strategy: str
    n_train: int
    n_test: int
    model_kind: str = "mlp"
    target_column: str = TARGET_COLUMN
    train_rmse: float = float("nan")
    test_rmse: float = float("nan")


def time_based_split(
    frame: pd.DataFrame, test_fraction: float = 0.2
) -> tuple[pd.Index, pd.Index]:
    """Split rows chronologically within each pile/scenario series.

    This is the default split. Only two piles exist in this dataset, so a
    leave-one-pile-out split trains on a single asset (a different
    species, exposure class, and contact/knot profile) and tests on a
    structurally unrelated one -- an extreme domain shift that mostly
    measures how different PileA and PileB happen to be. A time-based
    split instead tests whether the model can extrapolate a given pile's
    own trajectory into later years, which matches how the app is
    actually used: forecasting a known pile forward in time.
    """
    train_parts, test_parts = [], []
    for _, group in frame.groupby(["asset_code", "scenario_label"], sort=False):
        ordered = group.sort_values("year")
        cut = max(1, int(round(len(ordered) * (1 - test_fraction))))
        train_parts.append(ordered.iloc[:cut])
        test_parts.append(ordered.iloc[cut:])
    return pd.concat(train_parts).index, pd.concat(test_parts).index


def pile_based_split(
    frame: pd.DataFrame, test_pile: str = "PileB"
) -> tuple[pd.Index, pd.Index]:
    """Leave-one-pile-out split, offered as an alternate/diagnostic strategy.

    With only two piles this trains on a single asset and evaluates on a
    structurally different one. It is a useful generalization stress test
    but is not the default: the resulting metrics mostly describe how
    different PileA and PileB are, not forecast quality for a given pile.
    """
    test_mask = frame["asset_code"] == test_pile
    return frame.index[~test_mask], frame.index[test_mask]


def build_pipeline(model_kind: str = "mlp", random_state: int = 42, tuned: bool = False) -> object:
    """Build an untrained estimator for the requested model kind.

    ``tuned=True`` returns the rate-target surrogate with the stored tuned
    hyperparameters (see ``build_tuned_model``). Otherwise, "mlp" returns the
    original StandardScaler + MLPRegressor sklearn Pipeline and "gbm" a bare
    LightGBM/XGBoost regressor (see src/models/timber_model_gbm.py) -- tree
    ensembles need no feature scaling, and all of these expose the same
    fit/predict interface, so callers don't need to care which they got.
    The plain estimators stay the default here because some callers (e.g.
    src/models/timber_residual.py) train on targets other than the raw
    diameter, which the rate conversion does not apply to.
    """
    if tuned:
        return build_tuned_model(model_kind, random_state=random_state)
    if model_kind == "mlp":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "mlp",
                    MLPRegressor(
                        hidden_layer_sizes=(64, 32),
                        activation="relu",
                        solver="adam",
                        alpha=1e-3,
                        max_iter=4000,
                        early_stopping=True,
                        n_iter_no_change=25,
                        random_state=random_state,
                    ),
                ),
            ]
        )
    if model_kind == "gbm":
        from src.models.timber_model_gbm import build_gbm_regressor

        return build_gbm_regressor(random_state=random_state)
    raise ValueError(f"Unknown model_kind: {model_kind!r} (expected one of {MODEL_KINDS})")


def train(
    frame: pd.DataFrame,
    split_strategy: str = "time",
    model_kind: str = "mlp",
    target_column: str = TARGET_COLUMN,
    feature_columns: list[str] | None = None,
    include_categorical: bool = False,
    random_state: int = 42,
    tuned: bool = True,
) -> TrainResult:
    """Fit the requested estimator and score it on a held-out split.

    ``target_column`` and ``feature_columns`` default to the standard
    raw-diameter target and feature set (see src/features/timber_features.py);
    pass alternates (see src/features/timber_targets.py,
    DECAY_DOSE_FEATURE_COLUMNS) to train against a different target or
    feature set. ``tuned`` uses the rate-target surrogate with tuned
    hyperparameters (``build_tuned_model``); it only applies to the raw
    diameter target with age/diameter/water_factor among the features, and
    silently falls back to the plain estimator otherwise.
    """
    features = build_feature_matrix(
        frame, feature_columns=feature_columns, include_categorical=include_categorical
    )
    target = frame[target_column]
    tuned = (
        tuned
        and target_column == TARGET_COLUMN
        and set(RATE_REQUIRED_COLUMNS).issubset(features.columns)
    )

    if split_strategy == "time":
        train_idx, test_idx = time_based_split(frame)
    elif split_strategy == "pile":
        train_idx, test_idx = pile_based_split(frame)
    else:
        raise ValueError(f"Unknown split_strategy: {split_strategy!r}")

    pipeline = build_pipeline(model_kind=model_kind, random_state=random_state, tuned=tuned)
    pipeline.fit(features.loc[train_idx], target.loc[train_idx])

    def _score(index: pd.Index) -> tuple[float, float, float]:
        predictions = pipeline.predict(features.loc[index])
        return (
            mean_absolute_error(target.loc[index], predictions),
            mean_squared_error(target.loc[index], predictions) ** 0.5,
            r2_score(target.loc[index], predictions),
        )

    train_mae, train_rmse, train_r2 = _score(train_idx)
    test_mae, test_rmse, test_r2 = _score(test_idx)

    return TrainResult(
        pipeline=pipeline,
        train_mae=train_mae,
        train_r2=train_r2,
        test_mae=test_mae,
        test_r2=test_r2,
        split_strategy=split_strategy,
        n_train=len(train_idx),
        n_test=len(test_idx),
        model_kind=model_kind,
        target_column=target_column,
        train_rmse=train_rmse,
        test_rmse=test_rmse,
    )


def save_model(pipeline: object, path: Path = MODEL_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)


def load_model(path: Path = MODEL_PATH) -> object:
    return joblib.load(path)


def main() -> None:
    frame = load_timber_data(DATA_PATH)
    result = train(frame, split_strategy="time")

    print(f"Split strategy: {result.split_strategy}")
    print(f"Train rows: {result.n_train}  Test rows: {result.n_test}")
    print(
        f"Train MAE: {result.train_mae:.2f} mm  RMSE: {result.train_rmse:.2f} mm  "
        f"R2: {result.train_r2:.3f}"
    )
    print(
        f"Test  MAE: {result.test_mae:.2f} mm  RMSE: {result.test_rmse:.2f} mm  "
        f"R2: {result.test_r2:.3f}"
    )
    print(
        "NOTE: only 2 piles (PileA, PileB) are present in this dataset. "
        "These metrics describe fit quality for those two assets and "
        "should not be read as evidence of generalization to a wider "
        "pile population."
    )

    save_model(result.pipeline)
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
