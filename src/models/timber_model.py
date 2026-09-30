"""Train an MLPRegressor surrogate for the timber pile deterioration model.

Run directly to train, print metrics, and cache the model:

    python -m src.timber.model
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import MODEL_DIR
from src.timber.data_prep import DATA_PATH, load_timber_data
from src.timber.features import TARGET_COLUMN, build_feature_matrix

MODEL_PATH = MODEL_DIR / "timber_condition_model.joblib"
MODEL_KINDS = ("mlp", "gbm")


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


def build_pipeline(model_kind: str = "mlp", random_state: int = 42) -> object:
    """Build an untrained estimator for the requested model kind.

    "mlp" returns a StandardScaler + MLPRegressor sklearn Pipeline. "gbm"
    returns a bare LightGBM/XGBoost regressor (see src/timber/model_gbm.py)
    -- tree ensembles need no feature scaling, and expose the same
    fit/predict interface, so callers don't need to care which they got.
    """
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
        from src.timber.model_gbm import build_gbm_regressor

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
) -> TrainResult:
    """Fit the requested estimator and score it on a held-out split.

    ``target_column`` and ``feature_columns`` default to the standard
    raw-diameter target and feature set (see src/timber/features.py) so
    existing callers (the Streamlit app, tests) are unaffected; pass
    alternates (see src/timber/targets.py, DECAY_DOSE_FEATURE_COLUMNS) to
    train against a different target or feature set.
    """
    features = build_feature_matrix(
        frame, feature_columns=feature_columns, include_categorical=include_categorical
    )
    target = frame[target_column]

    if split_strategy == "time":
        train_idx, test_idx = time_based_split(frame)
    elif split_strategy == "pile":
        train_idx, test_idx = pile_based_split(frame)
    else:
        raise ValueError(f"Unknown split_strategy: {split_strategy!r}")

    pipeline = build_pipeline(model_kind=model_kind, random_state=random_state)
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
    print(f"Train MAE: {result.train_mae:.2f} mm  R2: {result.train_r2:.3f}")
    print(f"Test  MAE: {result.test_mae:.2f} mm  R2: {result.test_r2:.3f}")
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
