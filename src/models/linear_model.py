"""Timber linear model training entry point.

Run from the repository root: python -m src.models.linear_model
See docs/timber_linear_svr_pipeline.md for training and prediction commands.
"""

from pathlib import Path

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline

from src.models.timber_eval import (
    CUTOFF_YEAR,
    run_cli,
    train_and_evaluate,
)
from src.models.timber_eval import build_pipeline as _build_pipeline


def build_pipeline(**parameters) -> Pipeline:
    """Define ordinary linear regression with shared timber preprocessing."""
    return _build_pipeline(LinearRegression(**parameters))


def build_ridge_pipeline(alpha: float = 1000.0, **parameters) -> Pipeline:
    """Define the regularized linear model selected by historical tuning."""
    return _build_pipeline(Ridge(alpha=alpha, **parameters))


def train(
    data_path: Path,
    output_path: Path,
    cutoff: int = CUTOFF_YEAR,
    runs: tuple[str, ...] = ("primary", "scenario", "asset"),
) -> dict:
    """Save the primary cutoff-trained model; score independent benchmark folds."""
    return train_and_evaluate(data_path, output_path, build_pipeline(), "linear", cutoff, runs)


def main() -> None:
    run_cli("linear", train)


if __name__ == "__main__":
    main()
