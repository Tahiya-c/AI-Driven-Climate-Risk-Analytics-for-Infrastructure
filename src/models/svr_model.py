"""Timber support vector regression training entry point.

Run from the repository root: python -m src.models.svr_model
See docs/timber_linear_svr_pipeline.md for training and prediction commands.
"""

from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.svm import SVR

from src.models.timber_eval import (
    CUTOFF_YEAR,
    run_cli,
    train_and_evaluate,
)
from src.models.timber_eval import build_pipeline as _build_pipeline


def build_pipeline(
    c: float = 100.0, epsilon: float = 0.1, gamma: str | float = "scale",
    **parameters,
) -> Pipeline:
    """Define SVR with shared preprocessing; accept sklearn's C spelling too."""
    parameters.setdefault("C", c)
    parameters.setdefault("kernel", "rbf")
    return _build_pipeline(SVR(epsilon=epsilon, gamma=gamma, **parameters))


def train(
    data_path: Path,
    output_path: Path,
    cutoff: int = CUTOFF_YEAR,
    c: float = 100.0,
    epsilon: float = 0.1,
    gamma: str | float = "scale",
    runs: tuple[str, ...] = ("primary", "scenario", "asset"),
) -> dict:
    """Use exactly the same splits, rollout and depth metrics as the linear model."""
    return train_and_evaluate(
        data_path, output_path, build_pipeline(c, epsilon, gamma), "svr", cutoff, runs,
    )


def main() -> None:
    run_cli("svr", train)


if __name__ == "__main__":
    main()
