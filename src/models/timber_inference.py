"""Load saved timber models and forecast depth; counterpart to models/inference.py."""

import argparse
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.models.timber_eval import FEATURE_COLUMNS, TARGET, load_series, rollout


def predict_deterioration(model: Any, features: pd.DataFrame) -> pd.Series:
    """Return model predictions with the input index preserved."""
    return pd.Series(model.predict(features), index=features.index, name="prediction")


def predict_timber_until_year(
    artifact_path: Path,
    data_path: Path,
    end_year: int,
    start_year: int | None = None,
    anchor_year: int | None = None,
) -> pd.DataFrame:
    """Return depth forecasts per pile/scenario, recursively from the saved cutoff.

    start_year filters the output; it never re-anchors on future true depths.
    Future climate and other model inputs must exist through end_year.
    """
    if isinstance(end_year, bool) or not isinstance(end_year, int):
        raise TypeError("end_year must be an integer.")
    if start_year is not None and (
        isinstance(start_year, bool) or not isinstance(start_year, int)
    ):
        raise TypeError("start_year must be an integer or None.")

    artifact = joblib.load(artifact_path)
    required_keys = {"model", "features", "target", "cutoff_year", "model_type"}
    missing = sorted(required_keys - set(artifact))
    if missing:
        raise ValueError(f"Model artifact is missing: {', '.join(missing)}. Retrain a depth model.")
    if artifact["target"] != TARGET or artifact["features"] != FEATURE_COLUMNS:
        raise ValueError("Incompatible timber artifact. Retrain with the current depth pipeline.")
    cutoff = artifact["cutoff_year"] if anchor_year is None else anchor_year
    if artifact.get("training_scope") == "all_available_history" and anchor_year is None:
        raise ValueError("Final models require an explicit anchor_year (CLI: --anchor-year).")
    first_year = cutoff + 1 if start_year is None else start_year
    if first_year <= cutoff or first_year > end_year:
        raise ValueError("start_year must be after the training cutoff and <= end_year.")
    predictions = [
        rollout(artifact["model"], group, cutoff, end_year)
        for group in load_series(data_path)
    ]
    result = pd.concat(predictions, ignore_index=True)
    result["method"] = artifact["model_type"]
    return result.loc[result["year"] >= first_year].reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict timber deterioration depth in mm.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--end-year", type=int, required=True)
    parser.add_argument("--start-year", type=int)
    parser.add_argument("--anchor-year", type=int)
    parser.add_argument("--output", type=Path, help="Optional dashboard-schema CSV output.")
    arguments = parser.parse_args()
    predictions = predict_timber_until_year(
        artifact_path=arguments.model,
        data_path=arguments.data,
        end_year=arguments.end_year,
        start_year=arguments.start_year,
        anchor_year=arguments.anchor_year,
    )
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(arguments.output, index=False)
        print(f"Saved {len(predictions)} predictions to {arguments.output}")
    else:
        print(predictions.to_string(index=False))


if __name__ == "__main__":
    main()
