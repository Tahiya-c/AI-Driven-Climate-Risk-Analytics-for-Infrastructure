"""Prediction interfaces for trained deterioration models."""

import argparse
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.models.timber_scenarios import predict_general_scenarios, prepare_projection_rows


def predict_deterioration(model: Any, features: pd.DataFrame) -> pd.Series:
    """Return model predictions with the input index preserved."""
    return pd.Series(model.predict(features), index=features.index, name="prediction")


def predict_timber_until_year(
    artifact_path: Path,
    data_path: Path,
    end_year: int,
    start_year: int | None = None,
) -> pd.DataFrame:
    """Predict three general timber scenarios per year through ``end_year``."""
    if isinstance(end_year, bool) or not isinstance(end_year, int):
        raise TypeError("end_year must be an integer.")
    if start_year is not None and (
        isinstance(start_year, bool) or not isinstance(start_year, int)
    ):
        raise TypeError("start_year must be an integer or None.")

    artifact = joblib.load(artifact_path)
    required_keys = {"model", "features", "scenario_labels"}
    missing_keys = sorted(required_keys - set(artifact))
    if missing_keys:
        raise ValueError(f"Model artifact is missing: {', '.join(missing_keys)}")

    # replacement_time_years is intentionally never selected for prediction.
    frame = prepare_projection_rows(pd.read_csv(data_path))
    available_years = sorted(int(year) for year in frame["year"].dropna().unique())
    minimum_year, maximum_year = available_years[0], available_years[-1]
    first_year = minimum_year if start_year is None else start_year

    if first_year > end_year:
        raise ValueError("start_year must be less than or equal to end_year.")
    if first_year < minimum_year or end_year > maximum_year:
        raise ValueError(
            f"Requested range {first_year}-{end_year} is outside the available "
            f"projection range {minimum_year}-{maximum_year}."
        )

    requested_years = set(range(first_year, end_year + 1))
    missing_years = sorted(requested_years - set(available_years))
    if missing_years:
        raise ValueError(
            "Projection inputs are unavailable for years: "
            + ", ".join(str(year) for year in missing_years)
        )

    prediction_rows = frame.loc[frame["year"].isin(requested_years)]
    return predict_general_scenarios(
        artifact["model"], prediction_rows, artifact["features"]
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict three general timber scenarios through an ending year."
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--end-year", type=int, required=True)
    parser.add_argument("--start-year", type=int)
    arguments = parser.parse_args()
    predictions = predict_timber_until_year(
        artifact_path=arguments.model,
        data_path=arguments.data,
        end_year=arguments.end_year,
        start_year=arguments.start_year,
    )
    print(predictions.to_string(index=False))


if __name__ == "__main__":
    main()
