"""Train a time-aware linear model for timber replacement time. (TSLM model)"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import MODEL_DIR
from src.models.timber_scenarios import (
    SCENARIO_COLUMN,
    SCENARIO_LABELS,
    predict_general_scenarios,
    prepare_projection_rows,
)

TARGET = "replacement_time_years"
TIME_COLUMN = "year"

# Use source attributes and deterioration-model inputs, but not calculated outputs
# downstream of those inputs. This avoids leaking the target calculation into training.
FEATURE_COLUMNS = [
    "timber_type",
    "year_installed",
    "zone",
    "exposure",
    "heartwood_class",
    "sapwood_of_hardwood",
    "maintenance_measure",
    "contact",
    "knot_presence",
    "initial_diameter_mm",
    "layer_of_sapwood_mm",
    "replacement_diameter_mm",
    "emissions_pathway",
    "climate_model",
    "scenario_label",
    "year",
    "age_years",
    "salinity",
    "summer_sea_temp_c",
    "design_time_years",
    "water_factor",
    "timber_exposure_factor",
    "model_hazard_wood",
    "model_service_water",
    "salinity_factor",
    "shelter_factor",
    "protection_factor",
    "model_contact_factor",
    "model_knot_factor",
    "environment_factor",
    "construction_factor",
]


# Since the model forecasts future outcomes, reserve the latest years as test data
# to simulate unseen future conditions.
def chronological_split(
    frame: pd.DataFrame, test_fraction: float
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Put the latest fraction of unique projection years in the test set."""
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1.")
    years = np.sort(frame[TIME_COLUMN].dropna().unique())
    if len(years) < 2:
        raise ValueError("At least two unique years are required for a chronological split.")
    test_year_count = min(max(1, math.ceil(len(years) * test_fraction)), len(years) - 1)
    first_test_year = int(years[-test_year_count])
    train = frame.loc[frame[TIME_COLUMN] < first_test_year].copy()
    test = frame.loc[frame[TIME_COLUMN] >= first_test_year].copy()
    return train, test, first_test_year


def build_pipeline(frame: pd.DataFrame) -> Pipeline:
    """Build preprocessing and ordinary least-squares regression as one pipeline."""
    categorical = frame[FEATURE_COLUMNS].select_dtypes(exclude="number").columns.tolist()
    numerical = [column for column in FEATURE_COLUMNS if column not in categorical]
    preprocessing = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numerical,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("one_hot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical,
            ),
        ]
    )
    return Pipeline([("preprocessing", preprocessing), ("model", LinearRegression())])


def train(data_path: Path, output_path: Path, test_fraction: float = 0.2) -> dict[str, object]:
    """Train and evaluate one pooled model with three scenario outputs per year."""
    frame = prepare_projection_rows(pd.read_csv(data_path))
    required = set(FEATURE_COLUMNS) | {TARGET}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    frame = frame.dropna(subset=[TARGET, TIME_COLUMN]).sort_values(TIME_COLUMN)
    train_frame, test_frame, first_test_year = chronological_split(frame, test_fraction)

    pipeline = build_pipeline(train_frame)
    pipeline.fit(train_frame[FEATURE_COLUMNS], train_frame[TARGET])
    row_predictions = pipeline.predict(test_frame[FEATURE_COLUMNS])
    row_metrics = {
        "mae_years": float(mean_absolute_error(test_frame[TARGET], row_predictions)),
        "rmse_years": float(np.sqrt(mean_squared_error(test_frame[TARGET], row_predictions))),
        "r2": float(r2_score(test_frame[TARGET], row_predictions)),
    }
    scenario_predictions = predict_general_scenarios(pipeline, test_frame, FEATURE_COLUMNS)
    scenario_actuals = (
        test_frame.groupby([TIME_COLUMN, SCENARIO_COLUMN], as_index=False, observed=True)[TARGET]
        .mean()
    )
    scenario_evaluation = scenario_actuals.merge(
        scenario_predictions,
        on=[TIME_COLUMN, SCENARIO_COLUMN],
        validate="one_to_one",
    )
    scenario_values = scenario_evaluation["predicted_replacement_time_years"]
    metrics = {
        "mae_years": float(mean_absolute_error(scenario_evaluation[TARGET], scenario_values)),
        "rmse_years": float(
            np.sqrt(mean_squared_error(scenario_evaluation[TARGET], scenario_values))
        ),
        "r2": float(r2_score(scenario_evaluation[TARGET], scenario_values)),
    }
    artifact = {
        "model": pipeline,
        "model_type": "general_timber_time_aware_linear_regression",
        "target": TARGET,
        "features": FEATURE_COLUMNS,
        "scenario_column": SCENARIO_COLUMN,
        "scenario_labels": list(SCENARIO_LABELS),
        "prediction_granularity": "one pooled prediction per year and scenario",
        "first_test_year": first_test_year,
        "train_rows": len(train_frame),
        "test_rows": len(test_frame),
        "metrics": metrics,
        "row_metrics": row_metrics,
        "test_scenario_predictions": scenario_predictions.to_dict(orient="records"),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output_path)
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/processed/timber_kembla_v6_3_consolidated.csv"),
    )
    parser.add_argument(
        "--output", type=Path, default=MODEL_DIR / "timber_replacement_linear.joblib"
    )
    parser.add_argument("--test-fraction", type=float, default=0.2)
    args = parser.parse_args()
    result = train(args.data, args.output, args.test_fraction)
    summary = {
        key: value
        for key, value in result.items()
        if key not in {"model", "test_scenario_predictions"}
    }
    summary["test_scenario_prediction_rows"] = len(result["test_scenario_predictions"])
    print(json.dumps(summary, indent=2))
    print(f"Saved model to {args.output}")


if __name__ == "__main__":
    main()
