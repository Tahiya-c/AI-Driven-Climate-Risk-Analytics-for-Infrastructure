"""Shared timber data, training, recursive rollout and depth evaluation.

Follows concrete_eval.py: fit annual increments on <= cutoff history, feed
predicted increments forward, and score cumulative depth for each series.
The workbook depths are simulator outputs, not field observations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import MODEL_DIR, ROOT_DIR

DATA_PATH = ROOT_DIR / "data" / "processed" / "timber_kembla_v6_3_consolidated.csv"

# Inputs owned by this pipeline; no dependency on the separate MLP/GBM package.
# Exclude deterministic decay rates and calculated depths to avoid target leakage.
TIMBER_FEATURES = [
    "age_years",
    "summer_sea_temp_c",
    "salinity",
    "water_factor",
    "timber_exposure_factor",
    "salinity_factor",
    "shelter_factor",
    "protection_factor",
    "model_contact_factor",
    "model_knot_factor",
    "environment_factor",
    "construction_factor",
    "initial_diameter_mm",
]

CUTOFF_YEAR = 2026
SERIES_COLUMNS = ["asset_code", "emissions_pathway", "climate_model"]
DEPTH_COLUMN = "mean_deterioration_mm"
TARGET = "annual_rate_mm_year"
LAG_COLUMN = "lag_annual_rate"
FEATURE_COLUMNS = [*TIMBER_FEATURES, LAG_COLUMN]
RUNS = ("primary", "scenario", "asset")
ALL_MODELS = ("linear", "ridge", "svr")


def make_model(name: str, **parameters) -> Pipeline:
    """Select a model defined in the model modules, as concrete_eval does.

    Import locally because individual model training entry points also use
    this module's shared fitting and evaluation functions.
    """
    from src.models.linear_model import build_pipeline as linear_pipeline
    from src.models.linear_model import build_ridge_pipeline
    from src.models.svr_model import build_pipeline as svr_pipeline

    registry = {"linear": linear_pipeline, "ridge": build_ridge_pipeline, "svr": svr_pipeline}
    if name not in registry:
        raise ValueError(f"Unknown timber model {name!r}; choose from {ALL_MODELS}.")
    return registry[name](**parameters)


def load_series(data_path: str | Path) -> list[pd.DataFrame]:
    """Keep separate pile/pathway/climate series, including the baseline.

    Depth is optional for future rows during inference; fit/score validate
    their own required depths. Series must be annual, without duplicates.
    """
    frame = pd.read_csv(data_path)
    required = {*SERIES_COLUMNS, "year", DEPTH_COLUMN, *TIMBER_FEATURES}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    if frame.empty or frame[SERIES_COLUMNS].isna().any().any():
        raise ValueError("Data must contain non-empty, complete series identifiers.")
    for column in ["year", DEPTH_COLUMN, *TIMBER_FEATURES]:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    if not np.isfinite(frame["year"]).all() or (frame["year"] % 1 != 0).any():
        raise ValueError("year must contain finite integers.")
    frame["year"] = frame["year"].astype(int)
    series = []
    for key, group in frame.groupby(SERIES_COLUMNS, sort=True):
        group = group.sort_values("year").reset_index(drop=True)
        if len(group) < 3 or not group["year"].diff().iloc[1:].eq(1).all():
            raise ValueError(f"Series {key} must have at least 3 consecutive annual rows.")
        if not np.isfinite(group[TIMBER_FEATURES].to_numpy(dtype=float)).all():
            raise ValueError(f"Series {key} has missing or non-finite forecast inputs.")
        series.append(group)
    return series


def build_pipeline(estimator) -> Pipeline:
    """Fit preprocessing only on training rows, with the estimator in one artifact."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", estimator),
    ])


def _history(series: pd.DataFrame, cutoff: int) -> pd.DataFrame:
    history = series.loc[series["year"] <= cutoff].copy()
    if len(history) < 3 or history["year"].iloc[-1] != cutoff:
        raise ValueError(f"Each series needs at least 3 history years ending at cutoff {cutoff}.")
    if not np.isfinite(history[DEPTH_COLUMN]).all():
        raise ValueError("Historical depths through cutoff must be finite.")
    return history


def fit_model(pipeline: Pipeline, series: list[pd.DataFrame], cutoff: int) -> Pipeline:
    rows = []
    for group in series:
        history = _history(group, cutoff)
        history[TARGET] = history[DEPTH_COLUMN].diff()
        history[LAG_COLUMN] = history[TARGET].shift(1)
        # First row has no increment; second has no preceding increment.
        rows.append(history.iloc[2:])
    training = pd.concat(rows, ignore_index=True)
    pipeline.fit(training[FEATURE_COLUMNS], training[TARGET])
    return pipeline


def rollout(
    pipeline: Pipeline, series: pd.DataFrame, cutoff: int, end_year: int | None = None,
) -> pd.DataFrame:
    """Forecast from the cutoff anchor without ever reading future true depths."""
    history = _history(series, cutoff)
    last_year = int(series["year"].max()) if end_year is None else end_year
    if last_year <= cutoff or last_year > int(series["year"].max()):
        raise ValueError("Requested end year is outside the available projection range.")
    future = series.loc[series["year"].between(cutoff + 1, last_year)]
    lag = float(history[DEPTH_COLUMN].iloc[-1] - history[DEPTH_COLUMN].iloc[-2])
    depth = float(history[DEPTH_COLUMN].iloc[-1])
    records = []
    for _, row in future.iterrows():
        values = [float(row[column]) for column in TIMBER_FEATURES] + [lag]
        inputs = pd.DataFrame([values], columns=FEATURE_COLUMNS)
        lag = float(pipeline.predict(inputs)[0])
        depth += lag
        if not np.isfinite(depth):
            raise ValueError("Recursive forecast produced a non-finite depth.")
        records.append({
            "material": "timber", "asset_id": row["asset_code"],
            "scenario": f"{row['emissions_pathway']}|{row['climate_model']}",
            "year": int(row["year"]), "value": depth,
            "unit": "mm", "metric": "deterioration_depth",
        })
    return pd.DataFrame(records)


def _score(pipeline, series, cutoff, model_name, run, held_out):
    details, forecasts = [], []
    for group in series:
        prediction = rollout(pipeline, group, cutoff)
        actual = group.loc[group["year"] > cutoff, DEPTH_COLUMN].to_numpy()
        if not np.isfinite(actual).all():
            raise ValueError("Evaluation requires finite future reference depths.")
        values = prediction["value"].to_numpy()
        details.append({
            "run": run, "held_out": held_out, "model": model_name,
            "asset": prediction["asset_id"].iloc[0],
            "scenario": prediction["scenario"].iloc[0],
            "mae_mm": float(mean_absolute_error(actual, values)),
            "rmse_mm": float(np.sqrt(mean_squared_error(actual, values))),
            "r2": float(r2_score(actual, values)) if len(actual) > 1 else np.nan,
            "final_abs_error_mm": float(abs(actual[-1] - values[-1])),
            "n_years": len(actual),
        })
        prediction["method"] = model_name
        forecasts.append(prediction)
    return details, forecasts


def _evaluate(series, pipeline, model_name, cutoff, runs):
    if not runs or set(runs) - set(RUNS):
        raise ValueError(f"runs must contain only {RUNS}.")
    primary = fit_model(clone(pipeline), series, cutoff)
    details, forecasts = [], []
    if "primary" in runs:
        details, forecasts = _score(primary, series, cutoff, model_name, "primary", "ALL")
    for run, columns in (("scenario", SERIES_COLUMNS[1:]), ("asset", SERIES_COLUMNS[:1])):
        if run not in runs:
            continue
        keys = [tuple(group[column].iloc[0] for column in columns) for group in series]
        if len(set(keys)) < 2:
            raise ValueError(f"The {run} benchmark needs at least two distinct groups.")
        for key in sorted(set(keys)):
            training = [group for group, value in zip(series, keys) if value != key]
            testing = [group for group, value in zip(series, keys) if value == key]
            fitted = fit_model(clone(pipeline), training, cutoff)
            fold_details, _ = _score(fitted, testing, cutoff, model_name, run, "|".join(key))
            details.extend(fold_details)
    detail = pd.DataFrame(details)
    summary = detail.groupby(["run", "held_out", "model"], sort=False).agg(
        n_series=("mae_mm", "size"), mae_mm=("mae_mm", "mean"),
        mae_std_mm=("mae_mm", "std"), rmse_mm=("rmse_mm", "mean"),
        r2=("r2", "mean"), final_mae_mm=("final_abs_error_mm", "mean"),
    ).reset_index()
    return primary, summary, detail, forecasts


def run_benchmark(csv_path, model_names, runs=RUNS, cutoff=CUTOFF_YEAR):
    """Concrete-compatible benchmark API; returns tables without saving fitted models."""
    series = load_series(csv_path)
    summaries, details = [], []
    for name in model_names:
        _, summary, detail, _ = _evaluate(series, make_model(name), name, cutoff, runs)
        summaries.append(summary)
        details.append(detail)
    names = {
        "mae_mm": "mae", "mae_std_mm": "mae_std", "rmse_mm": "rmse",
        "final_mae_mm": "final_mae", "final_abs_error_mm": "final_abs_err",
    }
    return (
        pd.concat(summaries, ignore_index=True).rename(columns=names),
        pd.concat(details, ignore_index=True).rename(columns=names),
        [],
    )


def train_and_evaluate(data_path, output_path, pipeline, model_name, cutoff, runs) -> dict:
    """Legacy/tuning adapter saving a cutoff-trained research artifact."""
    if "primary" not in runs:
        raise ValueError("Research artifacts require a primary evaluation.")
    series = load_series(data_path)
    primary, summary, detail, forecasts = _evaluate(series, pipeline, model_name, cutoff, runs)
    artifact = {
        "artifact_version": 2, "model": primary, "model_type": model_name,
        "features": FEATURE_COLUMNS, "target": TARGET, "depth_column": DEPTH_COLUMN,
        "metric": "deterioration_depth", "unit": "mm", "cutoff_year": cutoff,
        "first_test_year": cutoff + 1,
        "train_rows": sum(len(group.loc[group["year"] <= cutoff]) - 2 for group in series),
        "test_rows": sum(len(group.loc[group["year"] > cutoff]) for group in series),
        "requires_future_inputs": True,
        "metrics": summary.loc[summary["run"] == "primary"].iloc[0].to_dict(),
        "evaluation": "Recursive depth forecasts; metrics averaged across series.",
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output_path)
    summary.to_csv(output_path.with_suffix(".metrics.csv"), index=False)
    detail.to_csv(output_path.with_suffix(".detail.csv"), index=False)
    pd.concat(forecasts, ignore_index=True).to_csv(
        output_path.with_suffix(".predictions.csv"), index=False,
    )
    return artifact


def run_cli(model_name, trainer) -> None:
    parser = argparse.ArgumentParser(
        description="Train annual timber increments; score depth in mm.",
    )
    parser.add_argument("--data", "--csv", type=Path, default=DATA_PATH)
    parser.add_argument(
        "--output", type=Path, default=MODEL_DIR / f"timber_depth_{model_name}.joblib",
    )
    parser.add_argument("--cutoff", type=int, default=CUTOFF_YEAR)
    parser.add_argument("--runs", default=",".join(RUNS), help="primary,scenario,asset or primary")
    if model_name == "svr":
        parser.add_argument("--c", type=float, default=100.0)
        parser.add_argument("--epsilon", type=float, default=0.1)
        parser.add_argument("--gamma", default="scale")
    args = parser.parse_args()
    parameters = {}
    if model_name == "svr":
        gamma = args.gamma if args.gamma in {"scale", "auto"} else float(args.gamma)
        parameters = {"c": args.c, "epsilon": args.epsilon, "gamma": gamma}
    artifact = trainer(
        args.data, args.output, cutoff=args.cutoff,
        runs=tuple(run.strip() for run in args.runs.split(",")), **parameters,
    )
    print(json.dumps({key: value for key, value in artifact.items() if key != "model"}, indent=2))
    print(f"Saved model to {args.output}")
    print("Saved .metrics.csv, .detail.csv and .predictions.csv beside the model.")
