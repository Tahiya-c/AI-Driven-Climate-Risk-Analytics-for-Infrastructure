"""Retrain the Kembla MLP/GBM surrogates, save them, and evaluate against Sydney.

One-off pipeline run (not wired into the shared timber pipeline modules) that:

1. Retrains MLP and GBM on the Kembla dataset (``src/models/timber_model.py::train``),
   reporting the same headline metrics as ``docs/timber_model_results.md``.
2. Saves both fitted pipelines to ``artifacts/``.
3. Loads ``data/processed/timber_sydney_v6_3_consolidated.csv`` and scores both
   Kembla-trained models against it -- full-set, and split at age_years <= 15
   ("observed window") vs > 15 ("forecast beyond the window").
4. Writes a dashboard-schema CSV (same 8 columns as
   ``src/models/build_dashboard_output.py``) for Sydney: age<=15 rows as
   method="observed" (Sydney's own simulated ground truth), age>15 rows as
   method="mlp"/"gbm" (the Kembla-trained model's forecast).

Run with: python -m scripts.retrain_and_eval_sydney
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.config import MODEL_DIR, ROOT_DIR
from src.data.timber_data_prep import (
    DATA_PATH,
    MIROC_LABEL,
    _clean_dtypes,
    _validate_scenario_labels,
    load_timber_data,
)
from src.features.timber_features import build_feature_matrix
from src.models.timber_model import save_model, train


def load_sydney_data(path: Path) -> pd.DataFrame:
    """Load Sydney's CSV, harmonizing its scenario_label spelling with Kembla's.

    ``build_timber_sydney.py`` labels the MIROC-Medres scenario "Cooler & Wetter";
    ``src/data/timber_data_prep.py`` (built against Kembla's export) expects "Colder &
    Wetter" -- same scenario, inconsistent spelling between the two builder
    scripts. Remapped in memory (the source CSV on disk is left untouched) rather
    than changing the shared ``load_timber_data``, since that function's
    behaviour for the Kembla path shouldn't change for this one-off harmonization.
    """
    frame = pd.read_csv(path)
    frame.loc[frame["scenario_label"] == "Cooler & Wetter", "scenario_label"] = MIROC_LABEL
    frame = _clean_dtypes(frame)
    _validate_scenario_labels(frame)
    return frame

SYDNEY_PATH = ROOT_DIR / "data" / "processed" / "timber_sydney_v6_3_consolidated.csv"
MLP_MODEL_PATH = MODEL_DIR / "timber_condition_model_mlp.joblib"
GBM_MODEL_PATH = MODEL_DIR / "timber_condition_model_gbm.joblib"
DASHBOARD_OUT = ROOT_DIR / "data" / "dashboard" / "dashboard_data_timber_sydney.csv"
METRICS_OUT = ROOT_DIR / "scripts" / "retrain_and_eval_sydney_metrics.json"

SCHEMA = ["material", "asset_id", "scenario", "year", "value", "unit", "metric", "method"]
MATERIAL, UNIT, METRIC = "timber", "mm", "deterioration_depth"
AGE_SPLIT_YEARS = 15


def _score(actual: pd.Series, predicted) -> dict[str, float]:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(mean_squared_error(actual, predicted) ** 0.5),
        "r2": float(r2_score(actual, predicted)),
        "n": int(len(actual)),
    }


def main() -> None:
    metrics: dict[str, object] = {}

    # 1-2. Retrain + save
    kembla = load_timber_data(DATA_PATH)
    results = {}
    for kind, path in (("mlp", MLP_MODEL_PATH), ("gbm", GBM_MODEL_PATH)):
        result = train(kembla, split_strategy="time", model_kind=kind)
        save_model(result.pipeline, path=path)
        results[kind] = result
        metrics[f"kembla_{kind}"] = {
            "train_mae": result.train_mae,
            "train_rmse": result.train_rmse,
            "train_r2": result.train_r2,
            "test_mae": result.test_mae,
            "test_rmse": result.test_rmse,
            "test_r2": result.test_r2,
            "n_train": result.n_train,
            "n_test": result.n_test,
        }
        print(f"[{kind}] Kembla train MAE={result.train_mae:.2f} R2={result.train_r2:.3f}"
              f"  test MAE={result.test_mae:.2f} R2={result.test_r2:.3f}")
        print(f"[{kind}] saved -> {path}")

    # 3. Sydney evaluation
    sydney = load_sydney_data(SYDNEY_PATH)
    sydney_features = build_feature_matrix(sydney)
    sydney_target = sydney["remaining_diameter_mm"]

    for kind, result in results.items():
        pred_all = result.pipeline.predict(sydney_features)
        metrics[f"sydney_{kind}_full"] = _score(sydney_target, pred_all)

        observed_mask = sydney["age_years"] <= AGE_SPLIT_YEARS
        forecast_mask = ~observed_mask

        metrics[f"sydney_{kind}_age_le_{AGE_SPLIT_YEARS}"] = _score(
            sydney_target[observed_mask], result.pipeline.predict(sydney_features[observed_mask])
        )
        metrics[f"sydney_{kind}_age_gt_{AGE_SPLIT_YEARS}"] = _score(
            sydney_target[forecast_mask], result.pipeline.predict(sydney_features[forecast_mask])
        )
        print(
            f"[{kind}] Sydney full MAE={metrics[f'sydney_{kind}_full']['mae']:.2f} "
            f"R2={metrics[f'sydney_{kind}_full']['r2']:.3f}"
        )
        print(
            f"[{kind}] Sydney age<={AGE_SPLIT_YEARS} MAE="
            f"{metrics[f'sydney_{kind}_age_le_{AGE_SPLIT_YEARS}']['mae']:.2f}  "
            f"age>{AGE_SPLIT_YEARS} MAE="
            f"{metrics[f'sydney_{kind}_age_gt_{AGE_SPLIT_YEARS}']['mae']:.2f}"
        )

    # 4. Dashboard-schema CSV for Sydney
    rows: list[dict[str, object]] = []
    sydney = sydney.copy()
    sydney["scenario"] = sydney["emissions_pathway"] + "|" + sydney["climate_model"]
    observed_mask = sydney["age_years"] <= AGE_SPLIT_YEARS

    observed = sydney[observed_mask]
    for _, row in observed.iterrows():
        rows.append(
            {
                "material": MATERIAL,
                "asset_id": row["asset_code"],
                "scenario": row["scenario"],
                "year": int(row["year"]),
                "value": row["mean_deterioration_mm"],
                "unit": UNIT,
                "metric": METRIC,
                "method": "observed",
            }
        )

    forecast = sydney[~observed_mask]
    forecast_features = build_feature_matrix(forecast)
    for kind, result in results.items():
        predicted_remaining = result.pipeline.predict(forecast_features)
        deterioration_depth = (forecast["initial_diameter_mm"].to_numpy() - predicted_remaining) / 2
        for (_, row), value in zip(forecast.iterrows(), deterioration_depth):
            rows.append(
                {
                    "material": MATERIAL,
                    "asset_id": row["asset_code"],
                    "scenario": row["scenario"],
                    "year": int(row["year"]),
                    "value": value,
                    "unit": UNIT,
                    "metric": METRIC,
                    "method": kind,
                }
            )

    dashboard_df = pd.DataFrame(rows)
    dashboard_df["value"] = dashboard_df["value"].round(4)
    dashboard_df = dashboard_df[SCHEMA].sort_values(
        ["material", "asset_id", "scenario", "method", "year"]
    ).reset_index(drop=True)
    DASHBOARD_OUT.parent.mkdir(parents=True, exist_ok=True)
    dashboard_df.to_csv(DASHBOARD_OUT, index=False)
    print(f"wrote {len(dashboard_df)} rows -> {DASHBOARD_OUT}")
    print(dashboard_df.groupby("method").size().to_string())

    METRICS_OUT.write_text(json.dumps(metrics, indent=2))
    print(f"wrote metrics -> {METRICS_OUT}")


if __name__ == "__main__":
    main()
