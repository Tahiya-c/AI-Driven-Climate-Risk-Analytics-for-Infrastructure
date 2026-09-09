"""Train on PileA only (Monte Carlo cross-validated), predict PileB.

1. Monte Carlo cross-validate MLP and GBM on PileA's own rows (repeated random
   80/20 sub-sampling via ``sklearn.model_selection.ShuffleSplit`` -- this is
   what "Monte Carlo cross-validation" means: many random splits, not one).
   This estimates how robust each architecture's fit to PileA is.
2. Refit each model on ALL of PileA's rows -- Monte Carlo CV estimates
   generalization error, it doesn't itself produce the model you deploy.
3. Use that PileA-only-trained model to predict PileB's entire trajectory,
   reusing ``src/timber/scenarios.py::build_pile_trajectories`` unchanged:
   a single shared line up to the split year (today, 2026 -- Baseline, no
   further climate forcing), then 3 diverging scenario lines afterward
   (Hotter & Drier / Most Likely / Colder & Wetter). Every point on both
   segments is a PREDICTION from the PileA-trained model, not PileB's own
   simulated ground truth -- this is a leave-one-pile-out generalization
   demo, not a history/forecast split of real data.
4. Write dashboard-schema CSVs for both models.

Run with: python -m scripts.train_pileA_predict_pileB
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import ShuffleSplit

from src.config import ROOT_DIR
from src.timber.data_prep import DATA_PATH, load_timber_data
from src.timber.features import TARGET_COLUMN, build_feature_matrix
from src.timber.model import build_pipeline
from src.timber.scenarios import build_pile_trajectories, default_split_year

TRAIN_ASSET = "PileA"
PREDICT_ASSET = "PileB"
MC_SPLITS = 30
MC_TEST_FRACTION = 0.2
RANDOM_STATE = 42

SCHEMA = ["material", "asset_id", "scenario", "year", "value", "unit", "metric", "method"]
MATERIAL, UNIT, METRIC = "timber", "mm", "deterioration_depth"

OUT_PATHS = {
    "mlp": ROOT_DIR / "data" / "dashboard" / "dashboard_data_timber_pileB_from_pileA_mlp.csv",
    "gbm": ROOT_DIR / "data" / "dashboard" / "dashboard_data_timber_pileB_from_pileA_gbm.csv",
}
METRICS_OUT = ROOT_DIR / "scripts" / "train_pileA_predict_pileB_metrics.json"


def monte_carlo_cv(
    features: pd.DataFrame,
    target: pd.Series,
    model_kind: str,
    n_splits: int = MC_SPLITS,
    test_fraction: float = MC_TEST_FRACTION,
    random_state: int = RANDOM_STATE,
) -> dict[str, float]:
    """Repeated random train/test sub-sampling -- i.e. Monte Carlo CV."""
    splitter = ShuffleSplit(n_splits=n_splits, test_size=test_fraction, random_state=random_state)
    mae_scores, r2_scores = [], []
    for train_idx, test_idx in splitter.split(features):
        pipeline = build_pipeline(model_kind=model_kind, random_state=random_state)
        pipeline.fit(features.iloc[train_idx], target.iloc[train_idx])
        predictions = pipeline.predict(features.iloc[test_idx])
        mae_scores.append(mean_absolute_error(target.iloc[test_idx], predictions))
        r2_scores.append(r2_score(target.iloc[test_idx], predictions))
    return {
        "mae_mean": float(np.mean(mae_scores)),
        "mae_std": float(np.std(mae_scores)),
        "r2_mean": float(np.mean(r2_scores)),
        "r2_std": float(np.std(r2_scores)),
        "n_splits": n_splits,
    }


def main() -> None:
    frame = load_timber_data(DATA_PATH)
    pileA = frame[frame["asset_code"] == TRAIN_ASSET]
    features_A = build_feature_matrix(pileA)
    target_A = pileA[TARGET_COLUMN]

    metrics: dict[str, object] = {}
    final_models = {}
    for kind in ("mlp", "gbm"):
        mc = monte_carlo_cv(features_A, target_A, kind)
        metrics[f"pileA_mc_cv_{kind}"] = mc
        print(
            f"[{kind}] PileA Monte Carlo CV ({mc['n_splits']} splits): "
            f"MAE={mc['mae_mean']:.2f}+/-{mc['mae_std']:.2f}mm  "
            f"R2={mc['r2_mean']:.3f}+/-{mc['r2_std']:.3f}"
        )

        final_pipeline = build_pipeline(model_kind=kind, random_state=RANDOM_STATE)
        final_pipeline.fit(features_A, target_A)
        final_models[kind] = final_pipeline

    pileB_diameter = float(
        frame.loc[frame["asset_code"] == PREDICT_ASSET, "initial_diameter_mm"].iloc[0]
    )
    split_year = default_split_year(frame[frame["asset_code"] == PREDICT_ASSET])
    print(f"Split year (today's calendar year, clamped to data range): {split_year}")

    for kind, model in final_models.items():
        trajectories, _ = build_pile_trajectories(
            frame, model, PREDICT_ASSET, split_year=split_year
        )
        rows = []
        for scenario, trajectory in trajectories.items():
            deterioration_depth = (pileB_diameter - trajectory.remaining_diameter_mm) / 2
            for year, value in zip(trajectory.years, deterioration_depth):
                rows.append(
                    {
                        "material": MATERIAL,
                        "asset_id": PREDICT_ASSET,
                        "scenario": scenario,
                        "year": int(year),
                        "value": value,
                        "unit": UNIT,
                        "metric": METRIC,
                        "method": kind,
                    }
                )
        df = pd.DataFrame(rows)
        df["value"] = df["value"].round(4)
        df = df[SCHEMA].sort_values(["scenario", "year"]).reset_index(drop=True)
        out_path = OUT_PATHS[kind]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"[{kind}] wrote {len(df)} rows -> {out_path}")
        metrics[f"pileB_{kind}_rows"] = len(df)
        for scenario, trajectory in trajectories.items():
            metrics[f"pileB_{kind}_{scenario}_replacement_year"] = trajectory.replacement_year

    METRICS_OUT.write_text(json.dumps(metrics, indent=2))
    print(f"wrote metrics -> {METRICS_OUT}")


if __name__ == "__main__":
    main()
