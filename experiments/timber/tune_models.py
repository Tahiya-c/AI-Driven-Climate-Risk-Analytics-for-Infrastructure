"""Tune timber models on historical recursive forecasts, then evaluate once.

python -m experiments.timber.tune_models
The post-cutoff period is excluded from every parameter-selection fold.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import ParameterGrid

from src.config import MODEL_DIR
from src.models.timber_eval import (
    CUTOFF_YEAR,
    DATA_PATH,
    DEPTH_COLUMN,
    FEATURE_COLUMNS,
    TIMBER_FEATURES,
    fit_model,
    load_series,
    make_model,
    train_and_evaluate,
)


def candidates(model_name: str) -> list[tuple[dict, object]]:
    """Fixed, reproducible search space; includes the original default model."""
    result = []
    if model_name == "linear":
        for params in ParameterGrid({"fit_intercept": [True, False], "positive": [False, True]}):
            result.append(({"estimator": "LinearRegression", **params},
                           make_model("linear", **params)))
        for params in ParameterGrid({
            "alpha": [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
            "fit_intercept": [True, False],
        }):
            result.append(({"estimator": "Ridge", **params},
                           make_model("ridge", **params)))
    elif model_name == "svr":
        for params in ParameterGrid({
            "C": [1.0, 10.0, 100.0, 1000.0],
            "epsilon": [0.01, 0.1, 0.5],
            "gamma": ["scale", 0.001, 0.01, 0.1],
        }):
            result.append(({"estimator": "SVR", "kernel": "rbf", **params},
                           make_model("svr", kernel="rbf", **params)))
    else:
        raise ValueError("model_name must be linear or svr.")
    return result


def validation_mae(pipeline, series, origin: int, end_year: int) -> float:
    """Batch independent series per year; feed only predicted increments forward.

    All series must cover the same validation years. This is the same rollout
    as timber_eval.rollout, with one batch prediction per year for search speed.
    """
    future = [g.loc[g.year.between(origin + 1, end_year)] for g in series]
    expected = np.arange(origin + 1, end_year + 1)
    if any(not np.array_equal(g.year.to_numpy(), expected) for g in future):
        raise ValueError("Every series must cover all validation years.")
    histories = [g.loc[g.year <= origin, DEPTH_COLUMN].to_numpy() for g in series]
    depth = np.array([h[-1] for h in histories], dtype=float)
    lag = np.array([h[-1] - h[-2] for h in histories], dtype=float)
    inputs = np.stack([g[TIMBER_FEATURES].to_numpy(dtype=float) for g in future])
    actual = np.stack([g[DEPTH_COLUMN].to_numpy(dtype=float) for g in future])
    if not np.isfinite(actual).all():
        raise ValueError("Validation depths must be finite.")
    errors = []
    for index in range(len(expected)):
        rows = pd.DataFrame(np.column_stack([inputs[:, index, :], lag]), columns=FEATURE_COLUMNS)
        lag = pipeline.predict(rows)
        depth = depth + lag
        if not np.isfinite(depth).all():
            raise ValueError("Non-finite recursive validation forecast.")
        errors.append(np.abs(actual[:, index] - depth))
    return float(np.mean(errors))


def select_parameters(series, model_name, cutoff=CUTOFF_YEAR, search=None):
    """Three expanding training folds, each forecasting to the training cutoff.

    With cutoff=2026: train through 2011/2016/2021, validate through 2026.
    Equal weight per origin prevents the longest horizon dominating selection.
    Future rows are removed before any fit or scoring operation.
    """
    history = [g.loc[g.year <= cutoff].copy() for g in series]
    origins = [cutoff - 15, cutoff - 10, cutoff - 5]
    if any(len(g.loc[g.year <= origins[0]]) < 3 for g in history):
        raise ValueError("Tuning needs at least 18 annual history rows through the cutoff.")
    search = candidates(model_name) if search is None else search
    rows = []
    best_score, best_pipeline, best_params = np.inf, None, None
    for index, (params, template) in enumerate(search):
        scores = []
        for origin in origins:
            fitted = fit_model(clone(template), history, origin)
            scores.append(validation_mae(fitted, history, origin, cutoff))
        score = float(np.mean(scores))
        rows.append({
            "candidate": index, "parameters": json.dumps(params, sort_keys=True),
            **{f"mae_from_{origin + 1}_mm": value for origin, value in zip(origins, scores)},
            "validation_mae_mm": score,
        })
        if score < best_score:
            best_score, best_pipeline, best_params = score, clone(template), params
        if (index + 1) % 8 == 0 or index + 1 == len(search):
            print(f"{model_name}: evaluated {index + 1}/{len(search)} candidates", flush=True)
    if best_pipeline is None:
        raise ValueError("No finite tuning candidate was found.")
    metadata = {
        "selected_parameters": best_params, "validation_mae_mm": best_score,
        "validation_origins": origins, "validation_end_year": cutoff,
        "selection_metric": "mean recursive depth MAE across three historical origins",
        "candidate_count": len(search),
    }
    return best_pipeline, metadata, pd.DataFrame(rows).sort_values("validation_mae_mm")


def tune(data_path, output_dir, model_names=("linear", "svr"), cutoff=CUTOFF_YEAR):
    series = load_series(data_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    comparisons = []
    for name in model_names:
        selected, metadata, trials = select_parameters(series, name, cutoff)
        trials.to_csv(output_dir / f"timber_depth_{name}.tuning.csv", index=False)
        # Only after selection is locked do we access the final test depths.
        baseline = make_model(name)
        for variant, pipeline in (("baseline", baseline), ("tuned", selected)):
            path = output_dir / f"timber_depth_{name}_{variant}.joblib"
            artifact = train_and_evaluate(
                data_path, path, pipeline, name, cutoff, ("primary",),
            )
            if variant == "tuned":
                artifact["tuning"] = metadata
                joblib.dump(artifact, path)
                path.with_suffix(".tuning.json").write_text(
                    json.dumps(metadata, indent=2), encoding="utf-8",
                )
            comparisons.append({"variant": variant, **artifact["metrics"]})
    comparison = pd.DataFrame(comparisons)
    comparison.to_csv(output_dir / "comparison.csv", index=False)
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=MODEL_DIR / "timber_tuned")
    parser.add_argument("--models", nargs="+", choices=["linear", "svr"], default=["linear", "svr"])
    parser.add_argument("--cutoff", type=int, default=CUTOFF_YEAR)
    args = parser.parse_args()
    print(tune(args.data, args.output_dir, args.models, args.cutoff).to_string(index=False))


if __name__ == "__main__":
    main()
