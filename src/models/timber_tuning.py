"""Optuna hyperparameter search and evaluation for the timber MLP/GBM surrogates.

    python -m src.models.timber_tuning --model gbm --trials 100   # tune one model
    python -m src.models.timber_tuning --model mlp --trials 60
    python -m src.models.timber_tuning --evaluate                 # ablation + final metrics

Tuning never sees the test rows. ``time_based_split`` first holds out the
last 20% of years of every pile/scenario series as the test set (exactly as
``train()`` does); Optuna then scores each trial with blocked, expanding-window
time-series CV *inside the remaining training years* (``blocked_time_folds``),
so the chosen hyperparameters are selected on forecasting later training
years, not on the test years they are finally reported against. Paste the
printed best params into TUNED_MLP_PARAMS (src/models/timber_model.py) or
TUNED_GBM_PARAMS (src/models/timber_model_gbm.py). Write-up and results:
docs/timber_model_tuning.md.
"""

from __future__ import annotations

import argparse
import json
import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import MODEL_DIR
from src.data.timber_data_prep import DATA_PATH, load_timber_data
from src.features.timber_features import TARGET_COLUMN, add_thermal_dose, build_feature_matrix
from src.models.timber_model import (
    MODEL_KINDS,
    DeteriorationRateRegressor,
    build_estimator,
    build_pipeline,
    build_tuned_model,
    pile_based_split,
    suggest_mlp_params,
    time_based_split,
)
from src.models.timber_model_gbm import suggest_gbm_params

RESULTS_PATH = MODEL_DIR / "timber_tuning_results.json"
SERIES_KEY = ["asset_code", "scenario_label"]


def blocked_time_folds(
    frame: pd.DataFrame, n_folds: int = 3, fold_fraction: float = 0.15
) -> list[tuple[pd.Index, pd.Index]]:
    """Expanding-window folds over each series' own years (forward chaining).

    Fold k validates on a block of ``fold_fraction`` of each series' years
    and trains on every year before it, with the blocks walking back from the
    end of ``frame``. Validation years are always later than training years,
    mirroring the forecast-forward test split instead of shuffled K-fold,
    which would let a model interpolate between neighbouring years.
    """
    folds = []
    for k in range(n_folds, 0, -1):
        train_parts, val_parts = [], []
        for _, group in frame.groupby(SERIES_KEY, sort=False):
            years = np.sort(group["year"].unique())
            n = len(years)
            val_end = n - int(round((k - 1) * fold_fraction * n))
            val_start = val_end - int(round(fold_fraction * n))
            train_parts.append(group.index[group["year"] < years[val_start]])
            val_parts.append(group.index[group["year"].isin(years[val_start:val_end])])
        folds.append((pd.Index(np.concatenate(train_parts)), pd.Index(np.concatenate(val_parts))))
    return folds


def _rmse(y_true, y_pred) -> float:
    return float(mean_squared_error(y_true, y_pred) ** 0.5)


def metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": _rmse(y_true, y_pred),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def tune(model_kind: str, n_trials: int, frame: pd.DataFrame, seed: int = 42) -> dict[str, Any]:
    """Run an Optuna TPE search; return the best params and CV score."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    train_idx, _ = time_based_split(frame)
    train_frame = frame.loc[train_idx]
    features = build_feature_matrix(train_frame)
    target = train_frame[TARGET_COLUMN]
    folds = blocked_time_folds(train_frame)
    suggest = suggest_mlp_params if model_kind == "mlp" else suggest_gbm_params

    def objective(trial) -> float:
        params = suggest(trial)
        scores = []
        for step, (fit_idx, val_idx) in enumerate(folds):
            model = DeteriorationRateRegressor(
                build_estimator(model_kind, params, random_state=seed)
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(features.loc[fit_idx], target.loc[fit_idx])
            scores.append(_rmse(target.loc[val_idx], model.predict(features.loc[val_idx])))
            trial.report(float(np.mean(scores)), step)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(scores))

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = suggest(optuna.trial.FixedTrial(study.best_params))
    return {
        "model_kind": model_kind,
        "n_trials": n_trials,
        "cv_rmse": study.best_value,
        "optuna_params": study.best_params,
        "estimator_params": best,
    }


def _fit_score(model, features, target, train_idx, test_idx) -> dict[str, float]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(features.loc[train_idx], target.loc[train_idx])
    return metrics(target.loc[test_idx], model.predict(features.loc[test_idx]))


class _WithThermalDose:
    """Raw-diameter estimator that only adds thermal_dose (ablation step 2)."""

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y):
        self.estimator.fit(add_thermal_dose(X), y)
        return self

    def predict(self, X):
        return self.estimator.predict(add_thermal_dose(X))


def evaluate(frame: pd.DataFrame, seeds: tuple[int, ...] = (0, 1, 2, 3, 42)) -> dict[str, Any]:
    """Ablation (one change at a time) on both splits, plus seed stability of the final models."""
    features = build_feature_matrix(frame)
    target = frame[TARGET_COLUMN]
    splits = {"time": time_based_split(frame), "pile": pile_based_split(frame)}
    steps = {
        "1_baseline": lambda kind: build_pipeline(kind),
        "2_thermal_dose": lambda kind: _WithThermalDose(build_pipeline(kind)),
        "3_rate_target": lambda kind: DeteriorationRateRegressor(build_estimator(kind)),
        "4_rate_target_tuned": lambda kind: build_tuned_model(kind),
    }
    rows = []
    for split_name, (train_idx, test_idx) in splits.items():
        for step, make in steps.items():
            for kind in MODEL_KINDS:
                train_scores = _fit_score(make(kind), features, target, train_idx, train_idx)
                test_scores = _fit_score(make(kind), features, target, train_idx, test_idx)
                rows.append(
                    {
                        "split": split_name,
                        "step": step,
                        "model": kind,
                        **{f"train_{k}": v for k, v in train_scores.items()},
                        **{f"test_{k}": v for k, v in test_scores.items()},
                    }
                )

    train_idx, test_idx = splits["time"]
    stability = {}
    for kind in MODEL_KINDS:
        per_seed = [
            _fit_score(
                build_tuned_model(kind, random_state=s), features, target, train_idx, test_idx
            )
            for s in seeds
        ]
        stability[kind] = {
            m: {
                "mean": float(np.mean([p[m] for p in per_seed])),
                "std": float(np.std([p[m] for p in per_seed])),
            }
            for m in ("r2", "rmse", "mae")
        }
    return {"ablation": rows, "seed_stability_time_split": stability, "seeds": list(seeds)}


def _print_ablation(rows: list[dict[str, Any]]) -> None:
    table = pd.DataFrame(rows)
    cols = ["split", "step", "model", "test_r2", "test_rmse", "test_mae", "train_r2"]
    print(table[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", choices=MODEL_KINDS)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()

    frame = load_timber_data(DATA_PATH)
    results = json.loads(RESULTS_PATH.read_text()) if RESULTS_PATH.exists() else {}
    if args.model:
        best = tune(args.model, args.trials, frame)
        print(f"Best blocked-CV RMSE: {best['cv_rmse']:.2f} mm")
        print(f"Paste into TUNED_{args.model.upper()}_PARAMS: {best['estimator_params']!r}")
        results[f"tuning_{args.model}"] = {
            **best,
            "estimator_params": repr(best["estimator_params"]),
        }
    if args.evaluate:
        evaluation = evaluate(frame)
        _print_ablation(evaluation["ablation"])
        print(
            "Seed stability (time split, tuned):",
            json.dumps(evaluation["seed_stability_time_split"], indent=2),
        )
        results["evaluation"] = evaluation
    if not (args.model or args.evaluate):
        parser.error("pass --model and/or --evaluate")
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    print(f"Saved results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
