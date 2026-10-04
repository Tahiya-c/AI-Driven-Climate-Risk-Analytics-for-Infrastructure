"""Optuna hyperparameter search for the concrete carbonation models.

    python -m src.models.concrete_tuning --model rnn --trials 40
    python -m src.models.concrete_tuning --model ridge,random_forest,catboost --trials 40
    python -m src.models.concrete_tuning --model all --trials 40

Covers the RNN plus the three tabular models (ridge, random_forest, catboost)
that are the real, correctly-trained equivalents of Aisha's original
experiments/concrete/ scripts (those were archived: synthetic data, wrong
leaky target -- see that directory's header comments). sqrt_law/wc_regress/
persistence/mean have no real hyperparameters to tune and are left alone.

Tuning never scores against years after CUTOFF_YEAR (2026) -- that period
(2027-2070) is the real forecast horizon concrete_benchmark_all.py reports
against, and must stay unseen by the search. Instead each trial is scored
with INNER leave-one-scenario-out folds at pseudo-cutoffs strictly earlier
than 2026 (INNER_CUTOFFS): train on a series' history up to the pseudo-cutoff,
roll forward to 2026 with concrete_eval.rollout exactly as the real benchmark
does, and score against the (real, workbook-modelled) depths in between.
Same rollout/scoring code as concrete_eval.py, just earlier in time -- so
model selection is honest and the real 2027-2070 benchmark numbers it is
checked against afterwards are still an unbiased outer score.

Paste the printed best params into the matching TUNED_*_PARAMS dict in
src/models/concrete_eval.py, then re-run
`python -m src.models.concrete_benchmark_all` to confirm the tuned model
actually improves the real (outer) leaderboard before keeping the change.
Write-up: docs/concrete_rnn_tuning.md.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np

from src.models.concrete_eval import (
    CatBoostModel,
    RandomForestModel,
    RidgeModel,
    RnnModel,
    rollout,
    score_forecast,
)
from src.models.rnn_model import load_series

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "processed" / "concrete_kembla_v6_consolidated.csv"
RESULTS_PATH = ROOT / "artifacts" / "concrete_tuning_results.json"

# Expanding window: tune once with a 5-year-out inner test (2022-2026), once
# with an 8-year-out one (2019-2026), so a trial can't just win by overfitting
# to one horizon length. Both stay strictly before CUTOFF_YEAR.
INNER_CUTOFFS = (2018, 2021)
# A subset of the 7 scenario groups (not all 7) to keep a trial's cost down --
# "small search space, only 49 series" per the project's tuning conventions.
INNER_HELD_OUT = ("A1FI|CSIRO Mk3.5", "A1B|MIROC(hires)", "Base|(Reference)")

MODEL_NAMES = ("rnn", "ridge", "random_forest", "catboost")


def suggest_rnn_params(trial) -> dict:
    """Centred on rnn_model.py's current constants (window=12, hidden=16,
    lr=5e-3, patience=30) with room either side."""
    return {
        "window": trial.suggest_int("window", 6, 16),
        "hidden": trial.suggest_int("hidden", 8, 48, log=True),
        "lr": trial.suggest_float("lr", 5e-4, 2e-2, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-7, 1e-2, log=True),
        "patience": trial.suggest_int("patience", 10, 40),
    }


def suggest_ridge_params(trial) -> dict:
    return {"alpha": trial.suggest_float("alpha", 1e-3, 1e3, log=True)}


def suggest_rf_params(trial) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500, log=True),
        "max_depth": trial.suggest_categorical("max_depth", [3, 5, 8, 12, None]),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 1.0]),
    }


def suggest_catboost_params(trial) -> dict:
    return {
        "iterations": trial.suggest_int("iterations", 100, 800, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "depth": trial.suggest_int("depth", 3, 8),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-1, 10.0, log=True),
    }


SUGGEST = {
    "rnn": suggest_rnn_params,
    "ridge": suggest_ridge_params,
    "random_forest": suggest_rf_params,
    "catboost": suggest_catboost_params,
}
BUILD = {
    "rnn": lambda params, seed: RnnModel(seeds=(seed,), params=params),
    "ridge": lambda params, seed: RidgeModel(params=params),
    "random_forest": lambda params, seed: RandomForestModel(params=params),
    "catboost": lambda params, seed: CatBoostModel(params=params),
}


def _inner_fold_mae(
    model_name: str, series: list[dict], params: dict, cutoff: int, held_out: str, seed: int
) -> float:
    train = [s for s in series if s["scenario"] != held_out]
    test = [s for s in series if s["scenario"] == held_out]
    model = BUILD[model_name](params, seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(train, cutoff)
    maes = []
    for s in test:
        years, depth = rollout(model, s, cutoff)
        maes.append(score_forecast(years, depth, s)["mae"])
    return float(np.mean(maes))


def tune(model_name: str, n_trials: int, series: list[dict], seed: int = 42) -> dict:
    """Run an Optuna TPE search over INNER_CUTOFFS x INNER_HELD_OUT; return the best params."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    suggest = SUGGEST[model_name]

    def objective(trial) -> float:
        params = suggest(trial)
        scores = []
        for step, cutoff in enumerate(INNER_CUTOFFS):
            for held_out in INNER_HELD_OUT:
                scores.append(_inner_fold_mae(model_name, series, params, cutoff, held_out, seed))
            trial.report(float(np.mean(scores)), step)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(scores))

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=8),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return {
        "model": model_name,
        "n_trials": n_trials,
        "inner_cv_mae": study.best_value,
        "params": study.best_params,
        "inner_cutoffs": list(INNER_CUTOFFS),
        "inner_held_out": list(INNER_HELD_OUT),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", default="rnn",
                         help=f"comma list from {MODEL_NAMES}, or 'all'")
    parser.add_argument("--trials", type=int, default=40)
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    args = parser.parse_args()

    models = list(MODEL_NAMES) if args.model == "all" else args.model.split(",")
    unknown = [m for m in models if m not in MODEL_NAMES]
    if unknown:
        parser.error(f"unknown model(s) {unknown}; choose from {MODEL_NAMES}")
    if "catboost" in models:
        try:
            import catboost  # noqa: F401
        except ImportError:
            print("catboost not installed -- skipping it.")
            models = [m for m in models if m != "catboost"]

    series, _ = load_series(args.data)
    print(f"{len(series)} series loaded; tuning over {len(INNER_CUTOFFS)} cutoffs x "
          f"{len(INNER_HELD_OUT)} held-out scenarios = {len(INNER_CUTOFFS) * len(INNER_HELD_OUT)} "
          f"fits/trial x {args.trials} trials, for: {', '.join(models)}\n")

    results = json.loads(RESULTS_PATH.read_text()) if RESULTS_PATH.exists() else {}
    for model_name in models:
        print(f"--- tuning {model_name} ---")
        best = tune(model_name, args.trials, series)
        print(f"Best inner-CV MAE: {best['inner_cv_mae']:.4f} mm")
        tuned_name = {
            "rnn": "TUNED_RNN_PARAMS", "ridge": "TUNED_RIDGE_PARAMS",
            "random_forest": "TUNED_RF_PARAMS", "catboost": "TUNED_CATBOOST_PARAMS",
        }[model_name]
        print(f"Paste into {tuned_name} (src/models/concrete_eval.py): {best['params']!r}\n")
        results[model_name] = best

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    print(f"Saved results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
