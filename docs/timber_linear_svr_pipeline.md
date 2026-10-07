# Timber model pipeline

Timber follows the concrete naming and separation of responsibilities in
`src/models/`. Material-specific names avoid colliding with concrete's existing
`train_final_model.py` and `inference.py`.

| Concrete file | Timber counterpart | Responsibility |
|---|---|---|
| `rnn_model.py` | `linear_model.py`, `svr_model.py` | Model definitions and training entry points |
| `concrete_eval.py` | `timber_eval.py` | Model selection, shared fitting, rollout, scoring and benchmark folds |
| `concrete_benchmark_all.py` | `timber_benchmark_all.py` | Compare all models; write benchmark CSVs |
| `train_final_model.py` | `train_final_timber_model.py` | Fit deployment models on all available history |
| `inference.py` | `timber_inference.py` | Load artifacts and produce depth forecasts |
| `experiments/concrete/` | `experiments/timber/` | Experiments; timber tuning is `tune_models.py` |

`linear_model.py` and `svr_model.py` replace `train_timber_linear.py` and
`train_timber_svr.py`. Model selection (`ALL_MODELS`, `make_model`) lives in
`timber_eval.py`. Linear Regression and Ridge are defined in `linear_model.py`;
SVR is defined in `svr_model.py`. There is no separate `timber_model.py`.
The redundant `predict.py` and `tune_timber.py` adapters have been removed.
Prediction functions live in `timber_inference.py`, matching concrete's
`inference.py`; tuning runs via `experiments.timber.tune_models`.
Existing `src/timber/` MLP/GBM and dashboard paths retain their own behavior.

Individual training commands are `python -m src.models.linear_model` and
`python -m src.models.svr_model`; their arguments and behavior are unchanged.

## Forecast quantity and mechanics

Output is radial deterioration depth in **mm**, the same quantity as the
MLP/GBM dashboard: `(initial_diameter_mm - remaining_diameter_mm) / 2`.
The supervised target is annual change in `mean_deterioration_mm`.
Models use the 13 numeric timber features plus the previous annual increment.
Both the feature list (`TIMBER_FEATURES`) and default CSV path (`DATA_PATH`)
are defined locally in `src/models/timber_eval.py`. Linear Regression, Ridge,
SVR, their benchmark, final trainer, inference and tuning do not import
`src/timber/`. They share only the processed CSV with that separate pipeline.

Series remain separate by pile, emissions pathway and climate model. With the
current dataset there are 14 series and 994 rows. Benchmark training uses years
through 2026, then recursively predicts increments for 2027–2070 and accumulates
depth from the 2026 anchor. Predicted increments become subsequent lags; future
true depths are never used in rollout. Future environmental inputs are required.

## Benchmark (accuracy)

Run from the repository root in PowerShell:

```powershell
.venv/Scripts/python.exe -m src.models.timber_benchmark_all
```

As with concrete, flags are `--csv`, `--models`, `--runs`, `--cutoff`, and
`--out`. Model and run selections are comma-separated:

```powershell
.venv/Scripts/python.exe -m src.models.timber_benchmark_all --models linear,ridge,svr --runs primary --cutoff 2026
```

Defaults run primary, leave-one-scenario-out, and leave-one-asset-out evaluations.
Outputs follow concrete's naming and metric column convention:

- `results_timber_benchmark.csv`: per-fold summary.
- `results_timber_benchmark_detail.csv`: per-series scores.
- Metrics: `mae`, `rmse`, `r2`, `final_mae` (errors are mm).
- Benchmarking does not write deployment model artifacts.

Held-out series supply their historical depth anchor and last increment; these
are not history-free predictions. Summary metrics average per-series metrics.

## Final training (deployment artifacts)

```powershell
.venv/Scripts/python.exe -m src.models.train_final_timber_model --models linear,ridge,svr --out artifacts
```

Like concrete's final trainer, this fits on **all available history**, currently
through 2070, and reports **no held-out accuracy**. An optional positional CSV
path and `--out` match the concrete final trainer convention.

Artifacts: `artifacts/timber_depth_<model>.joblib` and a matching `.json`
metadata file. Ridge uses the historically selected `alpha=1000`; SVR retains
the original `C=100, epsilon=0.1, gamma=scale` because tuning did not improve
its final-test performance.

## Inference

To reproduce a trajectory with an all-history final model:

```powershell
.venv/Scripts/python.exe -m src.models.timber_inference --model artifacts/timber_depth_ridge.joblib --data data/processed/timber_kembla_v6_3_consolidated.csv --anchor-year 2026 --end-year 2070 --output artifacts/timber_ridge_forecast.csv
```

This is a reconstruction using a model fitted through 2070, **not a held-out
forecast**. Use the benchmark command for accuracy. Forecasting beyond 2070
requires additional future input rows and historical depths up to the chosen
anchor. The caller must specify `--anchor-year` for final-model artifacts.

Research artifacts from tuning retain their training cutoff and can be replayed
without overriding the anchor:

```powershell
.venv/Scripts/python.exe -m src.models.timber_inference --model artifacts/timber_tuned/timber_depth_linear_tuned.joblib --data data/processed/timber_kembla_v6_3_consolidated.csv --end-year 2070 --output artifacts/timber_tuned/ridge_forecast.csv
```

Output columns are `material, asset_id, scenario, year, value, unit, metric, method`.
`--start-year` filters output only; it does not re-anchor the forecast.
Future reference depths may be empty; required historical depths and future
input variables must be present. Output is not automatically installed into the dashboard.

## Tuning experiment

```powershell
.venv/Scripts/python.exe -m experiments.timber.tune_models
```

The search evaluates 16 linear-family configurations (ordinary regression and
Ridge) and 48 RBF SVR configurations. Three expanding training windows end in
2011, 2016 and 2021; each forecasts recursively through 2026. Parameters minimize
the mean validation depth MAE across these origins. Post-2026 data is excluded
from selection, then evaluated once with the selected configuration.

| Configuration | Test MAE (mm) | Test RMSE (mm) | Mean per-series R² |
|---|---:|---:|---:|
| Original Linear Regression | 142.64 | 204.85 | -15.586 |
| Selected Ridge (alpha=1000, intercept=True) | 28.40 | 33.57 | 0.553 |
| Original SVR | 30.41 | 40.20 | 0.623 |
| Selected SVR (C=1, epsilon=0.01, gamma=0.01) | 43.87 | 59.02 | 0.112 |

Ridge improved the linear family; selected SVR worsened final-test performance.
Search results, cutoff-trained research artifacts, and `comparison.csv` are
saved under `artifacts/timber_tuned/`. These are primary benchmark results, not
nested tuned asset/scenario holdout scores.

## Tests

A fresh temporary directory avoids the Windows pytest-folder permission error:

```powershell
$timberTestTemp = Join-Path (Get-Location).Path ('artifacts/pytest-' + [guid]::NewGuid().ToString('N'))
.venv/Scripts/python.exe -m pytest -q -p no:cacheprovider --basetemp "$timberTestTemp"
```

Tests verify recursive feedback, exclusion of future targets, tuning isolation,
saved-model replay, input validation, holdout splits, and separation of benchmark
and final training. Smoke-test outputs are under `artifacts/timber_structure_check/`.

The reference depths are deterministic workbook outputs, not field observations.
Neither regressor imposes physical depth bounds. Existing MLP/GBM accuracy
numbers use different splits and are not directly comparable.
