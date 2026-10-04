# Port Pile Deterioration Dashboard

A Streamlit dashboard for predicting deterioration levels of concrete, steel, and timber piles at seaports. The first version follows the supplied low-fidelity layout: side navigation, prediction parameters, condition indicators, and a historical-to-future forecast chart.

## Quick start

Requires Python 3.11 or newer.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Repository structure

```text
FYP/
├── .streamlit/             # Shared Streamlit theme and server configuration
├── artifacts/              # Produced/trained model files (not committed)
├── docs/                   # Architecture notes, data dictionary, and team decisions
├── notebooks/              # Exploratory analysis and model experiments. 
├── src/
│   ├── data/               # CSV ingestion, validation, and data cleaning
│   ├── features/           # Reusable feature-engineering transformations
│   ├── models/             # Training, evaluation, persistence, and inference code
│   ├── ui/                 # Streamlit pages and reusable interface components
│   ├── visualization/      # Plotly chart and future map builders
│   ├── timber/             # Timber pile pipeline + its own streamlit/ app (see below)
│   └── config.py           # Central project paths and application settings
├── tests/                  # Automated tests mirroring production modules
├── app.py                  # Concrete-pile dashboard entry point
├── pyproject.toml          # Test and lint tool configuration
└── requirements.txt        # Reproducible application dependencies
```

## Suggested four-person ownership

| Role | Primary area | Initial responsibility |
|---|---|---|
| Data engineer | `src/data`, `docs` | Define the uploaded CSV schema, validation, and missing-value policy |
| ML engineer | `src/features`, `src/models`, `notebooks` | Establish target labels, train/evaluate models, and expose inference |
| Dashboard developer | `src/ui`, `app.py` | Build Streamlit workflows, controls, state, and accessibility |
| Visualization/QA developer | `src/visualization`, `tests`, `docs` | Create charts/maps, test integration, and maintain documentation |

Use short-lived feature branches and pull requests. Keep notebook logic exploratory; move any code used by the app into `src/` with tests. Do not commit uploaded datasets.

## Development checks

```bash
pytest
ruff check .
```

## Dataset integration checklist

1. Add a data dictionary to `docs/` describing units, missing values, pile identifiers, inspection dates, coordinates, and the deterioration target.
2. Add schema validation for processed uploads in `src/data/`.
3. Move repeatable transformations from notebooks to `src/features/`.
4. Save the produced model under `artifacts/` and connect it through `src/models/predict.py`.
5. Replace the demonstration series in `src/visualization/charts.py` with model output and add map layers under the Geospatial Mapping page.

## Timber pile replacement forecast (separate app)

A second pipeline forecasts when each timber pile will need replacing
under three climate scenarios. Its modules live alongside the concrete
ones in `src/data/`, `src/features/`, `src/models/`, and `src/ui/`.

The core pipeline (`timber_data_prep.py`, `timber_features.py`, `timber_model.py`,
`timber_scenarios.py`) has no Streamlit dependency, so it can be trained,
retrained, and analysed entirely from the command line -- the Streamlit
UI is isolated in `src/ui/timber_app.py`:

```text
src/data/
└── timber_data_prep.py           # load the CSV, fix the MIROC-Medres scenario_label gap, clean dtypes
src/features/
├── timber_features.py            # feature/target columns, decay_dose engineering, feature matrix builder
└── timber_targets.py             # alternate targets: distance-to-threshold, residual-correction
src/models/
├── timber_model.py               # StandardScaler + MLPRegressor pipeline, train/save/load, splits
├── timber_model_gbm.py           # LightGBM (falls back to XGBoost) regressor + quantile builders
├── timber_residual.py            # residual-correction model vs. direct surrogate comparison
├── timber_quantile.py            # 10th/50th/90th percentile GBM models, replacement-year bands
├── timber_baselines.py           # trivial non-fitted comparison baseline
├── timber_reporting.py           # error-by-scenario/pile/age breakdown helpers
├── timber_shap_importance.py     # SHAP feature importance + constant-within-pile flagging
├── timber_compare.py             # CLI tying the above together, see below
├── timber_scenarios.py           # per-pile scenario trajectories and replacement-year detection
├── timber_engine.py              # verbatim port of the Timber_kembla_v6.3 workbook calculation
├── timber_coefficients.py        # workbook Coefficients-sheet lookup tables used by timber_engine.py
├── timber_inference.py           # batch inference for the timber dashboard CSV export
├── timber_tuning.py              # Optuna tuning on the time-based split (docs/timber_model_tuning.md)
└── timber_transfer.py            # train PileA -> test PileB, MC CV tuning (docs/timber_pileA_to_pileB_report.md)
src/ui/
├── timber_app.py                 # Streamlit entry point: pile selector, scenario chart, summary
└── timber_results_app.py         # PileB material-loss forecast from the PileA-trained models
tests/test_timber_pipeline.py
tests/test_timber_compare.py
```

### Run the model / do analysis (no Streamlit needed)

```bash
python -m src.models.timber_model
```

Trains the pipeline, prints the split strategy, row counts, and train/test
MAE/R² to the console, and saves the model to
`artifacts/timber_condition_model.joblib`. For anything beyond the
headline metrics -- error by scenario/pile/age, permutation feature
importance, per-pile replacement-year predictions -- import the modules
directly in a script or notebook, e.g.:

```python
from src.data.timber_data_prep import load_timber_data, DATA_PATH
from src.models.timber_model import train
from src.models.timber_scenarios import summarize_pile

frame = load_timber_data(DATA_PATH)
result = train(frame, split_strategy="time")
print(summarize_pile(frame, result.pipeline, "PileA"))
```

See [`docs/timber_model_results.md`](docs/timber_model_results.md) for a
full write-up (accuracy, error breakdowns, feature importance, and
per-pile replacement-year predictions) from the last training run,
including how each number there was produced.

### Compare model architectures

```bash
python -m src.models.timber_compare --model gbm                                    # MLP vs GBM vs naive baseline, GBM deep-dive
python -m src.models.timber_compare --model mlp --target distance --features decay_dose
python -m src.models.timber_compare --model residual --residual-base gbm            # residual-correction vs. direct surrogate
python -m src.models.timber_compare --model gbm --quantile                          # + 10/50/90th percentile replacement-year bands
```

Every mode reuses `time_based_split`/`pile_based_split` from
`src/models/timber_model.py` unchanged, so results are directly comparable to
the MLP baseline in `docs/timber_model_results.md`. See
[`docs/timber_model_comparison.md`](docs/timber_model_comparison.md) for
the full MLP-vs-GBM write-up (head-to-head tables, individual deep dives,
residual-correction, quantile bands, and the `decay_dose`/distance-target
experiments) with a recommendation section for the team, or
[`docs/timber_project_report.md`](docs/timber_project_report.md) for a
shorter, single-file project report (dataset, target/features, models,
results, findings, accuracy summary, limitations).

`--model` always prints the shared MLP-vs-GBM-vs-naive comparison table
first, then a deep-dive for the requested model: train/test MAE/RMSE/R², error by
`scenario_label`/`asset_code`/age bucket, the leave-one-pile-out
diagnostic, and SHAP feature importance (flagging columns that are
constant within a single `asset_code`, since those are confounded with
pile identity rather than real trend signal). `--target distance` trains
against `remaining_diameter_mm - replacement_diameter_mm` instead of raw
diameter; `--features decay_dose` swaps `age_years`,
`summer_sea_temp_c`, `salinity_factor`, and `timber_exposure_factor` for
a single cumulative `decay_dose` feature (see `add_decay_dose` in
`src/features/timber_features.py`). GBM is LightGBM by default, falling back to
XGBoost if LightGBM isn't installed (`src/models/timber_model_gbm.py`).

### Run the Streamlit app

```bash
streamlit run src/ui/timber_app.py
```

The first run trains the MLP automatically (there is no committed model
binary) and caches it to `artifacts/timber_condition_model.joblib`.
Subsequent runs load the cached model; press **Retrain model** in the
sidebar to re-fit and overwrite it. Train/test MAE and R² are printed to
the sidebar whenever the model is (re)trained.

### Assumptions worth challenging

- **MIROC-Medres relabeling.** The source spreadsheet exports
  `scenario_label` as blank for every `climate_model == "MIROC-Medres"`
  row. `src/data/timber_data_prep.py` fills these with `"Colder & Wetter"`,
  inferred from this project's three-GCM naming convention (CSIRO Mk3.5 =
  Hotter & Drier, MRI-CGCM2.3.2 = Most Likely) and the sibling concrete
  dataset's provenance notes in
  `docs/concrete_kembla_v6_data_dictionary.md`, which document the
  equivalent GCM as `"MIROC(hires) (Wetter)"`. This has **not** been
  verified against the original `Timber_kembla_v6.3` workbook's scenario
  sheets — re-check there before treating the label as authoritative.
- **Only two piles exist in the dataset (PileA, PileB).** All 994 rows are
  71 years × up to 2 emissions pathways × 4 scenarios for those two
  assets. The MLP's reported MAE/R² describe how well it fits *these two
  piles' own trajectories* (using a chronological train/test split, see
  `time_based_split` in `src/models/timber_model.py`); they are not evidence the
  model would generalize to a different timber species, exposure class,
  or port. A leave-one-pile-out split is also available
  (`train(df, split_strategy="pile")`) as a harder diagnostic check, and
  performs noticeably worse in practice — expected, since it trains on a
  single asset and tests on a structurally different one.
- **Scenario trajectories are forced non-increasing.** The MLP is a
  smooth regressor with no physical constraint, so `src/models/timber_scenarios.py`
  applies a running minimum across each scenario's predicted years before
  looking for the replacement-threshold crossing, since a pile's diameter
  cannot physically recover once it starts deteriorating.
- **History/projection split point.** The chart splices each scenario's
  future rows onto the shared `Baseline` scenario as "history" up to the
  current calendar year (clamped into the data's 2000–2070 range), then
  lets the three scenarios diverge from there — see
  `default_split_year()` in `src/models/timber_scenarios.py`.
