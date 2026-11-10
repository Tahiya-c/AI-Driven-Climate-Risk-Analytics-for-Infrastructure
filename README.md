# AI-Driven Climate Risk Analytics for Infrastructure

A machine-learning dashboard that forecasts how concrete and timber port piles deteriorate under different climate change scenarios, helping port infrastructure teams plan inspections and replacements ahead of failure.

This project was built as a **final year university capstone project**, developed by a four-person team over roughly a year of research, model development, and testing — from scoping the problem and the underlying engineering datasets, through iterating on multiple model architectures, to a working Streamlit dashboard backed by a Supabase database.

## Overview

Seaport piles (concrete and timber) degrade over time due to environmental exposure — salinity, sea temperature, moisture, and other climate-linked factors. Port operators traditionally rely on deterministic engineering models and manual inspection schedules to estimate when a pile needs replacing. This project explores whether machine learning surrogate models can reproduce and extend those forecasts, and presents the results through an interactive dashboard that:

- Visualises each pile's predicted condition over time under multiple climate scenarios (`Baseline`, `Hotter & Drier`, `Most Likely`, `Colder & Wetter`).
- Forecasts the year a pile is expected to cross its replacement threshold.
- Lets engineers import new inspection data and compare predictions against the existing deterministic models.

## Key Features

- **Two independent forecasting pipelines** — a concrete carbonation-depth model (RNN / Ridge / CatBoost / Random Forest, benchmarked against each other) and a timber diameter-loss model (MLP / GBM, with quantile uncertainty bands).
- **Scenario-based forecasting** across multiple climate projections, with historical-to-future trend charts.
- **Data import pipeline** with schema validation for uploading new concrete/timber inspection datasets (`.xlsx`/`.csv`).
- **Supabase-backed persistence** for imported data, forecasts, and dashboard state.
- **Reproducible evaluation** — every model is trained and scored through the same benchmarking harness, with results and methodology written up in [`docs/`](docs).

## Tech Stack

| Layer | Technology |
|---|---|
| Dashboard / UI | [Streamlit](https://streamlit.io/), [Plotly](https://plotly.com/python/) |
| Modelling | scikit-learn, PyTorch (RNN), LightGBM/XGBoost, CatBoost, Optuna (hyperparameter tuning), SHAP (feature importance) |
| Data | pandas, NumPy, openpyxl |
| Backend / storage | [Supabase](https://supabase.com/) |
| Testing / tooling | pytest, ruff |

## Getting Started

Requires Python 3.11 or newer.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Supabase integration (data import/persistence) is optional for local exploration — see `src/db/client.py` for the environment variables required to connect it (`SUPABASE_URL`, `SUPABASE_SECRET_KEY`).

### Running tests and linting

```bash
pytest
ruff check .
```

## Repository Structure

```text
.
├── app.py                  # Concrete-pile dashboard entry point
├── src/
│   ├── data/                # CSV/XLSX ingestion, validation, and cleaning
│   ├── features/             # Feature-engineering transformations
│   ├── models/               # Training, evaluation, persistence, and inference
│   ├── db/                   # Supabase client, repository, and import pipeline
│   ├── ui/                   # Streamlit pages and dashboard components
│   └── visualization/        # Plotly chart builders
├── artifacts/                # Trained model files (models of record are committed)
├── data/                     # Raw, processed, sample, and template datasets
├── docs/                     # Methodology write-ups, data dictionaries, model reports
├── notebooks/                # Exploratory analysis and experimentation
├── experiments/               # Benchmarking and diagnostic scripts
├── scripts/                   # One-off retraining / evaluation entry points
└── tests/                    # Automated tests mirroring the production modules
```

## Modelling Summary

**Concrete piles** — forecast carbonation depth gain over time. Five architectures were benchmarked head-to-head (RNN, Ridge, Random Forest, CatBoost, plus a persistence baseline) across both a known-pile and an unseen-pile (leave-one-pile-out) evaluation. See [`docs/concrete_model_comparison.md`](docs/concrete_model_comparison.md) for the full results and discussion.

**Timber piles** — forecast remaining diameter and the year each pile crosses its replacement threshold, under three climate scenarios. MLP and GBM surrogates were compared against a naive baseline, with residual-correction and quantile-uncertainty variants also evaluated. See [`docs/timber_project_report.md`](docs/timber_project_report.md) for the full write-up, or [`docs/timber_model_comparison.md`](docs/timber_model_comparison.md) for the detailed MLP-vs-GBM comparison.

Both pipelines are usable from the command line independent of the dashboard — see the relevant docs above for the training/evaluation commands.

## Known Limitations

- Both datasets are the **output of existing deterministic engineering models**, not raw field measurements — the ML models are surrogates learning to reproduce those models' behaviour, not independent physical measurements.
- The timber dataset covers only **two piles**, which limits how confidently results generalise across pile types, species, or ports; this is documented and stress-tested via a leave-one-pile-out evaluation (see `docs/timber_project_report.md`).
- A spreadsheet labelling gap in the timber dataset was inferred and relabelled rather than confirmed against the source workbook — see `src/data/timber_data_prep.py` and `docs/timber_model_results.md` for details.

## Documentation

Further methodology, data dictionaries, and model reports are in [`docs/`](docs), including data dictionaries, model tuning notes, and per-model evaluation reports for both the concrete and timber pipelines.

## Team

Built by a four-person team covering data engineering, machine learning, dashboard development, and visualisation/QA.
