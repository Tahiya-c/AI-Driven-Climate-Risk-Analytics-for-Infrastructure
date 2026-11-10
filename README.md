# AI-Driven Climate Risk Analytics for Infrastructure

An AI-driven analytics platform that forecasts how concrete and timber seaport piles deteriorate under three climate change scenarios — **Hotter & Drier**, **Most Likely**, and **Colder & Wetter** — trained on real pile inspection and engineering data from **Port Kembla, Australia**, paired with multi-decade climate projections running through 2070. The goal is to give port infrastructure teams an evidence-based, per-asset forecast to plan inspections and replacements ahead of failure, rather than relying on fixed inspection intervals alone.

This project was built as a **final year university capstone project**: a four-person team spent roughly a year taking it from first principles — scoping the problem against real engineering datasets and deterministic reference models — through benchmarking multiple model architectures per material, to a production-style Streamlit dashboard backed by a Supabase database.

## Overview

Seaport piles degrade over time due to environmental exposure — salinity, sea temperature, moisture, and other climate-linked stressors. Port operators traditionally rely on deterministic engineering models (built in the style of AS 5604) and fixed manual inspection schedules to estimate when a pile needs replacing. This project trains machine-learning surrogate models on the real Port Kembla data underlying those deterministic models, to test whether they can match or extend their forecasting power, and presents the results through an interactive dashboard that:

- Visualises each pile's predicted condition over time under each climate scenario, spliced from historical values into a future projection.
- Forecasts the year a pile is expected to cross its replacement threshold.
- Lets engineers import new inspection data and compare model predictions against the existing deterministic reference models.

## Key Features

- **Real-world dataset.** Trained and evaluated on genuine Port Kembla concrete and timber pile inspection/engineering records — 7 concrete piles × 7 scenario combinations (49 series) and 2 timber piles tracked across 71 years (2000–2070) under 4 scenario labels — not synthetic or toy data.
- **Model architecture chosen per material based on benchmarked performance.** Concrete carbonation depth is forecast with an **RNN**, selected after being benchmarked head-to-head against **CatBoost, Random Forest, and Ridge regression** (plus a persistence baseline) on both a known-pile split and a harder leave-one-pile-out generalisation test. Timber diameter loss is forecast with a **LightGBM / MLP** comparison, with residual-correction and quantile-uncertainty variants also evaluated.
- **Continuous regression output, not a coarse risk label.** Every model predicts an actual deterioration-rate / remaining-diameter trajectory per pile, per year, per scenario — giving engineers a real forecast curve and projected replacement year rather than a single "low/medium/high" risk class.
- **Scenario-based forecasting** across four labelled climate trajectories (`Baseline`, `Hotter & Drier`, `Most Likely`, `Colder & Wetter`), with historical-to-future trend charts.
- **Data import pipeline** with schema validation for uploading new concrete/timber inspection datasets (`.xlsx`/`.csv`).
- **Python backend** — a Streamlit dashboard, Supabase for persisted imports/forecasts/dashboard state, and a pandas/scikit-learn/PyTorch/LightGBM modelling stack with Optuna-tuned hyperparameters and SHAP-based feature importance.
- **Reproducible evaluation** — every model is trained and scored through the same benchmarking harness, with full results and methodology written up in [`docs/`](docs).

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
