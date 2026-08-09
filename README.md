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
│   └── config.py           # Central project paths and application settings
├── tests/                  # Automated tests mirroring production modules
├── app.py                  # Streamlit application entry point
├── pyproject.toml          # Test and lint tool configuration
└── requirements.txt        # Reproducible application dependencies
```
