# Port Pile Deterioration Dashboard

A Streamlit dashboard for exploring concrete pile inspection data and predicting deterioration levels at seaports. The first version follows the supplied low-fidelity layout: side navigation, prediction parameters, condition indicators, and a historical-to-future forecast chart.

The app currently provides a working UI scaffold and deterministic demonstration forecast. Once the project CSV schema and trained model are finalized, the demo forecast can be replaced through the isolated data, feature, and model modules.

## Quick start

Requires Python 3.11 or newer.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Upload the processed CSV from the dashboard sidebar. Uploaded data is handled in memory and is not stored in the repository. Trained model binaries are ignored by Git to avoid accidentally committing large files.

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
