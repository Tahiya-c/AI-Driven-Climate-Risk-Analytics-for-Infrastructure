"""Batch inference for the timber dashboard export.

Mirrors the interface the concrete team's ``src/models/inference.py`` exposes
(``MATERIAL``/``UNIT``/``METRIC``/``DEFAULT_HORIZON``/``METHODS``/``ROOT``,
``list_assets``/``list_scenarios``/``get_history``/``predict_batch``) so
``build_dashboard_output.py`` can call either module the same way, per the
shared dashboard data contract (``dashboard_data_<material>.csv``, 8 columns:
material, asset_id, scenario, year, value, unit, metric, method).

There is no live inference API -- like the concrete side, this produces a
static CSV the frontend reads with ``pd.read_csv`` + filter. "Projected" rows
(``method="mlp"``/``"gbm"``) come from the trained surrogate in
``src/models/timber_model.py``; "observed" rows are the deterministic AS 5604-style
workbook's own output for years before the anchor year, not real field
inspection data -- see the module docstring in ``src/data/timber_data_prep.py``.
Reported accuracy for the projected rows is therefore surrogate-vs-simulator
fidelity, not validated predictive accuracy against reality.
"""

from __future__ import annotations

import pandas as pd

from src.config import ROOT_DIR
from src.data.timber_data_prep import DATA_PATH, load_timber_data
from src.features.timber_features import build_feature_matrix
from src.models.timber_model import train

MATERIAL = "timber"
UNIT = "mm"
METRIC = "deterioration_depth"
METHODS = {"mlp", "gbm"}
ROOT = ROOT_DIR

# Matches the team's agreed anchor: history is 2000-2049 (observed), the
# projection is the 21-year 2050-2070 tail.
ANCHOR_YEAR = 2050
LAST_DATA_YEAR = 2070
DEFAULT_HORIZON = LAST_DATA_YEAR - ANCHOR_YEAR + 1

_frame_cache: pd.DataFrame | None = None
_pipeline_cache: dict[str, object] = {}


def _load_frame() -> pd.DataFrame:
    global _frame_cache
    if _frame_cache is None:
        _frame_cache = load_timber_data(DATA_PATH)
    return _frame_cache


def _scenario_id(pathway: pd.Series, climate_model: pd.Series) -> pd.Series:
    return pathway.astype(str) + "|" + climate_model.astype(str)


def list_assets() -> list[str]:
    return sorted(_load_frame()["asset_code"].unique())


def list_scenarios() -> list[str]:
    """The 7 "<emissions_pathway>|<climate_model>" series, e.g. "Base|(Reference)"."""
    combos = _load_frame()[["emissions_pathway", "climate_model"]].drop_duplicates()
    return [
        f"{pathway}|{climate_model}" for pathway, climate_model in combos.itertuples(index=False)
    ]


def _pipeline(method: str) -> object:
    if method not in METHODS:
        raise ValueError(f"Unknown method {method!r} (expected one of {sorted(METHODS)})")
    if method not in _pipeline_cache:
        result = train(_load_frame(), split_strategy="time", model_kind=method)
        _pipeline_cache[method] = result.pipeline
    return _pipeline_cache[method]


def get_history(asset: str, scenario: str) -> tuple[list[int], list[float]]:
    """Years and deterioration-depth values before ``ANCHOR_YEAR`` for one asset/scenario."""
    pathway, climate_model = scenario.split("|", 1)
    frame = _load_frame()
    subset = frame[
        (frame["asset_code"] == asset)
        & (frame["emissions_pathway"] == pathway)
        & (frame["climate_model"] == climate_model)
        & (frame["year"] < ANCHOR_YEAR)
    ].sort_values("year")
    return subset["year"].tolist(), subset["mean_deterioration_mm"].tolist()


def predict_batch(horizon: int = DEFAULT_HORIZON, method: str = "mlp") -> pd.DataFrame:
    """Predicted deterioration depth for every asset/scenario over ``horizon`` years.

    Returns a DataFrame already carrying all 8 dashboard schema columns
    (material, asset_id, scenario, year, value, unit, metric, method), matching
    what ``build_dashboard_output.build_timber`` expects from ``build_concrete``'s
    ``inference.predict_batch``.
    """
    last_year = ANCHOR_YEAR + horizon - 1
    if last_year > LAST_DATA_YEAR:
        raise ValueError(
            f"horizon {horizon} requests year {last_year}, past the last simulated "
            f"year {LAST_DATA_YEAR}"
        )

    frame = _load_frame()
    window = frame[(frame["year"] >= ANCHOR_YEAR) & (frame["year"] <= last_year)].copy()

    pipeline = _pipeline(method)
    features = build_feature_matrix(window)
    predicted_remaining_mm = pipeline.predict(features)
    # remaining_diameter_mm = initial_diameter_mm - 2 * mean_deterioration_mm, so
    # invert it here to report the same cumulative-depth quantity as get_history.
    deterioration_depth = (window["initial_diameter_mm"].to_numpy() - predicted_remaining_mm) / 2

    scenario = _scenario_id(window["emissions_pathway"], window["climate_model"])
    return pd.DataFrame(
        {
            "material": MATERIAL,
            "asset_id": window["asset_code"].to_numpy(),
            "scenario": scenario.to_numpy(),
            "year": window["year"].to_numpy(),
            "value": deterioration_depth,
            "unit": UNIT,
            "metric": METRIC,
            "method": method,
        }
    ).reset_index(drop=True)
