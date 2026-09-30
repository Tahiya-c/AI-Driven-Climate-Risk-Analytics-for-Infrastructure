"""Load and clean the timber pile deterioration dataset.

The CSV is the OUTPUT of an existing deterministic AS 5604-style timber
durability model, not raw sensor data. This module fixes one known gap in
that export (see ``MIROC_LABEL`` below) and normalises column dtypes.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import ROOT_DIR

DATA_PATH = ROOT_DIR / "data" / "processed" / "timber_kembla_v6_3_consolidated.csv"

# MIROC-Medres rows are exported with scenario_label left blank in the
# source spreadsheet. It is the cool/wet GCM in this project's three-model
# framework (CSIRO Mk3.5 = Hotter & Drier, MRI-CGCM2.3.2 = Most Likely,
# MIROC-Medres = Colder & Wetter), consistent with the sibling concrete
# dataset's documented GCM naming ("MIROC(hires) (Wetter)" in
# docs/concrete_kembla_v6_data_dictionary.md). This label is an assumption
# made from that naming convention, not confirmed against the original
# Timber_kembla_v6.3 workbook's scenario sheets -- re-verify there before
# treating "Colder & Wetter" as authoritative.
MIROC_LABEL = "Colder & Wetter"
EXPECTED_SCENARIO_LABELS = {"Baseline", "Hotter & Drier", "Most Likely", MIROC_LABEL}

NUMERIC_COLUMNS = [
    "year_installed",
    "initial_diameter_mm",
    "layer_of_sapwood_mm",
    "replacement_diameter_mm",
    "year",
    "age_years",
    "salinity",
    "summer_sea_temp_c",
    "design_time_years",
    "model_salinity_ppt",
    "water_factor",
    "timber_exposure_factor",
    "model_hazard_wood",
    "model_service_water",
    "salinity_factor",
    "shelter_factor",
    "protection_factor",
    "model_contact_factor",
    "model_knot_factor",
    "environment_factor",
    "construction_factor",
    "sapwood_rate_mm_year",
    "sapwood_lag_years",
    "heartwood_rate_mm_year",
    "heartwood_lag_years",
    "sapwood_duration_years",
    "mean_deterioration_mm",
    "remaining_diameter_mm",
    "failure_probability",
    "design_deterioration_mm",
    "design_diameter_mm",
]


def load_timber_data(path: str | Path = DATA_PATH) -> pd.DataFrame:
    """Load the timber CSV, fix the MIROC-Medres label gap, and clean dtypes."""
    frame = pd.read_csv(path)
    frame = _relabel_miroc_scenario(frame)
    frame = _clean_dtypes(frame)
    _validate_scenario_labels(frame)
    return frame


def _relabel_miroc_scenario(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    is_miroc = frame["climate_model"] == "MIROC-Medres"
    frame.loc[is_miroc, "scenario_label"] = frame.loc[is_miroc, "scenario_label"].fillna(
        MIROC_LABEL
    )
    return frame


def _clean_dtypes(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    present = [column for column in NUMERIC_COLUMNS if column in frame.columns]
    frame[present] = frame[present].apply(pd.to_numeric, errors="coerce")
    frame["scenario_label"] = frame["scenario_label"].astype("string")
    return frame


def _validate_scenario_labels(frame: pd.DataFrame) -> None:
    if frame["scenario_label"].isna().any():
        missing_models = frame.loc[frame["scenario_label"].isna(), "climate_model"].unique()
        raise ValueError(f"scenario_label is still NaN for climate_model(s): {missing_models}")
    unexpected = set(frame["scenario_label"].unique()) - EXPECTED_SCENARIO_LABELS
    if unexpected:
        raise ValueError(f"Unexpected scenario_label values: {unexpected}")


def remaining_diameter_floor(series: pd.Series) -> pd.Series:
    """Clip remaining_diameter_mm at zero for display/evaluation.

    The deterministic model over-extrapolates past failure for far-future
    rows and produces negative "remaining diameter" values. Rows are kept
    for training -- they still encode a valid downward trend -- but any
    negative value is floored at 0 wherever it is displayed or scored.
    """
    return series.clip(lower=0)
