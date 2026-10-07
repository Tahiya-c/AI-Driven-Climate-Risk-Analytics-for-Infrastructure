"""Load and clean the timber pile deterioration dataset.

The CSV is the OUTPUT of an existing deterministic AS 5604-style timber
durability model, not raw sensor data. This module fixes two known problems
in that export -- mislabelled k-factor headers (Defect D9, see
``D9_COLUMN_RENAMES``) and a blank MIROC-Medres scenario label (see
``MIROC_LABEL``) -- and normalises column dtypes.
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

# Defect D9 (docs/timber_kembla_model_spec_and_defects.md): exports built before
# the fix labelled workbook columns AP-AY one column out of step with the
# workbook's own headers, so e.g. "salinity_factor" actually held k_shelter and
# "construction_factor" held k0. Verified against the workbook's row-7 headers
# and against src/models/timber_engine.py's recomputed k factors (every value
# matches). Old header -> correct name; applied as one simultaneous rename
# because several old names reappear as correct names for different columns.
D9_COLUMN_RENAMES = {
    "timber_exposure_factor": "heartwood_factor",  # AP: ktimber HW
    "model_hazard_wood": "sapwood_factor",  # AQ: ktimber SW
    "model_service_water": "salinity_factor",  # AR: ksal
    "salinity_factor": "shelter_factor",  # AS: kshelter
    "shelter_factor": "protection_factor",  # AT: kprotect
    "protection_factor": "contact_factor",  # AU: kcontact
    "model_contact_factor": "knot_factor",  # AV: kknot
    "model_knot_factor": "environment_factor",  # AW: kenv
    "environment_factor": "construction_factor",  # AX: kconst
    "construction_factor": "combined_factor",  # AY: k0
    "replacement_time_years": "design_life_years",  # BJ: re-estimated design life
}

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
    "heartwood_factor",
    "sapwood_factor",
    "salinity_factor",
    "shelter_factor",
    "protection_factor",
    "contact_factor",
    "knot_factor",
    "environment_factor",
    "construction_factor",
    "combined_factor",
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
    "design_life_years",
]


def load_timber_data(path: str | Path = DATA_PATH) -> pd.DataFrame:
    """Load the timber CSV, fix D9 headers and the MIROC-Medres label gap, clean dtypes."""
    frame = fix_d9_column_names(pd.read_csv(path))
    frame = _relabel_miroc_scenario(frame)
    frame = _clean_dtypes(frame)
    _validate_scenario_labels(frame)
    return frame


def fix_d9_column_names(frame: pd.DataFrame) -> pd.DataFrame:
    """Rename pre-fix export headers to their real meaning (no-op on fixed exports).

    A pre-fix export is recognised by ``timber_exposure_factor``, a header that
    only ever existed in the mislabelled layout.
    """
    if "timber_exposure_factor" not in frame.columns:
        return frame
    return frame.rename(columns=D9_COLUMN_RENAMES)


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
