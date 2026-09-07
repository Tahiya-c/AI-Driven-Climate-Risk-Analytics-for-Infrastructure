"""Fidelity tests: does `src/timber/engine.py` reproduce the workbook's own CSV export?

Phase 1 of the rebuild (docs/timber_kembla_model_spec_and_defects.md) requires the port
to be verbatim, defects included. If the port disagrees with the CSV anywhere, the port
is wrong -- these tests are the acceptance gate, not a general regression suite.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.timber.data_prep import DATA_PATH
from src.timber.engine import ASSETS, SCENARIOS, simulate

requires_data_file = pytest.mark.skipif(
    not DATA_PATH.exists(), reason="timber CSV not present on disk"
)

# (asset_code, csv climate_model, csv emissions_pathway) -> (ClimateScenario, engine pathway)
# Baseline's own anchors are zero for every pathway, so "A1B" is passed arbitrarily for
# the CSV's "Base" pathway label.
_CSV_TO_SCENARIO = {
    "(Reference)": ("Baseline", "A1B"),
    "CSIRO Mk3.5": ("Hotter & Drier", None),
    "MRI-CGCM2.3.2": ("Most Likely", None),
    "MIROC-Medres": ("Cooler & Wetter", None),
}

_FIDELITY_COLUMNS = {
    "remaining_diameter_mm": "D_t",
    "mean_deterioration_mm": "d_mean",
    "sapwood_rate_mm_year": "r_sap",
    "heartwood_rate_mm_year": "r_heart",
    "failure_probability": "prob",
    "design_deterioration_mm": "d_design",
    "replacement_time_years": "t_replace",
}


@pytest.fixture(scope="module")
def csv_frame() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def _engine_frame_for_group(
    asset_code: str, climate_model: str, emissions_pathway: str
) -> pd.DataFrame:
    scenario_name, forced_pathway = _CSV_TO_SCENARIO[climate_model]
    pathway = forced_pathway or emissions_pathway
    asset = ASSETS[asset_code]
    scenario = SCENARIOS[scenario_name]
    years = range(2000, 2071)
    return simulate(asset, scenario, pathway, years)


@requires_data_file
@pytest.mark.parametrize("asset_code", ["PileA", "PileB"])
def test_reproduces_csv_to_1e9(csv_frame: pd.DataFrame, asset_code: str) -> None:
    groups = csv_frame[csv_frame["asset_code"] == asset_code].groupby(
        ["climate_model", "emissions_pathway"], sort=False
    )
    assert len(groups) == 7, "expected 7 climate streams (Baseline + 3 GCMs x 2 pathways)"

    for (climate_model, emissions_pathway), csv_group in groups:
        csv_group = csv_group.sort_values("year").reset_index(drop=True)
        engine_group = _engine_frame_for_group(asset_code, climate_model, emissions_pathway)

        assert list(csv_group["year"]) == list(engine_group["year"])

        for csv_col, engine_col in _FIDELITY_COLUMNS.items():
            diff = (csv_group[csv_col] - engine_group[engine_col]).abs()
            assert diff.max() < 1e-9, (
                f"{asset_code}/{climate_model}/{emissions_pathway}: "
                f"{csv_col} vs engine {engine_col} max abs diff {diff.max():.3e}"
            )


def test_pilea_baseline_year_2000_known_values() -> None:
    frame = simulate(ASSETS["PileA"], SCENARIOS["Baseline"], "A1B", [2000])
    row = frame.iloc[0]

    assert row["r_heart"] == pytest.approx(2.127141737910438, abs=1e-9)
    assert row["t_replace"] == pytest.approx(39.88535825630865, abs=1e-9)


@pytest.mark.parametrize(
    ("asset_code", "scenario_name", "pathway", "expected_year"),
    [
        ("PileA", "Baseline", "A1B", 2045),
        ("PileA", "Hotter & Drier", "A1FI", 2039),
        ("PileA", "Cooler & Wetter", "A1FI", 2039),
        ("PileA", "Most Likely", "A1FI", 2041),
        ("PileB", "Baseline", "A1B", 2034),
        ("PileB", "Hotter & Drier", "A1FI", 2031),
        ("PileB", "Cooler & Wetter", "A1FI", 2031),
        ("PileB", "Most Likely", "A1FI", 2032),
    ],
)
def test_threshold_crossing_years(
    asset_code: str, scenario_name: str, pathway: str, expected_year: int
) -> None:
    asset = ASSETS[asset_code]
    frame = simulate(asset, SCENARIOS[scenario_name], pathway, range(2000, 2071))

    crossed = frame[frame["D_t"] <= asset.replacement_diameter_mm]
    assert not crossed.empty, f"{asset_code}/{scenario_name}/{pathway} never crosses D_replace"
    assert int(crossed.iloc[0]["year"]) == expected_year


def test_pileb_is_flagged_synthetic() -> None:
    assert ASSETS["PileB"].synthetic is True
    assert ASSETS["PileA"].synthetic is False


def test_undefined_codes_warn_and_approximate_match_last_entry() -> None:
    with pytest.warns(UserWarning, match="DEFECT D3"):
        frame = simulate(ASSETS["PileB"], SCENARIOS["Baseline"], "A1B", [2010])
    row = frame.iloc[0]
    # MTN3 -> MTN2 (1.0), CNT3 -> CNT2 (1.0), KNT4 -> KNT3 (1.0)
    assert row["k_protect"] == pytest.approx(1.0)
    assert row["k_contact"] == pytest.approx(1.0)
    assert row["k_knot"] == pytest.approx(1.0)
