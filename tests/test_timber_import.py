"""Tests for the user import path: skeleton, validation, climate expansion, forecasting."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from src.data.timber_data_prep import DATA_PATH, load_timber_data
from src.data.timber_import import (
    _example_tables,
    to_model_tables,
    validate_import,
    yearly_climate,
)
from src.models.timber_forecast import forecast_pile, saved_model_path


def _errors(result) -> list[str]:
    return [f"{i.sheet}:{i.column}:{i.message}" for i in result.errors]


def test_example_import_is_valid() -> None:
    result = validate_import(_example_tables())
    assert result.ok, _errors(result)


def test_yearly_climate_reproduces_the_workbook_climate() -> None:
    tables = _example_tables()
    climate = yearly_climate(tables["site_climate"], tables["climate_changes"])
    workbook = load_timber_data(DATA_PATH)
    workbook = workbook.loc[workbook["asset_code"] == "PileA"]
    for pathway, label, scenario_id in (
        ("A1FI", "Hotter & Drier", "A1FI-hotter_drier"),
        ("A1B", "Colder & Wetter", "A1B-colder_wetter"),
        ("A1FI", "Most Likely", "A1FI-most_likely"),
        ("A1B", "Most Likely", "A1B-most_likely"),
        ("A1FI", "Colder & Wetter", "A1FI-colder_wetter"),
        ("A1B", "Hotter & Drier", "A1B-hotter_drier"),
    ):
        expected = workbook.loc[
            (workbook["emissions_pathway"] == pathway) & (workbook["scenario_label"] == label)
        ].sort_values("year")
        got = climate.loc[climate["scenario_id"] == scenario_id].sort_values("year")
        assert np.allclose(got["summer_sea_temp_c"], expected["summer_sea_temp_c"], atol=1e-6)
        assert np.allclose(got["salinity_ppt"], expected["salinity"], atol=1e-6)


@pytest.mark.parametrize(
    ("sheet", "column", "value", "expected"),
    [
        ("timber_assets", "k_heartwood", "abc", "Not a number"),
        ("timber_assets", "k_protect", 0, "outside the allowed minimum"),
        ("timber_assets", "zone", "Z", "not an allowed value"),
        ("timber_assets", "initial_diameter_mm", "abc", "Not a number"),
        ("timber_assets", "replacement_diameter_mm", 999, "smaller than initial_diameter_mm"),
        ("timber_observations", "remaining_diameter_mm", 9999, "larger than initial_diameter_mm"),
        ("timber_observations", "asset_id", "NOPE", "not in timber_assets"),
    ],
)
def test_validation_catches_bad_values(sheet, column, value, expected) -> None:
    tables = _example_tables()
    tables[sheet] = tables[sheet].astype(object)
    tables[sheet].loc[0, column] = value
    result = validate_import(tables)
    assert not result.ok
    assert any(expected in issue.message for issue in result.errors), _errors(result)


def _warnings(result) -> list[str]:
    return [i.message for i in result.issues if i.level == "warning"]


def test_unusual_coefficients_and_pre_2000_inspections_warn_but_stay_valid() -> None:
    tables = _example_tables()
    tables["timber_assets"].loc[0, "k_sapwood"] = 7.5  # not in the AS 5604 table
    tables["timber_assets"].loc[0, "k_heartwood"] = 1.1  # zones A-C value, but zone is E
    observations = tables["timber_observations"]
    tables["timber_assets"].loc[0, "year_installed"] = 1990
    tables["timber_observations"] = pd.concat(
        [observations.head(1).assign(inspection_year=1995, remaining_diameter_mm=440.0), observations]
    )

    result = validate_import(tables)

    assert result.ok, _errors(result)
    messages = _warnings(result)
    assert any("k_sapwood = 7.5 is not a value" in m for m in messages), messages
    assert any("wrong zone column" in m for m in messages), messages
    assert any("before 2000 are not used" in m for m in messages), messages


def test_coefficient_numbers_reach_the_model_factors() -> None:
    from src.data.timber_template import assemble_timber_frame

    tables = _example_tables()
    tables["timber_assets"].loc[0, ["k_heartwood", "k_contact"]] = [5.2, 2.0]
    model_tables = to_model_tables(validate_import(tables))
    frame = assemble_timber_frame(
        model_tables["assets"], model_tables["climate"], model_tables["scenarios"]
    )
    assert (frame["heartwood_factor"] == 5.2).all()
    assert (frame["contact_factor"] == 2.0).all()
    assert np.allclose(frame["combined_factor"], 5.2 * 2.0)  # k_sal, shelter, protect, knot = 1


def test_missing_sheet_and_anchor_years_are_errors() -> None:
    tables = _example_tables()
    del tables["site_climate"]
    assert any("Required sheet is missing" in i.message for i in validate_import(tables).errors)

    tables = _example_tables()
    changes = tables["climate_changes"]
    tables["climate_changes"] = changes.loc[
        ~((changes["scenario_label"] == "Most Likely") & (changes["anchor_year"] == 2050))
    ]
    assert any("missing anchor years [2050]" in i.message for i in validate_import(tables).errors)


@pytest.mark.skipif(not saved_model_path("gbm").exists(), reason="saved models not present")
def test_valid_import_forecasts_three_scenarios_to_2070() -> None:
    model_tables = to_model_tables(validate_import(_example_tables()))
    for kind in ("gbm", "mlp"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = forecast_pile(**model_tables, model_kind=kind)
        forecast = out.loc[out["method"] != "observed"]
        assert forecast["year"].min() == 2023 and forecast["year"].max() == 2070
        assert forecast["scenario"].nunique() == 3
        assert (forecast["value"] >= 0).all()
        assert (forecast["value"] <= 450 / 2).all()
        assert set(out.loc[out["method"] == "observed", "year"]) == {2010, 2014, 2018, 2022}
    assert isinstance(out, pd.DataFrame)
