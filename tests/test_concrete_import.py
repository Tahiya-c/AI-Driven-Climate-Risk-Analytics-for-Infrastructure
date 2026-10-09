"""Tests for the concrete user-import path: skeleton, validation, fill policy, forecasting."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.concrete_import import (
    SHEETS,
    _example_tables,
    to_model_series,
    validate_import,
)
from src.models.concrete_forecast import forecast_series


def _errors(result) -> list[str]:
    return [f"{i.sheet}:{i.column}:{i.message}" for i in result.errors]


def _warnings(result) -> list[str]:
    return [i.message for i in result.issues if i.level == "warning"]


def test_example_import_is_valid() -> None:
    result = validate_import(_example_tables())
    assert result.ok, _errors(result)


def test_missing_mean_rh_pct_is_filled_from_the_year_lookup_not_blocked() -> None:
    """mean_rh_pct is the one column this format allows to be absent (fill="year_mean")."""
    tables = _example_tables()
    assert (tables["concrete_climate"]["mean_rh_pct"] == "").all()  # example leaves it blank
    result = validate_import(tables)
    assert result.ok
    filled = result.tables["concrete_climate"]["mean_rh_pct"]
    assert filled.notna().all()
    assert any("mean_rh_pct" in w and "filled with the typical value" in w for w in _warnings(result))


@pytest.mark.parametrize(
    ("sheet", "column", "value", "expected"),
    [
        ("concrete_assets", "wc_ratio", "abc", "Not a number"),
        ("concrete_assets", "wc_ratio", 5.0, "outside the allowed maximum"),
        ("concrete_assets", "zone", "underwater", "not an allowed value"),
        ("concrete_climate", "mean_temp_c", "abc", "Not a number"),
        ("concrete_climate", "emissions_pathway", "RCP8.5", "not an allowed value"),
        ("concrete_observations", "asset_id", "NOPE", "not in concrete_assets"),
    ],
)
def test_validation_catches_bad_values(sheet, column, value, expected) -> None:
    tables = _example_tables()
    tables[sheet] = tables[sheet].astype(object)
    tables[sheet].loc[0, column] = value
    result = validate_import(tables)
    assert not result.ok
    assert any(expected in issue.message for issue in result.errors), _errors(result)


def test_missing_required_sheet_is_an_error() -> None:
    tables = _example_tables()
    del tables["concrete_assets"]
    result = validate_import(tables)
    assert not result.ok
    assert any("Required sheet is missing" in i.message for i in result.errors)


def test_optional_observations_sheet_can_be_omitted() -> None:
    tables = _example_tables()
    del tables["concrete_observations"]
    result = validate_import(tables)
    assert result.ok, _errors(result)
    assert any("no inspections" in w for w in _warnings(result))


def test_to_model_series_matches_the_trained_feature_width() -> None:
    """X must be 9 columns wide (3 climate + t_exposure + 2 static + 3 zone one-hot),
    matching concrete_eval/rnn_model's feat_cols -- regardless of how many zones
    appear in a given import (a single-zone import must not shrink the one-hot block)."""
    result = validate_import(_example_tables())
    series = to_model_series(result)
    assert len(series) == 1
    s = series[0]
    # 71 rows, not 7: the example names a known reference scenario (A1FI|MRI-CGCM2.3.2)
    # with only 7 years of climate, so validate_import auto-extends it to 2070 -- see
    # test_climate_auto_fill_extends_a_known_scenario_to_2070 below.
    assert s["X"].shape == (71, 9)
    assert s["asset"] == "EXAMPLE_01"
    assert s["last_observed_year"] == 2026


def _richer_tables(n_history_years: int = 27, n_future_years: int = 4) -> dict[str, pd.DataFrame]:
    """A synthetic import with enough history for the RNN's lookback window (WINDOW=12)."""
    years = list(range(2000, 2000 + n_history_years + n_future_years))
    split = 2000 + n_history_years - 1
    return {
        "concrete_assets": pd.DataFrame(
            [{"asset_id": "TestPile", "zone": "splash", "wc_ratio": 0.45, "fc_mpa": 50.0}]
        ),
        "concrete_climate": pd.DataFrame(
            [
                {
                    "site": "Port Kembla", "scenario_label": "Most Likely",
                    "climate_model": "MRI-CGCM2.3.2", "emissions_pathway": "A1FI", "year": y,
                    "mean_temp_c": 19.1 + 0.03 * (y - 2000), "co2_ppm": 368.0 + 3.0 * (y - 2000),
                    "mean_rh_pct": 60.0,
                }
                for y in years
            ]
        ),
        "concrete_observations": pd.DataFrame(
            [
                {"asset_id": "TestPile", "year": y, "carb_depth_mm": round(1.2 + 0.08 * (y - 2000), 3)}
                for y in years
                if y <= split
            ]
        ),
    }


def test_forecast_series_raises_a_clear_error_when_history_is_too_short() -> None:
    """A genuinely custom scenario (not in the reference data) has nothing to auto-extend
    from, so one that stops at the last inspection year still can't be forecast."""
    tables = _example_tables()
    tables["concrete_climate"]["climate_model"] = "MyCustomGCM"  # no longer a known reference pair
    series = to_model_series(validate_import(tables))
    with pytest.raises(ValueError, match="no climate years after last_observed_year"):
        forecast_series(series, method="rnn")


def test_forecast_series_with_enough_history_continues_smoothly_from_observed() -> None:
    series = to_model_series(validate_import(_richer_tables()))
    out = forecast_series(series, method="rnn")
    assert list(out.columns) == [
        "material", "asset_id", "scenario", "year", "value", "unit", "metric", "method",
    ]
    observed = out[out["method"] == "observed"]
    forecast = out[out["method"] == "rnn"]
    assert len(observed) == 27
    # 44, not 4: this names a known reference scenario, so climate auto-extends to 2070
    # regardless of how many years _richer_tables itself supplied -- see §"climate auto-fill".
    assert len(forecast) == 44
    assert forecast["year"].max() == 2070
    assert forecast["year"].min() == observed["year"].max() + 1
    # continuity: the forecast's first value should be close to the last observed one
    # (one year's plausible carbonation growth, not a jump)
    gap = forecast.iloc[0]["value"] - observed.iloc[-1]["value"]
    assert 0 <= gap < 1.0


def test_climate_sheet_omitted_gives_all_7_reference_scenarios_to_2070() -> None:
    tables = _richer_tables()
    del tables["concrete_climate"]
    result = validate_import(tables)
    assert result.ok
    climate = result.tables["concrete_climate"]
    combos = set(zip(climate["emissions_pathway"], climate["climate_model"]))
    assert len(combos) == 7
    assert climate["year"].max() == 2070
    series = to_model_series(result)
    assert len(series) == 7
    assert all(s["years"].max() == 2070 for s in series)


def test_climate_auto_fill_extends_a_known_scenario_to_2070() -> None:
    """A known (climate_model, emissions_pathway) pair with partial years gets the rest
    from the reference; the user's own years are kept, not overwritten."""
    tables = _richer_tables(n_history_years=27, n_future_years=0)  # climate stops at 2026
    result = validate_import(tables)
    assert result.ok
    assert any(
        "added" in i.message and "from the reference series" in i.message
        for i in result.issues
        if i.level == "warning"
    )
    climate = result.tables["concrete_climate"]
    assert climate["year"].max() == 2070
    # the user's own 2026 value must survive unchanged, not be overwritten by the reference
    mine = tables["concrete_climate"]
    assert climate.loc[climate["year"] == 2026, "mean_temp_c"].iloc[0] == pytest.approx(
        mine.loc[mine["year"] == 2026, "mean_temp_c"].iloc[0]
    )


def test_custom_scenario_with_no_reference_match_is_left_as_supplied() -> None:
    tables = _richer_tables(n_history_years=27, n_future_years=0)
    tables["concrete_climate"]["climate_model"] = "MyCustomGCM"
    result = validate_import(tables)
    assert result.ok
    climate = result.tables["concrete_climate"]
    assert climate["year"].max() == 2026  # nothing to extend it with
    assert not any(
        "added" in i.message and "from the reference series" in i.message
        for i in result.issues
        if i.level == "warning"
    )


def test_forecast_series_wc_regress_needs_no_lookback_window() -> None:
    """wc_regress has no WINDOW requirement, so even the tiny example could use it --
    if it had any future climate years at all, which it doesn't; this just checks the
    richer fixture works for both methods identically."""
    series = to_model_series(validate_import(_richer_tables()))
    out = forecast_series(series, method="wc_regress")
    assert (out["method"].isin(["observed", "wc_regress"])).all()


def test_all_sheets_declared_in_schema_round_trip_through_example() -> None:
    assert set(_example_tables()) == set(SHEETS)
