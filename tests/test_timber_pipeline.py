"""Tests for the timber pile replacement forecasting pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from src.data.timber_data_prep import DATA_PATH, EXPECTED_SCENARIO_LABELS, load_timber_data
from src.features.timber_features import FEATURE_COLUMNS, TARGET_COLUMN, build_feature_matrix
from src.models.timber_model import train
from src.models.timber_scenarios import build_pile_trajectories, find_replacement_year

requires_data_file = pytest.mark.skipif(
    not DATA_PATH.exists(), reason="timber CSV not present on disk"
)


@requires_data_file
def test_load_timber_data_relabels_miroc_with_no_nans() -> None:
    frame = load_timber_data(DATA_PATH)

    assert not frame["scenario_label"].isna().any()
    assert set(frame["scenario_label"].unique()) == EXPECTED_SCENARIO_LABELS
    miroc_rows = frame.loc[frame["climate_model"] == "MIROC-Medres", "scenario_label"]
    assert (miroc_rows == "Colder & Wetter").all()


@requires_data_file
def test_build_feature_matrix_matches_expected_columns_with_no_nans() -> None:
    frame = load_timber_data(DATA_PATH)
    matrix = build_feature_matrix(frame)

    assert list(matrix.columns) == FEATURE_COLUMNS
    assert not matrix.isna().any().any()


def test_feature_columns_exclude_leaky_model_outputs() -> None:
    assert TARGET_COLUMN not in FEATURE_COLUMNS
    assert "failure_probability" not in FEATURE_COLUMNS
    assert "design_life_years" not in FEATURE_COLUMNS


@requires_data_file
def test_model_trains_without_error_and_reports_finite_metrics() -> None:
    frame = load_timber_data(DATA_PATH)
    result = train(frame, split_strategy="time")

    assert result.n_train > 0
    assert result.n_test > 0
    assert np.isfinite(result.test_mae)
    assert result.test_mae >= 0


@requires_data_file
def test_scenario_trajectories_are_monotonic_non_increasing_and_non_negative() -> None:
    frame = load_timber_data(DATA_PATH)
    result = train(frame, split_strategy="time")
    pile = frame["asset_code"].iloc[0]

    trajectories, _split_year = build_pile_trajectories(frame, result.pipeline, pile)

    for trajectory in trajectories.values():
        diffs = np.diff(trajectory.remaining_diameter_mm)
        assert (diffs <= 1e-9).all(), f"{trajectory.scenario} trajectory is not non-increasing"
        assert (trajectory.remaining_diameter_mm >= 0).all()


def test_find_replacement_year_interpolates_the_crossing_point() -> None:
    years = np.array([2000, 2010, 2020, 2030])
    values = np.array([400.0, 300.0, 150.0, 50.0])

    replacement_year = find_replacement_year(years, values, threshold=200.0)

    assert replacement_year is not None
    assert 2010 < replacement_year < 2020


def test_find_replacement_year_returns_none_when_never_crossed() -> None:
    years = np.array([2000, 2010, 2020])
    values = np.array([400.0, 380.0, 360.0])

    assert find_replacement_year(years, values, threshold=200.0) is None


def test_find_replacement_year_returns_first_year_when_already_below_threshold() -> None:
    years = np.array([2000, 2010])
    values = np.array([150.0, 100.0])

    assert find_replacement_year(years, values, threshold=200.0) == 2000.0


def test_sapwood_loss_is_zero_before_installation_and_the_full_layer_after() -> None:
    import pandas as pd

    from src.models.timber_transfer import SAPWOOD_COLUMN, sapwood_loss

    rows = pd.DataFrame({"age_years": [-3, 0, 1, 10, 40], SAPWOOD_COLUMN: [20.0, 20.0, 20.0, 20.0, 35.0]})

    np.testing.assert_allclose(sapwood_loss(rows), [0.0, 0.0, 20.0, 20.0, 35.0])


def test_pileA_trained_model_needs_under_5mm_shift_on_pileB() -> None:
    import warnings

    from src.models.timber_transfer import (
        LOSS_COLUMN,
        MODEL_INPUT_COLUMNS,
        build_transfer_model,
        load_transfer_frame,
        shift_forecast,
    )

    frame = load_transfer_frame()
    pile_a = frame.loc[frame["asset_code"] == "PileA"]
    pile_b = frame.loc[frame["asset_code"] == "PileB"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = build_transfer_model("mlp").fit(pile_a[MODEL_INPUT_COLUMNS], pile_a[LOSS_COLUMN])
        _, record = shift_forecast(pile_b, model.predict_uncapped(pile_b[MODEL_INPUT_COLUMNS]))

    offsets = np.array(list(record["offsets_mm"].values()))
    assert abs(offsets.mean()) < 5.0
    assert record["large_offsets"] == []
