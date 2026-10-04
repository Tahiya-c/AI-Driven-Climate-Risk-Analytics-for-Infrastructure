"""Tests for timber feature/target helpers, the GBM builder and quantile utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models import timber_model_gbm as model_gbm
from src.data.timber_data_prep import DATA_PATH, load_timber_data
from src.features.timber_features import (
    DECAY_DOSE_COLUMN,
    DECAY_DOSE_FEATURE_COLUMNS,
    FEATURE_COLUMNS,
    add_decay_dose,
    constant_within_pile_features,
)
from src.models.timber_quantile import pinball_loss
from src.features.timber_targets import (
    CLIPPED_DIAMETER_COLUMN,
    DISTANCE_TARGET,
    RESIDUAL_TARGET,
    add_distance_to_threshold,
    add_residual_correction_target,
)

requires_data_file = pytest.mark.skipif(
    not DATA_PATH.exists(), reason="timber CSV not present on disk"
)
requires_gbm = pytest.mark.skipif(
    model_gbm.GBM_BACKEND is None, reason="neither lightgbm nor xgboost is installed"
)


# --------------------------------------------------------------------------
# Pure/synthetic unit tests: no CSV or trained model needed.
# --------------------------------------------------------------------------


def test_constant_within_pile_features_flags_only_truly_constant_columns() -> None:
    frame = pd.DataFrame(
        {
            "asset_code": ["PileA", "PileA", "PileB", "PileB"],
            "same_everywhere": [1, 1, 1, 1],
            "constant_per_pile_but_differs_between_piles": [5, 5, 9, 9],
            "varies_within_pile": [1, 2, 3, 4],
        }
    )
    columns = [
        "same_everywhere",
        "constant_per_pile_but_differs_between_piles",
        "varies_within_pile",
    ]
    flagged = constant_within_pile_features(frame, columns)
    assert set(flagged) == {"same_everywhere", "constant_per_pile_but_differs_between_piles"}
    assert "varies_within_pile" not in flagged


def test_add_decay_dose_cumulates_within_trajectory_and_restarts_per_group() -> None:
    frame = pd.DataFrame(
        {
            "asset_code": ["PileA"] * 4,
            "scenario_label": ["Hotter & Drier"] * 2 + ["Most Likely"] * 2,
            "emissions_pathway": ["A1B"] * 4,
            "age_years": [1, 2, 1, 2],
            "salinity_factor": [1.0, 1.0, 2.0, 2.0],
            "summer_sea_temp_c": [10.0, 10.0, 10.0, 10.0],
            "heartwood_factor": [1.0, 1.0, 1.0, 1.0],
        }
    )
    result = add_decay_dose(frame).sort_values(["scenario_label", "age_years"])

    hotter_drier = result[result["scenario_label"] == "Hotter & Drier"][DECAY_DOSE_COLUMN].tolist()
    most_likely = result[result["scenario_label"] == "Most Likely"][DECAY_DOSE_COLUMN].tolist()

    assert hotter_drier == [10.0, 20.0]
    # A different (scenario_label, emissions_pathway) trajectory restarts the
    # cumulative sum rather than continuing Hotter & Drier's running total.
    assert most_likely == [20.0, 40.0]


def test_decay_dose_feature_columns_swap_out_the_raw_linear_inputs() -> None:
    for column in ("age_years", "summer_sea_temp_c", "salinity_factor", "heartwood_factor"):
        assert column not in DECAY_DOSE_FEATURE_COLUMNS
    assert DECAY_DOSE_COLUMN in DECAY_DOSE_FEATURE_COLUMNS
    # Every other standard feature is preserved.
    untouched = set(FEATURE_COLUMNS) - {
        "age_years",
        "summer_sea_temp_c",
        "salinity_factor",
        "heartwood_factor",
    }
    assert untouched.issubset(set(DECAY_DOSE_FEATURE_COLUMNS))


def test_add_distance_to_threshold_is_diameter_minus_replacement_threshold() -> None:
    frame = pd.DataFrame(
        {"remaining_diameter_mm": [250.0, 150.0], "replacement_diameter_mm": [200.0, 200.0]}
    )
    result = add_distance_to_threshold(frame)
    assert result[DISTANCE_TARGET].tolist() == [50.0, -50.0]


def test_add_residual_correction_target_is_zero_when_already_valid() -> None:
    frame = pd.DataFrame({"remaining_diameter_mm": [300.0, -50.0, 0.0]})
    result = add_residual_correction_target(frame)
    assert result[CLIPPED_DIAMETER_COLUMN].tolist() == [300.0, 0.0, 0.0]
    assert result[RESIDUAL_TARGET].tolist() == [0.0, 50.0, 0.0]


def test_pinball_loss_is_zero_for_a_perfect_prediction() -> None:
    y_true = np.array([10.0, 20.0, 30.0])
    assert pinball_loss(y_true, y_true, alpha=0.1) == pytest.approx(0.0)


def test_pinball_loss_penalizes_underprediction_more_at_high_quantiles() -> None:
    y_true = np.array([100.0])
    under_prediction = np.array([90.0])  # predicted below the truth
    loss_p10 = pinball_loss(y_true, under_prediction, alpha=0.1)
    loss_p90 = pinball_loss(y_true, under_prediction, alpha=0.9)
    assert loss_p90 > loss_p10


# --------------------------------------------------------------------------
# Integration tests against the real dataset (skipped if it isn't on disk).
# --------------------------------------------------------------------------


@requires_data_file
@requires_gbm
def test_gbm_trains_and_reports_finite_metrics() -> None:
    from src.models.timber_model import train

    frame = load_timber_data(DATA_PATH)
    result = train(frame, split_strategy="time", model_kind="gbm")

    assert result.model_kind == "gbm"
    assert np.isfinite(result.test_mae)
    assert np.isfinite(result.test_rmse)
    assert result.n_train > 0
    assert result.n_test > 0


@requires_data_file
@requires_gbm
def test_quantile_models_produce_an_ordered_replacement_year_band_table() -> None:
    from src.models.timber_quantile import replacement_year_bands, train_quantile_models

    frame = load_timber_data(DATA_PATH)
    quantile_models = train_quantile_models(frame, split_strategy="time")

    assert set(quantile_models.models) == {0.1, 0.5, 0.9}

    pile = frame["asset_code"].iloc[0]
    band_table = replacement_year_bands(frame, quantile_models, pile)
    assert list(band_table.columns) == [0.1, 0.5, 0.9]
    assert len(band_table) == 3  # one row per scenario
