import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge

from experiments.timber.tune_models import select_parameters, validation_mae
from src.models.timber_eval import DEPTH_COLUMN, TIMBER_FEATURES, build_pipeline, fit_model, rollout


def series_data():
    groups = []
    for asset in range(2):
        years = np.arange(2000, 2031)
        group = pd.DataFrame({column: np.ones(len(years)) for column in TIMBER_FEATURES})
        group["year"] = years
        group["age_years"] = years - 2000
        group["asset_code"] = f"Pile{asset}"
        group["emissions_pathway"] = "A1FI"
        group["climate_model"] = "test"
        group[DEPTH_COLUMN] = (1 + asset) * (years - 2000) + 0.01 * (years - 2000) ** 2
        groups.append(group)
    return groups


def test_batch_validation_matches_production_recursive_rollout():
    series = series_data()
    fitted = fit_model(build_pipeline(Ridge(alpha=1)), series, 2011)
    errors = []
    for group in series:
        prediction = rollout(fitted, group, 2011, 2026)
        actual = group.loc[group.year.between(2012, 2026), DEPTH_COLUMN].to_numpy()
        errors.append(np.mean(np.abs(prediction.value.to_numpy() - actual)))
    assert validation_mae(fitted, series, 2011, 2026) == pytest.approx(np.mean(errors))


def test_tuning_never_uses_final_test_rows():
    series = series_data()
    search = [({"alpha": alpha}, build_pipeline(Ridge(alpha=alpha))) for alpha in [0.1, 10]]
    _, metadata, scores = select_parameters(series, "linear", search=search)
    for group in series:
        group.loc[group.year > 2026, TIMBER_FEATURES + [DEPTH_COLUMN]] = np.nan
    _, changed_metadata, changed_scores = select_parameters(series, "linear", search=search)
    assert metadata == changed_metadata
    pd.testing.assert_frame_equal(scores, changed_scores)
    assert metadata["validation_origins"] == [2011, 2016, 2021]
    assert metadata["validation_mae_mm"] == scores.validation_mae_mm.min()


def test_tuning_rejects_insufficient_history():
    with pytest.raises(ValueError, match="18 annual history rows"):
        select_parameters(series_data(), "linear", cutoff=2010)
