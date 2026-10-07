import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import mean_absolute_error, r2_score

from src.models import timber_eval
from src.models.linear_model import build_pipeline
from src.models.linear_model import train as train_linear
from src.models.svr_model import train as train_svr
from src.models.timber_eval import (
    DEPTH_COLUMN,
    FEATURE_COLUMNS,
    TIMBER_FEATURES,
    fit_model,
    load_series,
    rollout,
)
from src.models.timber_inference import predict_timber_until_year


def _training_frame() -> pd.DataFrame:
    rows = []
    for asset_index, asset in enumerate(("PileA", "PileB")):
        for scenario_index, climate in enumerate(("(Reference)", "CSIRO Mk3.5")):
            for year in range(2000, 2011):
                age = year - 2000
                row = {column: float(index + 1) for index, column in enumerate(TIMBER_FEATURES)}
                row.update(
                    asset_code=asset, emissions_pathway="Base" if scenario_index == 0 else "A1FI",
                    climate_model=climate, year=year, age_years=age,
                    initial_diameter_mm=400 + 100 * asset_index,
                    summer_sea_temp_c=20 + 0.2 * age + scenario_index,
                    mean_deterioration_mm=(1 + asset_index + scenario_index) * age,
                )
                rows.append(row)
    return pd.DataFrame(rows)


@pytest.mark.parametrize("trainer", [train_linear, train_svr], ids=["linear", "svr"])
def test_training_roundtrip_depth_metrics_and_future_target_independence(tmp_path, trainer):
    frame = _training_frame()
    data = tmp_path / "timber.csv"
    output = tmp_path / "model.joblib"
    frame.to_csv(data, index=False)
    artifact = trainer(data, output, cutoff=2006)
    assert artifact["train_rows"] == 20
    assert artifact["test_rows"] == 16
    assert artifact["target"] == "annual_rate_mm_year"
    assert artifact["unit"] == "mm"
    predictions = predict_timber_until_year(output, data, end_year=2010)
    saved = pd.read_csv(output.with_suffix(".predictions.csv"))
    pd.testing.assert_frame_equal(predictions, saved)
    detail = pd.read_csv(output.with_suffix(".detail.csv"))
    assert set(detail["run"]) == {"primary", "scenario", "asset"}
    assert len(detail) == 12
    errors = []
    for (asset, scenario), group in predictions.groupby(["asset_id", "scenario"]):
        pathway, climate = scenario.split("|")
        actual = frame.loc[
            (frame.asset_code == asset) & (frame.emissions_pathway == pathway)
            & (frame.climate_model == climate) & (frame.year > 2006), DEPTH_COLUMN
        ]
        errors.append(mean_absolute_error(actual, group.value))
        score = detail.loc[
            (detail.run == "primary") & (detail.asset == asset) & (detail.scenario == scenario)
        ].iloc[0]
        assert score.r2 == pytest.approx(r2_score(actual, group.value))
    assert artifact["metrics"]["mae_mm"] == pytest.approx(np.mean(errors))
    # Forecasting must work even when all future true depths are absent.
    frame.loc[frame.year > 2006, DEPTH_COLUMN] = np.nan
    frame.to_csv(data, index=False)
    replay = predict_timber_until_year(output, data, end_year=2010)
    pd.testing.assert_frame_equal(predictions, replay)
    tail = predict_timber_until_year(output, data, start_year=2009, end_year=2010)
    pd.testing.assert_frame_equal(
        tail, predictions.loc[predictions.year >= 2009].reset_index(drop=True),
    )
    with pytest.raises(ValueError, match="outside the available projection range"):
        predict_timber_until_year(output, data, end_year=2011)
    with pytest.raises(ValueError, match="after the training cutoff"):
        predict_timber_until_year(output, data, start_year=2006, end_year=2010)


def test_rollout_uses_own_previous_increment(tmp_path):
    frame = _training_frame()
    data = tmp_path / "timber.csv"
    frame.to_csv(data, index=False)
    group = load_series(data)[0]
    group.loc[group.year > 2006, DEPTH_COLUMN] = np.nan

    class LagPlusOne:
        def predict(self, inputs):
            return inputs["lag_annual_rate"].to_numpy() + 1

    forecast = rollout(LagPlusOne(), group, 2006)
    history = group.loc[group.year <= 2006, DEPTH_COLUMN]
    lag = history.iloc[-1] - history.iloc[-2]
    expected = history.iloc[-1] + np.cumsum(lag + np.arange(1, 5))
    np.testing.assert_allclose(forecast.value, expected)


def test_fit_and_scaler_never_see_future_rows(tmp_path):
    frame = _training_frame()
    data = tmp_path / "timber.csv"
    frame.to_csv(data, index=False)
    original = load_series(data)
    first = fit_model(build_pipeline(), original, 2006)
    frame.loc[frame.year > 2006, TIMBER_FEATURES + [DEPTH_COLUMN]] = 999999.0
    frame.to_csv(data, index=False)
    changed = fit_model(build_pipeline(), load_series(data), 2006)
    np.testing.assert_allclose(first["scaler"].mean_, changed["scaler"].mean_)
    np.testing.assert_allclose(first["model"].coef_, changed["model"].coef_)
    assert first["scaler"].mean_[FEATURE_COLUMNS.index("age_years")] == pytest.approx(4.0)
    assert DEPTH_COLUMN not in FEATURE_COLUMNS
    assert "heartwood_rate_mm_year" not in FEATURE_COLUMNS


def test_holdout_folds_exclude_held_out_groups(tmp_path, monkeypatch):
    data = tmp_path / "timber.csv"
    _training_frame().to_csv(data, index=False)
    calls = []
    original_fit = timber_eval.fit_model

    def recording_fit(pipeline, series, cutoff):
        calls.append({
            (group.asset_code.iloc[0], group.climate_model.iloc[0]) for group in series
        })
        return original_fit(pipeline, series, cutoff)

    monkeypatch.setattr(timber_eval, "fit_model", recording_fit)
    artifact = train_linear(data, tmp_path / "model.joblib", cutoff=2006)
    assert len(calls) == 5  # primary, two scenarios, two assets
    assert len(calls[0]) == 4
    assert all(len({climate for _, climate in call}) == 1 for call in calls[1:3])
    assert all(len({asset for asset, _ in call}) == 1 for call in calls[3:])
    saved = joblib.load(tmp_path / "model.joblib")
    np.testing.assert_allclose(saved["model"]["model"].coef_, artifact["model"]["model"].coef_)


@pytest.mark.parametrize("bad_data", ["duplicate", "gap", "missing_feature", "missing_history"])
def test_invalid_input_fails_clearly(tmp_path: Path, bad_data):
    frame = _training_frame()
    if bad_data == "duplicate":
        frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    elif bad_data == "gap":
        frame = frame.drop(index=3)
    elif bad_data == "missing_feature":
        frame = frame.drop(columns="salinity")
    else:
        frame.loc[frame.year == 2006, DEPTH_COLUMN] = np.nan
    data = tmp_path / "timber.csv"
    frame.to_csv(data, index=False)
    with pytest.raises(ValueError):
        train_linear(data, tmp_path / "model.joblib", cutoff=2006, runs=("primary",))


def test_benchmark_and_final_training_are_separate(tmp_path):
    from src.models.timber_eval import run_benchmark
    from src.models.train_final_timber_model import train_final

    data = tmp_path / "timber.csv"
    _training_frame().to_csv(data, index=False)
    summary, detail, skipped = run_benchmark(data, ["linear", "ridge", "svr"], cutoff=2006)
    assert set(summary.model) == {"linear", "ridge", "svr"}
    assert {"mae", "rmse", "r2", "final_mae"} <= set(summary.columns)
    assert len(detail) == 36
    assert skipped == []
    assert list(tmp_path.glob("*.joblib")) == []
    paths = train_final(data, tmp_path / "final", ["ridge", "svr"])
    for path in paths:
        artifact = joblib.load(path)
        assert artifact["train_rows"] == 36
        assert artifact["cutoff_year"] == 2010
        assert "metrics" not in artifact
        assert path.with_suffix(".json").exists()
        with pytest.raises(ValueError, match="explicit anchor_year"):
            predict_timber_until_year(path, data, end_year=2010)
        replay = predict_timber_until_year(path, data, end_year=2010, anchor_year=2006)
        assert len(replay) == 16


def test_models_pipeline_runs_without_src_timber(tmp_path):
    """Use a fresh interpreter to catch both direct and transitive dependencies."""
    data = tmp_path / "timber.csv"
    _training_frame().to_csv(data, index=False)
    script = """
import importlib.abc
import sys

class BlockTimber(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'src.timber' or fullname.startswith('src.timber.'):
            raise ImportError('src.timber must not be required by the models pipeline')

sys.meta_path.insert(0, BlockTimber())
from experiments.timber.tune_models import candidates
from src.models import timber_benchmark_all
from src.models.timber_eval import run_benchmark
from src.models.train_final_timber_model import train_final
from src.models.timber_inference import predict_timber_until_year

summary, _, _ = run_benchmark(sys.argv[1], ['linear', 'ridge', 'svr'], ('primary',), 2006)
assert len(summary) == 3
for model in train_final(sys.argv[1], sys.argv[2]):
    assert len(predict_timber_until_year(model, sys.argv[1], 2010, anchor_year=2006)) == 16
assert len(candidates('linear')) == 16
assert len(candidates('svr')) == 48
assert not any(name == 'src.timber' or name.startswith('src.timber.') for name in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(data), str(tmp_path / "models")],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
