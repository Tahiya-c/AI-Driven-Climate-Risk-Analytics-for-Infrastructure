from pathlib import Path

import pandas as pd
import pytest
from sklearn.metrics import r2_score

from src.models.predict import predict_timber_until_year
from src.models.timber_scenarios import SCENARIO_LABELS
from src.models.train_timber_linear import FEATURE_COLUMNS, train


def _training_frame() -> pd.DataFrame:
    models = {
        "Most Likely": "MRI-CGCM2.3.2",
        "Hotter & Drier": "CSIRO Mk3.5",
        "Colder & Wetter": "MIROC-Medres",
    }
    rows: list[dict[str, object]] = []
    for year in range(2000, 2005):
        for asset_index, asset in enumerate(("PileA", "PileB")):
            for pathway_index, pathway in enumerate(("A1B", "A1FI")):
                for scenario_index, (scenario, climate_model) in enumerate(models.items()):
                    row: dict[str, object] = {
                        column: float(index + 1)
                        for index, column in enumerate(FEATURE_COLUMNS)
                    }
                    row.update(
                        {
                            "asset_code": asset,
                            "timber_type": "Hardwood",
                            "zone": "Marine",
                            "exposure": "Exposed",
                            "heartwood_class": "1",
                            "sapwood_of_hardwood": "Yes",
                            "maintenance_measure": "No maintenance",
                            "contact": "Water",
                            "knot_presence": "No",
                            "emissions_pathway": pathway,
                            "climate_model": climate_model,
                            "scenario_label": scenario,
                            "year": year,
                            "year_installed": 1990,
                            "age_years": year - 1990,
                            "replacement_time_years": (
                                10 + year - 2000 + asset_index + pathway_index + scenario_index
                            ),
                        }
                    )
                    rows.append(row)
    return pd.DataFrame(rows)


def test_prediction_accepts_user_end_year_and_ignores_target(tmp_path: Path) -> None:
    frame = _training_frame()
    data_path = tmp_path / "timber.csv"
    artifact_path = tmp_path / "model.joblib"
    frame.to_csv(data_path, index=False)
    artifact = train(data_path, artifact_path, test_fraction=0.2)

    saved_predictions = pd.DataFrame(artifact["test_scenario_predictions"])
    actuals = (
        frame.loc[frame["year"] >= artifact["first_test_year"]]
        .groupby(["year", "scenario_label"], as_index=False)["replacement_time_years"]
        .mean()
    )
    evaluation = actuals.merge(saved_predictions, on=["year", "scenario_label"])
    expected_r2 = r2_score(
        evaluation["replacement_time_years"],
        evaluation["predicted_replacement_time_years"],
    )
    assert artifact["metrics"]["r2"] == pytest.approx(expected_r2)

    frame["replacement_time_years"] = "not a prediction input"
    frame.to_csv(data_path, index=False)
    predictions = predict_timber_until_year(
        artifact_path, data_path, start_year=2002, end_year=2004
    )

    assert predictions.groupby("year").size().eq(3).all()
    assert predictions["year"].nunique() == 3
    assert set(predictions["scenario_label"]) == set(SCENARIO_LABELS)


def test_prediction_rejects_end_year_outside_available_inputs(tmp_path: Path) -> None:
    frame = _training_frame()
    data_path = tmp_path / "timber.csv"
    artifact_path = tmp_path / "model.joblib"
    frame.to_csv(data_path, index=False)
    train(data_path, artifact_path, test_fraction=0.2)

    with pytest.raises(ValueError, match="outside the available projection range"):
        predict_timber_until_year(artifact_path, data_path, end_year=2005)
