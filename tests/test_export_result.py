from src.ui.dashboard import (
    RESULT_FILES,
    _build_concrete_export_data,
    _build_timber_export_data,
)

from src.visualization.charts import (
    dashboard_concrete_model_data,
    dashboard_timber_model_data,
)


# =========================================================
# GENERAL RESULT FILE TESTS
# =========================================================

def test_all_result_files_exist():
    for path in RESULT_FILES.values():
        assert path.exists()


# =========================================================
# CONCRETE EXPORT TESTS
# =========================================================

def test_concrete_export_columns():

    pile = "PileA"

    data = dashboard_concrete_model_data(
        pile
    )

    export = _build_concrete_export_data(
        data,
        pile,
    )

    assert list(export.columns) == [
        "Pile",
        "Year",
        "Scenario",
        "Carbonation Depth (mm)",
        "Model",
        "Data Type",
    ]


def test_concrete_export_contains_both_models():

    pile = "PileA"

    data = dashboard_concrete_model_data(
        pile
    )

    export = _build_concrete_export_data(
        data,
        pile,
    )

    assert set(export["Model"]) == {
        "RNN",
        "Ridge",
    }


def test_concrete_export_only_contains_selected_pile():

    pile = "PileA"

    data = dashboard_concrete_model_data(
        pile
    )

    export = _build_concrete_export_data(
        data,
        pile,
    )

    assert set(export["Pile"]) == {
        pile
    }


def test_concrete_export_contains_historical_and_forecast():

    pile = "PileA"

    data = dashboard_concrete_model_data(
        pile
    )

    export = _build_concrete_export_data(
        data,
        pile,
    )

    assert {
        "Historical",
        "AI Forecast",
    }.issubset(
        set(export["Data Type"])
    )


# =========================================================
# TIMBER EXPORT TESTS
# =========================================================

def test_timber_export_columns():

    data = dashboard_timber_model_data()

    export = _build_timber_export_data(
        data
    )

    assert list(export.columns) == [
        "Year",
        "Scenario",
        "Material Loss",
        "Model",
        "Data Type",
    ]


def test_timber_export_contains_both_models():

    data = dashboard_timber_model_data()

    export = _build_timber_export_data(
        data
    )

    assert set(export["Model"]) == {
        "GBM",
        "MLP",
    }


def test_timber_export_contains_historical_and_forecast():

    data = dashboard_timber_model_data()

    export = _build_timber_export_data(
        data
    )

    assert {
        "Historical",
        "AI Forecast",
    }.issubset(
        set(export["Data Type"])
    )


def test_timber_export_is_not_empty():

    data = dashboard_timber_model_data()

    export = _build_timber_export_data(
        data
    )

    assert len(export) > 0