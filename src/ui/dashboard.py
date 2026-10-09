"""Main Streamlit dashboard matching the supplied low-fidelity layout."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from src.config import ROOT_DIR
from src.ui.ai_summary import render_ai_summary
from src.visualization.charts import (
    comparison_data_from_payload,
    concrete_asset_ids,
    dashboard_concrete_model_data,
    dashboard_timber_model_data,
    material_forecast_chart,
)


# ---------------------------------------------------------
# Approved result files that users are allowed to download
# ---------------------------------------------------------

RESULT_FILES = {
    "Concrete Model Benchmark": ROOT_DIR / "results_concrete_benchmark.csv",
    "Concrete Benchmark Details": ROOT_DIR / "results_concrete_benchmark_detail.csv",
    "RNN Carbonation Results": ROOT_DIR / "results_rnn_carbonation.csv",
    "RNN Trajectory Results": ROOT_DIR / "results_rnn_trajectory.csv",
    "Benchmark by Asset": ROOT_DIR / "results_benchmark_asset.csv",
    "Benchmark by Scenario": ROOT_DIR / "results_benchmark_scenario.csv",
}


def _apply_styles() -> None:
    st.markdown(
        """
        <style>
        html, body {overflow-x: hidden;}
        *, *::before, *::after {box-sizing: border-box;}

        .block-container {
            padding: 1.5rem 1rem 2rem;
            width: auto;
            max-width: 1450px;
            margin: 0 auto;
        }

        [data-testid="stSidebar"] {
            border-right: 1px solid #D8E0E8;
        }

        [data-testid="stMetric"] {
            background: white;
            border: 1px solid #E2E8F0;
            border-radius: 10px;
            padding: 0.9rem;
        }

        div[data-testid="stHorizontalBlock"] {
            width: 100%;
            min-width: 0;
        }

        div[data-testid="stHorizontalBlock"] > div {
            min-width: 0 !important;
        }

        div[data-testid="stPlotlyChart"] {
            width: 100%;
            max-width: 100%;
            min-width: 0;
            overflow: hidden;
            background: white;
            border: 1px solid #E2E8F0;
            border-radius: 12px;
            padding: 0.5rem;
        }

        div[data-testid="stPlotlyChart"] > div {
            width: 100% !important;
            min-width: 0 !important;
        }

        div[data-testid="stPlotlyChart"] .modebar-container,
        div[data-testid="stPlotlyChart"] .modebar {
            position: absolute !important;
            top: 0.4rem !important;
            right: auto !important;
            left: 7rem !important;
        }

        .modebar-btn[data-title="Autoscale"],
        .modebar-btn[data-title="Reset axes"] {
            display: none !important;
        }

        @media (max-width: 900px) {
            div[data-testid="stHorizontalBlock"] {
                flex-wrap: wrap !important;
            }

            div[data-testid="stHorizontalBlock"] > div {
                flex: 1 1 100% !important;
                width: 100% !important;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.dialog("Expanded chart", width="large")
def _show_expanded_chart(
    material: str,
    historical: Any,
    projection: Any,
) -> None:
    """Show a material chart in a larger modal dialog."""

    figure = material_forecast_chart(
        material,
        historical,
        projection,
    )

    figure.update_layout(height=680)

    st.plotly_chart(
        figure,
        use_container_width=True,
        config={
            "displayModeBar": False,
            "responsive": True,
        },
    )


def render_dashboard(
    comparison_payload: Mapping[str, Any] | None = None,
) -> None:
    """Render the main dashboard."""

    st.set_page_config(
        page_title="Port Pile Deterioration",
        page_icon="🏗️",
        layout="wide",
    )

    _apply_styles()

    with st.sidebar:
        st.title("⚓ Port Pile Deterioration")

        page = st.radio(
            "Material",
            ["Concrete", "Timber"],
            index=1,
            label_visibility="collapsed",
        )

        st.divider()

    if page == "Timber":
        _render_timber_page(comparison_payload)
    else:
        _render_concrete_page()


# =========================================================
# EXPORT HELPER FUNCTIONS
# =========================================================

def _build_timber_export_data(
    data: Mapping[
        str,
        Mapping[str, pd.DataFrame],
    ],
) -> pd.DataFrame:
    """
    Create a clean CSV containing the currently displayed
    timber results from the GBM and MLP models.
    """

    export_frames: list[pd.DataFrame] = []

    for model_name, model_data in data.items():

        # -----------------------------
        # Historical data
        # -----------------------------
        historical = model_data["historical"].copy()

        historical["Scenario"] = "Historical"
        historical["Model"] = model_name
        historical["Data Type"] = "Historical"

        historical = historical.rename(
            columns={
                "year": "Year",
                "value": "Material Loss",
            }
        )

        # -----------------------------
        # Forecast data
        # -----------------------------
        projection = model_data["projection"].copy()

        projection = projection.rename(
            columns={
                "year": "Year",
                "condition": "Scenario",
                "value": "Material Loss",
            }
        )

        projection["Model"] = model_name
        projection["Data Type"] = "AI Forecast"

        export_frames.append(historical)
        export_frames.append(projection)

    export_data = pd.concat(
        export_frames,
        ignore_index=True,
    )

    export_data = export_data[
        [
            "Year",
            "Scenario",
            "Material Loss",
            "Model",
            "Data Type",
        ]
    ]

    export_data = export_data.sort_values(
        [
            "Model",
            "Year",
            "Scenario",
        ],
        ignore_index=True,
    )

    return export_data


def _build_concrete_export_data(
    data: Mapping[
        str,
        Mapping[str, pd.DataFrame],
    ],
    pile: str,
) -> pd.DataFrame:
    """
    Create a clean CSV containing the currently displayed
    concrete results for the selected pile.
    """

    export_frames: list[pd.DataFrame] = []

    for model_name, model_data in data.items():

        historical = model_data["historical"].copy()

        historical["Scenario"] = "Historical"
        historical["Model"] = model_name
        historical["Data Type"] = "Historical"

        historical = historical.rename(
            columns={
                "year": "Year",
                "value": "Carbonation Depth (mm)",
            }
        )

        projection = model_data["projection"].copy()

        projection = projection.rename(
            columns={
                "year": "Year",
                "condition": "Scenario",
                "value": "Carbonation Depth (mm)",
            }
        )

        projection["Model"] = model_name
        projection["Data Type"] = "AI Forecast"

        export_frames.append(historical)
        export_frames.append(projection)

    export_data = pd.concat(
        export_frames,
        ignore_index=True,
    )

    export_data.insert(
        0,
        "Pile",
        pile,
    )

    export_data = export_data[
        [
            "Pile",
            "Year",
            "Scenario",
            "Carbonation Depth (mm)",
            "Model",
            "Data Type",
        ]
    ]

    export_data = export_data.sort_values(
        [
            "Model",
            "Year",
            "Scenario",
        ],
        ignore_index=True,
    )

    return export_data


def _render_result_file_export() -> None:
    """
    Allow the user to download approved result datasets.
    """

    st.subheader("Download result dataset")

    st.caption(
        "Choose an available analysis result to export as a CSV file."
    )

    selected_result = st.selectbox(
        "Result dataset",
        list(RESULT_FILES.keys()),
        key="result_dataset_selector",
    )

    selected_path = RESULT_FILES[selected_result]

    if not selected_path.exists():
        st.warning(
            f"{selected_result} is not currently available."
        )
        return

    try:
        result_df = pd.read_csv(selected_path)

    except (
        pd.errors.ParserError,
        OSError,
    ) as exc:
        st.error(
            f"Could not load this result file: {exc}"
        )
        return

    with st.expander("Preview selected dataset"):
        st.dataframe(
            result_df.head(20),
            use_container_width=True,
        )

        st.caption(
            f"{len(result_df):,} rows × "
            f"{len(result_df.columns):,} columns"
        )

    csv_data = (
        result_df
        .to_csv(index=False)
        .encode("utf-8")
    )

    st.download_button(
        label=f"Download {selected_result}",
        data=csv_data,
        file_name=selected_path.name,
        mime="text/csv",
        icon=":material/download:",
        use_container_width=True,
        key="download_result_dataset",
    )


# =========================================================
# TIMBER PAGE
# =========================================================

def _render_timber_page(
    comparison_payload: Mapping[str, Any] | None = None,
) -> None:
    """Render GBM and MLP Timber forecasts side by side."""

    st.title("Timber outlook")

    st.caption(
        "Compare material deterioration projections "
        "from the GBM and MLP models."
    )

    if comparison_payload is None:
        data = dashboard_timber_model_data()

    else:
        try:
            data = comparison_data_from_payload(
                comparison_payload
            )["Timber"]

        except ValueError as exc:
            st.error(
                f"Could not load comparison data: {exc}"
            )
            return

    # --------------------------------------------------
    # Timber charts
    # --------------------------------------------------

    gbm, mlp = st.columns(
        2,
        gap="small",
    )

    for container, model_name in (
        (gbm, "GBM"),
        (mlp, "MLP"),
    ):

        model_data = data[model_name]

        with container:

            st.plotly_chart(
                material_forecast_chart(
                    f"Timber - {model_name}",
                    model_data["historical"],
                    model_data["projection"],
                ),
                use_container_width=True,
                config={
                    "displayModeBar": False,
                    "responsive": True,
                },
            )

            if st.button(
                "Open larger view",
                key=f"expand_timber_{model_name.lower()}",
                icon=":material/open_in_full:",
                use_container_width=True,
            ):
                _show_expanded_chart(
                    f"Timber - {model_name}",
                    model_data["historical"],
                    model_data["projection"],
                )

    # ==================================================
    # TIMBER EXPORT
    # ==================================================

    st.divider()

    st.header("Export results")

    st.caption(
        "Download the historical data and AI forecasts "
        "currently displayed for Timber."
    )

    timber_export_data = _build_timber_export_data(
        data
    )

    timber_csv = (
        timber_export_data
        .to_csv(index=False)
        .encode("utf-8")
    )

    st.download_button(
        label="Export Timber Results",
        data=timber_csv,
        file_name="timber_forecast_results.csv",
        mime="text/csv",
        icon=":material/download:",
        use_container_width=True,
        key="download_timber_results",
    )

    with st.expander("Preview Timber export"):

        st.dataframe(
            timber_export_data.head(20),
            use_container_width=True,
        )

        st.caption(
            f"{len(timber_export_data):,} rows × "
            f"{len(timber_export_data.columns):,} columns"
        )

    # --------------------------------------------------
    # AI summary
    # --------------------------------------------------

    st.divider()

    render_ai_summary(
        "Timber",
        data,
    )

    # --------------------------------------------------
    # Backend contract
    # --------------------------------------------------

    with st.expander(
        "Backend response contract"
    ):
        st.code(
            '''{
  "concrete": {
    "historical": [
      {
        "year": 2000,
        "value": 86.0
      }
    ],
    "projection": [
      {
        "year": 2027,
        "condition": "Most Likely",
        "value": 70.0
      }
    ]
  },
  "timber": {
    "historical": [],
    "projection": []
  }
}''',
            language="json",
        )


# =========================================================
# CONCRETE PAGE
# =========================================================

def _render_concrete_page() -> None:
    """Render RNN and Ridge concrete forecasts side by side."""

    st.title("Concrete outlook")

    st.caption(
        "Compare concrete deterioration projections "
        "from the RNN and Ridge models."
    )

    # --------------------------------------------------
    # Load piles
    # --------------------------------------------------

    try:
        piles = concrete_asset_ids()

    except (
        FileNotFoundError,
        OSError,
    ) as exc:

        st.error(
            f"Could not load concrete data: {exc}"
        )
        return

    pile = st.selectbox(
        "Pile",
        piles,
        index=0,
    )

    # --------------------------------------------------
    # Load pile data
    # --------------------------------------------------

    try:
        data = dashboard_concrete_model_data(
            pile
        )

    except (
        FileNotFoundError,
        ValueError,
        OSError,
    ) as exc:

        st.error(
            f"Could not load concrete data: {exc}"
        )
        return

    # --------------------------------------------------
    # Shared chart ranges
    # --------------------------------------------------

    values = []
    years = []

    for model_data in data.values():

        values += (
            model_data["historical"]
            ["value"]
            .tolist()
        )

        values += (
            model_data["projection"]
            ["value"]
            .tolist()
        )

        years += (
            model_data["historical"]
            ["year"]
            .tolist()
        )

        years += (
            model_data["projection"]
            ["year"]
            .tolist()
        )

    pad = (
        max(values)
        - min(values)
    ) * 0.05 or 0.1

    y_range = [
        min(values) - pad,
        max(values) + pad,
    ]

    x_range = [
        min(years),
        max(years),
    ]

    # --------------------------------------------------
    # Concrete charts
    # --------------------------------------------------

    rnn_col, ridge_col = st.columns(
        2,
        gap="small",
    )

    for container, model_name in (
        (rnn_col, "RNN"),
        (ridge_col, "Ridge"),
    ):

        model_data = data[model_name]

        with container:

            figure = material_forecast_chart(
                f"Concrete - {model_name}",
                model_data["historical"],
                model_data["projection"],
            )

            figure.update_layout(
                yaxis_title=(
                    "Concrete deterioration "
                    "(carbonation depth, mm)"
                )
            )

            figure.update_yaxes(
                range=y_range
            )

            figure.update_xaxes(
                range=x_range
            )

            st.plotly_chart(
                figure,
                use_container_width=True,
                config={
                    "displayModeBar": False,
                    "responsive": True,
                },
            )

            if st.button(
                "Open larger view",
                key=(
                    "expand_concrete_"
                    f"{model_name.lower()}"
                ),
                icon=":material/open_in_full:",
                use_container_width=True,
            ):

                _show_expanded_chart(
                    f"Concrete - {model_name}",
                    model_data["historical"],
                    model_data["projection"],
                )

    # ==================================================
    # CONCRETE EXPORT
    # ==================================================

    st.divider()

    st.header("Export results")

    st.caption(
        "Download the results currently being viewed, "
        "or choose another available result dataset."
    )

    current_export_col, dataset_export_col = st.columns(
        2,
        gap="large",
    )

    # --------------------------------------------------
    # Current pile export
    # --------------------------------------------------

    with current_export_col:

        st.subheader("Current pile")

        st.caption(
            "Download the historical data and AI forecasts "
            f"currently displayed for {pile}. "
            "The CSV contains both RNN and Ridge results."
        )

        export_data = _build_concrete_export_data(
            data,
            pile,
        )

        current_csv = (
            export_data
            .to_csv(index=False)
            .encode("utf-8")
        )

        st.download_button(
            label=f"Export {pile} results",
            data=current_csv,
            file_name=(
                f"{pile}_concrete_forecast.csv"
            ),
            mime="text/csv",
            icon=":material/download:",
            use_container_width=True,
            key="download_current_pile",
        )

        with st.expander(
            "Preview current export"
        ):

            st.dataframe(
                export_data.head(20),
                use_container_width=True,
            )

            st.caption(
                f"{len(export_data):,} rows × "
                f"{len(export_data.columns):,} columns"
            )

    # --------------------------------------------------
    # Other result datasets
    # --------------------------------------------------

    with dataset_export_col:
        _render_result_file_export()

    # --------------------------------------------------
    # AI summary
    # --------------------------------------------------

    st.divider()

    render_ai_summary(
        "Concrete",
        data,
        pile,
    )