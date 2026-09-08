"""Main Streamlit dashboard matching the supplied low-fidelity layout."""

from collections.abc import Mapping
from typing import Any

import streamlit as st

from src.visualization.charts import (
    comparison_data_from_payload,
    dashboard_timber_model_data,
    material_forecast_chart,
)


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
        [data-testid="stSidebar"] {border-right: 1px solid #D8E0E8;}
        [data-testid="stMetric"] {background: white; border: 1px solid #E2E8F0;
            border-radius: 10px; padding: 0.9rem;}
        div[data-testid="stHorizontalBlock"] {width: 100%; min-width: 0;}
        div[data-testid="stHorizontalBlock"] > div {min-width: 0 !important;}
        div[data-testid="stPlotlyChart"] {width: 100%; max-width: 100%; min-width: 0;
            overflow: hidden;
            background: white; border: 1px solid #E2E8F0;
            border-radius: 12px; padding: 0.5rem;}
        div[data-testid="stPlotlyChart"] > div {width: 100% !important; min-width: 0 !important;}
        div[data-testid="stPlotlyChart"] .modebar-container,
        div[data-testid="stPlotlyChart"] .modebar {
            position: absolute !important;
            top: 0.4rem !important;
            right: auto !important;
            left: 7rem !important;
        }
        .modebar-btn[data-title="Autoscale"],
        .modebar-btn[data-title="Reset axes"] {display: none !important;}
        @media (max-width: 900px) {
            div[data-testid="stHorizontalBlock"] {flex-wrap: wrap !important;}
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
    figure = material_forecast_chart(material, historical, projection)
    figure.update_layout(height=680)
    st.plotly_chart(
        figure,
        use_container_width=True,
        config={
            "displayModeBar": False,
            "responsive": True,
        },
    )


def render_dashboard(comparison_payload: Mapping[str, Any] | None = None) -> None:
    st.set_page_config(page_title="Port Pile Deterioration", page_icon="🏗️", layout="wide")
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


def _render_timber_page(
    comparison_payload: Mapping[str, Any] | None = None,
) -> None:
    """Render GBM and MLP Timber forecasts side by side."""
    st.title("Timber outlook")
    st.caption(
        "Compare material loss projections from the GBM and MLP models."
    )

    if comparison_payload is None:
        data = dashboard_timber_model_data()
    else:
        try:
            data = comparison_data_from_payload(comparison_payload)["Timber"]
        except ValueError as exc:
            st.error(f"Could not load comparison data: {exc}")
            return

    gbm, mlp = st.columns(2, gap="small")
    for container, model_name in ((gbm, "GBM"), (mlp, "MLP")):
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

    with st.expander("Backend response contract"):
        st.code(
            '''{
  "concrete": {
    "historical": [{"year": 2000, "value": 86.0}],
    "projection": [{"year": 2027, "condition": "Most Likely", "value": 70.0}]
  },
  "timber": {"historical": [], "projection": []}
}''',
            language="json",
        )


def _render_concrete_page() -> None:
    """Render the reserved Concrete page shell."""
    st.title("Concrete outlook")
    st.caption("Concrete model results will be displayed here.")
    st.columns(2, gap="small")

