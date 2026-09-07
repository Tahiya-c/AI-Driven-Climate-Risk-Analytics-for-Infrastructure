"""Main Streamlit dashboard matching the supplied low-fidelity layout."""

from collections.abc import Mapping
from typing import Any

import streamlit as st

from src.visualization.charts import (
    comparison_data_from_payload,
    demo_material_comparison_data,
    material_forecast_chart,
)


def _apply_styles() -> None:
    st.markdown(
        """
        <style>
        .block-container {padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1450px;}
        [data-testid="stSidebar"] {border-right: 1px solid #D8E0E8;}
        [data-testid="stMetric"] {background: white; border: 1px solid #E2E8F0;
            border-radius: 10px; padding: 0.9rem;}
        div[data-testid="stPlotlyChart"] {background: white; border: 1px solid #E2E8F0;
            border-radius: 12px; padding: 0.5rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_dashboard(comparison_payload: Mapping[str, Any] | None = None) -> None:
    st.set_page_config(page_title="Port Pile Deterioration", page_icon="🏗️", layout="wide")
    _apply_styles()

    with st.sidebar:
        st.title("⚓ Port Pile Deterioration")
        st.divider()
    _render_materials_comparison(comparison_payload)


def _render_materials_comparison(
    comparison_payload: Mapping[str, Any] | None = None,
) -> None:
    """Render the concrete and timber history-to-projection comparison page."""
    st.title("Concrete and timber outlook")
    st.caption(
        "Compare historical replacement-time predictions with a 21-year projection."
    )

    if comparison_payload is None:
        data = demo_material_comparison_data()
        st.info("Concrete uses deterministic demo data; timber uses the saved linear model.")
    else:
        try:
            data = comparison_data_from_payload(comparison_payload)
        except ValueError as exc:
            st.error(f"Could not load comparison data: {exc}")
            return

    st.subheader("Prediction parameters")
    parameter_columns = st.columns(4)
    available = numeric_columns(frame) if frame is not None else []
    defaults = ["Pile age", "Chloride level", "Crack width", "Exposure zone"]
    for index, container in enumerate(parameter_columns):
        with container:
            if available:
                selected = st.selectbox(
                    f"Parameter {index + 1}", available, index=index % len(available)
                )
                st.number_input(
                    f"{selected} value", value=float(frame[selected].median()),
                    key=f"value_{index}",
                )
            else:
                st.number_input(defaults[index], min_value=0.0, value=0.0, key=f"demo_{index}")

    concrete, timber = st.columns(2, gap="large")
    for container, material in ((concrete, "Concrete"), (timber, "Timber")):
        material_data = data[material]
        with container:
            st.plotly_chart(
                material_forecast_chart(
                    material,
                    material_data["historical"],
                    material_data["projection"],
                ),
                use_container_width=True,
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

