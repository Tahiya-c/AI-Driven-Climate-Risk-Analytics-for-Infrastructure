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

    st.subheader("Projection conditions")
    st.caption("The same three conditions are shown for both materials to make comparison direct.")
    summary_columns = st.columns(4)
    summary_columns[0].metric("Historical window", "2000 - present")
    summary_columns[1].metric("Projection window", "21 years")
    summary_columns[2].metric("Materials", "2")
    summary_columns[3].metric("Conditions", "3")

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

