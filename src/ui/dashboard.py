"""Main Streamlit dashboard matching the supplied low-fidelity layout."""

import streamlit as st

from src.data.loader import load_csv, numeric_columns
from src.visualization.charts import condition_forecast_chart


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


def render_dashboard() -> None:
    st.set_page_config(page_title="Port Pile Health", page_icon="🏗️", layout="wide")
    _apply_styles()

    with st.sidebar:
        st.title("⚓ Port Pile Health")
        page = st.radio(
            "Navigation", ["Predictive Model", "Geospatial Mapping", "Settings"],
            label_visibility="collapsed",
        )
        st.divider()
        uploaded_file = st.file_uploader("Concrete pile dataset", type="csv")
        st.caption("Upload the project CSV to populate parameter controls.")

    st.title("Dashboard")
    st.caption("Monitor structural condition and explore future deterioration scenarios.")

    if page != "Predictive Model":
        st.info(f"{page} is scaffolded for the next development phase.")
        return

    frame = None
    if uploaded_file is not None:
        try:
            frame = load_csv(uploaded_file)
            st.success(f"Loaded {len(frame):,} pile inspection records.")
        except (ValueError, OSError) as exc:
            st.error(f"Could not load the CSV: {exc}")

    st.subheader("Prediction parameters")
    parameter_columns = st.columns(4)
    available = numeric_columns(frame) if frame is not None else []
    defaults = ["Pile age", "Chloride level", "Crack width", "Exposure zone"]
    for index, container in enumerate(parameter_columns):
        with container:
            if available:
                selected = st.selectbox(f"Parameter {index + 1}", available, index=index % len(available))
                st.number_input(
                    f"{selected} value", value=float(frame[selected].median()),
                    key=f"value_{index}",
                )
            else:
                st.number_input(defaults[index], min_value=0.0, value=0.0, key=f"demo_{index}")

    st.divider()
    metric_columns = st.columns(4)
    metric_columns[0].metric("Predicted level", "Moderate", "Demo")
    metric_columns[1].metric("Condition index", "62 / 100", "-4.1%")
    metric_columns[2].metric("Remaining life", "12.4 years")
    metric_columns[3].metric("Inspection priority", "Medium")

    st.subheader("Historical change and future prediction")
    st.plotly_chart(condition_forecast_chart(), use_container_width=True)
    st.caption("Illustrative forecast until the trained model and agreed CSV schema are connected.")

