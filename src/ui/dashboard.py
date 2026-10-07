"""Main Streamlit dashboard matching the supplied low-fidelity layout."""

from collections.abc import Mapping
from math import isfinite
from typing import Any

import streamlit as st

from src.ui.ai_summary import render_ai_summary
from src.visualization.charts import (
    comparison_data_from_payload,
    concrete_asset_ids,
    dashboard_concrete_model_data,
    dashboard_timber_model_data,
    material_forecast_chart,
    timber_asset_ids,
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
    flags: list[dict[str, Any]] | None = None,
) -> None:
    """Show a material chart in a larger modal dialog."""
    figure = material_forecast_chart(material, historical, projection, flags)
    if material.startswith("Concrete"):
        figure.update_layout(yaxis_title="Carbonation depth (mm)")
    figure.update_layout(height=680)
    st.plotly_chart(
        figure,
        use_container_width=True,
        config={
            "displayModeBar": False,
            "responsive": True,
        },
    )


@st.dialog("Add flag")
def _add_flag_dialog(material: str, pile: str) -> None:
    """Collect a user-defined threshold for the selected material and pile."""
    metric = "Material loss" if material == "Timber" else "Carbonation depth"
    st.caption(f"{material} · {pile}")
    with st.form(f"flag_form_{material}_{pile}"):
        purpose = st.selectbox("Flag purpose", ["Maintenance", "Replacement"])
        value = st.number_input(f"{metric} (mm)", min_value=0.0, value=None, step=1.0)
        submitted = st.form_submit_button("Add flag", type="primary")
    if submitted:
        if value is None or not isfinite(value):
            st.error("Enter a finite, non-negative value in mm.")
            return
        flags = st.session_state.setdefault("user_flags", {})
        pile_flags = flags.setdefault((material, pile), [])
        flag = {"purpose": purpose, "value": value}
        if flag not in pile_flags:
            pile_flags.append(flag)
        st.rerun()


def _pile_controls(material: str, piles: list[str]) -> tuple[str, list[dict[str, Any]]]:
    """Separate pile selection from the chart threshold controls."""
    pile = st.selectbox("Pile", piles, index=0, key=f"{material.lower()}_pile")
    st.divider()
    st.markdown("**Maintenance and replacement flags**")
    st.caption("Add a threshold line to both charts for the selected pile.")
    if st.button("Add flag", key=f"add_flag_{material}"):
        _add_flag_dialog(material, pile)
    flags = st.session_state.get("user_flags", {}).get((material, pile), [])
    if flags:
        st.caption("Flags are saved for this pile during the current session.")
        for index, flag in enumerate(flags):
            label, remove = st.columns([4, 1])
            label.write(f"{flag['purpose']}: {flag['value']:g} mm")
            if remove.button("Remove", key=f"remove_flag_{material}_{pile}_{index}"):
                flags.pop(index)
                st.rerun()
    return pile, flags


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

    try:
        piles = timber_asset_ids()
    except (FileNotFoundError, ValueError, OSError, KeyError) as exc:
        st.error(f"Could not load timber data: {exc}")
        return
    if not piles:
        st.info("No timber piles have forecasts available for both models.")
        return

    pile, flags = _pile_controls("Timber", piles)

    try:
        if comparison_payload is None:
            data = dashboard_timber_model_data(pile)
        else:
            data = comparison_data_from_payload(comparison_payload)["Timber"]
    except (FileNotFoundError, ValueError, OSError) as exc:
        st.error(f"Could not load timber data: {exc}")
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
                    flags,
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
                    flags,
                )

    render_ai_summary("Timber", data, pile)

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
    """Render RNN and Ridge concrete carbonation forecasts side by side."""
    st.title("Concrete outlook")
    st.caption(
        "Compare carbonation depth projections from the RNN and Ridge models."
    )

    try:
        piles = concrete_asset_ids()
    except (FileNotFoundError, OSError) as exc:
        st.error(f"Could not load concrete data: {exc}")
        return

    pile, flags = _pile_controls("Concrete", piles)

    try:
        data = dashboard_concrete_model_data(pile)
    except (FileNotFoundError, ValueError, OSError) as exc:
        st.error(f"Could not load concrete data: {exc}")
        return

    rnn_col, ridge_col = st.columns(2, gap="small")
    for container, model_name in ((rnn_col, "RNN"), (ridge_col, "Ridge")):
        model_data = data[model_name]
        with container:
            figure = material_forecast_chart(
                f"Concrete - {model_name}",
                model_data["historical"],
                model_data["projection"],
                flags,
            )
            figure.update_layout(yaxis_title="Carbonation depth (mm)")
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
                key=f"expand_concrete_{model_name.lower()}",
                icon=":material/open_in_full:",
                use_container_width=True,
            ):
                _show_expanded_chart(
                    f"Concrete - {model_name}",
                    model_data["historical"],
                    model_data["projection"],
                    flags,
                )

    render_ai_summary("Concrete", data, pile)

