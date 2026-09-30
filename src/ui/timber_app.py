"""Standalone Streamlit app: timber pile replacement forecasting.

Run with: streamlit run src/ui/timber_app.py

This is a separate entry point from app.py (the concrete-pile dashboard).
It uses the timber data/feature/model/scenario modules (src/data/timber_data_prep.py,
src/features/timber_features.py, src/models/timber_model.py, src/models/timber_scenarios.py). The
Streamlit UI itself lives under src/ui/ so those pipeline modules stay free
of the streamlit import and can be run or analysed without it -- see
src/models/timber_model.py and docs/timber_model_results.md.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.data.timber_data_prep import DATA_PATH, load_timber_data
from src.models.timber_model import MODEL_PATH, TrainResult, load_model, save_model, train
from src.models.timber_scenarios import SCENARIOS, build_pile_trajectories, summarize_pile

SCENARIO_COLORS = {
    "Hotter & Drier": "#DC2626",
    "Most Likely": "#2563EB",
    "Colder & Wetter": "#0F766E",
}


@st.cache_data(show_spinner="Loading timber pile dataset...")
def _load_data() -> pd.DataFrame:
    return load_timber_data(DATA_PATH)


@st.cache_resource(show_spinner="Loading cached model...")
def _cached_load_model():
    return load_model()


def _get_model(frame: pd.DataFrame, retrain: bool) -> tuple[object, TrainResult | None]:
    """Return a fitted pipeline, retraining if asked or if no cache exists."""
    if retrain or not MODEL_PATH.exists():
        result = train(frame, split_strategy="time")
        save_model(result.pipeline)
        _cached_load_model.clear()
        return result.pipeline, result
    return _cached_load_model(), None


def _build_chart(
    frame: pd.DataFrame, model: object, asset_code: str
) -> tuple[go.Figure, pd.DataFrame]:
    trajectories, split_year = build_pile_trajectories(frame, model, asset_code)
    threshold = float(
        frame.loc[frame["asset_code"] == asset_code, "replacement_diameter_mm"].iloc[0]
    )

    figure = go.Figure()
    for scenario in SCENARIOS:
        trajectory = trajectories[scenario]
        figure.add_trace(
            go.Scatter(
                x=trajectory.years,
                y=trajectory.remaining_diameter_mm,
                mode="lines",
                name=scenario,
                line={"color": SCENARIO_COLORS[scenario], "width": 2.5},
            )
        )
    figure.add_hline(
        y=threshold,
        line_dash="dash",
        line_color="#94A3B8",
        annotation_text=f"Replacement threshold ({threshold:.0f} mm)",
        annotation_position="bottom right",
    )
    figure.add_vline(x=split_year, line_dash="dot", line_color="#64748B")
    figure.add_annotation(
        x=split_year, y=1.05, yref="paper", text="History | Projection", showarrow=False
    )
    figure.update_layout(
        height=460,
        margin={"l": 25, "r": 25, "t": 50, "b": 20},
        xaxis_title="Year",
        yaxis_title="Remaining diameter (mm)",
        hovermode="x unified",
        legend={"orientation": "h", "y": -0.2},
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    figure.update_yaxes(gridcolor="#E2E8F0", rangemode="tozero")
    figure.update_xaxes(showgrid=False)

    summary = summarize_pile(frame, model, asset_code)
    return figure, summary


def main() -> None:
    st.set_page_config(page_title="Timber Pile Replacement Forecast", page_icon="🪵", layout="wide")

    with st.sidebar:
        st.title("🪵 Timber Pile Forecast")
        st.caption(
            "Neural-network surrogate (scikit-learn MLPRegressor) trained on "
            "the AS 5604-style deterministic timber deterioration model output."
        )
        retrain = st.button(
            "Retrain model", help="Re-fit the MLP and overwrite the cached model file."
        )

    frame = _load_data()
    pipeline, train_result = _get_model(frame, retrain=retrain)

    with st.sidebar:
        if train_result is not None:
            st.success("Model trained this session.")
            st.metric("Test MAE (mm)", f"{train_result.test_mae:.2f}")
            st.metric("Test R²", f"{train_result.test_r2:.3f}")
            st.caption(
                f"Split strategy: {train_result.split_strategy} · "
                f"n_train={train_result.n_train} · n_test={train_result.n_test}"
            )
        else:
            st.info("Using cached model. Press Retrain to refresh metrics.")
        st.warning(
            "Only 2 piles exist in this dataset (PileA, PileB). These "
            "metrics describe fit to those two assets, not generalization "
            "to a wider pile population.",
            icon="⚠️",
        )

    st.title("Timber Pile Replacement Forecast")
    st.caption(
        "Select a pile to see its predicted remaining diameter under three "
        "climate scenarios, and the year each scenario is predicted to "
        "cross the replacement threshold."
    )

    piles = sorted(frame["asset_code"].unique())
    asset_code = st.selectbox("Pile", piles)

    figure, summary = _build_chart(frame, pipeline, asset_code)

    st.subheader("Predicted replacement year by scenario")
    metric_columns = st.columns(len(summary))
    for column, (_, row) in zip(metric_columns, summary.iterrows(), strict=True):
        column.metric(row["Scenario"], row["Predicted replacement year"])

    st.subheader("Remaining diameter: history and scenario projections")
    st.plotly_chart(figure, use_container_width=True)

    with st.expander("Scenario detail table"):
        st.dataframe(summary, hide_index=True, use_container_width=True)


if __name__ == "__main__":
    main()
