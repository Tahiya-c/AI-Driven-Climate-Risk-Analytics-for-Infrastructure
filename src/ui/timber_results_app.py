"""Standalone Streamlit page: train on PileA, forecast PileB's material loss.

Run with: streamlit run src/ui/timber_results_app.py

Same chart as the main dashboard's Timber page (material loss, history then three
climate scenarios, via src/visualization/charts.py::material_forecast_chart), driven
by the hazard-normalised MLP/GBM in src/models/timber_transfer.py:

- Both models are trained on PileA only and never see PileB.
- Charts use one emissions pathway (``PATHWAY``, A1FI), matching the dashboard CSVs
  built by ``build_dashboard_output --material timber --transfer``. "Historical" is
  PileB's engine output for 2000-2026; projections are model predictions for 2027-2070.
- An optional overlay shows the engine's own values for the forecast years.

Kept separate from app.py / src/ui/dashboard.py so the main dashboard is unaffected.
Write-up: docs/timber_pileA_to_pileB_report.md.
"""

from __future__ import annotations

import json
import warnings

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.models.timber_forecast import load_model
from src.models.timber_transfer import (
    LOSS_COLUMN,
    MODEL_INPUT_COLUMNS,
    RESULTS_PATH,
    SPLIT_YEAR,
    TEST_PILE,
    TRAIN_PILE,
    forecast_mask,
    load_transfer_frame,
    metrics,
)
from src.ui.dashboard import _apply_styles
from src.visualization.charts import SCENARIO_COLORS, SCENARIO_LABELS, material_forecast_chart

MODELS = ("GBM", "MLP")
PATHWAY = "A1FI"
STEP_LABELS = {
    "1_baseline": "1. Baseline (untuned)",
    "2_hazard_normalised": "2. + hazard normalisation",
    "3_tuned": "3. + Optuna (hold-out)",
}
CHART_CONFIG = {"displayModeBar": False, "responsive": True}


@st.cache_data(show_spinner="Loading the saved models, predicting PileB...")
def _pile_b_predictions() -> pd.DataFrame:
    frame = load_transfer_frame()
    pile_b = frame.loc[frame["asset_code"] == TEST_PILE].copy()
    for model_name in MODELS:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = load_model(model_name.lower())  # saved model of record (PileA-trained)
        pile_b[model_name] = model.predict(pile_b[MODEL_INPUT_COLUMNS])
    return pile_b


def _projection(rows: pd.DataFrame, value_column: str) -> pd.DataFrame:
    future = rows.loc[forecast_mask(rows)]
    return (
        future.groupby(["year", "scenario_label"], as_index=False)[value_column]
        .mean()
        .rename(columns={"scenario_label": "condition", value_column: "value"})
    )


def _forecast_chart(pile_b: pd.DataFrame, model_name: str, show_truth: bool) -> go.Figure:
    history = (
        pile_b.loc[pile_b["year"] <= SPLIT_YEAR]
        .groupby("year", as_index=False)[LOSS_COLUMN]
        .mean()
        .rename(columns={LOSS_COLUMN: "value"})
    )
    figure = material_forecast_chart(
        f"Timber - {model_name} ({TEST_PILE}, trained on {TRAIN_PILE})",
        history,
        _projection(pile_b, model_name),
    )
    if show_truth:
        truth = _projection(pile_b, LOSS_COLUMN)
        for condition in SCENARIO_LABELS:
            rows = truth.loc[truth["condition"] == condition]
            figure.add_trace(
                go.Scatter(
                    x=rows["year"],
                    y=rows["value"],
                    name=f"{condition} (engine)",
                    mode="lines",
                    line={"color": SCENARIO_COLORS[condition], "width": 1},
                    opacity=0.45,
                )
            )
    radius = float(pile_b["initial_diameter_mm"].iloc[0]) / 2
    figure.add_hline(
        y=radius,
        line_dash="dot",
        line_color="#94A3B8",
        annotation_text=f"Full radius ({radius:.0f} mm)",
        annotation_position="top left",
    )
    return figure


def _metric_tiles(pile_b: pd.DataFrame) -> None:
    forecast = pile_b.loc[forecast_mask(pile_b)]
    st.markdown(
        f"**{TEST_PILE} accuracy** (material loss vs. engine output; models never saw "
        f"{TEST_PILE}). All years: {len(pile_b)} rows. "
        f"Forecast {SPLIT_YEAR + 1}-{int(forecast['year'].max())}: {len(forecast)} rows."
    )
    for model_name in MODELS:
        all_rows = metrics(pile_b[LOSS_COLUMN], pile_b[model_name])
        future = metrics(forecast[LOSS_COLUMN], forecast[model_name])
        label, r2, r2_future, rmse, mae = st.columns([1, 2, 2, 2, 2], gap="small")
        label.markdown(f"#### {model_name}")
        r2.metric("R² (all years)", f"{all_rows['r2']:.3f}")
        r2_future.metric("R² (forecast years)", f"{future['r2']:.3f}")
        rmse.metric("RMSE (mm, all years)", f"{all_rows['rmse']:.1f}")
        mae.metric("MAE (mm, all years)", f"{all_rows['mae']:.1f}")


def _ablation_chart() -> go.Figure | None:
    if not RESULTS_PATH.exists():
        return None
    results = json.loads(RESULTS_PATH.read_text())
    if "evaluation" not in results:
        return None
    rows = pd.DataFrame(results["evaluation"]["ablation"])
    figure = go.Figure()
    for model_name, color in (("GBM", "#0F766E"), ("MLP", "#2563EB")):
        model_rows = rows.loc[rows["model"] == model_name.lower()]
        figure.add_trace(
            go.Bar(
                x=[STEP_LABELS[step] for step in model_rows["step"]],
                y=model_rows["pileB_all_r2"].clip(lower=0),
                name=model_name,
                marker_color=color,
                text=[f"{value:.3f}" for value in model_rows["pileB_all_r2"]],
                textposition="outside",
            )
        )
    figure.update_layout(
        title=f"{TEST_PILE} R² (all years) after each step",
        barmode="group",
        height=420,
        margin={"l": 30, "r": 30, "t": 55, "b": 70},
        yaxis_title="R²",
        legend={"orientation": "h", "y": -0.18},
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    figure.update_yaxes(range=[0, 1.1], gridcolor="#E2E8F0")
    return figure


def render() -> None:
    st.set_page_config(page_title="Timber Model Results", page_icon="🪵", layout="wide")
    _apply_styles()

    with st.sidebar:
        st.title("🪵 Timber model results")
        show_truth = st.toggle("Show engine output for forecast years", value=True)
        st.divider()
        st.caption(
            f"Hazard-normalised MLP/GBM trained on {TRAIN_PILE} only, tuned with Optuna + "
            f"a chronological hold-out of {TRAIN_PILE}. See docs/timber_model_updated.md."
        )

    st.title(f"Timber outlook: {TEST_PILE} forecast from a {TRAIN_PILE}-trained model")
    st.caption(
        f"Material loss history ({TEST_PILE} engine output, 2000-{SPLIT_YEAR}), then model "
        f"projections for three climate scenarios under the {PATHWAY} emissions pathway. "
        "The scores below cover both pathways."
    )

    pile_b = _pile_b_predictions()
    _metric_tiles(pile_b)
    chart_rows = pile_b.loc[pile_b["emissions_pathway"] == PATHWAY]

    gbm_col, mlp_col = st.columns(2, gap="small")
    for container, model_name in ((gbm_col, "GBM"), (mlp_col, "MLP")):
        with container:
            st.plotly_chart(
                _forecast_chart(chart_rows, model_name, show_truth),
                width="stretch",
                config=CHART_CONFIG,
            )

    with st.expander("What each step contributed"):
        ablation = _ablation_chart()
        if ablation is None:
            st.info("Run `python -m src.models.timber_transfer --evaluate` to generate this chart.")
        else:
            st.plotly_chart(ablation, width="stretch", config=CHART_CONFIG)

    st.warning(
        "Scores compare each model with the deterministic AS 5604-style engine, not with "
        "real inspection data, and rest on a single held-out pile. The hazard "
        "normalisation assumes decay scales with the k0 factor, as in the engine."
    )


render()
