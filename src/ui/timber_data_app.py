"""Standalone Streamlit page: explore the timber dataset.

Run with: streamlit run src/ui/timber_data_app.py

Visualises data/processed/timber_kembla_v6_3_consolidated.csv (D9-corrected): the two
piles' attributes, material loss and remaining diameter per climate scenario, the
climate drivers, and why hazard normalisation (loss / k0) lets PileA's data describe
PileB (docs/timber_model_updated.md), and the GBM vs MLP forecasts for PileB.
Kept separate from app.py / src/ui/dashboard.py.
"""

from __future__ import annotations

import warnings

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.models.timber_forecast import load_model
from src.models.timber_transfer import (
    LOSS_COLUMN,
    MODEL_INPUT_COLUMNS,
    SPLIT_YEAR,
    TEST_PILE,
    TRAIN_PILE,
    anchor_forecast,
    load_transfer_frame,
    metrics,
)
from src.ui.dashboard import _apply_styles
from src.ui.timber_import_page import render_import_section
from src.visualization.charts import SCENARIO_COLORS, SCENARIO_LABELS, material_forecast_chart

FORECAST_MODELS = {"GBM": "gbm", "MLP": "mlp"}
FORECAST_PATHWAY = "A1FI"

SCENARIO_COLORS_ALL = {"Baseline": "#334155", **SCENARIO_COLORS}
PILE_COLORS = {"PileA": "#0F766E", "PileB": "#D97706"}
PATHWAY_DASH = {"Base": "solid", "A1B": "solid", "A1FI": "dash"}
CHART_CONFIG = {"displayModeBar": False, "responsive": True}
ATTRIBUTE_ROWS = {
    "timber_type": "Timber type",
    "year_installed": "Year installed",
    "initial_diameter_mm": "Initial diameter (mm)",
    "replacement_diameter_mm": "Replacement diameter (mm)",
    "zone": "Zone",
    "exposure": "Exposure code",
    "heartwood_class": "Heartwood class",
    "maintenance_measure": "Maintenance code",
    "contact": "Contact code",
    "knot_presence": "Knot code",
    "heartwood_factor": "k_HW (heartwood)",
    "shelter_factor": "k_shelter",
    "salinity_factor": "k_sal",
    "combined_factor": "k0 (combined hazard)",
}


@st.cache_data(show_spinner=False)
def _load() -> pd.DataFrame:
    frame = load_transfer_frame()
    frame["loss_per_k0"] = frame[LOSS_COLUMN] / frame["combined_factor"]
    frame["series"] = frame["scenario_label"] + " (" + frame["emissions_pathway"] + ")"
    return frame


def _style(figure: go.Figure, title: str, yaxis_title: str, xaxis_title: str = "Year"):
    figure.update_layout(
        title=title,
        height=440,
        autosize=True,
        margin={"l": 30, "r": 30, "t": 55, "b": 70},
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        hovermode="x unified",
        legend={"orientation": "h", "y": -0.2},
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    figure.update_xaxes(showgrid=False)
    figure.update_yaxes(gridcolor="#E2E8F0")
    return figure


def _scenario_lines(rows: pd.DataFrame, column: str) -> go.Figure:
    """One line per scenario/pathway series for a single pile."""
    figure = go.Figure()
    for (scenario, pathway), series in rows.groupby(["scenario_label", "emissions_pathway"]):
        series = series.sort_values("year")
        figure.add_trace(
            go.Scatter(
                x=series["year"],
                y=series[column],
                name=f"{scenario} ({pathway})",
                mode="lines",
                line={
                    "color": SCENARIO_COLORS_ALL[scenario],
                    "width": 2,
                    "dash": PATHWAY_DASH.get(pathway, "solid"),
                },
            )
        )
    figure.add_vline(x=SPLIT_YEAR, line_dash="dot", line_color="#94A3B8")
    return figure


def _overview(frame: pd.DataFrame) -> None:
    piles, series, rows = st.columns(3, gap="small")
    piles.metric("Piles", frame["asset_code"].nunique())
    series.metric("Climate series per pile", frame.groupby("asset_code")["series"].nunique().max())
    rows.metric("Rows", f"{len(frame)} ({int(frame['year'].min())}-{int(frame['year'].max())})")
    st.caption(
        "Every row is output from the Timber_kembla_v6.3 spreadsheet's AS 5604-style decay "
        "formula, not a field measurement."
    )
    attributes = (
        frame.groupby("asset_code")[list(ATTRIBUTE_ROWS)].first().T.rename(index=ATTRIBUTE_ROWS)
    )
    st.dataframe(attributes.astype(str), width="stretch")


def _per_pile_charts(frame: pd.DataFrame, column: str, label: str, threshold: bool) -> None:
    left, right = st.columns(2, gap="small")
    for container, pile in ((left, "PileA"), (right, "PileB")):
        rows = frame.loc[frame["asset_code"] == pile]
        figure = _scenario_lines(rows, column)
        if threshold:
            replacement = float(rows["replacement_diameter_mm"].iloc[0])
            figure.add_hline(
                y=replacement,
                line_dash="dot",
                line_color="#DC2626",
                annotation_text=f"Replacement threshold ({replacement:.0f} mm)",
                annotation_position="bottom left",
            )
        else:
            radius = float(rows["initial_diameter_mm"].iloc[0]) / 2
            figure.add_hline(
                y=radius,
                line_dash="dot",
                line_color="#94A3B8",
                annotation_text=f"Full radius ({radius:.0f} mm)",
                annotation_position="top left",
            )
        with container:
            st.plotly_chart(_style(figure, pile, label), width="stretch", config=CHART_CONFIG)


def _climate_charts(frame: pd.DataFrame) -> None:
    climate = frame.loc[frame["asset_code"] == "PileA"]  # identical climate for both piles
    left, right = st.columns(2, gap="small")
    for container, column, label in (
        (left, "summer_sea_temp_c", "Summer sea temperature (°C)"),
        (right, "salinity", "Salinity (ppt)"),
    ):
        with container:
            st.plotly_chart(
                _style(_scenario_lines(climate, column), label, label),
                width="stretch",
                config=CHART_CONFIG,
            )


def _comparison_charts(frame: pd.DataFrame) -> None:
    """Most Likely (A1B): raw loss vs. loss / k0 for both piles, against age."""
    rows = frame.loc[
        (frame["scenario_label"] == "Most Likely") & (frame["emissions_pathway"] == "A1B")
    ]
    left, right = st.columns(2, gap="small")
    for container, column, title, label in (
        (left, LOSS_COLUMN, "Material loss by age", "Material loss (mm)"),
        (right, "loss_per_k0", "Material loss ÷ k0 by age", "Loss per unit hazard (mm)"),
    ):
        figure = go.Figure()
        for pile, series in rows.groupby("asset_code"):
            series = series.sort_values("age_years")
            k0 = float(series["combined_factor"].iloc[0])
            figure.add_trace(
                go.Scatter(
                    x=series["age_years"],
                    y=series[column],
                    name=f"{pile} (k0 = {k0:g})",
                    mode="lines",
                    line={"color": PILE_COLORS[pile], "width": 3},
                )
            )
        with container:
            st.plotly_chart(
                _style(figure, title, label, xaxis_title="Age (years since installation)"),
                width="stretch",
                config=CHART_CONFIG,
            )
    st.caption(
        "Left: PileB loses material ~2.4x faster than PileA, because its k0 is 3.12 vs. 1.3. "
        "Right: divided by k0, the two piles rise at the same rate (parallel lines), which is "
        "why a model trained on PileA's loss ÷ k0 can forecast PileB. The constant ~9 mm gap "
        "between them is the fixed 20 mm sapwood loss, which does not scale with k0 "
        "(20/1.3 − 20/3.12 ≈ 9). Scaled back up by PileB's k0, it becomes the ~28 mm "
        "over-prediction in the report. PileB's line flattens once its loss hits the "
        "250 mm radius."
    )


@st.cache_data(show_spinner="Loading the saved GBM and MLP models...")
def _model_predictions() -> pd.DataFrame:
    """PileB rows with each saved model's predicted material loss (trained on PileA only)."""
    frame = load_transfer_frame()
    pile_b = frame.loc[frame["asset_code"] == TEST_PILE].copy()
    for name, kind in FORECAST_MODELS.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = load_model(kind)  # the saved model of record, not a fresh fit
        pile_b[name] = model.predict(pile_b[MODEL_INPUT_COLUMNS])
        pile_b[f"{name} anchored"] = anchor_forecast(
            pile_b, model.predict_uncapped(pile_b[MODEL_INPUT_COLUMNS])
        )
    return pile_b


def _forecast_chart(
    rows: pd.DataFrame, model_name: str, column: str, show_truth: bool
) -> go.Figure:
    """History, 3 scenario forecasts and the Baseline (no climate change) forecast."""
    history = (
        rows.loc[(rows["year"] <= SPLIT_YEAR) & (rows["scenario_label"] != "Baseline")]
        .groupby("year", as_index=False)[LOSS_COLUMN]
        .mean()
        .rename(columns={LOSS_COLUMN: "value"})
    )
    future = rows.loc[rows["year"] > SPLIT_YEAR]
    projection = (
        future.loc[future["scenario_label"].isin(SCENARIO_LABELS)]
        .groupby(["year", "scenario_label"], as_index=False)[column]
        .mean()
        .rename(columns={"scenario_label": "condition", column: "value"})
    )
    figure = material_forecast_chart(f"Timber - {model_name}", history, projection)
    baseline = future.loc[future["scenario_label"] == "Baseline"].sort_values("year")
    figure.add_trace(
        go.Scatter(
            x=baseline["year"],
            y=baseline[column],
            name="Baseline (no climate change)",
            mode="lines",
            line={"color": SCENARIO_COLORS_ALL["Baseline"], "width": 2, "dash": "dash"},
        )
    )
    if show_truth:
        for label in [*SCENARIO_LABELS, "Baseline"]:
            truth = future.loc[future["scenario_label"] == label].sort_values("year")
            figure.add_trace(
                go.Scatter(
                    x=truth["year"],
                    y=truth[LOSS_COLUMN],
                    name=f"{label} (actual)",
                    mode="lines",
                    line={"color": SCENARIO_COLORS_ALL[label], "width": 1},
                    opacity=0.4,
                )
            )
    radius = float(rows["initial_diameter_mm"].iloc[0]) / 2
    figure.add_hline(
        y=radius,
        line_dash="dot",
        line_color="#94A3B8",
        annotation_text=f"Full radius ({radius:.0f} mm)",
        annotation_position="top left",
    )
    return figure


def _model_forecasts() -> None:
    anchored = st.toggle(
        f"Anchor forecasts to the {SPLIT_YEAR} actual value",
        value=True,
        help=f"Shifts each scenario's forecast by (actual - predicted) at {SPLIT_YEAR}, "
        "removing the jump caused by the models' ~28 mm bias. Uses one real "
        f"{TEST_PILE} observation, so it is no longer a pure {TRAIN_PILE}-only forecast.",
    )
    show_truth = st.toggle("Show actual values for the forecast years", value=False)
    suffix = " anchored" if anchored else ""
    pile_b = _model_predictions()
    rows = pile_b.loc[pile_b["emissions_pathway"].isin([FORECAST_PATHWAY, "Base"])]
    forecast = rows.loc[(rows["year"] > SPLIT_YEAR) & rows["scenario_label"].isin(SCENARIO_LABELS)]
    st.markdown(
        f"**{TEST_PILE} forecast, {SPLIT_YEAR + 1}-{int(forecast['year'].max())}, "
        f"{FORECAST_PATHWAY} pathway** (models trained on {TRAIN_PILE} only"
        + (f", forecasts anchored to {TEST_PILE}'s {SPLIT_YEAR} value" if anchored else "")
        + f"; scored on the 3 climate scenarios, {len(forecast)} rows)"
    )
    raw_r2 = {
        name: metrics(forecast[LOSS_COLUMN], forecast[name])["r2"] for name in FORECAST_MODELS
    }
    for name in FORECAST_MODELS:
        scores = metrics(forecast[LOSS_COLUMN], forecast[name + suffix])
        label, r2, rmse, mae = st.columns([1, 2, 2, 2], gap="small")
        label.markdown(f"#### {name}")
        r2.metric("R²", f"{scores['r2']:.3f}")
        rmse.metric("RMSE (mm)", f"{scores['rmse']:.1f}")
        mae.metric("MAE (mm)", f"{scores['mae']:.1f}")

    left, right = st.columns(2, gap="small")
    for container, name in zip((left, right), FORECAST_MODELS):
        with container:
            st.plotly_chart(
                _forecast_chart(rows, name, name + suffix, show_truth),
                width="stretch",
                config=CHART_CONFIG,
            )
    st.caption(
        "Both models use the same features, target and techniques: material loss (mm) as "
        "the target, thermal_dose, the zero-variance filter, hazard normalisation by k0, and "
        f"Optuna tuning on a chronological hold-out of {TRAIN_PILE} (2057-2070). "
        "The dashed grey line is each model's forecast with no climate change (Baseline "
        f"scenario). History is {TEST_PILE}'s actual material loss for 2000-{SPLIT_YEAR}."
    )
    if anchored:
        st.info(
            f"Anchored view: both models over-predict {TEST_PILE} by a near-constant ~28 mm "
            "(the fixed 20 mm sapwood loss scaled by k0). Anchoring removes that offset "
            f"using {TEST_PILE}'s real {SPLIT_YEAR} value, which is why the lines join "
            "smoothly and the scores are close to 1. Switch it off to see each model's own "
            "forecast: "
            + ", ".join(f"R² {r2:.3f} ({name})" for name, r2 in raw_r2.items())
            + f", {TRAIN_PILE}-only."
        )


def _data_table(frame: pd.DataFrame) -> None:
    piles = st.multiselect("Pile", sorted(frame["asset_code"].unique()), default=["PileA", "PileB"])
    series = st.multiselect("Series", sorted(frame["series"].unique()))
    rows = frame.loc[frame["asset_code"].isin(piles)]
    if series:
        rows = rows.loc[rows["series"].isin(series)]
    columns = [
        "asset_code",
        "scenario_label",
        "emissions_pathway",
        "year",
        "age_years",
        "summer_sea_temp_c",
        "salinity",
        "water_factor",
        "combined_factor",
        "remaining_diameter_mm",
        LOSS_COLUMN,
    ]
    st.dataframe(rows[columns], width="stretch", hide_index=True, height=380)
    st.download_button(
        "Download filtered rows (CSV)",
        rows.to_csv(index=False).encode("utf-8"),
        file_name="timber_data_filtered.csv",
        mime="text/csv",
        icon=":material/download:",
    )


def render() -> None:
    st.set_page_config(page_title="Timber Data Explorer", page_icon="🪵", layout="wide")
    _apply_styles()
    frame = _load()

    with st.sidebar:
        st.title("🪵 Timber data explorer")
        section = st.radio(
            "Section",
            [
                "Overview",
                "Material loss",
                "Remaining diameter",
                "Climate drivers",
                "PileA vs PileB",
                "Model forecasts (GBM vs MLP)",
                "Import a pile",
                "Data table",
            ],
            label_visibility="collapsed",
        )
        st.divider()
        st.caption(
            "Solid lines: A1B pathway (and Baseline). Dashed lines: A1FI. "
            f"Dotted vertical line: {SPLIT_YEAR}, the dashboard's history/forecast boundary."
        )

    st.title("Timber data explorer")
    if section == "Overview":
        _overview(frame)
    elif section == "Material loss":
        st.caption("Radial material loss: (initial diameter − remaining diameter) / 2.")
        _per_pile_charts(frame, LOSS_COLUMN, "Material loss (mm)", threshold=False)
    elif section == "Remaining diameter":
        st.caption(
            "The spreadsheet does not floor diameter at 0 (Defect D7), so PileB goes negative."
        )
        _per_pile_charts(frame, "remaining_diameter_mm", "Remaining diameter (mm)", threshold=True)
    elif section == "Climate drivers":
        st.caption("Both piles share the same climate series.")
        _climate_charts(frame)
    elif section == "PileA vs PileB":
        _comparison_charts(frame)
    elif section == "Model forecasts (GBM vs MLP)":
        _model_forecasts()
    elif section == "Import a pile":
        render_import_section()
    else:
        _data_table(frame)


render()
