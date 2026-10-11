"""Plotly chart builders for historical condition and forecasts."""

from collections.abc import Mapping
from datetime import date
from typing import Any

import pandas as pd
import plotly.graph_objects as go

from src.config import ROOT_DIR

SCENARIO_LABELS = ("Most Likely", "Hotter & Drier", "Colder & Wetter")
SCENARIO_COLORS = {
    "Most Likely": "#2563EB",
    "Hotter & Drier": "#DC2626",
    "Colder & Wetter": "#0891B2",
}
# Climate model (GCM) in a scenario name such as 'A1FI|CSIRO Mk3.5' -> chart label.
GCM_CONDITIONS = {"CSIRO": "Hotter & Drier", "MRI": "Most Likely", "MIROC": "Colder & Wetter"}


def scenario_condition(scenario: str) -> str:
    """'A1FI|MIROC(hires)' -> 'Colder & Wetter'; friendly labels pass through."""
    if scenario in SCENARIO_LABELS:
        return scenario
    return next(
        (label for gcm, label in GCM_CONDITIONS.items() if gcm in scenario), "Most Likely"
    )


def chart_frames(points: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Dashboard-schema rows for one pile and one model -> chart history + projections.

    Rows with method 'observed' are the history (averaged over scenarios: a pile has
    one measured history); every other row is a forecast, labelled by its scenario.
    """
    frame = points.assign(condition=points["scenario"].map(scenario_condition))
    observed = frame["method"] == "observed"
    historical = frame.loc[observed].groupby("year", as_index=False)["value"].mean()
    projection = (
        frame.loc[~observed].groupby(["year", "condition"], as_index=False)["value"].mean()
    )
    if historical.empty or projection.empty:
        raise ValueError("Forecast data needs both observed and forecast rows.")
    return {"historical": historical, "projection": projection}


# The functions below read the local CSV exports in data/dashboard/. The dashboard
# itself reads Supabase (src/db/dashboard_data.py); these remain for offline tests.


def timber_asset_ids() -> list[str]:
    """Pile ids with forecasts available in both timber dashboard exports."""
    asset_ids = []
    for model in ("gbm", "mlp"):
        filename = f"dashboard_data_timber_pileB_from_pileA_{model}.csv"
        path = ROOT_DIR / "data" / "dashboard" / filename
        asset_ids.append(set(pd.read_csv(path)["asset_id"].dropna().unique()))
    return sorted(asset_ids[0] & asset_ids[1])


def dashboard_timber_model_data(
    asset_id: str | None = None,
) -> dict[str, dict[str, pd.DataFrame]]:
    """Load the GBM and MLP timber forecasts, optionally for one pile."""
    return {
        "GBM": _load_dashboard_csv("timber_pileB_from_pileA_gbm", asset_id),
        "MLP": _load_dashboard_csv("timber_pileB_from_pileA_mlp", asset_id),
    }


def concrete_asset_ids() -> list[str]:
    """Pile ids available in the concrete dashboard export."""
    path = ROOT_DIR / "data" / "dashboard" / "dashboard_data_concrete_rnn.csv"
    return sorted(pd.read_csv(path)["asset_id"].unique())


def dashboard_concrete_model_data(
    asset_id: str | None = None,
) -> dict[str, dict[str, pd.DataFrame]]:
    """Load the RNN and Ridge concrete forecasts, optionally for one pile."""
    return {
        "RNN": _load_concrete_csv("concrete_rnn", asset_id),
        "Ridge": _load_concrete_csv("concrete_ridge", asset_id),
    }


def _load_concrete_csv(name: str, asset_id: str | None) -> dict[str, pd.DataFrame]:
    """Concrete export -> chart-ready history and projections (same shape as the timber loader)."""
    path = ROOT_DIR / "data" / "dashboard" / f"dashboard_data_{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Dashboard data file not found: {path}")

    frame = pd.read_csv(path)
    if asset_id is not None:
        frame = frame.loc[frame["asset_id"] == asset_id]
    frame = frame.rename(columns={"scenario": "condition"})

    current_year = date.today().year
    historical = frame.loc[frame["year"] <= current_year]
    projection = frame.loc[frame["year"] > current_year]
    if historical.empty or projection.empty:
        raise ValueError(f"Concrete CSV {name} needs rows on both sides of {current_year}.")

    historical = historical.groupby("year", as_index=False)["value"].mean()
    projection = projection.groupby(["year", "condition"], as_index=False)["value"].mean()
    return {"historical": historical, "projection": projection}


def _load_dashboard_csv(
    material: str, asset_id: str | None = None,
) -> dict[str, pd.DataFrame]:
    """Convert one shared dashboard CSV into chart-ready history and projections."""
    path = ROOT_DIR / "data" / "dashboard" / f"dashboard_data_{material}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Dashboard data file not found: {path}")

    frame = pd.read_csv(path)
    required = {"material", "asset_id", "scenario", "year", "value", "method"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Dashboard CSV is missing columns: {', '.join(missing)}")

    if asset_id is not None:
        frame = frame.loc[frame["asset_id"] == asset_id].copy()

    frame["condition"] = frame["scenario"].map(scenario_condition)
    current_year = date.today().year
    if (frame["method"] == "observed").any():
        historical = frame.loc[frame["method"] == "observed"]
        projection = frame.loc[frame["method"] != "observed"]
    else:
        historical = frame.loc[frame["year"] <= current_year]
        projection = frame.loc[frame["year"] > current_year]
    if historical.empty or projection.empty:
        raise ValueError(f"Dashboard CSV for {material} needs observed and projected rows.")

    historical = historical.groupby("year", as_index=False)["value"].mean()
    projection = projection.groupby(["year", "condition"], as_index=False)["value"].mean()
    return {"historical": historical, "projection": projection}


def material_forecast_chart(
    material: str,
    historical: pd.DataFrame,
    projection: pd.DataFrame,
    flags: list[dict[str, Any]] | None = None,
) -> go.Figure:
    """Build one material's historical series and three projection conditions."""
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=historical["year"],
            y=historical["value"],
            name="Historical",
            mode="lines",
            line={"color": "#334155", "width": 3},
        )
    )
    for condition in SCENARIO_LABELS:
        rows = projection.loc[projection["condition"] == condition]
        # Start each forecast trace at the final historical point so the
        # history-to-projection transition is visually continuous.
        boundary_year = int(historical["year"].max())
        boundary_value = float(historical.loc[historical["year"] == boundary_year, "value"].iloc[0])
        figure.add_trace(
            go.Scatter(
                x=pd.concat([pd.Series([boundary_year]), rows["year"]], ignore_index=True),
                y=pd.concat([pd.Series([boundary_value]), rows["value"]], ignore_index=True),
                name=condition,
                mode="lines",
                line={"color": SCENARIO_COLORS[condition], "width": 2, "dash": "dot"},
            )
        )
    boundary_year = int(historical["year"].max())
    figure.add_vline(x=boundary_year, line_dash="dash", line_color="#94A3B8")
    for flag in flags or []:
        purpose = flag["purpose"]
        value = float(flag["value"])
        figure.add_hline(
            y=value,
            line_dash="dash",
            line_color="#D97706" if purpose == "Maintenance" else "#7C3AED",
            annotation_text=f"{purpose}: {value:g} mm",
            annotation_position="top left",
        )
    figure.update_layout(
        title=material,
        height=520,
        autosize=True,
        margin={"l": 30, "r": 30, "t": 55, "b": 70},
        xaxis_title="Year",
        yaxis_title="Material loss (mm)",
        hovermode="x unified",
        legend={"orientation": "h", "y": -0.18},
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    figure.update_xaxes(
        showgrid=False,
        range=[int(historical["year"].min()), int(projection["year"].max())],
    )
    figure.update_yaxes(gridcolor="#E2E8F0")
    return figure


def comparison_data_from_payload(
    payload: Mapping[str, Any],
) -> dict[str, dict[str, pd.DataFrame]]:
    """Normalize a backend response into the comparison chart contract."""
    normalized: dict[str, dict[str, pd.DataFrame]] = {}
    for material in ("concrete", "timber"):
        material_payload = payload.get(material)
        if not isinstance(material_payload, Mapping):
            raise ValueError(f"Backend response is missing '{material}' data.")
        historical = pd.DataFrame(material_payload.get("historical", []))
        projection = pd.DataFrame(material_payload.get("projection", []))
        if not {"year", "value"}.issubset(historical.columns):
            raise ValueError(f"{material} historical data needs year and value columns.")
        if not {"year", "condition", "value"}.issubset(projection.columns):
            raise ValueError(
                f"{material} projection data needs year, condition, and value columns."
            )
        normalized[material.title()] = {
            "historical": historical,
            "projection": projection,
        }
    return normalized

