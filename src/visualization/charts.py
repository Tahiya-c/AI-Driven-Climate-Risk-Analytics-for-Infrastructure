"""Plotly chart builders for historical condition and forecasts."""

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.config import ROOT_DIR

SCENARIO_LABELS = ("Most Likely", "Hotter & Drier", "Colder & Wetter")
SCENARIO_COLORS = {
    "Most Likely": "#2563EB",
    "Hotter & Drier": "#DC2626",
    "Colder & Wetter": "#0891B2",
}


def demo_condition_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create deterministic demonstration data until a trained model is connected."""
    rng = np.random.default_rng(7)
    historical_time = np.arange(2005, 2025)
    historical_value = 24 + np.cumsum(rng.normal(1.15, 1.2, len(historical_time)))
    future_time = np.arange(2024, 2036)
    base = historical_value[-1]
    scenarios = pd.DataFrame(
        {
            "Year": future_time,
            "Low deterioration": base + np.linspace(0, 18, len(future_time)),
            "Expected deterioration": base + np.linspace(0, 9, len(future_time)),
            "High deterioration": base - np.linspace(0, 10, len(future_time)),
        }
    )
    historical = pd.DataFrame({"Year": historical_time, "Condition index": historical_value})
    return historical, scenarios


def condition_forecast_chart() -> go.Figure:
    """Build the main history and forecast scenario chart."""
    historical, forecast = demo_condition_data()
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=historical["Year"], y=historical["Condition index"],
            name="Historical data", mode="lines", line={"color": "#0F766E", "width": 3},
        )
    )
    colors = ["#2563EB", "#D97706", "#DC2626"]
    for color, column in zip(colors, forecast.columns[1:], strict=True):
        figure.add_trace(
            go.Scatter(
                x=forecast["Year"], y=forecast[column], name=column, mode="lines",
                line={"color": color, "width": 2, "dash": "dot"},
            )
        )
    figure.add_vline(x=2024, line_dash="dash", line_color="#64748B")
    figure.add_annotation(x=2014, y=1.08, yref="paper", text="Historical", showarrow=False)
    figure.add_annotation(x=2029, y=1.08, yref="paper", text="Future prediction", showarrow=False)
    figure.update_layout(
        height=440, margin={"l": 25, "r": 25, "t": 50, "b": 20},
        xaxis_title="Year", yaxis_title="Condition index", hovermode="x unified",
        legend={"orientation": "h", "y": -0.2}, paper_bgcolor="white", plot_bgcolor="white",
    )
    figure.update_xaxes(showgrid=False)
    figure.update_yaxes(gridcolor="#E2E8F0")
    return figure


def dashboard_material_comparison_data() -> dict[str, dict[str, pd.DataFrame]]:
    """Load dashboard CSV data and retain a concrete fallback until its export exists."""
    timber = _load_dashboard_csv("timber")
    current_year = int(timber["historical"]["year"].max())
    historical_years = np.arange(2000, current_year + 1)
    projection_years = np.arange(
        int(timber["projection"]["year"].min()),
        int(timber["projection"]["year"].max()) + 1,
    )
    years_from_start = historical_years - historical_years[0]
    rng = np.random.default_rng(12)

    concrete_values = np.clip(
        86.0
        - 0.42 * years_from_start
        + np.cumsum(rng.normal(0, 1.1, len(historical_years))),
        0,
        100,
    )
    concrete_last_value = float(concrete_values[-1])
    concrete_projection = {
        "Most Likely": concrete_last_value - np.linspace(0, 5.0, len(projection_years)),
        "Hotter & Drier": concrete_last_value - np.linspace(0, 12.0, len(projection_years)),
        "Colder & Wetter": concrete_last_value - np.linspace(0, 2.5, len(projection_years)),
    }
    data: dict[str, dict[str, pd.DataFrame]] = {
        "Concrete": {
            "historical": pd.DataFrame({"year": historical_years, "value": concrete_values}),
            "projection": _scenario_frame(projection_years, concrete_projection),
        }
    }
    data["Timber"] = timber
    return data


def _load_dashboard_csv(material: str) -> dict[str, pd.DataFrame]:
    """Convert one shared dashboard CSV into chart-ready history and projections."""
    path = ROOT_DIR / "data" / "dashboard" / f"dashboard_data_{material}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Dashboard data file not found: {path}")

    frame = pd.read_csv(path)
    required = {"material", "asset_id", "scenario", "year", "value", "method"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Dashboard CSV is missing columns: {', '.join(missing)}")

    scenario_map = {
        "CSIRO Mk3.5": "Hotter & Drier",
        "MRI-CGCM2.3.2": "Most Likely",
        "MIROC-Medres": "Colder & Wetter",
    }
    frame["condition"] = frame["scenario"].map(
        lambda value: next(
            (label for model, label in scenario_map.items() if model in value),
            "Most Likely",
        )
    )
    historical = frame.loc[frame["method"] == "observed"]
    projection = frame.loc[frame["method"] != "observed"]
    if historical.empty or projection.empty:
        raise ValueError(f"Dashboard CSV for {material} needs observed and projected rows.")

    historical = historical.groupby("year", as_index=False)["value"].mean()
    projection = projection.groupby(["year", "condition"], as_index=False)["value"].mean()
    return {"historical": historical, "projection": projection}


def _scenario_frame(years: np.ndarray, values: dict[str, np.ndarray]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"year": year, "condition": condition, "value": scenario_values[index]}
            for index, year in enumerate(years)
            for condition, scenario_values in values.items()
        ]
    )


def material_forecast_chart(
    material: str,
    historical: pd.DataFrame,
    projection: pd.DataFrame,
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

