"""Plotly chart builders for historical condition and forecasts."""

from collections.abc import Mapping
from datetime import date
from typing import Any

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.config import MODEL_DIR

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


def demo_material_comparison_data() -> dict[str, dict[str, pd.DataFrame]]:
    """Return concrete demo data and timber predictions from the linear model."""
    current_year = date.today().year
    historical_years = np.arange(2000, current_year + 1)
    projection_years = np.arange(current_year + 1, current_year + 22)
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
    data = {
        "Concrete": {
            "historical": pd.DataFrame({"year": historical_years, "value": concrete_values}),
            "projection": _scenario_frame(projection_years, concrete_projection),
        }
    }
    data["Timber"] = _timber_model_data(current_year, projection_years)
    return data


def _scenario_frame(years: np.ndarray, values: dict[str, np.ndarray]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"year": year, "condition": condition, "value": scenario_values[index]}
            for index, year in enumerate(years)
            for condition, scenario_values in values.items()
        ]
    )


def _timber_model_data(
    current_year: int,
    projection_years: np.ndarray,
) -> dict[str, pd.DataFrame]:
    """Generate the timber comparison series using the saved linear model."""
    data_path = MODEL_DIR.parent / "data" / "processed" / "timber_kembla_v6_3_consolidated.csv"
    artifact_path = MODEL_DIR / "timber_replacement_linear.joblib"
    if not data_path.exists() or not artifact_path.exists():
        return _fallback_timber_data(current_year, projection_years)

    artifact = joblib.load(artifact_path)
    frame = pd.read_csv(data_path)
    scenario_map = {"Baseline": "Colder & Wetter"}
    frame["scenario_label"] = frame["scenario_label"].replace(scenario_map)
    frame = frame.loc[frame["scenario_label"].isin(SCENARIO_LABELS)].copy()
    frame["value"] = artifact["model"].predict(frame[artifact["features"]])
    historical = (
        frame.loc[frame["year"].between(2000, current_year)]
        .groupby("year", as_index=False)["value"]
        .mean()
    )
    future = frame.loc[frame["year"].isin(projection_years)].copy()
    projection = future[["year", "scenario_label", "value"]].rename(
        columns={"scenario_label": "condition"}
    )
    projection = projection.groupby(["year", "condition"], as_index=False)["value"].mean()
    return {"historical": historical, "projection": projection}


def _fallback_timber_data(
    current_year: int,
    projection_years: np.ndarray,
) -> dict[str, pd.DataFrame]:
    """Keep the page usable when the local model artifact is unavailable."""
    historical_years = np.arange(2000, current_year + 1)
    historical_values = 28.0 - np.linspace(0, 8, len(historical_years))
    values = {
        condition: historical_values[-1] - decline * np.linspace(0, 10, len(projection_years))
        for condition, decline in (
            ("Most Likely", 0.8),
            ("Hotter & Drier", 1.2),
            ("Colder & Wetter", 0.5),
        )
    }
    return {
        "historical": pd.DataFrame({"year": historical_years, "value": historical_values}),
        "projection": _scenario_frame(projection_years, values),
    }


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
        figure.add_trace(
            go.Scatter(
                x=rows["year"],
                y=rows["value"],
                name=condition,
                mode="lines",
                line={"color": SCENARIO_COLORS[condition], "width": 2, "dash": "dot"},
            )
        )
    boundary_year = int(historical["year"].max())
    figure.add_vline(x=boundary_year, line_dash="dash", line_color="#94A3B8")
    figure.update_layout(
        title=material,
        height=390,
        margin={"l": 20, "r": 20, "t": 55, "b": 20},
        xaxis_title="Year",
        yaxis_title="Replacement time (years)",
        hovermode="x unified",
        legend={"orientation": "h", "y": -0.2},
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    figure.update_xaxes(showgrid=False)
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

