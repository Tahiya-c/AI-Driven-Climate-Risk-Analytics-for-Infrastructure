"""Plotly chart builders for historical condition and forecasts."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go


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

