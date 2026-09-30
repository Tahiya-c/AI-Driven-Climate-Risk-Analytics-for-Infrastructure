"""Shared error-breakdown helpers used by the model comparison report."""

from __future__ import annotations

import pandas as pd

AGE_BUCKET_EDGES = [-10, 0, 10, 20, 30, 40, 50, 60, 70]


def add_predictions(
    frame: pd.DataFrame, index: pd.Index, predictions, target_column: str
) -> pd.DataFrame:
    """Attach model predictions and absolute error to the scored rows."""
    scored = frame.loc[index].copy()
    scored["prediction"] = predictions
    scored["abs_error"] = (scored["prediction"] - scored[target_column]).abs()
    return scored


def error_by_scenario(scored: pd.DataFrame) -> pd.DataFrame:
    return scored.groupby("scenario_label")["abs_error"].agg(["mean", "median", "max", "count"])


def error_by_pile(scored: pd.DataFrame) -> pd.DataFrame:
    return scored.groupby("asset_code")["abs_error"].agg(["mean", "median", "max", "count"])


def error_by_age(scored: pd.DataFrame, edges: list[int] = AGE_BUCKET_EDGES) -> pd.DataFrame:
    buckets = pd.cut(scored["age_years"], bins=edges)
    return scored.groupby(buckets, observed=True)["abs_error"].agg(["mean", "count"])
