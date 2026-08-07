"""Reusable feature preparation functions."""

import pandas as pd


def select_model_features(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Select model inputs and fail clearly when a column is unavailable."""
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    return frame.loc[:, columns].copy()

