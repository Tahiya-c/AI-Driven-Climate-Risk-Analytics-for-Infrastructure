"""Prediction interface placeholder for the trained deterioration model."""

from typing import Any

import pandas as pd


def predict_deterioration(model: Any, features: pd.DataFrame) -> pd.Series:
    """Return model predictions with the input index preserved."""
    return pd.Series(model.predict(features), index=features.index, name="prediction")

