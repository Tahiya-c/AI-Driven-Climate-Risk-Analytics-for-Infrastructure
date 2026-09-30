"""Alternate regression targets for the timber surrogate model."""

from __future__ import annotations

import pandas as pd

from src.timber.features import TARGET_COLUMN as DIAMETER_TARGET

DISTANCE_TARGET = "distance_to_threshold_mm"
CLIPPED_DIAMETER_COLUMN = "remaining_diameter_clipped_mm"
RESIDUAL_TARGET = "residual_correction_mm"


def add_distance_to_threshold(frame: pd.DataFrame) -> pd.DataFrame:
    """Add remaining_diameter_mm - replacement_diameter_mm as an alternate target.

    Framing the target as "how far above/below the replacement threshold"
    rather than raw diameter puts every pile on the same decision-relevant
    scale regardless of its own initial/threshold diameter.
    """
    frame = frame.copy()
    frame[DISTANCE_TARGET] = frame[DIAMETER_TARGET] - frame["replacement_diameter_mm"]
    return frame


def add_residual_correction_target(frame: pd.DataFrame) -> pd.DataFrame:
    """Add the correction needed to bring remaining_diameter_mm into range.

    ``remaining_diameter_clipped_mm`` floors the deterministic model's raw
    (sometimes physically invalid, negative) output at 0.
    ``residual_correction_mm`` is what a residual-correction model is
    trained to predict: 0 for every row already in range, and a positive
    correction for rows where the deterministic model has over-extrapolated
    past failure.
    """
    frame = frame.copy()
    frame[CLIPPED_DIAMETER_COLUMN] = frame[DIAMETER_TARGET].clip(lower=0)
    frame[RESIDUAL_TARGET] = frame[CLIPPED_DIAMETER_COLUMN] - frame[DIAMETER_TARGET]
    return frame
