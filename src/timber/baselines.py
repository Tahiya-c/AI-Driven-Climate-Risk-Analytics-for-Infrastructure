"""Trivial, unfitted baselines to sanity-check the trained models against."""

from __future__ import annotations

import pandas as pd


def naive_diameter_baseline(frame: pd.DataFrame) -> pd.Series:
    """``initial_diameter_mm - mean_deterioration_mm``, with no fitting at all.

    This deliberately uses ``mean_deterioration_mm``, one of the
    deterministic model's own OUTPUT columns (see ``LEAKY_COLUMNS`` in
    ``src/timber/features.py``). That is fine here specifically because
    this is a fixed arithmetic reference point, not a trained-on feature:
    it exists to answer "is the trained model beating simple arithmetic
    on the deterministic model's own numbers, at all?" It must never be
    passed into ``build_feature_matrix`` or used as a model input.
    """
    return frame["initial_diameter_mm"] - frame["mean_deterioration_mm"]
