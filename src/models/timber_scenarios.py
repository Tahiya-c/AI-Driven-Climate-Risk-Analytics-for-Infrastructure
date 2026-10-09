"""Per-pile scenario trajectories and replacement-year prediction.

For a chosen pile, this module splices the shared "Baseline" history
(remaining diameter with no further climate forcing) with each climate
scenario's own future rows, runs the spliced trajectory through the
trained surrogate model, and finds when each scenario first crosses the
pile's replacement threshold.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from src.data.timber_data_prep import remaining_diameter_floor
from src.features.timber_features import FEATURE_COLUMNS

SCENARIOS = ["Hotter & Drier", "Most Likely", "Colder & Wetter"]
BASELINE_LABEL = "Baseline"


@dataclass
class ScenarioTrajectory:
    """A spliced history+projection line for one scenario, one pile."""

    scenario: str
    years: np.ndarray
    remaining_diameter_mm: np.ndarray  # floored at 0, monotonic non-increasing
    replacement_year: float | None


def default_split_year(frame: pd.DataFrame) -> int:
    """Year the chart switches from shared history to diverging scenarios.

    Defaults to the current calendar year, clamped into the data's
    available year range. Baseline stands in for "no further climate
    forcing" and is used as the common historical line up to today; each
    scenario only meaningfully diverges from that point forward.
    """
    year_min, year_max = int(frame["year"].min()), int(frame["year"].max())
    return min(max(date.today().year, year_min), year_max)


def _aggregate_by_year(frame: pd.DataFrame) -> pd.DataFrame:
    """Collapse duplicate emissions-pathway rows (A1B/A1FI) to one per year.

    Each non-baseline scenario_label carries two emissions_pathway rows
    per pile-year (A1B, A1FI); a single display/prediction line per
    scenario is produced by averaging their feature values.
    """
    return frame.groupby("year", as_index=False)[FEATURE_COLUMNS].mean()


def build_pile_trajectories(
    frame: pd.DataFrame,
    model: Any,
    asset_code: str,
    split_year: int | None = None,
) -> tuple[dict[str, ScenarioTrajectory], int]:
    """Predict, per scenario, the remaining-diameter trajectory for one pile."""
    pile_frame = frame[frame["asset_code"] == asset_code]
    if pile_frame.empty:
        raise ValueError(f"No rows found for asset_code={asset_code!r}")

    if split_year is None:
        split_year = default_split_year(pile_frame)

    threshold = float(pile_frame["replacement_diameter_mm"].iloc[0])

    history = pile_frame[
        (pile_frame["scenario_label"] == BASELINE_LABEL) & (pile_frame["year"] <= split_year)
    ]
    history = _aggregate_by_year(history).sort_values("year")
    history_prediction = model.predict(history[FEATURE_COLUMNS])

    trajectories: dict[str, ScenarioTrajectory] = {}
    for scenario in SCENARIOS:
        future = pile_frame[
            (pile_frame["scenario_label"] == scenario) & (pile_frame["year"] >= split_year)
        ]
        future = _aggregate_by_year(future).sort_values("year")
        future_prediction = model.predict(future[FEATURE_COLUMNS])

        years = np.concatenate([history["year"].to_numpy(), future["year"].to_numpy()])
        values = np.concatenate([history_prediction, future_prediction])

        # Keep the first occurrence of the shared split-year point (from
        # history) so the historical and projected segments join smoothly.
        years, first_occurrence = np.unique(years, return_index=True)
        values = values[first_occurrence]

        # Deterioration is physically irreversible: enforce a non-increasing
        # line, since the MLP regressor itself has no such constraint.
        values = np.minimum.accumulate(values)
        values = remaining_diameter_floor(pd.Series(values)).to_numpy()

        replacement_year = find_replacement_year(years, values, threshold)
        trajectories[scenario] = ScenarioTrajectory(scenario, years, values, replacement_year)

    return trajectories, split_year


def find_replacement_year(
    years: np.ndarray, values: np.ndarray, threshold: float
) -> float | None:
    """First year (linearly interpolated) the trajectory crosses the threshold."""
    below_threshold = values <= threshold
    if not below_threshold.any():
        return None

    first_index = int(np.argmax(below_threshold))
    if first_index == 0:
        return float(years[0])

    year_before, year_after = years[first_index - 1], years[first_index]
    value_before, value_after = values[first_index - 1], values[first_index]
    if value_before == value_after:
        return float(year_after)

    fraction = (value_before - threshold) / (value_before - value_after)
    return float(year_before + fraction * (year_after - year_before))


def summarize_pile(
    frame: pd.DataFrame, model: Any, asset_code: str, split_year: int | None = None
) -> pd.DataFrame:
    """Replacement-year summary table for one pile, one row per scenario."""
    trajectories, _ = build_pile_trajectories(frame, model, asset_code, split_year)
    rows = []
    for scenario in SCENARIOS:
        trajectory = trajectories[scenario]
        replacement_label = (
            f"{trajectory.replacement_year:.0f}"
            if trajectory.replacement_year is not None
            else "Not reached by 2070"
        )
        rows.append(
            {
                "Scenario": scenario,
                "Predicted replacement year": replacement_label,
                "Remaining diameter at 2070 (mm)": round(
                    float(trajectory.remaining_diameter_mm[-1]), 1
                ),
            }
        )
    return pd.DataFrame(rows)
