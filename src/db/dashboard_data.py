"""Chart-ready data for the dashboard, read from Supabase.

Same shape as the old CSV loaders in src/visualization/charts.py:
``{model_label: {"historical": df(year, value), "projection": df(year, condition, value)}}``.
"""

from __future__ import annotations

import pandas as pd

from src.db import repository as repo
from src.db.import_pipeline import MODELS
from src.visualization.charts import chart_frames, scenario_condition


def pile_options(material: str) -> pd.DataFrame:
    """Every pile with a forecast, plus a ``label`` for the pile dropdown."""
    piles = repo.list_piles(material)
    if piles.empty:
        return piles.assign(label=pd.Series(dtype=str))

    def label(row) -> str:
        if row["is_reference"]:
            return f"{row['asset_id']} (reference)"
        when = pd.Timestamp(row["created_at"]).strftime("%d %b %Y %H:%M")
        who = f", {row['uploaded_by']}" if row["uploaded_by"] else ""
        return f"{row['asset_id']} (import {when}{who})"

    piles = piles.assign(label=piles.apply(label, axis=1))
    # Two uploads of the same pile in the same minute would share a label.
    repeat = piles.groupby("label").cumcount()
    piles.loc[repeat > 0, "label"] += " #" + (repeat[repeat > 0] + 1).astype(str)
    return piles


def _model_labels(material: str) -> dict[str, str]:
    return {kind: label for label, kind in MODELS[material]}


def pile_model_data(
    material: str, import_id: str, asset_id: str
) -> dict[str, dict[str, pd.DataFrame]]:
    """Both models' history and forecasts for one pile, keyed by model label."""
    runs = repo.runs_for_import(import_id)
    if runs.empty:
        raise ValueError("No forecasts saved for this import.")
    points = repo.points_for_pile(runs["run_id"].tolist(), asset_id)
    if points.empty:
        raise ValueError(f"No forecast points saved for {asset_id}.")
    data = {}
    for label, kind in MODELS[material]:
        run_ids = runs.loc[runs["method"] == kind, "run_id"]
        if run_ids.empty:
            raise ValueError(f"No {label} forecast saved for this import.")
        data[label] = chart_frames(points.loc[points["run_id"] == run_ids.iloc[0]])
    return data


def pile_summary(material: str, import_id: str, asset_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(per-model inspection summary, per-model and scenario outcomes) for one pile."""
    runs = repo.runs_for_import(import_id)
    if runs.empty:
        return pd.DataFrame(), pd.DataFrame()
    model_of = runs.set_index("run_id")["method"].map(_model_labels(material))
    results, outcomes = repo.results_for_pile(runs["run_id"].tolist(), asset_id)
    if not results.empty:
        results.insert(0, "model", results["run_id"].map(model_of))
        results = results.drop(columns=["run_id"])
    if not outcomes.empty:
        outcomes.insert(0, "model", outcomes["run_id"].map(model_of))
        outcomes.insert(1, "condition", outcomes["scenario"].map(scenario_condition))
        outcomes = outcomes.drop(columns=["run_id"]).sort_values(["model", "condition"])
    return results, outcomes.reset_index(drop=True)
