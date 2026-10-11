"""Every Supabase read and write the app makes, in one place.

Tables and columns are defined in docs/database_schema.sql. Functions take and return
plain dicts / DataFrames so the dashboard never builds queries itself.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from src.db.client import get_client

BUCKET = "imports"
REFERENCE_UPLOADER = "Reference data"  # uploaded_by of the seeded reference piles
_PAGE = 1000  # the Data API returns at most this many rows per request
_BATCH = 500


# ------------------------------------------------------------------------ helpers
def _clean(value: Any) -> Any:
    """JSON-safe scalar: numpy -> python, NaN -> None, whole floats -> int.

    Whole floats become ints because pandas turns an integer column with a blank into
    floats (2045.0), which Postgres rejects for an integer column; a double column
    accepts the int just the same.
    """
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        if math.isnan(value):
            return None
        return int(value) if value.is_integer() else value
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if value is pd.NA or value is pd.NaT:
        return None
    return value


def records(frame: pd.DataFrame) -> list[dict]:
    return [{k: _clean(v) for k, v in row.items()} for row in frame.to_dict("records")]


def _insert(table: str, rows: list[dict]) -> None:
    client = get_client()
    for start in range(0, len(rows), _BATCH):
        client.table(table).insert(rows[start:start + _BATCH]).execute()


def _select_all(query_factory) -> list[dict]:
    """Page through a query; ``query_factory()`` must return a fresh filtered query."""
    rows: list[dict] = []
    start = 0
    while True:
        page = query_factory().range(start, start + _PAGE - 1).execute().data
        rows += page
        if len(page) < _PAGE:
            return rows
        start += _PAGE


# ------------------------------------------------------------------------ imports
def create_import(
    material: str,
    file_name: str | None,
    uploaded_by: str | None,
    status: str,
    n_errors: int,
    n_warnings: int,
    climate_input: Any = None,
) -> str:
    row = {
        "material": material,
        "file_name": file_name,
        "uploaded_by": uploaded_by or None,
        "status": status,
        "n_errors": n_errors,
        "n_warnings": n_warnings,
        "climate_input": climate_input,
    }
    data = get_client().table("imports").insert(row).execute().data
    return data[0]["import_id"]


def set_import_status(import_id: str, status: str, storage_path: str | None = None) -> None:
    update: dict[str, Any] = {"status": status}
    if storage_path:
        update["storage_path"] = storage_path
    get_client().table("imports").update(update).eq("import_id", import_id).execute()


def upload_file(material: str, import_id: str, file_name: str, content: bytes) -> str:
    path = f"{material}/{import_id}/{file_name}"
    content_type = (
        "text/csv"
        if file_name.lower().endswith(".csv")
        else "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    get_client().storage.from_(BUCKET).upload(path, content, {"content-type": content_type})
    return path


def save_issues(import_id: str, issues: pd.DataFrame) -> None:
    if issues.empty:
        return
    rows = [
        {
            "import_id": import_id,
            "level": row["level"],
            "sheet": row["sheet"],
            "sheet_row": row["row"],
            "sheet_column": row["column"],
            "message": row["message"],
        }
        for row in records(issues)
    ]
    _insert("import_issues", rows)


def save_assets(material: str, import_id: str, assets: pd.DataFrame, columns: list[str]) -> None:
    frame = assets[[c for c in columns if c in assets.columns]].assign(import_id=import_id)
    _insert(f"{material}_assets", records(frame))


def save_observations(material: str, import_id: str, observations: pd.DataFrame) -> None:
    if observations is None or observations.empty:
        return
    _insert(f"{material}_observations", records(observations.assign(import_id=import_id)))


def save_run(
    import_id: str,
    model_version: str,
    method: str,
    pathway: str | None,
    anchored: bool,
    anchor: dict | None,
    points: pd.DataFrame,
    pile_results: pd.DataFrame | None = None,
    outcomes: pd.DataFrame | None = None,
) -> str:
    run = {
        "import_id": import_id,
        "model_version": model_version,
        "method": method,
        "pathway": pathway,
        "anchored": anchored,
        "anchor": anchor,
    }
    run_id = get_client().table("forecast_runs").insert(run).execute().data[0]["run_id"]
    cols = ["asset_id", "scenario", "year", "value", "unit", "metric", "method"]
    _insert("forecast_points", records(points[cols].assign(run_id=run_id)))
    if pile_results is not None and not pile_results.empty:
        _insert("pile_results", records(pile_results.assign(run_id=run_id)))
    if outcomes is not None and not outcomes.empty:
        _insert("scenario_outcomes", records(outcomes.assign(run_id=run_id)))
    return run_id


def delete_import(import_id: str) -> None:
    """Delete an import and everything under it (rows cascade), plus its stored file."""
    client = get_client()
    found = client.table("imports").select("storage_path").eq("import_id", import_id).execute().data
    if found and found[0].get("storage_path"):
        try:
            client.storage.from_(BUCKET).remove([found[0]["storage_path"]])
        except Exception:  # the rows matter more than the file; don't block on storage
            pass
    client.table("imports").delete().eq("import_id", import_id).execute()


def delete_pile(material: str, import_id: str, asset_id: str) -> bool:
    """Remove one pile from an import: its forecasts, results, flags, inspections and
    properties. If it was the import's last pile, the whole import (and its stored
    file) is deleted too. Returns True when the import was deleted."""
    client = get_client()
    run_ids = [
        row["run_id"]
        for row in client.table("forecast_runs").select("run_id")
        .eq("import_id", import_id).execute().data
    ]
    if run_ids:
        for table in ("forecast_points", "pile_results", "scenario_outcomes"):
            client.table(table).delete().in_("run_id", run_ids).eq("asset_id", asset_id).execute()
    client.table("pile_flags").delete().eq("material", material).eq(
        "import_id", import_id).eq("asset_id", asset_id).execute()
    # Deleting the asset row cascades to its inspections.
    client.table(f"{material}_assets").delete().eq("import_id", import_id).eq(
        "asset_id", asset_id).execute()
    remaining = (
        client.table(f"{material}_assets").select("asset_id")
        .eq("import_id", import_id).limit(1).execute().data
    )
    if not remaining:
        delete_import(import_id)
        return True
    return False


def delete_reference_imports(material: str) -> int:
    client = get_client()
    found = (
        client.table("imports").select("import_id")
        .eq("material", material).eq("uploaded_by", REFERENCE_UPLOADER).execute().data
    )
    for row in found:
        client.table("imports").delete().eq("import_id", row["import_id"]).execute()
    return len(found)


# ------------------------------------------------------------------------ reads
def list_piles(material: str) -> pd.DataFrame:
    """Every pile with a saved forecast: reference piles first, then uploads, newest first.

    Columns: import_id, asset_id, file_name, uploaded_by, created_at, is_reference.
    """
    client = get_client()
    imports = pd.DataFrame(
        client.table("imports")
        .select("import_id, file_name, uploaded_by, created_at")
        .eq("material", material).eq("status", "forecast")
        .order("created_at", desc=True).execute().data
    )
    columns = ["import_id", "asset_id", "file_name", "uploaded_by", "created_at", "is_reference"]
    if imports.empty:
        return pd.DataFrame(columns=columns)
    assets = pd.DataFrame(
        _select_all(
            lambda: client.table(f"{material}_assets")
            .select("import_id, asset_id").in_("import_id", imports["import_id"].tolist())
        )
    )
    if assets.empty:
        return pd.DataFrame(columns=columns)
    piles = assets.merge(imports, on="import_id")
    piles["is_reference"] = piles["uploaded_by"] == REFERENCE_UPLOADER
    piles = piles.sort_values(
        ["is_reference", "created_at", "asset_id"], ascending=[False, False, True]
    )
    return piles[columns].reset_index(drop=True)


def list_imports(material: str | None = None) -> pd.DataFrame:
    query = get_client().table("imports").select(
        "import_id, material, file_name, uploaded_by, status, n_errors, n_warnings, created_at"
    )
    if material:
        query = query.eq("material", material)
    return pd.DataFrame(query.order("created_at", desc=True).execute().data)


def runs_for_import(import_id: str) -> pd.DataFrame:
    return pd.DataFrame(
        get_client().table("forecast_runs")
        .select("run_id, model_version, method, pathway, anchored, anchor, created_at")
        .eq("import_id", import_id).execute().data
    )


def points_for_pile(run_ids: list[str], asset_id: str) -> pd.DataFrame:
    client = get_client()
    rows = _select_all(
        lambda: client.table("forecast_points")
        .select("run_id, asset_id, scenario, year, value, unit, metric, method")
        .in_("run_id", run_ids).eq("asset_id", asset_id)
        .order("run_id").order("scenario").order("method").order("year")
    )
    return pd.DataFrame(rows)


def results_for_pile(run_ids: list[str], asset_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    client = get_client()
    results = client.table("pile_results").select("*").in_("run_id", run_ids).eq(
        "asset_id", asset_id).execute().data
    outcomes = client.table("scenario_outcomes").select("*").in_("run_id", run_ids).eq(
        "asset_id", asset_id).execute().data
    return pd.DataFrame(results), pd.DataFrame(outcomes)


def model_scores(material: str) -> pd.DataFrame:
    client = get_client()
    versions = client.table("model_versions").select("model_version, model_kind").eq(
        "material", material).eq("is_active", True).execute().data
    if not versions:
        return pd.DataFrame()
    kinds = {v["model_version"]: v["model_kind"] for v in versions}
    scores = pd.DataFrame(
        client.table("model_evaluations")
        .select("model_version, evaluation, asset_id, n_rows, r2, rmse_mm, mae_mm, notes")
        .in_("model_version", list(kinds)).execute().data
    )
    if not scores.empty:
        scores.insert(0, "model", scores["model_version"].map(kinds).str.upper())
    return scores


# ------------------------------------------------------------------------ flags
def list_flags(material: str, asset_id: str, import_id: str) -> list[dict]:
    rows = (
        get_client().table("pile_flags").select("id, purpose, value_mm")
        .eq("material", material).eq("asset_id", asset_id).eq("import_id", import_id)
        .order("created_at").execute().data
    )
    return [{"id": r["id"], "purpose": r["purpose"], "value": r["value_mm"]} for r in rows]


def add_flag(material: str, asset_id: str, import_id: str, purpose: str, value_mm: float) -> None:
    row = {
        "material": material, "asset_id": asset_id, "import_id": import_id,
        "purpose": purpose, "value_mm": float(value_mm),
    }
    get_client().table("pile_flags").upsert(
        row, on_conflict="material,asset_id,import_id,purpose,value_mm", ignore_duplicates=True
    ).execute()


def delete_flag(flag_id: int) -> None:
    get_client().table("pile_flags").delete().eq("id", flag_id).execute()
