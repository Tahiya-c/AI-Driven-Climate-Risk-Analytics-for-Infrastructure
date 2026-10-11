"""Upload -> validate -> forecast with both models -> save to Supabase.

The dashboard's Import page calls these in order. Nothing here is Streamlit-specific,
so the same steps can be run from a script or a test.

    timber   : GBM and MLP (artifacts/timber_transfer_*.joblib), anchored to inspections
    concrete : RNN and Ridge (artifacts/concrete_*.pt / .joblib)

Both materials forecast the A1FI pathway's three climate scenarios, the same as the
reference piles on the dashboard.
"""

from __future__ import annotations

import json
import warnings
import zipfile
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path

import pandas as pd

from src.data import concrete_import, timber_import
from src.data.import_schema import ImportResult
from src.db import repository as repo

PATHWAY = "A1FI"
MODELS = {
    "timber": (("GBM", "gbm"), ("MLP", "mlp")),
    "concrete": (("RNN", "rnn"), ("Ridge", "ridge")),
}
TIMBER_ASSET_COLUMNS = [
    "asset_id", "description", "timber_type", "year_installed", "zone",
    "initial_diameter_mm", "sapwood_layer_mm", "replacement_diameter_mm",
    "k_heartwood", "k_sapwood", "k_shelter", "k_protect", "k_contact", "k_knot",
]
CONCRETE_ASSET_COLUMNS = ["asset_id", "zone", "wc_ratio", "fc_mpa"]
# A single concrete CSV: one row per inspection, the pile's properties on every row.
SINGLE_CONCRETE_CSV = (*CONCRETE_ASSET_COLUMNS, "year", "carb_depth_mm")


@dataclass
class ModelForecast:
    label: str  # shown on the chart, e.g. "GBM"
    method: str  # gbm | mlp | rnn | ridge
    model_version: str
    points: pd.DataFrame  # dashboard schema
    anchored: bool = False
    anchor: dict | None = None
    pile_results: pd.DataFrame = field(default_factory=pd.DataFrame)
    outcomes: pd.DataFrame = field(default_factory=pd.DataFrame)


def _module(material: str):
    return timber_import if material == "timber" else concrete_import


# ------------------------------------------------------------------------ step 1: read
def required_features(material: str) -> pd.DataFrame:
    """Every column the engineer must (or may) supply, from the import definitions."""
    module = _module(material)
    rows = []
    for sheet, fields in module.SHEETS.items():
        for f in fields:
            if f.allowed:
                limits = ", ".join(f.allowed)
            elif f.minimum is not None or f.maximum is not None:
                limits = " to ".join(str(v) for v in (f.minimum, f.maximum) if v is not None)
            else:
                limits = ""
            rows.append(
                {
                    "sheet": sheet + (" (optional)" if sheet in module.OPTIONAL_SHEETS else ""),
                    "column": f.name,
                    "required": "yes" if f.required else "no",
                    "unit": f.unit,
                    "allowed values / range": limits,
                    "description": f.description,
                }
            )
    return pd.DataFrame(rows)


def template_download(material: str) -> tuple[str, bytes, str]:
    """(file name, content, mime type) of the CSV template the Import page offers.

    Concrete: one CSV, one row per inspection (``SINGLE_CONCRETE_CSV``).
    Timber: its 4 sheets as 4 CSVs in a zip; unzip and upload all 4 together.
    Both are filled with the skeleton's example pile, to be replaced.
    """
    skeleton = _module(material).SKELETON_DIR
    if material == "concrete":
        assets = pd.read_csv(skeleton / "concrete_assets.example.csv")
        observations = pd.read_csv(skeleton / "concrete_observations.example.csv")
        frame = observations.merge(assets, on="asset_id")[list(SINGLE_CONCRETE_CSV)]
        return "concrete_import_template.csv", frame.to_csv(index=False).encode(), "text/csv"
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for sheet in timber_import.SHEETS:
            archive.write(skeleton / f"{sheet}.example.csv", f"{sheet}.csv")
    return "timber_import_template_csvs.zip", buffer.getvalue(), "application/zip"


def read_upload(
    material: str, files: list[tuple[str, bytes]]
) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """One .xlsx (sheets named as in the template), one .csv per sheet named
    ``<sheet>.csv``, or (concrete only) a single .csv with ``SINGLE_CONCRETE_CSV``'s
    columns. Returns the tables and notes about files that were ignored."""
    module = _module(material)
    xlsx = [(name, data) for name, data in files if name.lower().endswith(".xlsx")]
    if xlsx:
        notes = [f"Only the first .xlsx was read ({xlsx[0][0]})."] if len(files) > 1 else []
        return module.read_import(BytesIO(xlsx[0][1])), notes
    tables, notes = {}, []
    if material == "concrete" and len(files) == 1:
        frame = pd.read_csv(BytesIO(files[0][1]))
        if set(SINGLE_CONCRETE_CSV) <= set(frame.columns):
            return split_concrete_csv(frame), notes
    for name, data in files:
        sheet = Path(name).stem.removesuffix(".example")
        if sheet in module.SHEETS:
            tables[sheet] = pd.read_csv(BytesIO(data))
        else:
            expected = ", ".join(f"{s}.csv" for s in module.SHEETS)
            notes.append(f"Ignored {name}: expected one of {expected}.")
    return tables, notes


def split_concrete_csv(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """One concrete CSV (one row per inspection, the pile's properties repeated on
    each row) -> the concrete_assets and concrete_observations sheets. A pile whose
    properties differ between rows appears twice in concrete_assets, which the
    validator reports as a duplicate asset_id."""
    assets = frame[list(CONCRETE_ASSET_COLUMNS)].drop_duplicates().reset_index(drop=True)
    observations = frame[["asset_id", "year", "carb_depth_mm"]].dropna(how="all")
    return {"concrete_assets": assets, "concrete_observations": observations}


def validate(material: str, tables: dict[str, pd.DataFrame]) -> ImportResult:
    return _module(material).validate_import(tables)


MAX_PILE_NAME = 60


def check_pile_names(names: dict[str, str]) -> list[str]:
    """Problems with new pile names ({name in file: new name}); empty if all fine."""
    problems = []
    cleaned = {old: str(new or "").strip() for old, new in names.items()}
    for old, new in cleaned.items():
        if not new:
            problems.append(f"{old}: the name can't be blank.")
        elif len(new) > MAX_PILE_NAME:
            problems.append(f"{old}: the name is longer than {MAX_PILE_NAME} characters.")
    used = [new for new in cleaned.values() if new]
    for name in sorted({n for n in used if used.count(n) > 1}):
        problems.append(f"Two piles are both named {name!r}; each pile needs its own name.")
    return problems


def rename_piles(result: ImportResult, names: dict[str, str]) -> ImportResult:
    """A copy of a valid import with piles renamed ({name in file: new name}) in every
    sheet that has an asset_id. Names left unchanged are kept."""
    mapping = {old: str(new).strip() for old, new in names.items() if str(new).strip() != old}
    if not mapping:
        return result
    tables = {
        sheet: frame.assign(asset_id=frame["asset_id"].replace(mapping))
        if "asset_id" in frame
        else frame
        for sheet, frame in result.tables.items()
    }
    return ImportResult(tables, list(result.issues))


# ------------------------------------------------------------------------ step 2: forecast
def _timber_forecasts(result: ImportResult) -> list[ModelForecast]:
    from src.models.timber_forecast import forecast_pile, load_model_metadata

    model_tables = timber_import.to_model_tables(result)
    assets = result.tables["timber_assets"].set_index("asset_id")
    observations = result.tables.get("timber_observations")
    out = []
    for label, kind in MODELS["timber"]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            points = forecast_pile(**model_tables, model_kind=kind, pathway=PATHWAY, anchor=True)
        anchor = points.attrs.get("anchor")
        forecast = ModelForecast(
            label=label,
            method=kind,
            model_version=load_model_metadata(kind).get("model_version", f"timber_transfer_{kind}"),
            points=points,
            anchored=anchor is not None,
            anchor=anchor,
        )
        forecast.pile_results = _pile_results(
            observations, "inspection_year", assets.index, anchor
        )
        forecast.outcomes = _outcomes(points, assets)
        out.append(forecast)
    return out


def _concrete_forecasts(result: ImportResult) -> list[ModelForecast]:
    from src.models import concrete_saved

    series = concrete_import.to_model_series(result)
    assets = result.tables["concrete_assets"].set_index("asset_id")
    observations = result.tables.get("concrete_observations")
    out = []
    for label, kind in MODELS["concrete"]:
        points = concrete_saved.forecast_imported(series, kind)
        forecast = ModelForecast(
            label=label,
            method=kind,
            model_version=concrete_saved.load_metadata(kind).get(
                "model_version", concrete_saved.MODEL_VERSIONS[kind]
            ),
            points=points,
        )
        forecast.pile_results = _pile_results(observations, "year", assets.index, None)
        forecast.outcomes = _outcomes(points, None)
        out.append(forecast)
    return out


def forecast(material: str, result: ImportResult) -> list[ModelForecast]:
    """Both models' forecasts for every pile in a valid import (no database writes)."""
    if not result.ok:
        raise ValueError("The file has errors; fix them before forecasting.")
    return _timber_forecasts(result) if material == "timber" else _concrete_forecasts(result)


def _pile_results(observations, year_column, asset_ids, anchor) -> pd.DataFrame:
    offsets = (anchor or {}).get("offsets_mm", {})
    large = {item["series"] for item in (anchor or {}).get("large_offsets", [])}
    rows = []
    for asset_id in asset_ids:
        years = (
            observations.loc[observations["asset_id"] == asset_id, year_column]
            if observations is not None and not observations.empty
            else pd.Series(dtype=float)
        )
        rows.append(
            {
                "asset_id": asset_id,
                "n_inspections": int(len(years)),
                "first_inspection": int(years.min()) if len(years) else None,
                "last_inspection": int(years.max()) if len(years) else None,
                "anchor_offset_mm": offsets.get(asset_id) if anchor else None,
                "large_offset": asset_id in large,
            }
        )
    return pd.DataFrame(rows)


def _outcomes(points: pd.DataFrame, timber_assets: pd.DataFrame | None) -> pd.DataFrame:
    """Per pile and scenario: the 2050 value, the last forecast year, and (timber) the
    year the remaining diameter first reaches the replacement diameter."""
    forecast = points.loc[points["method"] != "observed"]
    rows = []
    for (asset_id, scenario), group in forecast.groupby(["asset_id", "scenario"]):
        group = group.sort_values("year")
        at_2050 = group.loc[group["year"] == 2050, "value"]
        threshold_year = None
        if timber_assets is not None and asset_id in timber_assets.index:
            asset = timber_assets.loc[asset_id]
            remaining = asset["initial_diameter_mm"] - 2 * group["value"]
            reached = group.loc[remaining <= asset["replacement_diameter_mm"], "year"]
            threshold_year = int(reached.iloc[0]) if len(reached) else None
        rows.append(
            {
                "asset_id": asset_id,
                "scenario": scenario,
                "value_2050": float(at_2050.iloc[0]) if len(at_2050) else None,
                "final_year": int(group["year"].iloc[-1]),
                "final_value": float(group["value"].iloc[-1]),
                "threshold_year": threshold_year,
            }
        )
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------ step 3: save
def _climate_input(material: str, tables: dict[str, pd.DataFrame]) -> dict | None:
    sheets = ("site_climate", "climate_changes") if material == "timber" else ("concrete_climate",)
    supplied = {s: json.loads(tables[s].to_json(orient="records")) for s in sheets if s in tables}
    return supplied or None


def save_invalid(
    material: str, file_name: str, content: bytes, uploaded_by: str | None, result: ImportResult
) -> str:
    """Keep a record of a failed upload and its issues (it never reaches the dashboard)."""
    import_id = repo.create_import(
        material, file_name, uploaded_by, "invalid", len(result.errors),
        len(result.issues) - len(result.errors),
    )
    repo.save_issues(import_id, result.issues_frame())
    return import_id


def save(
    material: str,
    file_name: str | None,
    content: bytes | None,
    uploaded_by: str | None,
    result: ImportResult,
    forecasts: list[ModelForecast],
    raw_tables: dict[str, pd.DataFrame] | None = None,
) -> str:
    """Write the import, its piles, inspections, both forecasts and results."""
    n_errors = len(result.errors)
    import_id = repo.create_import(
        material, file_name, uploaded_by, "validated", n_errors,
        len(result.issues) - n_errors, _climate_input(material, raw_tables or {}),
    )
    try:
        repo.save_issues(import_id, result.issues_frame())
        tables = result.tables
        if material == "timber":
            repo.save_assets("timber", import_id, tables["timber_assets"], TIMBER_ASSET_COLUMNS)
            observations = tables.get("timber_observations")
            if observations is not None and not observations.empty:
                cols = [c for c in ("asset_id", "inspection_year", "remaining_diameter_mm", "notes")
                        if c in observations]
                repo.save_observations("timber", import_id, observations[cols])
        else:
            repo.save_assets(
                "concrete", import_id, tables["concrete_assets"], CONCRETE_ASSET_COLUMNS
            )
            observations = tables.get("concrete_observations")
            if observations is not None and not observations.empty:
                repo.save_observations(
                    "concrete", import_id, observations[["asset_id", "year", "carb_depth_mm"]]
                )
        for f in forecasts:
            repo.save_run(
                import_id, f.model_version, f.method, PATHWAY, f.anchored, f.anchor,
                f.points, f.pile_results, f.outcomes,
            )
        storage_path = (
            repo.upload_file(material, import_id, file_name, content)
            if file_name and content
            else None
        )
        repo.set_import_status(import_id, "forecast", storage_path)
    except Exception:
        repo.set_import_status(import_id, "failed")
        raise
    return import_id
