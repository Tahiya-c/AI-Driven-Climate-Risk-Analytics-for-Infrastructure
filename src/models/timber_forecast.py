"""Forecast an imported timber pile with a saved model (no retraining).

    python -m src.models.timber_forecast --input data/templates/port_kembla --model gbm
    python -m src.models.timber_forecast --input my_import/ --model mlp --pathway A1FI \\
        --out data/dashboard/dashboard_data_timber_import_mlp.csv

The input folder uses the template tables (see src/data/timber_template.py):
``timber_assets.csv`` (pile attributes and codes), ``climate.csv`` (each scenario's climate,
past and future) and, optionally, ``timber_observations.csv`` (measured remaining diameter).
``scenarios.csv`` is read from data/templates/.

The saved models (``python -m src.models.timber_transfer --save``) were trained on PileA.
They predict material loss for any pile from its attributes and climate. If inspection
history is supplied, the forecast is **anchored**: shifted by the model's mean error over
the last 5 years of inspections, so it continues from the pile's measured history. The
offsets are in ``result.attrs["anchor"]``; any above ``LARGE_OFFSET_MM`` are warned
about, because a large shift means the model is far from that pile. Output uses the
dashboard schema, so the existing dashboard chart can show it.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.data.timber_template import TEMPLATE_DIR, assemble_timber_frame
from src.features.timber_features import TARGET_COLUMN
from src.models.timber_transfer import (
    LARGE_OFFSET_MM,
    LOSS_COLUMN,
    MODEL_INPUT_COLUMNS,
    SHIFT_WINDOW,
    flag_large_offsets,
    library_versions,
    material_loss,
    saved_model_path,
)

SCHEMA = ["material", "asset_id", "scenario", "year", "value", "unit", "metric", "method"]
SHIFT_YEARS = SHIFT_WINDOW[1] - SHIFT_WINDOW[0] + 1  # 5, the same window as the dashboard
FORECAST_SCENARIOS = ("Most Likely", "Hotter & Drier", "Colder & Wetter")


def load_model_metadata(model_kind: str) -> dict:
    """The saved model's metadata JSON (version, features, params, scores)."""
    path = saved_model_path(model_kind).with_suffix(".json")
    return json.loads(path.read_text()) if path.exists() else {}


def load_model(model_kind: str):
    """Load the saved model of record. Never trains a new one.

    Warns if the installed libraries differ from those the model was saved with:
    joblib files can fail to load, or silently change predictions, across
    scikit-learn/LightGBM versions.
    """
    path = saved_model_path(model_kind)
    if not path.exists():
        raise FileNotFoundError(
            f"No saved {model_kind} model at {path}. "
            "Run: python -m src.models.timber_transfer --save"
        )
    saved_with = load_model_metadata(model_kind).get("library_versions", {})
    installed = library_versions()
    mismatched = {
        name: (version, installed.get(name))
        for name, version in saved_with.items()
        if name != "python" and installed.get(name) != version
    }
    if mismatched:
        warnings.warn(
            f"{model_kind} model was saved with different library versions "
            f"(saved, installed): {mismatched}. Predictions may differ; pin these versions.",
            stacklevel=2,
        )
    return joblib.load(path)


def _broadcast_inspections(
    observations: pd.DataFrame, assets: pd.DataFrame, climate: pd.DataFrame
) -> pd.DataFrame:
    """Rows with a blank ``scenario_id`` are real inspections: copy them to every scenario.

    A pile has one measured history, not one per climate future, so an import can leave
    scenario_id empty. Rows that do name a scenario (e.g. simulated data) are kept as-is.
    """
    if "scenario_id" not in observations:
        observations = observations.assign(scenario_id=np.nan)
    blank = observations["scenario_id"].isna()
    if not blank.any():
        return observations
    site_scenarios = climate[["site", "scenario_id"]].drop_duplicates()
    asset_sites = assets[["asset_id", "site"]]
    expanded = (
        observations.loc[blank]
        .drop(columns="scenario_id")
        .merge(asset_sites, on="asset_id")
        .merge(site_scenarios, on="site")
        .drop(columns="site")
    )
    return pd.concat([observations.loc[~blank], expanded], ignore_index=True)


def _anchor(
    frame: pd.DataFrame, uncapped: np.ndarray, max_offset: float = LARGE_OFFSET_MM
) -> tuple[np.ndarray, pd.Series, dict]:
    """Shift each asset's forecast so it continues from its measured history.

    The offset is the mean (actual - predicted) over the inspections in the last
    ``SHIFT_YEARS`` years up to the asset's latest inspection, averaged over every
    scenario series, so one set of inspections (usually recorded once, not per scenario)
    shifts every scenario and one bad inspection can't set it on its own. Same window
    length as the dashboard's ``timber_transfer.shift_forecast``. Applied before clipping
    to [0, radius] so forecasts can still reach the full radius. Offsets larger than
    ``max_offset`` mm are warned about and listed in the returned record.
    """
    frame = frame.assign(raw=uncapped)
    observed = frame.loc[frame[LOSS_COLUMN].notna()]
    anchor_year = observed.groupby("asset_code")["year"].max().rename("anchor_year")
    recent = observed.join(anchor_year, on="asset_code")
    recent = recent.loc[recent["year"] > recent["anchor_year"] - SHIFT_YEARS]
    offset = (recent[LOSS_COLUMN] - recent["raw"]).groupby(recent["asset_code"]).mean()
    record = {
        "window_years": SHIFT_YEARS,
        "threshold_mm": max_offset,
        "offsets_mm": {asset: float(value) for asset, value in offset.items()},
        "large_offsets": flag_large_offsets(offset, "import anchoring", max_offset),
    }
    shifted = frame["raw"] + frame["asset_code"].map(offset).fillna(0.0)
    clipped = np.clip(shifted, 0, frame["initial_diameter_mm"] / 2).to_numpy()
    return clipped, anchor_year, record


def forecast_pile(
    assets: pd.DataFrame,
    climate: pd.DataFrame,
    scenarios: pd.DataFrame,
    observations: pd.DataFrame | None = None,
    model_kind: str = "gbm",
    pathway: str = "A1FI",
    anchor: bool = True,
    allow_unknown_codes: bool = False,
) -> pd.DataFrame:
    """History (observed) + 3 scenario forecasts per asset, in the dashboard schema.

    Without observations there is no history to anchor to: every year is a model
    prediction, and the forecast starts at the first year after installation.
    """
    if observations is not None:
        observations = _broadcast_inspections(observations, assets, climate)
    frame = assemble_timber_frame(assets, climate, scenarios, observations, allow_unknown_codes)
    frame = frame.loc[
        (frame["emissions_pathway"] == pathway) & frame["scenario_label"].isin(FORECAST_SCENARIOS)
    ].reset_index(drop=True)
    if frame.empty:
        raise ValueError(
            f"No climate rows for pathway {pathway!r} and scenarios {FORECAST_SCENARIOS}"
        )

    if TARGET_COLUMN in frame:
        measured = frame[TARGET_COLUMN].notna()
        frame[LOSS_COLUMN] = np.where(measured, material_loss(frame), np.nan)
    else:
        frame[LOSS_COLUMN] = np.nan

    model = load_model(model_kind)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        uncapped = model.predict_uncapped(frame[MODEL_INPUT_COLUMNS])
    radius = frame["initial_diameter_mm"].to_numpy() / 2
    has_history = frame[LOSS_COLUMN].notna().any()
    anchor_record = None
    if has_history:
        anchored, anchor_year, anchor_record = _anchor(frame, uncapped)
        frame["prediction"] = anchored if anchor else np.clip(uncapped, 0, radius)
    else:
        frame["prediction"] = np.clip(uncapped, 0, radius)
        anchor_year = frame.groupby("asset_code")["year"].min().rename("anchor_year") - 1
    frame = frame.join(anchor_year, on="asset_code")

    is_history = frame["year"] <= frame["anchor_year"]
    history = frame.loc[is_history & frame[LOSS_COLUMN].notna()].assign(
        value=lambda rows: rows[LOSS_COLUMN], method="observed"
    )
    forecast = frame.loc[~is_history].assign(
        value=lambda rows: rows["prediction"],
        method=model_kind + ("_anchored" if anchor and has_history else ""),
    )
    out = pd.concat([history, forecast], ignore_index=True).assign(
        material="timber",
        asset_id=lambda rows: rows["asset_code"],
        scenario=lambda rows: rows["emissions_pathway"] + "|" + rows["climate_model"],
        unit="mm",
        metric="deterioration_depth",
    )
    out["value"] = out["value"].round(4)
    out = out[SCHEMA].sort_values(["asset_id", "scenario", "method", "year"]).reset_index(drop=True)
    out.attrs["anchor"] = anchor_record if anchor else None
    return out


def forecast_from_import(source: Path, **kwargs) -> pd.DataFrame:
    """Forecast from the engineer-facing import format (docs/timber_import_guide.md).

    ``source`` is the import .xlsx or a folder of its 4 CSVs (timber_assets,
    site_climate, climate_changes, timber_observations). It is validated first; any
    errors are raised together, in the same form the upload page shows them.
    """
    from src.data.timber_import import read_import, to_model_tables, validate_import

    result = validate_import(read_import(source))
    if not result.ok:
        raise ValueError("Import file has errors:\n" + result.issues_frame().to_string(index=False))
    return forecast_pile(**to_model_tables(result), **kwargs)


def forecast_from_folder(directory: Path, **kwargs) -> pd.DataFrame:
    """Forecast from a folder: the import format, or the yearly-climate template format.

    The import format is detected by ``site_climate.csv`` (or an .xlsx path); otherwise
    the folder must hold timber_assets.csv + a yearly climate.csv (src/data/timber_template.py).
    """
    directory = Path(directory)
    if directory.suffix.lower() == ".xlsx" or (directory / "site_climate.csv").exists():
        return forecast_from_import(directory, **kwargs)
    observations_path = directory / "timber_observations.csv"
    return forecast_pile(
        assets=pd.read_csv(directory / "timber_assets.csv"),
        climate=pd.read_csv(directory / "climate.csv"),
        scenarios=pd.read_csv(TEMPLATE_DIR / "scenarios.csv"),
        observations=pd.read_csv(observations_path) if observations_path.exists() else None,
        **kwargs,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="import .xlsx, a folder of its 4 CSVs, or a yearly-climate template folder",
    )
    parser.add_argument("--model", choices=("gbm", "mlp"), default="gbm")
    parser.add_argument("--pathway", choices=("A1FI", "A1B"), default="A1FI")
    parser.add_argument("--no-anchor", action="store_true", help="raw model output only")
    parser.add_argument(
        "--allow-unknown-codes",
        action="store_true",
        help="approximate-match undefined codes like the workbook (Defect D3) instead of failing",
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    df = forecast_from_folder(
        args.input,
        model_kind=args.model,
        pathway=args.pathway,
        anchor=not args.no_anchor,
        allow_unknown_codes=args.allow_unknown_codes,
    )
    print(df.groupby(["asset_id", "method"]).size().to_string())
    anchor_record = df.attrs.get("anchor")
    if anchor_record:
        print("anchor offsets (mm):", anchor_record["offsets_mm"])
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out, index=False)
        print(f"wrote {len(df)} rows -> {args.out}")
        if anchor_record:
            args.out.with_suffix(".anchor.json").write_text(json.dumps(anchor_record, indent=2))


if __name__ == "__main__":
    main()
