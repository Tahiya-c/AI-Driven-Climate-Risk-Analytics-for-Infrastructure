"""Import a user's concrete pile data: read, validate, and convert it for the saved models.

    python -m src.data.concrete_import --skeleton                 # write the blank skeleton files
    python -m src.data.concrete_import --validate my_piles.xlsx   # report every problem in a file

Mirrors ``src/data/timber_import.py``'s shape (same ``Field``/``Issue``/``ImportResult``
engine, from ``src/data/import_schema.py``) but scoped to concrete's much smaller feature
set -- see ``docs/concrete_kembla_v6_data_dictionary.md`` and ``[[concrete-modelling-reference]]``
for where these columns come from. Three sheets (one .xlsx with these sheet names, or a
folder of same-named .csv files):

    concrete_assets        <- one row per pile: zone, w/c ratio, concrete grade  (required)
    concrete_climate        <- one row per scenario/year: temperature, humidity, CO2 (required)
    concrete_observations   <- inspections: measured carbonation depth per year  (optional)

``to_model_tables`` turns a valid import into the series-shaped records
``src.models.concrete_forecast.forecast_pile`` expects (the same per-series dict shape
``concrete_eval.load_series`` builds from the training CSV, so the exact same
``rollout()``/model code runs on imported data as on the training data).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import ROOT_DIR
from src.data.import_schema import Field, ImportResult, Issue
from src.data.import_schema import check_sheet as _check_sheet
from src.data.import_schema import read_import as _read_import
from src.data.import_schema import apply_fill, build_year_lookup

TEMPLATE_DIR = ROOT_DIR / "data" / "templates"  # shared with timber_import.py's templates
SKELETON_DIR = TEMPLATE_DIR / "skeleton"  # concrete_*.csv alongside timber's timber_*.csv there
SKELETON_XLSX = TEMPLATE_DIR / "concrete_import_template.xlsx"
REFERENCE_CSV = ROOT_DIR / "data" / "processed" / "concrete_kembla_v6_consolidated.csv"

ZONES = ("atmospheric", "splash", "tidal")
PATHWAYS = ("A1FI", "A1B", "Base")
SERIES_START = 2000  # must match rnn_model.SERIES_START -- t_exposure's zero point
FIRST_YEAR, LAST_YEAR = 2000, 2070

ASSET_FIELDS = (
    Field("asset_id", "text", True, description="Unique pile ID, e.g. PileH"),
    Field(
        "zone",
        "code",
        True,
        allowed=ZONES,
        description="Exposure zone -- sets how fast carbonation progresses. No cross-pile "
        "default: concrete grade and exposure are what make a pile a pile, so this is "
        "never guessed. Only 3 values exist in the reference data (atmospheric/splash/tidal).",
    ),
    Field(
        "wc_ratio",
        "float",
        True,
        minimum=0.2,
        maximum=0.8,
        description="Water/cement ratio used when the pile was built (reference range 0.33-0.60)",
    ),
    Field(
        "fc_mpa",
        "float",
        True,
        "MPa",
        minimum=15,
        maximum=150,
        description="Concrete compressive strength grade (reference range 40-120 MPa)",
    ),
)
CLIMATE_FIELDS = (
    Field("site", "text", True, description="Must match the pile's site"),
    Field(
        "scenario_label",
        "text",
        True,
        description="Free label for the climate scenario, e.g. 'Most Likely', 'Base'",
    ),
    Field("climate_model", "text", True, description="GCM name, or '(Reference)' for the baseline"),
    Field("emissions_pathway", "code", True, allowed=PATHWAYS, description="A1FI, A1B, or Base"),
    Field("year", "int", True, "year", minimum=FIRST_YEAR, maximum=LAST_YEAR),
    Field(
        "mean_temp_c",
        "float",
        True,
        "°C",
        minimum=10,
        maximum=35,
        description="Required: the dashboard's climate driver, never guessed",
    ),
    Field(
        "co2_ppm",
        "float",
        True,
        "ppm",
        minimum=300,
        maximum=1200,
        description="Required: the dashboard's climate driver, never guessed",
    ),
    Field(
        "mean_rh_pct",
        "float",
        False,
        "%",
        minimum=20,
        maximum=100,
        description="Relative humidity. Optional: filled from that year's typical value "
        "across the reference scenarios if an import has temperature/CO2 but not this.",
        fill="year_mean",
    ),
)
OBSERVATION_FIELDS = (
    Field("asset_id", "text", True, description="Must match concrete_assets"),
    Field("year", "int", True, "year", minimum=FIRST_YEAR, maximum=LAST_YEAR),
    Field(
        "carb_depth_mm",
        "float",
        True,
        "mm",
        minimum=0,
        maximum=100,
        description="Measured carbonation depth (the target) -- used to anchor the forecast",
    ),
)
SHEETS: dict[str, tuple[Field, ...]] = {
    "concrete_assets": ASSET_FIELDS,
    "concrete_climate": CLIMATE_FIELDS,
    "concrete_observations": OBSERVATION_FIELDS,
}
OPTIONAL_SHEETS = {"concrete_observations"}
KEYS = {
    "concrete_assets": ["asset_id"],
    "concrete_climate": ["site", "scenario_label", "emissions_pathway", "year"],
    "concrete_observations": ["asset_id", "year"],
}


# --------------------------------------------------------------------------- reading


def read_import(source: str | Path) -> dict[str, pd.DataFrame]:
    """Read an .xlsx (path or file-like, e.g. a Streamlit upload) or a folder of CSVs."""
    return _read_import(source, SHEETS)


# --------------------------------------------------------------------------- validation


def validate_import(tables: dict[str, pd.DataFrame]) -> ImportResult:
    """Check every sheet and the links between them; report all problems at once."""
    issues: list[Issue] = []
    checked: dict[str, pd.DataFrame] = {}
    for sheet in SHEETS:
        if sheet not in tables:
            if sheet not in OPTIONAL_SHEETS:
                issues.append(Issue("error", sheet, "Required sheet is missing"))
            continue
        checked[sheet] = _check_sheet(sheet, tables[sheet], SHEETS[sheet], KEYS[sheet], issues)
    if any(
        i.level == "error" and i.column is None and "missing" in i.message.lower() for i in issues
    ):
        return ImportResult(checked, issues)

    assets = checked.get("concrete_assets", pd.DataFrame())
    climate = checked.get("concrete_climate", pd.DataFrame())
    observations = checked.get("concrete_observations")

    # mean_rh_pct: the one column this import format allows to be filled rather than
    # supplied -- see CLIMATE_FIELDS. Reference lookup built from the training CSV.
    rh_field = next(f for f in CLIMATE_FIELDS if f.name == "mean_rh_pct")
    if "year" in climate.columns and REFERENCE_CSV.exists():
        reference = pd.read_csv(REFERENCE_CSV)
        lookup = build_year_lookup(reference, "mean_rh_pct", "year")
        climate = apply_fill(climate, rh_field, "year", lookup, issues, "concrete_climate")
        checked["concrete_climate"] = climate

    # Links: every asset named in observations must exist in concrete_assets.
    if observations is not None and not observations.empty and "asset_id" in assets:
        unknown = sorted(set(observations["asset_id"]) - set(assets["asset_id"]))
        for asset_id in unknown:
            issues.append(
                Issue(
                    "error",
                    "concrete_observations",
                    f"asset_id {asset_id!r} not in concrete_assets",
                    column="asset_id",
                )
            )
    for asset_id in sorted(
        set(assets.get("asset_id", []))
        - set([] if observations is None else observations.get("asset_id", []))
    ):
        issues.append(
            Issue(
                "warning",
                "concrete_observations",
                f"{asset_id}: no inspections; forecast starts from 0mm depth at the "
                "earliest climate year instead of anchoring to a measured history",
            )
        )
    return ImportResult(checked, issues)


# --------------------------------------------------------------------------- conversion


def to_model_series(result: ImportResult) -> list[dict]:
    """Valid import -> the per-series dict shape ``concrete_eval``/``rnn_model`` use.

    One dict per (asset, scenario) combination, with exactly the fields
    ``concrete_eval.rollout`` and the model registry read: asset, scenario, X
    (climate + t_exposure + wc_ratio + fc_mpa + zone one-hot, one row per year),
    y (annual_rate, NaN after the anchor so a model never peeks at a "future" true
    rate), t (t_exposure), depth (carbonation depth, anchored to observations if any,
    else starting at 0), wc (wc_ratio, for wc_regress).
    """
    if not result.ok:
        raise ValueError("Import has errors; fix them before forecasting")
    tables = result.tables
    assets = tables["concrete_assets"].set_index("asset_id")
    climate = tables["concrete_climate"]
    observations = tables.get("concrete_observations")

    # categories=ZONES (not just the zones present in this import) so the one-hot
    # block is always 3 columns wide, matching what the saved model was trained on --
    # a single-pile import would otherwise produce a 1-column block and silently
    # misalign every feature after it.
    zone_categorical = pd.Categorical(assets["zone"], categories=ZONES)
    cat = pd.get_dummies(pd.DataFrame({"zone": zone_categorical}, index=assets.index), prefix="zone")

    series = []
    groups = climate.groupby(["site", "scenario_label", "climate_model", "emissions_pathway"])
    for (_site, scenario_label, climate_model, pathway), rows in groups:
        rows = rows.sort_values("year")
        for asset_id in assets.index:
            years = rows["year"].to_numpy(dtype=np.int64)
            t_exposure = (years - SERIES_START + 1).astype(np.float64)
            climate_block = rows[["mean_temp_c", "mean_rh_pct", "co2_ppm"]].to_numpy(dtype=np.float32)
            static_block = np.tile(
                assets.loc[asset_id, ["wc_ratio", "fc_mpa"]].to_numpy(dtype=np.float32), (len(rows), 1)
            )
            zone_block = np.tile(cat.loc[asset_id].to_numpy(dtype=np.float32), (len(rows), 1))
            X = np.concatenate(
                [climate_block, t_exposure.astype(np.float32)[:, None], static_block, zone_block], axis=1
            )

            depth = np.full(len(rows), np.nan)
            obs = (
                observations.loc[observations["asset_id"] == asset_id]
                if observations is not None
                else pd.DataFrame(columns=["year", "carb_depth_mm"])
            )
            if not obs.empty:
                by_year = obs.set_index("year")["carb_depth_mm"]
                known = np.isin(years, by_year.index)
                depth[known] = by_year.loc[years[known]].to_numpy()
                # Anchor: hold the first known depth back to year 0, so every earlier
                # row still has a real starting point to compute annual_rate from.
                # Years between inspections (or before the first one) are therefore
                # approximations, not measurements -- see last_observed_year below.
                first_known = np.flatnonzero(known)[0]
                depth[:first_known] = depth[first_known]
                last_observed_year = int(years[known].max())
            else:
                depth[0] = 0.0  # brand-new pile, no inspections: start at 0mm
                last_observed_year = int(years[0]) - 1  # i.e. "nothing is observed yet"

            # Forward-fill so every row up to the last known depth has a value;
            # annual_rate is only meaningful where two consecutive real depths exist.
            depth = pd.Series(depth).ffill().to_numpy()
            annual_rate = np.diff(depth, prepend=depth[0])
            annual_rate[0] = np.nan

            series.append(
                {
                    "asset": asset_id,
                    "scenario": f"{pathway}|{climate_model}",
                    "scenario_label": scenario_label,
                    "X": X.astype(np.float32),
                    "y": annual_rate.astype(np.float32),
                    "t": t_exposure,
                    "depth": depth,
                    "wc": float(assets.loc[asset_id, "wc_ratio"]),
                    "years": years,
                    # Last year with a real (not filled/interpolated) depth. Forecasting
                    # starts the year after this -- concrete_forecast.py needs it to know
                    # where "known" ends and "to predict" begins; it can't be inferred
                    # from NaN since depth/y above are already forward-filled.
                    "last_observed_year": last_observed_year,
                }
            )
    return series


# --------------------------------------------------------------------------- skeleton


def _example_tables() -> dict[str, pd.DataFrame]:
    """One example pile, climate copied from the real PileA|A1FI|CSIRO Mk3.5 series."""
    years = list(range(2020, 2027))
    return {
        "concrete_assets": pd.DataFrame(
            [{"asset_id": "EXAMPLE_01", "zone": "splash", "wc_ratio": 0.45, "fc_mpa": 50.0}]
        ),
        "concrete_climate": pd.DataFrame(
            [
                {
                    "site": "Port Kembla",
                    "scenario_label": "Most Likely",
                    "climate_model": "MRI-CGCM2.3.2",
                    "emissions_pathway": "A1FI",
                    "year": year,
                    "mean_temp_c": round(19.1 + 0.05 * i, 2),
                    "co2_ppm": round(368.0 + 5.0 * i, 1),
                    "mean_rh_pct": "",  # deliberately blank: demonstrates the year-fill
                }
                for i, year in enumerate(years)
            ]
        ),
        "concrete_observations": pd.DataFrame(
            [
                {"asset_id": "EXAMPLE_01", "year": year, "carb_depth_mm": round(1.4 + 0.3 * i, 2)}
                for i, year in enumerate(years)
            ]
        ),
    }


def write_skeleton() -> None:
    """Blank CSVs + example-filled CSVs, same shape as timber_import.write_skeleton."""
    SKELETON_DIR.mkdir(parents=True, exist_ok=True)
    for sheet, fields in SHEETS.items():
        pd.DataFrame(columns=[f.name for f in fields]).to_csv(
            SKELETON_DIR / f"{sheet}.csv", index=False
        )
    examples = _example_tables()
    for sheet, frame in examples.items():
        frame.to_csv(SKELETON_DIR / f"{sheet}.example.csv", index=False)
    with pd.ExcelWriter(SKELETON_XLSX, engine="openpyxl") as writer:
        for sheet, frame in examples.items():
            frame.to_excel(writer, sheet_name=sheet, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--skeleton", action="store_true", help="write the skeleton files")
    parser.add_argument("--validate", type=Path, help=".xlsx or folder of CSVs to check")
    args = parser.parse_args()
    if not (args.skeleton or args.validate):
        parser.error("pass --skeleton and/or --validate")
    if args.skeleton:
        write_skeleton()
        print(f"wrote {SKELETON_XLSX.relative_to(ROOT_DIR)} and {SKELETON_DIR.relative_to(ROOT_DIR)}/")
    if args.validate:
        result = validate_import(read_import(args.validate))
        report = result.issues_frame()
        print(report.to_string(index=False) if not report.empty else "No issues.")
        print("VALID" if result.ok else f"INVALID: {len(result.errors)} error(s)")


if __name__ == "__main__":
    main()
