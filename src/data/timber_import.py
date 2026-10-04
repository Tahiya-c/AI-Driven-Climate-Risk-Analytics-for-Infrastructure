"""Import a user's timber pile data: read, validate, and convert it for the saved models.

    python -m src.data.timber_import --skeleton                 # write the blank skeleton files
    python -m src.data.timber_import --validate my_piles.xlsx   # report every problem in a file

The import mirrors the Timber_kembla_v6.3 workbook, so an engineer can copy values
straight from it. Four sheets (one .xlsx with these sheet names, or a folder of
same-named .csv files):

    timber_assets        <- workbook "InputData" sheet: one row per pile, with the 6 pile
                            coefficients as numbers from the "Coefficients" sheet  (required)
    site_climate         <- workbook "Baseline" sheet: summer SST + salinity       (required)
    climate_changes      <- workbook "MostLikely"/"Hotter&Drier"/"Cooler&Wetter"
                            sheets: change from baseline at 1990/2010/2030/2050/2070 (required)
    timber_observations  <- inspections: measured remaining diameter per year     (optional)

``ASSET_FIELDS`` etc. are the single definition of every column (type, unit, allowed
values, Excel source). The validator, the skeleton files and
docs/timber_import_guide.md are all built from them. ``to_model_tables`` turns a
valid import into the tables ``src.models.timber_forecast.forecast_pile`` expects:
yearly climate is interpolated between the anchor years exactly as the workbook does.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import ROOT_DIR
from src.data.timber_template import TEMPLATE_DIR, scenario_id
from src.models.timber_coefficients import K_CONTACT, K_HW_SW, K_KNOT, K_PROTECT, K_SHELTER

SKELETON_DIR = TEMPLATE_DIR / "skeleton"
SKELETON_XLSX = TEMPLATE_DIR / "timber_import_template.xlsx"
FIRST_YEAR, LAST_YEAR = 2000, 2070
ANCHOR_YEARS = (1990, 2010, 2030, 2050, 2070)
REQUIRED_SCENARIOS = ("Most Likely", "Hotter & Drier", "Colder & Wetter")
PATHWAYS = ("A1FI", "A1B")
MIN_HISTORY_YEARS = 15  # below this an anchored forecast still runs, with a warning

ZONES = list("ABCDEFG")


@dataclass(frozen=True)
class Field:
    name: str
    dtype: str  # "text" | "int" | "float" | "code"
    required: bool
    unit: str = ""
    source: str = ""
    description: str = ""
    allowed: tuple[str, ...] = ()
    minimum: float | None = None
    maximum: float | None = None


ASSET_FIELDS = (
    Field(
        "asset_id",
        "text",
        True,
        source="InputData A (Asset Code)",
        description="Unique pile ID, e.g. PileA",
    ),
    Field(
        "site",
        "text",
        True,
        source="Baseline B1 (e.g. Port Kembla)",
        description="Must match a site in site_climate",
    ),
    Field(
        "description",
        "text",
        False,
        source="InputData C",
        description="Free text, e.g. Pile 32 @ Berth 201",
    ),
    Field(
        "timber_type",
        "text",
        False,
        source="InputData D (Type)",
        description="Species, e.g. Turpentine Gum",
    ),
    Field(
        "year_installed",
        "int",
        True,
        "year",
        "InputData B",
        minimum=1900,
        maximum=LAST_YEAR,
        description="Installation year",
    ),
    Field(
        "zone",
        "code",
        True,
        source="InputData M (Zone)",
        allowed=tuple(ZONES),
        description="AS 5604 hazard zone A-G (sets k_sal each year, and which k_wood column to read)",
    ),
    Field(
        "initial_diameter_mm",
        "float",
        True,
        "mm",
        "InputData G (Do)",
        minimum=50,
        maximum=2000,
        description="Diameter when installed",
    ),
    Field(
        "sapwood_layer_mm",
        "float",
        True,
        "mm",
        "InputData H (dsap)",
        minimum=0,
        maximum=200,
        description="Sapwood layer thickness",
    ),
    Field(
        "replacement_diameter_mm",
        "float",
        True,
        "mm",
        "InputData I (Dreplace)",
        minimum=0,
        maximum=2000,
        description="Replace the pile below this diameter",
    ),
    Field(
        "k_heartwood",
        "float",
        True,
        source="Coefficients: k_wood for the heartwood class (InputData E), zone column",
        minimum=0.1,
        maximum=50,
        description="Heartwood factor k_HW, e.g. HW1 = 1.1 (zones A-C) or 1.3 (D-G)",
    ),
    Field(
        "k_sapwood",
        "float",
        True,
        source="Coefficients: k_wood for the sapwood code (InputData F)",
        minimum=0.1,
        maximum=50,
        description="Sapwood factor k_SW, e.g. SPHU (untreated hardwood sapwood) = 25.0",
    ),
    Field(
        "k_shelter",
        "float",
        True,
        source="Coefficients: k_shelter for the exposure code (InputData N)",
        minimum=0.1,
        maximum=2,
        description="Water zone: EXP1 calm/sheltered = 1.0, EXP2 strong wave or surf = 0.6",
    ),
    Field(
        "k_protect",
        "float",
        True,
        source="Coefficients: k_protect for the maintenance code (InputData J)",
        minimum=0.1,
        maximum=2,
        description="Maintenance: MTN1 floating collar = 0.5, MTN2 none = 1.0",
    ),
    Field(
        "k_contact",
        "float",
        True,
        source="Coefficients: k_contact for the contact code (InputData K)",
        minimum=0.1,
        maximum=5,
        description="Contact: CNT1 with another timber member = 2.0, CNT2 none = 1.0",
    ),
    Field(
        "k_knot",
        "float",
        True,
        source="Coefficients: k_knot for the knot code (InputData L)",
        minimum=0.1,
        maximum=5,
        description="Knots: KNT1 without protective plate = 2.0, KNT2 with plate = 1.0, KNT3 none = 1.0",
    ),
)
# AS 5604 values each coefficient normally takes (the Coefficients sheet). A value
# outside its set is allowed -- e.g. a treatment not in the table -- but is warned about.
STANDARD_K_VALUES: dict[str, set[float]] = {
    "k_heartwood": {v for code, pair in K_HW_SW.items() if code.startswith("HW") for v in pair},
    "k_sapwood": {v for code, pair in K_HW_SW.items() if code.startswith("SP") for v in pair},
    "k_shelter": set(K_SHELTER.values()),
    "k_protect": set(K_PROTECT.values()),
    "k_contact": set(K_CONTACT.values()),
    "k_knot": set(K_KNOT.values()),
}
SITE_FIELDS = (
    Field("site", "text", True, source="Baseline B1", description="Site name"),
    Field(
        "baseline_summer_sst_c",
        "float",
        True,
        "°C",
        "Baseline C9 (Sea surface temperature, DJF)",
        minimum=-2,
        maximum=40,
        description="Summer (DJF) sea surface temperature, baseline period",
    ),
    Field(
        "baseline_salinity_ppt",
        "float",
        True,
        "ppt",
        "Baseline C8 (Sea surface salinity, DJF)",
        minimum=0,
        maximum=45,
        description="Summer sea surface salinity, baseline period",
    ),
)
CHANGE_FIELDS = (
    Field("site", "text", True, description="Must match site_climate"),
    Field(
        "scenario_label",
        "code",
        True,
        source="scenario sheet name",
        allowed=(*REQUIRED_SCENARIOS, "Baseline"),
        description="Climate scenario (Baseline = no change, optional)",
    ),
    Field(
        "climate_model",
        "text",
        True,
        source="scenario sheet B1 (Model)",
        description="GCM name, e.g. MRI-CGCM2.3.2",
    ),
    Field(
        "emissions_pathway",
        "code",
        True,
        source="scenario sheet row 4",
        allowed=PATHWAYS,
        description="A1FI or A1B",
    ),
    Field(
        "anchor_year",
        "int",
        True,
        "year",
        "scenario sheet A5:A9",
        allowed=tuple(str(year) for year in ANCHOR_YEARS),
        description="1990, 2010, 2030, 2050, 2070",
    ),
    Field(
        "summer_sst_change_c",
        "float",
        True,
        "°C",
        "scenario sheet L-M (Summer SST change)",
        minimum=-5,
        maximum=10,
        description="Change from baseline at anchor_year",
    ),
    Field(
        "salinity_change_ppt",
        "float",
        False,
        "ppt",
        "pile sheet column E (Salinity) minus baseline; the workbook takes it from "
        "Cooler&Wetter J-K (Annual SSS change) for every scenario",
        minimum=-10,
        maximum=10,
        description="Change from baseline salinity at anchor_year; blank = 0",
    ),
)
OBSERVATION_FIELDS = (
    Field("asset_id", "text", True, description="Must match timber_assets"),
    Field(
        "inspection_year",
        "int",
        True,
        "year",
        minimum=1900,
        maximum=LAST_YEAR,
        description="Year of the inspection",
    ),
    Field(
        "remaining_diameter_mm",
        "float",
        True,
        "mm",
        "pile sheet BF (Dt) / field inspection",
        minimum=0,
        description="Measured remaining sound diameter (the target)",
    ),
    Field("notes", "text", False, description="Optional"),
)
SHEETS: dict[str, tuple[Field, ...]] = {
    "timber_assets": ASSET_FIELDS,
    "site_climate": SITE_FIELDS,
    "climate_changes": CHANGE_FIELDS,
    "timber_observations": OBSERVATION_FIELDS,
}
OPTIONAL_SHEETS = {"timber_observations"}
KEYS = {
    "timber_assets": ["asset_id"],
    "site_climate": ["site"],
    "climate_changes": ["site", "scenario_label", "emissions_pathway", "anchor_year"],
    "timber_observations": ["asset_id", "inspection_year"],
}


@dataclass
class Issue:
    level: str  # "error" blocks forecasting; "warning" does not
    sheet: str
    message: str
    row: int | None = None  # 1-based data row (row 1 = first row under the header)
    column: str | None = None


@dataclass
class ImportResult:
    tables: dict[str, pd.DataFrame]
    issues: list[Issue] = field(default_factory=list)

    @property
    def errors(self) -> list[Issue]:
        return [issue for issue in self.issues if issue.level == "error"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def issues_frame(self) -> pd.DataFrame:
        frame = pd.DataFrame(
            [vars(issue) for issue in self.issues],
            columns=["level", "sheet", "row", "column", "message"],
        )
        frame["row"] = frame["row"].astype("Int64")
        return frame


# --------------------------------------------------------------------------- reading


def read_import(source: str | Path | Any) -> dict[str, pd.DataFrame]:
    """Read an .xlsx (path or file-like, e.g. a Streamlit upload) or a folder of CSVs."""
    if isinstance(source, (str, Path)) and Path(source).is_dir():
        directory = Path(source)
        return {
            sheet: pd.read_csv(directory / f"{sheet}.csv")
            for sheet in SHEETS
            if (directory / f"{sheet}.csv").exists()
        }
    workbook = pd.read_excel(source, sheet_name=None)
    return {sheet: frame for sheet, frame in workbook.items() if sheet in SHEETS}


# --------------------------------------------------------------------------- validation


def _check_sheet(sheet: str, frame: pd.DataFrame, issues: list[Issue]) -> pd.DataFrame:
    """Column presence, types, ranges and codes. Returns the frame with coerced dtypes."""
    fields = SHEETS[sheet]
    frame = frame.copy()
    frame.columns = [str(column).strip() for column in frame.columns]
    frame = frame.dropna(how="all")
    missing = [f.name for f in fields if f.required and f.name not in frame.columns]
    for name in missing:
        issues.append(Issue("error", sheet, f"Missing required column '{name}'", column=name))
    unknown = sorted(set(frame.columns) - {f.name for f in fields})
    if unknown:
        issues.append(Issue("warning", sheet, f"Ignoring unknown columns: {unknown}"))
    if frame.empty:
        issues.append(Issue("error", sheet, "Sheet has no data rows"))
        return frame

    for f in fields:
        if f.name not in frame.columns:
            continue
        column = frame[f.name]
        blank = column.isna() | (column.astype(str).str.strip() == "")
        if f.required:
            for row in np.flatnonzero(blank.to_numpy()):
                issues.append(Issue("error", sheet, "Required value is blank", row + 1, f.name))
        if f.dtype in ("int", "float"):
            numeric = pd.to_numeric(column, errors="coerce")
            for row in np.flatnonzero((numeric.isna() & ~blank).to_numpy()):
                issues.append(
                    Issue("error", sheet, f"Not a number: {column.iloc[row]!r}", row + 1, f.name)
                )
            if f.dtype == "int":
                fractional = numeric.notna() & (numeric % 1 != 0)
                for row in np.flatnonzero(fractional.to_numpy()):
                    issues.append(Issue("error", sheet, "Must be a whole number", row + 1, f.name))
            unit = f" {f.unit}" if f.unit else ""
            for bound, outside, limit in (
                (f.minimum, lambda b: numeric < b, "minimum"),
                (f.maximum, lambda b: numeric > b, "maximum"),
            ):
                if bound is None:
                    continue
                for row in np.flatnonzero(outside(bound).fillna(False).to_numpy()):
                    message = f"{numeric.iloc[row]} is outside the allowed {limit} {bound}{unit}"
                    issues.append(Issue("error", sheet, message, row + 1, f.name))
            frame[f.name] = numeric
        else:
            frame[f.name] = column.where(blank, column.astype(str).str.strip())
        if f.allowed:
            values = frame[f.name].astype("string")
            if f.dtype == "int":
                values = frame[f.name].map(lambda v: None if pd.isna(v) else str(int(v)))
            bad = values.notna() & ~values.isin(f.allowed)
            for row in np.flatnonzero(bad.fillna(False).to_numpy()):
                issues.append(
                    Issue(
                        "error",
                        sheet,
                        f"{values.iloc[row]!r} is not an allowed value {list(f.allowed)}",
                        row + 1,
                        f.name,
                    )
                )
    key = [column for column in KEYS[sheet] if column in frame.columns]
    if key and len(key) == len(KEYS[sheet]):
        duplicated = frame.duplicated(key, keep=False)
        for row in np.flatnonzero(duplicated.to_numpy()):
            issues.append(Issue("error", sheet, f"Duplicate key {key}", row + 1))
    return frame


def validate_import(tables: dict[str, pd.DataFrame]) -> ImportResult:
    """Check every sheet and the links between them; report all problems at once."""
    issues: list[Issue] = []
    checked: dict[str, pd.DataFrame] = {}
    for sheet in SHEETS:
        if sheet not in tables:
            if sheet not in OPTIONAL_SHEETS:
                issues.append(Issue("error", sheet, "Required sheet is missing"))
            continue
        checked[sheet] = _check_sheet(sheet, tables[sheet], issues)
    if any(
        i.level == "error" and i.column is None and "missing" in i.message.lower() for i in issues
    ):
        return ImportResult(checked, issues)

    assets = checked.get("timber_assets", pd.DataFrame())
    sites = checked.get("site_climate", pd.DataFrame())
    changes = checked.get("climate_changes", pd.DataFrame())
    observations = checked.get("timber_observations")

    # Cross-field checks within timber_assets.
    if {"initial_diameter_mm", "replacement_diameter_mm", "sapwood_layer_mm"} <= set(assets):
        for row, asset in enumerate(assets.itertuples(index=False), start=1):
            if asset.replacement_diameter_mm >= asset.initial_diameter_mm:
                issues.append(
                    Issue(
                        "error",
                        "timber_assets",
                        "replacement_diameter_mm must be smaller than initial_diameter_mm",
                        row,
                        "replacement_diameter_mm",
                    )
                )
            if 2 * asset.sapwood_layer_mm >= asset.initial_diameter_mm:
                issues.append(
                    Issue(
                        "error",
                        "timber_assets",
                        "sapwood_layer_mm is at least the radius",
                        row,
                        "sapwood_layer_mm",
                    )
                )

    # Coefficients: allowed if positive and in range, but flag values not in AS 5604's
    # tables, and heartwood values from the wrong zone column.
    for column, standard in STANDARD_K_VALUES.items():
        if column not in assets:
            continue
        for row, value in enumerate(assets[column], start=1):
            if pd.isna(value):
                continue
            if not any(abs(value - s) < 1e-9 for s in standard):
                issues.append(
                    Issue(
                        "warning",
                        "timber_assets",
                        f"{column} = {value:g} is not a value in the AS 5604 coefficient "
                        f"table {sorted(standard)}; check it, or keep it for a custom value",
                        row,
                        column,
                    )
                )
    if {"k_heartwood", "zone"} <= set(assets):
        for row, (value, zone) in enumerate(zip(assets["k_heartwood"], assets["zone"]), start=1):
            if pd.isna(value) or zone not in ZONES:
                continue
            column = 0 if zone in "ABC" else 1
            expected = [pair[column] for code, pair in K_HW_SW.items() if code.startswith("HW")]
            other = [pair[1 - column] for code, pair in K_HW_SW.items() if code.startswith("HW")]
            if not any(abs(value - e) < 1e-9 for e in expected) and any(
                abs(value - o) < 1e-9 for o in other
            ):
                group = "A-C" if column == 0 else "D-G"
                issues.append(
                    Issue(
                        "warning",
                        "timber_assets",
                        f"k_heartwood = {value:g} comes from the wrong zone column: zone "
                        f"{zone} uses the zones {group} column {sorted(expected)}",
                        row,
                        "k_heartwood",
                    )
                )

    # Links: every asset's site has a baseline and full scenario anchors.
    if {"site"} <= set(assets) and {"site"} <= set(sites):
        for row, site in enumerate(assets["site"], start=1):
            if site not in set(sites["site"]):
                issues.append(
                    Issue(
                        "error", "timber_assets", f"site {site!r} not in site_climate", row, "site"
                    )
                )
    if {"site", "scenario_label", "emissions_pathway", "anchor_year"} <= set(changes):
        for site in sorted(set(assets.get("site", []))):
            for scenario in REQUIRED_SCENARIOS:
                for pathway in PATHWAYS:
                    have = set(
                        changes.loc[
                            (changes["site"] == site)
                            & (changes["scenario_label"] == scenario)
                            & (changes["emissions_pathway"] == pathway),
                            "anchor_year",
                        ]
                        .dropna()
                        .astype(int)
                    )
                    missing = sorted(set(ANCHOR_YEARS) - have)
                    if not missing:
                        continue
                    level = "error" if pathway == "A1FI" else "warning"
                    issues.append(
                        Issue(
                            level,
                            "climate_changes",
                            f"{site} / {scenario} / "
                            f"{pathway} is missing anchor years {missing}"
                            + ("" if level == "error" else " (A1B charts unavailable)"),
                        )
                    )

    # Observations: link to assets, plausible values, enough history.
    if observations is not None and not observations.empty and {"asset_id"} <= set(assets):
        lookup = assets.set_index("asset_id")
        this_year = date.today().year
        for row, obs in enumerate(observations.itertuples(index=False), start=1):
            if obs.asset_id not in lookup.index:
                issues.append(
                    Issue(
                        "error",
                        "timber_observations",
                        f"asset_id {obs.asset_id!r} not in timber_assets",
                        row,
                        "asset_id",
                    )
                )
                continue
            asset = lookup.loc[obs.asset_id]
            if obs.inspection_year < asset["year_installed"]:
                issues.append(
                    Issue(
                        "error",
                        "timber_observations",
                        "Inspection is before year_installed",
                        row,
                        "inspection_year",
                    )
                )
            if obs.inspection_year < FIRST_YEAR:
                issues.append(
                    Issue(
                        "warning",
                        "timber_observations",
                        f"Inspections before {FIRST_YEAR} are not used: the climate "
                        f"series starts in {FIRST_YEAR}",
                        row,
                        "inspection_year",
                    )
                )
            if obs.inspection_year > this_year:
                issues.append(
                    Issue(
                        "error",
                        "timber_observations",
                        "Inspection year is in the future",
                        row,
                        "inspection_year",
                    )
                )
            if obs.remaining_diameter_mm > asset["initial_diameter_mm"]:
                issues.append(
                    Issue(
                        "error",
                        "timber_observations",
                        "remaining_diameter_mm is larger than initial_diameter_mm",
                        row,
                        "remaining_diameter_mm",
                    )
                )
        for asset_id, history in observations.groupby("asset_id"):
            if asset_id not in lookup.index:
                continue
            history = history.sort_values("inspection_year")
            if (history["remaining_diameter_mm"].diff() > 1.0).any():
                issues.append(
                    Issue(
                        "warning",
                        "timber_observations",
                        f"{asset_id}: remaining "
                        "diameter increases between inspections (measurement "
                        "error?)",
                    )
                )
            span = history["inspection_year"].max() - lookup.loc[asset_id, "year_installed"]
            if span < MIN_HISTORY_YEARS:
                issues.append(
                    Issue(
                        "warning",
                        "timber_observations",
                        f"{asset_id}: history "
                        f"covers {span} years since installation; at least "
                        f"{MIN_HISTORY_YEARS} recommended. Forecast will run but "
                        "is less reliable.",
                    )
                )
    for asset_id in sorted(
        set(assets.get("asset_id", []))
        - set([] if observations is None else observations.get("asset_id", []))
    ):
        issues.append(
            Issue(
                "warning",
                "timber_observations",
                f"{asset_id}: no inspections; "
                "forecast uses the model alone (no anchoring, no check against the pile)",
            )
        )
    return ImportResult(checked, issues)


# --------------------------------------------------------------------------- conversion


def yearly_climate(sites: pd.DataFrame, changes: pd.DataFrame) -> pd.DataFrame:
    """Expand baseline + anchor-year changes into one row per site/scenario/pathway/year.

    Linear interpolation between anchor years, flat outside them -- the workbook's own
    method (src/models/timber_engine.py::_interpolate_anchors). Baseline rows (no change)
    are added for each site if not supplied.
    """
    rows = []
    groups = changes.groupby(["site", "scenario_label", "climate_model", "emissions_pathway"])
    for (site, scenario, model, pathway), anchors in groups:
        base = sites.set_index("site").loc[site]
        anchors = anchors.sort_values("anchor_year")
        years = np.arange(FIRST_YEAR, LAST_YEAR + 1)
        sst = np.interp(years, anchors["anchor_year"], anchors["summer_sst_change_c"])
        salinity_change = (
            anchors["salinity_change_ppt"].fillna(0.0)
            if "salinity_change_ppt" in anchors
            else pd.Series(0.0, index=anchors.index)
        )
        sal = np.interp(years, anchors["anchor_year"], salinity_change)
        rows.append(
            pd.DataFrame(
                {
                    "site": site,
                    "scenario_id": scenario_id(pathway, scenario),
                    "climate_model": model,
                    "year": years,
                    "summer_sea_temp_c": base["baseline_summer_sst_c"] + sst,
                    "salinity_ppt": base["baseline_salinity_ppt"] + sal,
                }
            )
        )
    for site, base in sites.set_index("site").iterrows():
        if not ((changes["site"] == site) & (changes["scenario_label"] == "Baseline")).any():
            years = np.arange(FIRST_YEAR, LAST_YEAR + 1)
            rows.append(
                pd.DataFrame(
                    {
                        "site": site,
                        "scenario_id": scenario_id("Base", "Baseline"),
                        "climate_model": "(Reference)",
                        "year": years,
                        "summer_sea_temp_c": base["baseline_summer_sst_c"],
                        "salinity_ppt": base["baseline_salinity_ppt"],
                    }
                )
            )
    return pd.concat(rows, ignore_index=True)


def to_model_tables(result: ImportResult) -> dict[str, pd.DataFrame | None]:
    """Valid import -> keyword arguments for ``timber_forecast.forecast_pile``."""
    if not result.ok:
        raise ValueError("Import has errors; fix them before forecasting")
    tables = result.tables
    climate = yearly_climate(tables["site_climate"], tables["climate_changes"])
    scenarios = (
        climate[["scenario_id"]]
        .drop_duplicates()
        .assign(
            emissions_pathway=lambda f: (
                f["scenario_id"].str.split("-").str[0].replace({"BASE": "Base"})
            ),
            scenario_label=lambda f: f["scenario_id"].map(_label_from_id),
        )
        .reset_index(drop=True)
    )
    observations = tables.get("timber_observations")
    if observations is not None and not observations.empty:
        observations = observations.rename(columns={"inspection_year": "year"}).assign(
            scenario_id=np.nan
        )[["asset_id", "scenario_id", "year", "remaining_diameter_mm"]]
    else:
        observations = None
    assets = tables["timber_assets"].copy()
    for optional in ("description", "timber_type"):  # optional here, required downstream
        assets[optional] = assets[optional].fillna("") if optional in assets else ""
    return {
        "assets": assets,
        "climate": climate,
        "scenarios": scenarios,
        "observations": observations,
    }


def _label_from_id(value: str) -> str:
    labels = {
        scenario_id("A1FI", label).split("-", 1)[1]: label
        for label in (*REQUIRED_SCENARIOS, "Baseline")
    }
    return labels[value.split("-", 1)[1]]


# --------------------------------------------------------------------------- skeleton


# Salinity change from the 35.5 ppt baseline that the workbook's pile sheets (column E,
# the model's `salinity` input) actually use. Only MIROC has salinity projections, so
# the workbook borrows MIROC's *annual* SSS change for every scenario -- the A1FI values
# everywhere except Colder & Wetter A1B -- not the summer SSS columns that
# src/models/timber_engine.py uses. Verified against the processed CSV.
_MIROC_A1FI = dict(zip(ANCHOR_YEARS, (0.0, 0.05 / 3, 0.05, 0.10, 0.17), strict=True))
_MIROC_A1B = dict(zip(ANCHOR_YEARS, (0.0, 0.05 / 3, 0.05, 0.09, 0.12), strict=True))
KEMBLA_SALINITY_CHANGE = {
    (scenario, pathway): (
        _MIROC_A1B if (scenario, pathway) == ("Colder & Wetter", "A1B") else _MIROC_A1FI
    )
    for scenario in REQUIRED_SCENARIOS
    for pathway in PATHWAYS
}


def _example_tables() -> dict[str, pd.DataFrame]:
    """One example pile at Port Kembla, climate anchors copied from the workbook."""
    from src.models.timber_engine import SCENARIOS as ENGINE_SCENARIOS

    label = {"Cooler & Wetter": "Colder & Wetter"}
    change_rows = []
    for name, scenario in ENGINE_SCENARIOS.items():
        if name == "Baseline":
            continue
        scenario_label = label.get(name, name)
        for pathway in PATHWAYS:
            salinity = KEMBLA_SALINITY_CHANGE[(scenario_label, pathway)]
            for year in ANCHOR_YEARS:
                change_rows.append(
                    {
                        "site": "Port Kembla",
                        "scenario_label": scenario_label,
                        "climate_model": scenario.climate_model,
                        "emissions_pathway": pathway,
                        "anchor_year": year,
                        "summer_sst_change_c": round(scenario.sst_anchors[pathway][year], 7),
                        "salinity_change_ppt": round(salinity[year], 7),
                    }
                )
    return {
        "timber_assets": pd.DataFrame(
            [
                {
                    "asset_id": "EXAMPLE_01",
                    "site": "Port Kembla",
                    "description": "Pile 12 @ Berth 105 (example - replace)",
                    "timber_type": "Gum",
                    "year_installed": 2006,
                    "zone": "E",
                    "initial_diameter_mm": 450,
                    "sapwood_layer_mm": 20,
                    "replacement_diameter_mm": 200,
                    "k_heartwood": 1.3,  # HW1, zones D-G
                    "k_sapwood": 25.0,  # SPHU
                    "k_shelter": 1.0,  # EXP1
                    "k_protect": 1.0,  # MTN2
                    "k_contact": 1.0,  # CNT2
                    "k_knot": 1.0,  # KNT3
                }
            ]
        ),
        "site_climate": pd.DataFrame(
            [
                {
                    "site": "Port Kembla",
                    "baseline_summer_sst_c": 21.5,
                    "baseline_salinity_ppt": 35.5,
                }
            ]
        ),
        "climate_changes": pd.DataFrame(change_rows),
        "timber_observations": pd.DataFrame(
            [
                {
                    "asset_id": "EXAMPLE_01",
                    "inspection_year": year,
                    "remaining_diameter_mm": value,
                    "notes": "example - replace with real inspections",
                }
                # The workbook engine's values for this pile (A1FI Most Likely), to the mm.
                for year, value in ((2010, 402.0), (2014, 385.0), (2018, 367.0), (2022, 348.0))
            ]
        ),
    }


def _readme_rows() -> pd.DataFrame:
    rows = []
    for sheet, fields in SHEETS.items():
        for f in fields:
            allowed = ", ".join(f.allowed)
            limits = (
                " to ".join(str(v) for v in (f.minimum, f.maximum) if v is not None)
                if (f.minimum is not None or f.maximum is not None)
                else ""
            )
            rows.append(
                {
                    "sheet": sheet + (" (optional)" if sheet in OPTIONAL_SHEETS else ""),
                    "column": f.name,
                    "required": "yes" if f.required else "no",
                    "type": f.dtype,
                    "unit": f.unit,
                    "allowed values / range": allowed or limits,
                    "copy from workbook": f.source,
                    "description": f.description,
                }
            )
    return pd.DataFrame(rows)


_WOOD_DESCRIPTIONS = {
    "HW1": "Heartwood class 1",
    "HW2": "Heartwood class 2",
    "HW3": "Heartwood class 3",
    "HW4": "Heartwood class 4",
    "SPSU": "Softwood sapwood, untreated",
    "SPSCR1": "Softwood sapwood, creosote-treated 24%kg/kg",
    "SPSCR2": "Softwood sapwood, creosote-treated 40%kg/kg",
    "SPSCC1": "Softwood sapwood, CCA-treated 0.6%kg/kg",
    "SPSCC2": "Softwood sapwood, CCA-treated 1%kg/kg",
    "SPSCC3": "Softwood sapwood, CCA-treated 2%kg/kg",
    "SPSCC4": "Softwood sapwood, CCA-treated 5%kg/kg",
    "SPSDBT": "Softwood sapwood, double-treated CC2+CR2",
    "SPHU": "Hardwood sapwood, untreated",
    "SPHCR1": "Hardwood sapwood, creosote-treated 13%kg/kg",
    "SPHCR2": "Hardwood sapwood, creosote-treated 22%kg/kg",
    "SPHCC1": "Hardwood sapwood, CCA-treated 0.7%kg/kg",
    "SPHCC2": "Hardwood sapwood, CCA-treated 1.2%kg/kg",
    "SPHCC3": "Hardwood sapwood, CCA-treated 2.4%kg/kg",
    "SPHDBT": "Hardwood sapwood, double-treated CC2+CR2",
}
_OTHER_DESCRIPTIONS = {
    ("k_shelter", "EXP1"): "Calm or sheltered from wave",
    ("k_shelter", "EXP2"): "Exposed to strong wave or surf",
    ("k_protect", "MTN1"): "Floating collar",
    ("k_protect", "MTN2"): "None",
    ("k_contact", "CNT1"): "Contact with other timber member",
    ("k_contact", "CNT2"): "None",
    ("k_knot", "KNT1"): "Knots without protective plate",
    ("k_knot", "KNT2"): "Knots with protective plate",
    ("k_knot", "KNT3"): "None",
}


def coefficient_table() -> pd.DataFrame:
    """Every AS 5604 code with the number to type into its timber_assets column."""
    rows = [
        {
            "column": "k_heartwood" if code.startswith("HW") else "k_sapwood",
            "code": code,
            "meaning": _WOOD_DESCRIPTIONS[code],
            "value_zones_A_C": ac,
            "value_zones_D_G": dg,
        }
        for code, (ac, dg) in K_HW_SW.items()
    ]
    for (column, code), meaning in _OTHER_DESCRIPTIONS.items():
        table = {"k_shelter": K_SHELTER, "k_protect": K_PROTECT}.get(column) or (
            K_CONTACT if column == "k_contact" else K_KNOT
        )
        rows.append(
            {
                "column": column,
                "code": code,
                "meaning": meaning,
                "value_zones_A_C": table[code],
                "value_zones_D_G": table[code],
            }
        )
    return pd.DataFrame(rows)


def write_skeleton() -> None:
    """Blank CSVs + an .xlsx (README, example-filled sheets, coefficient lookup)."""
    SKELETON_DIR.mkdir(parents=True, exist_ok=True)
    for sheet, fields in SHEETS.items():
        pd.DataFrame(columns=[f.name for f in fields]).to_csv(
            SKELETON_DIR / f"{sheet}.csv", index=False
        )
    examples = _example_tables()
    for sheet, frame in examples.items():
        frame.to_csv(SKELETON_DIR / f"{sheet}.example.csv", index=False)
    coefficient_table().to_csv(SKELETON_DIR / "coefficients.csv", index=False)
    codes = pd.DataFrame(
        {
            "zone": pd.Series(ZONES),
            "scenario_label": pd.Series([*REQUIRED_SCENARIOS, "Baseline"]),
            "emissions_pathway": pd.Series(list(PATHWAYS)),
            "anchor_year": pd.Series(list(ANCHOR_YEARS)),
        }
    )
    with pd.ExcelWriter(SKELETON_XLSX, engine="openpyxl") as writer:
        _readme_rows().to_excel(writer, sheet_name="README", index=False)
        for sheet, frame in examples.items():
            frame.to_excel(writer, sheet_name=sheet, index=False)
        coefficient_table().to_excel(writer, sheet_name="coefficients", index=False)
        codes.to_excel(writer, sheet_name="allowed_codes", index=False)


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
        print(
            f"wrote {SKELETON_XLSX.relative_to(ROOT_DIR)} and {SKELETON_DIR.relative_to(ROOT_DIR)}/"
        )
    if args.validate:
        result = validate_import(read_import(args.validate))
        report = result.issues_frame()
        print(report.to_string(index=False) if not report.empty else "No issues.")
        print("VALID" if result.ok else f"INVALID: {len(result.errors)} error(s)")


if __name__ == "__main__":
    main()
