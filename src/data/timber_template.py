"""Generalised input-data template: shared scenario/climate tables + timber-specific tables.

    python -m src.data.timber_template --export   # write data/templates/ from the processed data
    python -m src.data.timber_template --check    # rebuild the model-ready frame and compare

The template splits the data into four tables joined by keys (layout and column
rules: docs/input_data_template.md):

    scenarios.csv              shared   scenario_id -> emissions_pathway, scenario_label
    climate.csv                shared   (site, scenario_id, year) -> climate drivers
    timber_assets.csv          timber   asset_id -> pile attributes and codes
    timber_observations.csv    timber   (asset_id, scenario_id, year) -> remaining diameter

``assemble_timber_frame`` joins them into the model-ready frame the timber pipeline uses
(FEATURE_COLUMNS + target). Everything the models need that the workbook calculates --
water_factor and every k factor -- is derived here from the codes and climate, so a user
only ever supplies plain inputs. Concrete can reuse scenarios.csv and climate.csv as-is
and add its own concrete_assets.csv / concrete_observations.csv.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import ROOT_DIR
from src.features.timber_features import FEATURE_COLUMNS, TARGET_COLUMN
from src.models.timber_coefficients import (
    K_CONTACT,
    K_HW_SW,
    K_KNOT,
    K_PROTECT,
    K_SHELTER,
    lookup_approx,
    lookup_hw_sw,
    lookup_salinity,
)

TEMPLATE_DIR = ROOT_DIR / "data" / "templates"
KEMBLA_DIR = TEMPLATE_DIR / "port_kembla"

SCENARIO_COLUMNS = ["scenario_id", "emissions_pathway", "scenario_label"]
CLIMATE_COLUMNS = ["site", "scenario_id", "climate_model", "year"]  # + driver columns
TIMBER_CLIMATE_DRIVERS = ["summer_sea_temp_c", "salinity_ppt"]
TIMBER_ASSET_COLUMNS = [
    "asset_id",
    "site",
    "description",
    "timber_type",
    "year_installed",
    "zone",
    "exposure",
    "heartwood_class",
    "sapwood_class",
    "maintenance",
    "contact",
    "knot",
    "initial_diameter_mm",
    "sapwood_layer_mm",
    "replacement_diameter_mm",
]
# The 6 code columns above can instead be given as their coefficients (the import does
# this): coefficient column -> the model's factor column. k_sal still comes from zone.
K_INPUT_COLUMNS = {
    "k_heartwood": "heartwood_factor",
    "k_sapwood": "sapwood_factor",
    "k_shelter": "shelter_factor",
    "k_protect": "protection_factor",
    "k_contact": "contact_factor",
    "k_knot": "knot_factor",
}
_CODE_COLUMNS = ["exposure", "heartwood_class", "sapwood_class", "maintenance", "contact", "knot"]
TIMBER_ASSET_COLUMNS_WITH_K = [
    *(c for c in TIMBER_ASSET_COLUMNS if c not in _CODE_COLUMNS),
    *K_INPUT_COLUMNS,
]
TIMBER_OBSERVATION_COLUMNS = ["asset_id", "scenario_id", "year", "remaining_diameter_mm", "source"]


def scenario_id(emissions_pathway: str, scenario_label: str) -> str:
    """e.g. ("A1B", "Most Likely") -> "A1B-most_likely"; Baseline -> "BASE-baseline"."""
    pathway = "BASE" if emissions_pathway == "Base" else emissions_pathway
    label = scenario_label.lower().replace(" & ", "_").replace(" ", "_")
    return f"{pathway}-{label}"


def validate(table: pd.DataFrame, required: list[str], name: str, key: list[str]) -> None:
    missing = sorted(set(required) - set(table.columns))
    if missing:
        raise ValueError(f"{name}: missing columns {missing}")
    empty = [column for column in required if table[column].isna().any()]
    if empty:
        raise ValueError(f"{name}: blank values in required columns {empty}")
    duplicated = table.duplicated(key)
    if duplicated.any():
        raise ValueError(f"{name}: {int(duplicated.sum())} duplicate rows for key {key}")


def _code_lookup(code: str, table: dict[str, float], name: str, allow_unknown: bool) -> float:
    if code in table or allow_unknown:
        return lookup_approx(code, table, name)
    raise ValueError(
        f"Unknown {name} code {code!r}; valid codes: {sorted(table)}. "
        "Pass allow_unknown_codes=True to reproduce the workbook's approximate match "
        "(Defect D3) instead."
    )


def derive_timber_factors(
    assets: pd.DataFrame, climate: pd.DataFrame, allow_unknown_codes: bool = False
) -> pd.DataFrame:
    """Asset x climate rows with water_factor and every k factor, as the workbook computes them.

    The asset k factors come from the coefficient columns (``K_INPUT_COLUMNS``) when the
    assets give them, otherwise from looking up the codes in the Coefficients tables.
    """
    rows = assets.merge(climate, on="site", how="inner")  # every asset x its site's climate
    rows["water_factor"] = 0.1 * np.exp(0.13 * rows["summer_sea_temp_c"])
    rows["salinity_factor"] = [
        lookup_salinity(s, z) for s, z in zip(rows["salinity_ppt"], rows["zone"])
    ]
    if set(K_INPUT_COLUMNS) <= set(rows.columns):
        for k_column, factor_column in K_INPUT_COLUMNS.items():
            rows[factor_column] = rows[k_column].astype(float)
    else:
        for column, code_table in (("heartwood_class", K_HW_SW), ("sapwood_class", K_HW_SW)):
            unknown = sorted(set(rows[column]) - set(code_table))
            if unknown:
                raise ValueError(f"Unknown {column} codes {unknown}; valid: {sorted(code_table)}")
        rows["heartwood_factor"] = [
            lookup_hw_sw(c, z) for c, z in zip(rows["heartwood_class"], rows["zone"])
        ]
        rows["sapwood_factor"] = [
            lookup_hw_sw(c, z) for c, z in zip(rows["sapwood_class"], rows["zone"])
        ]
        for column, code_column, code_table, name in (
            ("shelter_factor", "exposure", K_SHELTER, "K_SHELTER"),
            ("protection_factor", "maintenance", K_PROTECT, "K_PROTECT"),
            ("contact_factor", "contact", K_CONTACT, "K_CONTACT"),
            ("knot_factor", "knot", K_KNOT, "K_KNOT"),
        ):
            rows[column] = [
                _code_lookup(code, code_table, name, allow_unknown_codes)
                for code in rows[code_column]
            ]
    rows["environment_factor"] = rows["salinity_factor"] * rows["shelter_factor"]
    rows["construction_factor"] = (
        rows["protection_factor"] * rows["contact_factor"] * rows["knot_factor"]
    )
    rows["combined_factor"] = (
        rows["heartwood_factor"] * rows["environment_factor"] * rows["construction_factor"]
    )
    return rows


def assemble_timber_frame(
    assets: pd.DataFrame,
    climate: pd.DataFrame,
    scenarios: pd.DataFrame,
    observations: pd.DataFrame | None = None,
    allow_unknown_codes: bool = False,
) -> pd.DataFrame:
    """Join the template tables into the frame the timber models use.

    Without ``observations`` the result has every feature but no target -- that is
    the input for projecting a pile under each climate scenario.
    """
    validate(scenarios, SCENARIO_COLUMNS, "scenarios", ["scenario_id"])
    validate(
        climate,
        CLIMATE_COLUMNS + TIMBER_CLIMATE_DRIVERS,
        "climate",
        ["site", "scenario_id", "year"],
    )
    asset_columns = (
        TIMBER_ASSET_COLUMNS_WITH_K if set(K_INPUT_COLUMNS) <= set(assets) else TIMBER_ASSET_COLUMNS
    )
    validate(assets, asset_columns, "timber_assets", ["asset_id"])
    unknown = sorted(set(climate["scenario_id"]) - set(scenarios["scenario_id"]))
    if unknown:
        raise ValueError(f"climate: scenario_id not in scenarios table: {unknown}")

    climate = climate.merge(scenarios, on="scenario_id", how="left")
    frame = derive_timber_factors(assets, climate, allow_unknown_codes)
    frame["salinity"] = frame["salinity_ppt"]
    frame["age_years"] = frame["year"] - frame["year_installed"]
    frame = frame.rename(columns={"asset_id": "asset_code", "sapwood_layer_mm": "layer_of_sapwood_mm"})

    if observations is not None:
        validate(
            observations,
            TIMBER_OBSERVATION_COLUMNS[:4],
            "timber_observations",
            ["asset_id", "scenario_id", "year"],
        )
        frame = frame.merge(
            observations.rename(columns={"asset_id": "asset_code"}),
            on=["asset_code", "scenario_id", "year"],
            how="left",
        )
    keep = [
        "asset_code",
        "site",
        "scenario_id",
        "emissions_pathway",
        "climate_model",
        "scenario_label",
        "year",
        *FEATURE_COLUMNS,
        "layer_of_sapwood_mm",  # pile attribute the transfer models need (sapwood loss)
        "replacement_diameter_mm",
    ]
    if observations is not None:
        keep.append(TARGET_COLUMN)
    return frame[list(dict.fromkeys(keep))].reset_index(drop=True)


def load_template_dir(directory: Path = KEMBLA_DIR, allow_unknown_codes: bool = False):
    """Read a filled template folder and assemble it."""
    observations_path = directory / "timber_observations.csv"
    return assemble_timber_frame(
        assets=pd.read_csv(directory / "timber_assets.csv"),
        climate=pd.read_csv(directory / "climate.csv"),
        scenarios=pd.read_csv(TEMPLATE_DIR / "scenarios.csv"),
        observations=pd.read_csv(observations_path) if observations_path.exists() else None,
        allow_unknown_codes=allow_unknown_codes,
    )


def export_kembla_template(out_dir: Path = KEMBLA_DIR) -> dict[str, pd.DataFrame]:
    """Write the Port Kembla data (timber + concrete climate) in template form."""
    from src.data.timber_data_prep import DATA_PATH, load_timber_data

    timber = load_timber_data(DATA_PATH)
    timber["scenario_id"] = [
        scenario_id(p, s) for p, s in zip(timber["emissions_pathway"], timber["scenario_label"])
    ]
    timber["site"] = "Port Kembla"

    scenarios = (
        timber[SCENARIO_COLUMNS].drop_duplicates().sort_values("scenario_id").reset_index(drop=True)
    )
    climate = (
        timber[["site", "scenario_id", "climate_model", "year", "summer_sea_temp_c", "salinity"]]
        .rename(columns={"salinity": "salinity_ppt"})
        .drop_duplicates(["site", "scenario_id", "year"])
    )
    concrete_path = ROOT_DIR / "data" / "processed" / "concrete_kembla_v6_consolidated.csv"
    if concrete_path.exists():
        concrete = pd.read_csv(concrete_path)
        concrete["scenario_id"] = [
            scenario_id(p, s)
            for p, s in zip(concrete["emissions_pathway"], concrete["scenario_label"])
        ]
        air = concrete.drop_duplicates(["scenario_id", "year"])[
            ["scenario_id", "year", "mean_temp_c", "mean_rh_pct", "co2_ppm"]
        ].rename(columns={"mean_temp_c": "mean_air_temp_c"})
        climate = climate.merge(air, on=["scenario_id", "year"], how="left")
    climate = climate.sort_values(["scenario_id", "year"]).reset_index(drop=True)

    assets = (
        timber.drop_duplicates("asset_code")
        .rename(
            columns={
                "asset_code": "asset_id",
                "sapwood_of_hardwood": "sapwood_class",
                "maintenance_measure": "maintenance",
                "knot_presence": "knot",
                "layer_of_sapwood_mm": "sapwood_layer_mm",
            }
        )[TIMBER_ASSET_COLUMNS]
        .reset_index(drop=True)
    )
    observations = (
        timber.rename(columns={"asset_code": "asset_id"})
        .assign(source="simulated: Timber_kembla_v6.3 workbook")[TIMBER_OBSERVATION_COLUMNS]
        .sort_values(["asset_id", "scenario_id", "year"])
        .reset_index(drop=True)
    )

    tables = {
        "scenarios": scenarios,
        "climate": climate,
        "timber_assets": assets,
        "timber_observations": observations,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    scenarios.to_csv(TEMPLATE_DIR / "scenarios.csv", index=False)
    for name in ("climate", "timber_assets", "timber_observations"):
        tables[name].to_csv(out_dir / f"{name}.csv", index=False)
    return tables


def check_round_trip(directory: Path = KEMBLA_DIR) -> pd.DataFrame:
    """Max abs difference between the template-assembled frame and the processed CSV."""
    from src.data.timber_data_prep import DATA_PATH, load_timber_data

    original = load_timber_data(DATA_PATH)
    original["scenario_id"] = [
        scenario_id(p, s) for p, s in zip(original["emissions_pathway"], original["scenario_label"])
    ]
    rebuilt = load_template_dir(directory, allow_unknown_codes=True)
    merged = original.merge(
        rebuilt, on=["asset_code", "scenario_id", "year"], suffixes=("", "_template")
    )
    if len(merged) != len(original):
        raise ValueError(f"Row mismatch: {len(original)} original vs {len(merged)} matched")
    columns = [*FEATURE_COLUMNS, "replacement_diameter_mm", TARGET_COLUMN]
    return pd.DataFrame(
        {
            "column": columns,
            "max_abs_diff": [
                float((merged[c] - merged[f"{c}_template"]).abs().max()) for c in columns
            ],
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if not (args.export or args.check):
        parser.error("pass --export and/or --check")
    if args.export:
        for name, table in export_kembla_template().items():
            print(f"{name}: {len(table)} rows, columns {list(table.columns)}")
    if args.check:
        report = check_round_trip()
        print(report.to_string(index=False))
        worst = report["max_abs_diff"].max()
        print("ROUND TRIP OK" if math.isclose(worst, 0, abs_tol=1e-9) else f"MISMATCH {worst}")


if __name__ == "__main__":
    main()
