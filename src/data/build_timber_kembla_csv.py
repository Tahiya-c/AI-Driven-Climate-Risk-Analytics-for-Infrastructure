"""Consolidate Timber_kembla_v6.3.xlsm into one tidy timber CSV.

The source of truth: two per-asset worksheets, PileA and PileB. 
The workbook contains descriptive pile properties, environmental inputs, timber
deterioration factors, and calculated deterioration outputs in widely
separated Excel columns. Only rows with a numeric projection year are treated
as data; banner, header, and separator rows are skipped.
"""

import argparse
from pathlib import Path

import openpyxl
import pandas as pd

PILES = ["PileA", "PileB"]

# One-based Excel column number -> output CSV column name.
COLMAP = {
    # Scenario and environmental drivers
    2: "emissions_pathway",  # B: Scenario (Base / A1FI / A1B)
    3: "climate_model",  # C: Model
    4: "year",  # D: Year
    5: "salinity",  # E: Salinity
    6: "summer_sea_temp_c",  # F: Summer sea temperature (C)
    # Pile properties
    7: "year_installed",  # G
    8: "description",  # H
    9: "timber_type",  # I: Type
    10: "heartwood_class",  # J: HW
    11: "sapwood_of_hardwood",  # K: SW
    12: "initial_diameter_mm",  # L: D_o
    13: "layer_of_sapwood_mm",  # M: d_sap
    14: "replacement_diameter_mm",  # N: D_replace
    15: "maintenance_measure",  # O: Maintn
    16: "contact",  # P: Contact
    17: "knot_presence",  # Q: Knot
    18: "zone",  # R
    19: "exposure",  # S
    # Timber deterioration model inputs and factors
    39: "design_time_years",  # AM: t_design
    40: "model_salinity_ppt",  # AN: Salinity
    # Names follow the workbook's own row-7 headers (Defect D9: earlier exports
    # labelled AP-AY one column out of step, e.g. k_shelter as "salinity_factor").
    41: "water_factor",  # AO: kwater
    42: "heartwood_factor",  # AP: ktimber HW
    43: "sapwood_factor",  # AQ: ktimber SW
    44: "salinity_factor",  # AR: ksal
    45: "shelter_factor",  # AS: kshelter
    46: "protection_factor",  # AT: kprotect
    47: "contact_factor",  # AU: kcontact
    48: "knot_factor",  # AV: kknot
    49: "environment_factor",  # AW: kenv = ksal * kshelter
    50: "construction_factor",  # AX: kconst = kprotect * kcontact * kknot
    51: "combined_factor",  # AY: k0 = product of all k factors except kwater/kSW
    # Calculated deterioration outputs
    52: "sapwood_rate_mm_year",  # AZ: r_sap
    53: "sapwood_lag_years",  # BA: t_lag,sap
    54: "heartwood_rate_mm_year",  # BB: r_heart
    55: "heartwood_lag_years",  # BC: t_lag,heart
    56: "sapwood_duration_years",  # BD: t_sap
    57: "mean_deterioration_mm",  # BE: d_mean
    58: "remaining_diameter_mm",  # BF: D_t
    59: "failure_probability",  # BG: Prob
    60: "design_deterioration_mm",  # BH: d_design
    61: "design_diameter_mm",  # BI: D_design
    62: "design_life_years",  # BJ: t_replacement (re-estimated design life, D9)
}

SCENARIO_LABEL = {
    "(Reference)": "Baseline",
    "MRI-CGCM2.3.2": "Most Likely",
    "CSIRO Mk3.5": "Hotter & Drier",
    "MIROC(hires)": "Colder & Wetter",
}

MODEL_TYPOS = {"CSIRO Mk3.6": "CSIRO Mk3.5"}
MAX_COL = max(COLMAP)


def _is_projection_row(row: tuple[object, ...]) -> bool:
    """Return True only when Excel column D contains a numeric year."""
    year = pd.to_numeric(row[3], errors="coerce")
    return not pd.isna(year)


def main(src: Path, out: Path) -> pd.DataFrame:
    """Read both pile sheets, consolidate their projections, and write a CSV."""
    workbook = openpyxl.load_workbook(src, read_only=True, data_only=True)
    records: list[dict[str, object]] = []
    fixes: list[tuple[str, int, object, object]] = []

    missing_sheets = [sheet for sheet in PILES if sheet not in workbook.sheetnames]
    if missing_sheets:
        raise ValueError(f"Missing required worksheets: {', '.join(missing_sheets)}")

    for sheet in PILES:
        worksheet = workbook[sheet]
        for row_index, row in enumerate(
            worksheet.iter_rows(max_col=MAX_COL, values_only=True), start=1
        ):
            if not _is_projection_row(row):
                continue

            record = {"asset_code": sheet, "source_row": row_index}
            for column_number, output_name in COLMAP.items():
                record[output_name] = row[column_number - 1]

            model = record["climate_model"]
            if model in MODEL_TYPOS:
                corrected_model = MODEL_TYPOS[model]
                fixes.append((sheet, row_index, model, corrected_model))
                record["climate_model"] = corrected_model

            record["scenario_label"] = SCENARIO_LABEL.get(record["climate_model"])
            records.append(record)

    if not records:
        raise ValueError("No timber projection rows with a numeric year were found.")

    frame = pd.DataFrame(records)

    for column in ("year", "year_installed"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("Int64")
    frame["age_years"] = frame["year"] - frame["year_installed"]

    numeric_columns = [
        "salinity",
        "summer_sea_temp_c",
        "initial_diameter_mm",
        "layer_of_sapwood_mm",
        "replacement_diameter_mm",
        "design_time_years",
        "model_salinity_ppt",
        "water_factor",
        "heartwood_factor",
        "sapwood_factor",
        "salinity_factor",
        "shelter_factor",
        "protection_factor",
        "contact_factor",
        "knot_factor",
        "environment_factor",
        "construction_factor",
        "combined_factor",
        "sapwood_rate_mm_year",
        "sapwood_lag_years",
        "heartwood_rate_mm_year",
        "heartwood_lag_years",
        "sapwood_duration_years",
        "mean_deterioration_mm",
        "remaining_diameter_mm",
        "failure_probability",
        "design_deterioration_mm",
        "design_diameter_mm",
        "design_life_years",
    ]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    text_columns = [
        "description",
        "timber_type",
        "heartwood_class",
        "sapwood_of_hardwood",
        "maintenance_measure",
        "contact",
        "knot_presence",
        "zone",
        "exposure",
    ]
    for column in text_columns:
        frame[column] = frame[column].astype("string").str.strip()

    output_order = [
        "asset_code",
        "description",
        "timber_type",
        "year_installed",
        "zone",
        "exposure",
        "heartwood_class",
        "sapwood_of_hardwood",
        "maintenance_measure",
        "contact",
        "knot_presence",
        "initial_diameter_mm",
        "layer_of_sapwood_mm",
        "replacement_diameter_mm",
        "emissions_pathway",
        "climate_model",
        "scenario_label",
        "year",
        "age_years",
        "salinity",
        "summer_sea_temp_c",
        "design_time_years",
        "model_salinity_ppt",
        "water_factor",
        "heartwood_factor",
        "sapwood_factor",
        "salinity_factor",
        "shelter_factor",
        "protection_factor",
        "contact_factor",
        "knot_factor",
        "environment_factor",
        "construction_factor",
        "combined_factor",
        "sapwood_rate_mm_year",
        "sapwood_lag_years",
        "heartwood_rate_mm_year",
        "heartwood_lag_years",
        "sapwood_duration_years",
        "mean_deterioration_mm",
        "remaining_diameter_mm",
        "failure_probability",
        "design_deterioration_mm",
        "design_diameter_mm",
        "design_life_years",
        "source_row",
    ]
    frame = (
        frame[output_order]
        .sort_values(["asset_code", "emissions_pathway", "climate_model", "year"])
        .reset_index(drop=True)
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    print(f"wrote {out}")
    print("rows:", len(frame), "| cols:", frame.shape[1])
    print("\nassets:", sorted(frame.asset_code.dropna().unique()))
    print("pathways:", sorted(frame.emissions_pathway.dropna().unique()))
    print("models:", sorted(frame.climate_model.dropna().unique()))
    print("years:", frame.year.min(), "-", frame.year.max())
    print("\nrows per pathway x model:")
    print(frame.groupby(["emissions_pathway", "climate_model"]).size())
    print("\nmodel-name fixes applied:", len(fixes))
    print("\nnull counts (non-zero only):")
    null_counts = frame.isna().sum()
    print(null_counts[null_counts > 0])
    return frame


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Consolidate Timber_kembla_v6.3.xlsm into a tidy CSV."
    )
    parser.add_argument("source", type=Path, help="path to Timber_kembla_v6.3.xlsm")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("data/processed/timber_kembla_v6_3_consolidated.csv"),
        help="output CSV path (default: data/processed/timber_kembla_v6_3_consolidated.csv)",
    )
    arguments = parser.parse_args()
    if not arguments.source.exists():
        raise SystemExit(f"source file not found: {arguments.source}")
    main(arguments.source, arguments.output)
