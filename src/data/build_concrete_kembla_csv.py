"""
Consolidate Concrete_kembla_v6.xlsm -> single tidy CSV for ML training.

Source of truth: the 7 per-asset sheets PileA..PileG.
Excluded: Mastersheet (template/scratch, inputs do not match InputData),
          Baseline / 3 GCM sheets (already propagated into pile columns E,F,G),
          Coefficients / Chloride Model / Design service life / Port Kembla Cement
          (reference lookups, not row data).

Layout of each pile sheet:
  rows 22-24 = 3-tier header, row 25 = first banner
  data rows 26..528, split into 7 scenario blocks by banner rows
  (banner = value in col B, nothing in col C/D)
"""
import argparse
from pathlib import Path

import openpyxl
import pandas as pd

PILES = ["PileA", "PileB", "PileC", "PileD", "PileE", "PileF", "PileG"]

# 1-based column index in the pile sheets -> output column name
COLMAP = {
    # --- identifiers / scenario ---
    2:  "emissions_pathway",        # B  Scenario  (Base / A1FI / A1B)
    3:  "climate_model",            # C  Model
    4:  "year",                     # D  Year
    # --- climate drivers ---
    5:  "mean_temp_c",              # E
    6:  "mean_rh_pct",              # F
    7:  "co2_ppm",                  # G
    # --- asset attributes ---
    8:  "year_built",               # H
    9:  "description",              # I
    10: "zone",                     # J
    11: "dist_from_coast_km",       # K
    12: "exposure_class",           # L
    13: "carbonation_class",        # M
    14: "chloride_class",           # N
    15: "protection",               # O
    16: "cover_h_mm",               # P
    17: "member_diameter_mm",       # Q
    18: "fc_mpa",                   # R
    19: "wc_ratio",                 # S
    20: "cement_content_kgm3",      # T
    21: "rebar_diameter_mm",        # U
    # --- carbonation outputs ---
    56: "carb_depth_xc_mm",         # BD  xc(t')
    58: "carb_depth_increment_mm",  # BF  delta xc(t')  <- annual deterioration rate
    57: "carb_initiation_prob",     # BE
    62: "carb_icorr_ua_cm2",        # BJ  corrosion rate
    # --- chloride model inputs that vary by asset/zone ---
    89: "surface_chloride_co_kgm3",  # CK
    91: "chloride_diffusion_m2yr",   # CM
    # --- chloride outputs ---
    94: "chloride_c_at_xc_kgm3",    # CP  C(xc,t)
    95: "chloride_c_at_cover_kgm3", # CQ  C(h,t)
    97: "chloride_initiation_prob", # CS
    100: "chloride_icorr_ua_cm2",   # CV
    # --- combined ---
    133: "combined_initiation_prob",  # EC  "Combo"
}

# The workbook's own sheet names give the scenario semantics.
SCENARIO_LABEL = {
    "(Reference)":    "Baseline",
    "MRI-CGCM2.3.2":  "Most Likely",
    "CSIRO Mk3.5":    "Hotter & Drier",
    "MIROC(hires)":   "Colder & Wetter",
}

# Known defect: one row is labelled Mk3.6 in every pile sheet.
MODEL_TYPOS = {"CSIRO Mk3.6": "CSIRO Mk3.5"}

MAX_COL = max(COLMAP)
DATA_START, DATA_END = 26, 528


def main(src: Path, out: Path):
    wb = openpyxl.load_workbook(src, read_only=True, data_only=True)
    records = []
    fixes = []

    for sheet in PILES:
        ws = wb[sheet]
        for r_i, row in enumerate(
            ws.iter_rows(min_row=DATA_START, max_row=DATA_END,
                         max_col=MAX_COL, values_only=True),
            DATA_START,
        ):
            # banner rows carry a label in B but no Model/Year -> skip
            if row[2] is None and row[3] is None:
                continue
            if row[3] is None:          # no Year -> not a data row
                continue

            rec = {"asset_code": sheet, "source_row": r_i}
            for c, name in COLMAP.items():
                rec[name] = row[c - 1]

            model = rec["climate_model"]
            if model in MODEL_TYPOS:
                fixes.append((sheet, r_i, model, MODEL_TYPOS[model]))
                rec["climate_model"] = MODEL_TYPOS[model]

            rec["scenario_label"] = SCENARIO_LABEL.get(rec["climate_model"])
            records.append(rec)

    df = pd.DataFrame(records)

    # types
    for col in ("year", "year_built"):
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    df["age_years"] = df["year"] - df["year_built"]

    numeric = [
        "mean_temp_c", "mean_rh_pct", "co2_ppm", "dist_from_coast_km",
        "cover_h_mm", "member_diameter_mm", "fc_mpa", "wc_ratio",
        "cement_content_kgm3", "rebar_diameter_mm", "carb_depth_xc_mm",
        "carb_depth_increment_mm", "carb_initiation_prob", "carb_icorr_ua_cm2",
        "surface_chloride_co_kgm3", "chloride_diffusion_m2yr",
        "chloride_c_at_xc_kgm3", "chloride_c_at_cover_kgm3",
        "chloride_initiation_prob", "chloride_icorr_ua_cm2",
        "combined_initiation_prob",
    ]
    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ("zone", "exposure_class", "carbonation_class",
                "chloride_class", "protection", "description"):
        df[col] = df[col].astype(str).str.strip()

    # protection is 'NONE' in every pile sheet; keep as-is, don't invent a code
    order = [
        "asset_code", "description", "year_built", "zone", "dist_from_coast_km",
        "exposure_class", "carbonation_class", "chloride_class", "protection",
        "cover_h_mm", "member_diameter_mm", "fc_mpa", "wc_ratio",
        "cement_content_kgm3", "rebar_diameter_mm",
        "emissions_pathway", "climate_model", "scenario_label",
        "year", "age_years", "mean_temp_c", "mean_rh_pct", "co2_ppm",
        "surface_chloride_co_kgm3", "chloride_diffusion_m2yr",
        "carb_depth_xc_mm", "carb_depth_increment_mm", "carb_initiation_prob",
        "carb_icorr_ua_cm2", "chloride_c_at_xc_kgm3", "chloride_c_at_cover_kgm3",
        "chloride_initiation_prob", "chloride_icorr_ua_cm2",
        "combined_initiation_prob", "source_row",
    ]
    df = df[order].sort_values(
        ["asset_code", "emissions_pathway", "climate_model", "year"]
    ).reset_index(drop=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"wrote {out}")

    # ---- report ----
    print("rows:", len(df), "| cols:", df.shape[1])
    print("\nassets:", sorted(df.asset_code.unique()))
    print("pathways:", sorted(df.emissions_pathway.unique()))
    print("models:", sorted(df.climate_model.unique()))
    print("years:", df.year.min(), "-", df.year.max())
    print("\nrows per asset x pathway x model:")
    print(df.groupby(["emissions_pathway", "climate_model"]).size())
    print("\nmodel-name fixes applied:", len(fixes))
    print("\nnull counts (non-zero only):")
    n = df.isna().sum()
    print(n[n > 0])
    return df


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Consolidate Concrete_kembla_v6.xlsm into a tidy CSV."
    )
    p.add_argument("source", type=Path, help="path to Concrete_kembla_v6.xlsm")
    p.add_argument(
        "-o", "--output", type=Path,
        default=Path("data/processed/concrete_kembla_v6_consolidated.csv"),
        help="output CSV path (default: data/processed/...)",
    )
    a = p.parse_args()
    if not a.source.exists():
        raise SystemExit(f"source file not found: {a.source}")
    main(a.source, a.output)
