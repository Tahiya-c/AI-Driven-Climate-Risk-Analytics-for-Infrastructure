"""Write a 2-pile concrete import for trying the dashboard's Import page.

    python -m src.data.make_concrete_sample

Output (same content in both forms):
    data/samples/concrete_import_2_piles.xlsx       one workbook, 2 sheets
    data/samples/concrete_import_2_piles/*.csv      one CSV per sheet

PileH  copies PileB (atmospheric, w/c 0.37, 60 MPa) and its inspections every 4 years,
       so its forecast can be checked against the reference PileB on the dashboard.
PileI  a new splash-zone pile (w/c 0.45, 70 MPa); its inspections every 3 years are the
       mean of PileA (w/c 0.41) and PileG (w/c 0.50), which bracket it.

Inspection depths come from the reference workbook's Most Likely (A1FI, MRI-CGCM2.3.2)
series up to 2026. The climate sheet is left out on purpose: the importer then uses the
reference Port Kembla climate through 2070.
"""

from __future__ import annotations

import pandas as pd

from src.config import ROOT_DIR
from src.data.concrete_import import REFERENCE_CSV

OUT_DIR = ROOT_DIR / "data" / "samples"
XLSX = OUT_DIR / "concrete_import_2_piles.xlsx"
CSV_DIR = OUT_DIR / "concrete_import_2_piles"


def _history() -> pd.DataFrame:
    reference = pd.read_csv(REFERENCE_CSV)
    rows = reference.loc[
        (reference["emissions_pathway"] == "A1FI")
        & (reference["climate_model"] == "MRI-CGCM2.3.2")
        & (reference["year"] <= 2026)
    ]
    return rows.pivot(index="year", columns="asset_code", values="carb_depth_xc_mm")


def sample_tables() -> dict[str, pd.DataFrame]:
    history = _history()
    pile_h_years = list(range(2002, 2027, 4))
    pile_i_years = list(range(2002, 2027, 3))
    observations = pd.concat(
        [
            pd.DataFrame(
                {"asset_id": "PileH", "year": pile_h_years,
                 "carb_depth_mm": history.loc[pile_h_years, "PileB"].round(2).to_numpy()}
            ),
            pd.DataFrame(
                {"asset_id": "PileI", "year": pile_i_years,
                 "carb_depth_mm": history.loc[pile_i_years, ["PileA", "PileG"]]
                 .mean(axis=1).round(2).to_numpy()}
            ),
        ],
        ignore_index=True,
    )
    assets = pd.DataFrame(
        [
            {"asset_id": "PileH", "zone": "atmospheric", "wc_ratio": 0.37, "fc_mpa": 60.0},
            {"asset_id": "PileI", "zone": "splash", "wc_ratio": 0.45, "fc_mpa": 70.0},
        ]
    )
    return {"concrete_assets": assets, "concrete_observations": observations}


def main() -> None:
    tables = sample_tables()
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(XLSX, engine="openpyxl") as writer:
        for sheet, frame in tables.items():
            frame.to_excel(writer, sheet_name=sheet, index=False)
            frame.to_csv(CSV_DIR / f"{sheet}.csv", index=False)
    print(f"wrote {XLSX.relative_to(ROOT_DIR)} and {CSV_DIR.relative_to(ROOT_DIR)}/")


if __name__ == "__main__":
    main()
