"""
Write the concrete dashboard CSVs in the frontend's format (one file per model).

    python -m src.models.build_concrete_dashboard

Output: data/dashboard/dashboard_data_concrete_<model>.csv  for model in {rnn, ridge}
Columns: material, asset_id, scenario, year, value, unit, metric, method
  scenario : friendly label (Hotter & Drier / Colder & Wetter / Most Likely), A1FI pathway
  year     : 2001-2070
  value    : carbonation depth (mm); years <= 2026 are the workbook hindcast,
             years > 2026 are the model's autoregressive forecast
  method   : the model name -- the frontend splits history/forecast at the current year

RNN vs Ridge = best vs worst from src/models/concrete_benchmark_all.py, so the
concrete page shows a visible model comparison (mirrors the timber GBM/MLP page).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

try:
    from . import concrete_eval as ce
except ImportError:
    import concrete_eval as ce

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed" / "concrete_kembla_v6_consolidated.csv"
OUT_DIR = ROOT / "data" / "dashboard"

MODELS = ("rnn", "ridge")
PATHWAY = "A1FI"
GCM_LABEL = {
    "CSIRO Mk3.5": "Hotter & Drier",
    "MRI-CGCM2.3.2": "Most Likely",
    "MIROC(hires)": "Colder & Wetter",
}
COLUMNS = ["material", "asset_id", "scenario", "year", "value", "unit", "metric", "method"]


def build(model_key: str) -> pd.DataFrame:
    series, _ = ce.load_series(DATA)
    model = ce.make_model(model_key)
    if not model.available:
        raise SystemExit(f"{model_key} is not installed -- pip install -r requirements.txt")
    print(f"fitting {model_key} on {len(series)} series (<= {ce.CUTOFF_YEAR}) ...")
    model.fit(series, ce.CUTOFF_YEAR)

    rows: list[tuple] = []
    for s in series:
        pathway, gcm = s["scenario"].split("|")
        if pathway != PATHWAY or gcm not in GCM_LABEL:
            continue
        label = GCM_LABEL[gcm]
        years = ce.series_years(s)
        cut = ce.cut_index(s, ce.CUTOFF_YEAR)
        for yr, depth in zip(years[:cut], s["depth"][:cut]):
            rows.append(("concrete", s["asset"], label, int(yr), round(float(depth), 4),
                         "mm", "carbonation_depth", model_key))
        f_years, f_depth = ce.rollout(model, s, ce.CUTOFF_YEAR)
        for yr, depth in zip(f_years, f_depth):
            rows.append(("concrete", s["asset"], label, int(yr), round(float(depth), 4),
                         "mm", "carbonation_depth", model_key))
    return pd.DataFrame(rows, columns=COLUMNS)


def main() -> None:
    if not DATA.exists():
        raise SystemExit(f"{DATA} not found -- pull main.")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for key in MODELS:
        df = build(key).sort_values(["asset_id", "scenario", "year"]).reset_index(drop=True)
        out = OUT_DIR / f"dashboard_data_concrete_{key}.csv"
        df.to_csv(out, index=False)
        print(f"wrote {len(df)} rows -> {out}")


if __name__ == "__main__":
    main()
