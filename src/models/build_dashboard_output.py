"""
Generate the dashboard-facing CSV: one row per (material, asset, scenario, year).

The frontend reads this file directly (pd.read_csv + filter) -- there is no live
inference API. Schema, fixed with the frontend team:

    material   asset_id   scenario   year   value   unit   metric   method

    concrete   PileA      A1FI|CSIRO Mk3.5   2055   4.91   mm   carbonation_depth   rnn
    timber     PileA      A1FI|CSIRO Mk3.5   2055   4.91   mm   deterioration_depth mlp

Notes
-----
* Concrete and timber are built by separate functions (``build_concrete``,
  ``build_timber_transfer``) against their own models, and written to
  separate ``dashboard_data_<material>.csv`` files -- same schema, no frontend
  change either way. Select which with ``--material``.
* ``scenario`` is the full "<emissions_pathway>|<climate_model>" id -- all 7 series
  the workbook actually contains. Collapsing these to 3 chart lines is an open
  question for Dr. Daniel; do it in the frontend once decided, not here.
* Concrete: with the default 21-year horizon the projected rows cover 2050-2070.
  For timber, the "observed" rows are the deterministic AS 5604-style workbook's own
  output, not real field inspection data (see ``src/data/timber_data_prep.py``) --
  reported accuracy is surrogate-vs-simulator fidelity, not validated accuracy.
* ``--with-history`` also emits the observed depths (method="observed") so the
  frontend can draw the history/forecast split from a single file.
* ``--material timber`` builds the files the dashboard's Timber page reads
  (``dashboard_data_timber_pileB_from_pileA_{gbm,mlp}.csv``) from the final models in
  ``src/models/timber_transfer.py``: trained on PileA only, forecasting PileB. Rows up to
  ``timber_transfer.SPLIT_YEAR`` (2026) are PileB's own values (method="observed", always
  included); later rows are each model's prediction for the 3 climate scenarios (Most
  Likely, Hotter & Drier, Colder & Wetter; no Baseline). Only one emissions pathway is
  written, ``--pathway`` (default A1FI), so the dashboard draws exactly 3 forecast lines
  instead of averaging A1B and A1FI. Value = radial material loss in mm, capped at the
  pile radius. By default each forecast is shifted by the model's mean error against
  PileB's real 2022-2026 values (offsets written to ``<file>.shift.json``);
  ``--no-shift`` writes the raw model forecast. See docs/timber_model_updated.md.

Usage
-----
    python src/models/build_dashboard_output.py
    python src/models/build_dashboard_output.py --with-history --horizon 21
    python src/models/build_dashboard_output.py --method wc_regress --out data/dashboard/foo.csv
    python -m src.models.build_dashboard_output --material timber                 # gbm + mlp
    python -m src.models.build_dashboard_output --material timber --method gbm
    python -m src.models.build_dashboard_output --material timber --pathway A1B
    python -m src.models.build_dashboard_output --material timber --no-shift      # raw forecast
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

try:
    from . import inference
except ImportError:
    try:
        import inference
    except ImportError:
        # The concrete team's own src/models/inference.py isn't in this checkout yet
        # (see build_concrete). Deferred so `--material timber` still works without it.
        inference = None

from src.config import ROOT_DIR

SCHEMA = ["material", "asset_id", "scenario", "year", "value", "unit", "metric", "method"]
DEFAULT_OUT = (
    inference.ROOT / "data" / "dashboard" / "dashboard_data_concrete.csv" if inference else None
)
# Timber rows: radial material loss from the saved PileA-trained models (timber_transfer).
TIMBER_MATERIAL, TIMBER_UNIT, TIMBER_METRIC = "timber", "mm", "deterioration_depth"
TRANSFER_METHODS = ("gbm", "mlp")
TRANSFER_PATHWAYS = ("A1FI", "A1B")
TRANSFER_DEFAULT_OUT = {
    method: ROOT_DIR
    / "data"
    / "dashboard"
    / f"dashboard_data_timber_pileB_from_pileA_{method}.csv"
    for method in TRANSFER_METHODS
}

MATERIAL_DEFAULT_METHOD = {"concrete": "rnn", "timber": "mlp"}


def build_concrete(horizon: int, method: str, with_history: bool) -> pd.DataFrame:
    if inference is None:
        raise ModuleNotFoundError(
            "src/models/inference.py is not in this checkout yet -- pull the concrete "
            "team's inference module before building the concrete dashboard CSV."
        )
    df = inference.predict_batch(horizon=horizon, method=method)

    if with_history:
        first_projected = int(df["year"].min())
        hist_rows = []
        for asset in inference.list_assets():
            for scenario in inference.list_scenarios():
                years, values = inference.get_history(asset, scenario)
                for year, value in zip(years, values):
                    if year < first_projected:
                        hist_rows.append(
                            {
                                "material": inference.MATERIAL,
                                "asset_id": asset,
                                "scenario": scenario,
                                "year": year,
                                "value": value,
                                "unit": inference.UNIT,
                                "metric": inference.METRIC,
                                "method": "observed",
                            }
                        )
        df = pd.concat([df, pd.DataFrame(hist_rows)], ignore_index=True)

    df["value"] = df["value"].round(4)
    df = df[SCHEMA].sort_values(
        ["material", "asset_id", "scenario", "method", "year"]
    ).reset_index(drop=True)
    return df


def build_timber_transfer(method: str, pathway: str = "A1FI", shift: bool = True) -> pd.DataFrame:
    """PileB history + forecasts from the PileA-trained model in ``timber_transfer``.

    History rows are always included: the dashboard needs them to draw the
    history/forecast split, and they are PileB's own values, not predictions.
    Only ``pathway``'s three climate series are written, for history and forecast alike.
    ``shift`` moves each forecast by the model's mean error over ``tt.SHIFT_WINDOW``
    (``tt.shift_forecast``); the offsets are returned in ``df.attrs["shift"]``.
    """
    from src.models import timber_transfer as tt

    if method not in TRANSFER_METHODS:
        raise ValueError(f"timber supports {TRANSFER_METHODS}, got {method!r}")
    if pathway not in TRANSFER_PATHWAYS:
        raise ValueError(f"--pathway supports {TRANSFER_PATHWAYS}, got {pathway!r}")

    from src.models.timber_forecast import load_model

    frame = tt.load_transfer_frame()
    test = frame.loc[frame["asset_code"] == tt.TEST_PILE].copy()
    model = load_model(method)  # saved model of record (python -m src.models.timber_transfer --save)
    shift_record = None
    if shift:
        test["prediction"], shift_record = tt.shift_forecast(
            test, model.predict_uncapped(test[tt.MODEL_INPUT_COLUMNS])
        )
    else:
        test["prediction"] = model.predict(test[tt.MODEL_INPUT_COLUMNS])
    test["scenario"] = test["emissions_pathway"] + "|" + test["climate_model"]
    test = test.loc[test["emissions_pathway"] == pathway]

    history = test.loc[test["year"] <= tt.SPLIT_YEAR].assign(
        value=lambda rows: rows[tt.LOSS_COLUMN], method="observed"
    )
    forecast = test.loc[tt.forecast_mask(test)].assign(
        value=lambda rows: rows["prediction"], method=method
    )
    df = pd.concat([history, forecast], ignore_index=True).assign(
        material=TIMBER_MATERIAL,
        asset_id=lambda rows: rows["asset_code"],
        unit=TIMBER_UNIT,
        metric=TIMBER_METRIC,
    )
    df["value"] = df["value"].round(4)
    df = (
        df[SCHEMA]
        .sort_values(["material", "asset_id", "scenario", "method", "year"])
        .reset_index(drop=True)
    )
    df.attrs["shift"] = shift_record
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--material",
        choices=sorted(MATERIAL_DEFAULT_METHOD),
        default="concrete",
        help="which material's dashboard CSV to build (default: concrete)",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="years to project (default: the material's own DEFAULT_HORIZON, 21 -> 2050-2070)",
    )
    parser.add_argument(
        "--method",
        default=None,
        help="model name (default: 'rnn' for concrete, 'mlp' for timber)",
    )
    parser.add_argument("--with-history", action="store_true",
                        help="also emit observed depths as method='observed'")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--transfer",
        action="store_true",
        help="kept for compatibility: timber always uses the saved PileA-trained models "
        "(src/models/timber_transfer.py); builds gbm and mlp unless --method is given",
    )
    parser.add_argument(
        "--pathway",
        choices=TRANSFER_PATHWAYS,
        default="A1FI",
        help="timber: the one emissions pathway to write (default: A1FI)",
    )
    parser.add_argument(
        "--no-shift",
        dest="shift",
        action="store_false",
        help="timber: write the raw model forecast instead of shifting it to PileB's "
        "2022-2026 values",
    )
    args = parser.parse_args()

    if args.transfer and args.material != "timber":
        parser.error("--transfer is timber-only; add --material timber")
    if args.material == "timber":
        methods = [args.method] if args.method else list(TRANSFER_METHODS)
        if args.out and len(methods) > 1:
            parser.error("--out needs a single --method for timber")
        for method in methods:
            df = build_timber_transfer(method, args.pathway, args.shift)
            out = args.out or TRANSFER_DEFAULT_OUT[method]
            out.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(out, index=False)
            print(f"wrote {len(df)} rows -> {out}")
            shift_meta = out.with_suffix(".shift.json")
            if df.attrs.get("shift"):
                shift_meta.write_text(json.dumps(df.attrs["shift"], indent=2))
                print(f"wrote shift offsets -> {shift_meta}")
            elif shift_meta.exists():
                shift_meta.unlink()  # unshifted file: a leftover offsets file would mislead
            print(df.groupby("method").size().to_string())
            print(f"assets={df['asset_id'].nunique()}  scenarios={df['scenario'].nunique()}  "
                  f"years={df['year'].min()}-{df['year'].max()}")
        return

    if inference is None:
        parser.error(
            "src/models/inference.py is not in this checkout yet -- pull the "
            "concrete team's inference module before building the concrete "
            "dashboard CSV (--material timber works without it)."
        )
    module, builder, out = inference, build_concrete, args.out or DEFAULT_OUT

    horizon = args.horizon if args.horizon is not None else module.DEFAULT_HORIZON
    method = args.method if args.method is not None else MATERIAL_DEFAULT_METHOD[args.material]

    df = builder(horizon, method, args.with_history)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print(f"wrote {len(df)} rows -> {out}")
    print(df.groupby("method").size().to_string())
    print(f"assets={df['asset_id'].nunique()}  scenarios={df['scenario'].nunique()}  "
          f"years={df['year'].min()}-{df['year'].max()}")


if __name__ == "__main__":
    main()
