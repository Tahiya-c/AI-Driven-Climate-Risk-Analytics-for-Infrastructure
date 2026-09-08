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
  ``build_timber``) against their own ``inference`` module, and written to
  separate ``dashboard_data_<material>.csv`` files -- same schema, no frontend
  change either way. Select which with ``--material``.
* ``scenario`` is the full "<emissions_pathway>|<climate_model>" id -- all 7 series
  the workbook actually contains. Collapsing these to 3 chart lines is an open
  question for Dr. Daniel; do it in the frontend once decided, not here.
* With the default 21-year horizon the projected rows cover 2050-2070. That is the
  held-out tail of the workbook projection, not a genuine post-2070 forecast --
  projecting further needs the climate-futures workbook wired into the pipeline.
  For timber specifically, the "observed" 2000-2049 rows are the deterministic
  AS 5604-style workbook's own output, not real field inspection data (see
  ``src/timber/data_prep.py``) -- reported accuracy for the projected rows is
  surrogate-vs-simulator fidelity, not validated predictive accuracy.
* ``--with-history`` also emits the observed depths (method="observed") so the
  frontend can draw the history/forecast split from a single file.

Usage
-----
    python src/models/build_dashboard_output.py
    python src/models/build_dashboard_output.py --with-history --horizon 21
    python src/models/build_dashboard_output.py --method wc_regress --out data/dashboard/foo.csv
    python src/models/build_dashboard_output.py --material timber --with-history
    python src/models/build_dashboard_output.py --material timber --method gbm
"""

from __future__ import annotations

import argparse
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

from src.timber import inference as timber_inference

SCHEMA = ["material", "asset_id", "scenario", "year", "value", "unit", "metric", "method"]
DEFAULT_OUT = (
    inference.ROOT / "data" / "dashboard" / "dashboard_data_concrete.csv" if inference else None
)
TIMBER_DEFAULT_OUT = timber_inference.ROOT / "data" / "dashboard" / "dashboard_data_timber.csv"

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


def build_timber(horizon: int, method: str, with_history: bool) -> pd.DataFrame:
    """Same shape as ``build_concrete``, against ``src/timber/inference.py`` instead.

    Timber's "observed" years and projection anchor are fixed by
    ``timber_inference.ANCHOR_YEAR`` (2050) rather than derived from the
    projected frame's min year, since the surrogate can in principle be asked
    to predict any window -- see that module for why 2050 is the anchor.
    """
    df = timber_inference.predict_batch(horizon=horizon, method=method)

    if with_history:
        hist_rows = []
        for asset in timber_inference.list_assets():
            for scenario in timber_inference.list_scenarios():
                years, values = timber_inference.get_history(asset, scenario)
                for year, value in zip(years, values):
                    hist_rows.append(
                        {
                            "material": timber_inference.MATERIAL,
                            "asset_id": asset,
                            "scenario": scenario,
                            "year": year,
                            "value": value,
                            "unit": timber_inference.UNIT,
                            "metric": timber_inference.METRIC,
                            "method": "observed",
                        }
                    )
        df = pd.concat([df, pd.DataFrame(hist_rows)], ignore_index=True)

    df["value"] = df["value"].round(4)
    df = df[SCHEMA].sort_values(
        ["material", "asset_id", "scenario", "method", "year"]
    ).reset_index(drop=True)
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
    args = parser.parse_args()

    if args.material == "concrete":
        if inference is None:
            parser.error(
                "src/models/inference.py is not in this checkout yet -- pull the "
                "concrete team's inference module before building the concrete "
                "dashboard CSV (--material timber works without it)."
            )
        module, builder, out = inference, build_concrete, args.out or DEFAULT_OUT
    else:
        module, builder, out = timber_inference, build_timber, args.out or TIMBER_DEFAULT_OUT

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
