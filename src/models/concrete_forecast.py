"""Forecast an imported concrete pile with the deployed model (no retraining).

    python -m src.models.concrete_forecast --input data/templates/concrete/skeleton
    python -m src.models.concrete_forecast --input my_import.xlsx --method wc_regress \\
        --out data/dashboard/dashboard_data_concrete_import.csv

Mirrors ``src/models/timber_forecast.py``'s shape. The input is the import format from
``src/data/concrete_import.py`` (an .xlsx with 3 sheets, or a folder of their CSVs):
``concrete_assets``, ``concrete_climate``, optionally ``concrete_observations``.

Uses the SAME deployed RNN artifact (``artifacts/rnn_carbonation.pt``) the live
dashboard serves via ``src.models.inference`` -- this never retrains on an upload, it
only runs the existing model forward. The RNN needs ``inference.rnn_window()`` years
(read from the artifact's own metadata, so it always matches whatever was actually
deployed) of real-or-filled history immediately before the forecast starts, to build
its lookback window (see ``concrete_import.to_model_series``'s ``last_observed_year``);
an import without that much history raises a clear error telling you how many years
are missing, rather than silently producing a bad forecast.
``method="wc_regress"`` has no such requirement (no lookback, climate-free) but is the
cruder of the two -- see ``[[concrete-modelling-reference]]``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.concrete_import import read_import, to_model_series, validate_import
from src.models.inference import MATERIAL, METRIC, UNIT, _project_rnn, _project_wc
from src.models.inference import rnn_window as _rnn_window

SCHEMA = ["material", "asset_id", "scenario", "year", "value", "unit", "metric", "method"]
_PROJECTORS = {"rnn": _project_rnn, "wc_regress": _project_wc}


def _cut_index(s: dict) -> int:
    """First row index whose year is strictly after last_observed_year (forecast start)."""
    return int(np.searchsorted(s["years"], s["last_observed_year"] + 1))


def _check_series(s: dict, method: str) -> None:
    cut = _cut_index(s)
    if cut >= len(s["years"]):
        raise ValueError(
            f"{s['asset']} / {s['scenario']}: no climate years after "
            f"last_observed_year={s['last_observed_year']} to forecast -- add future "
            f"years to concrete_climate for this scenario."
        )
    if method == "rnn":
        window = _rnn_window()
        if cut < window:
            raise ValueError(
                f"{s['asset']} / {s['scenario']}: only {cut} year(s) of real-or-filled "
                f"history before the forecast start ({s['last_observed_year'] + 1}); the "
                f"deployed RNN needs at least {window} years for its lookback window "
                f"(artifacts/rnn_carbonation.json's 'window'). Supply more "
                f"climate/observation years, or pass method='wc_regress' (no lookback "
                f"needed, but cruder -- see concrete-modelling-reference)."
            )


def forecast_series(series_list: list[dict], method: str = "rnn") -> pd.DataFrame:
    """Imported series (``concrete_import.to_model_series`` output) -> dashboard-schema rows.

    Each series contributes its real/filled history (``method="observed"``) plus a
    forecast for every climate year after ``last_observed_year`` (``method=method``).
    """
    if method not in _PROJECTORS:
        raise ValueError(f"method must be one of {sorted(_PROJECTORS)}, got {method!r}")
    rows = []
    for s in series_list:
        _check_series(s, method)
        cut = _cut_index(s)
        horizon = len(s["years"]) - cut
        values = _PROJECTORS[method](s, horizon)

        history = s["years"] <= s["last_observed_year"]
        for year, value in zip(s["years"][history], s["depth"][history]):
            rows.append(
                {
                    "material": MATERIAL, "asset_id": s["asset"], "scenario": s["scenario"],
                    "year": int(year), "value": round(float(value), 4), "unit": UNIT,
                    "metric": METRIC, "method": "observed",
                }
            )
        for year, value in zip(s["years"][cut:], values):
            rows.append(
                {
                    "material": MATERIAL, "asset_id": s["asset"], "scenario": s["scenario"],
                    "year": int(year), "value": round(float(value), 4), "unit": UNIT,
                    "metric": METRIC, "method": method,
                }
            )
    out = pd.DataFrame(rows, columns=SCHEMA)
    return out.sort_values(["asset_id", "scenario", "method", "year"]).reset_index(drop=True)


def forecast_from_import(source: Path, method: str = "rnn") -> pd.DataFrame:
    """Validate ``source`` (an import .xlsx or folder of its CSVs), then forecast it.

    Raises with every validation problem listed together (same form the upload page
    would show) if the import has errors.
    """
    result = validate_import(read_import(source))
    if not result.ok:
        raise ValueError("Import file has errors:\n" + result.issues_frame().to_string(index=False))
    return forecast_series(to_model_series(result), method=method)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input", type=Path, required=True, help="import .xlsx or a folder of its CSVs")
    parser.add_argument("--method", choices=("rnn", "wc_regress"), default="rnn")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    df = forecast_from_import(args.input, method=args.method)
    print(df.groupby(["asset_id", "method"]).size().to_string())
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out, index=False)
        print(f"wrote {len(df)} rows -> {args.out}")


if __name__ == "__main__":
    main()
