"""The dashboard's two concrete models (RNN and Ridge): train once, save, reuse.

    python -m src.models.concrete_saved --save      # fit on the reference data and save

Both models are fitted exactly as ``build_concrete_dashboard.py`` fits them (all 49
reference series, years <= ``concrete_eval.CUTOFF_YEAR``), then saved to
``artifacts/`` so an imported pile is forecast by the *same* fitted models as the
reference piles, without retraining on every upload:

    artifacts/concrete_rnn.pt        the 3 seed networks' weights + their scaling
    artifacts/concrete_ridge.joblib  the fitted Ridge model
    artifacts/concrete_<kind>.json   version, params, library versions

Forecasts go through ``concrete_eval.rollout`` (fully autoregressive, the same loop
used for scoring), so reference and imported piles are treated identically. Output is
the dashboard schema (material, asset_id, scenario, year, value, unit, metric, method).
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from functools import cache
from importlib.metadata import PackageNotFoundError, version

import joblib
import numpy as np
import pandas as pd

from src.config import ROOT_DIR
from src.models import concrete_eval as ce

DATA = ROOT_DIR / "data" / "processed" / "concrete_kembla_v6_consolidated.csv"
ARTIFACT_DIR = ROOT_DIR / "artifacts"
MODEL_KINDS = ("rnn", "ridge")
# Names already registered in Supabase's model_versions table (docs/database_schema.sql).
MODEL_VERSIONS = {"rnn": "concrete_rnn_tuned_v1", "ridge": "concrete_ridge_v1"}
PATHWAY = "A1FI"
GCMS = ("MRI-CGCM2.3.2", "CSIRO Mk3.5", "MIROC(hires)")
SCHEMA = ["material", "asset_id", "scenario", "year", "value", "unit", "metric", "method"]


def _paths(kind: str):
    suffix = ".pt" if kind == "rnn" else ".joblib"
    return ARTIFACT_DIR / f"concrete_{kind}{suffix}", ARTIFACT_DIR / f"concrete_{kind}.json"


def _library_versions() -> dict[str, str]:
    out = {}
    for name in ("torch", "scikit-learn", "numpy"):
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            pass
    return out


# ------------------------------------------------------------------------ RNN weights
def _net(n_inputs: int, hidden: int):
    """Same layers as rnn_model.train_rnn's Net, so its state_dict loads unchanged."""
    import torch.nn as nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.rnn = nn.RNN(n_inputs, hidden, 1, nonlinearity="tanh", batch_first=True)
            self.head = nn.Linear(hidden, 1)

        def forward(self, x):
            o, _ = self.rnn(x)
            return self.head(o[:, -1, :]).squeeze(-1)

    return Net()


def _save_rnn(model: ce.RnnModel, path) -> None:
    import torch

    fits = [
        {
            "state": net.state_dict(),
            "n_inputs": int(net.rnn.input_size),
            "mu": mu, "sd": sd, "ymu": float(ymu), "ysd": float(ysd),
        }
        for net, mu, sd, ymu, ysd in model._fits
    ]
    torch.save({"window": model.window, "hidden": model.hidden, "fits": fits}, path)


def _load_rnn(path) -> ce.RnnModel:
    import torch

    saved = torch.load(path, weights_only=False)
    model = ce.RnnModel(params={"window": saved["window"], "hidden": saved["hidden"]})
    model._fits = []
    for fit in saved["fits"]:
        net = _net(fit["n_inputs"], saved["hidden"])
        net.load_state_dict(fit["state"])
        net.eval()
        model._fits.append((net, fit["mu"], fit["sd"], fit["ymu"], fit["ysd"]))
    return model


# ------------------------------------------------------------------------ train / load
def train_and_save() -> dict[str, dict]:
    """Fit both models on the reference series (<= CUTOFF_YEAR) and save them."""
    series, _ = ce.load_series(DATA)
    ARTIFACT_DIR.mkdir(exist_ok=True)
    saved = {}
    for kind in MODEL_KINDS:
        model = ce.make_model(kind)
        print(f"fitting {kind} on {len(series)} series (<= {ce.CUTOFF_YEAR}) ...")
        model.fit(series, ce.CUTOFF_YEAR)
        model_path, meta_path = _paths(kind)
        if kind == "rnn":
            _save_rnn(model, model_path)
            params = {"window": model.window, "hidden": model.hidden, "seeds": list(model.seeds)}
        else:
            joblib.dump(model, model_path)
            params = {"alpha": model.params.get("alpha", 1.0)}
        meta = {
            "model_version": MODEL_VERSIONS[kind],
            "model_kind": kind,
            "saved_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "trained_on": f"{len(series)} series, 7 piles, years <= {ce.CUTOFF_YEAR}",
            "params": params,
            "library_versions": _library_versions(),
        }
        meta_path.write_text(json.dumps(meta, indent=2))
        load_model.cache_clear()
        saved[kind] = meta
        print(f"saved {model_path.relative_to(ROOT_DIR)}")
    return saved


@cache
def load_model(kind: str) -> ce.Model:
    """The saved model of record. Never trains."""
    model_path, _ = _paths(kind)
    if not model_path.exists():
        raise FileNotFoundError(
            f"No saved concrete {kind} model at {model_path}. "
            "Run: python -m src.models.concrete_saved --save"
        )
    return _load_rnn(model_path) if kind == "rnn" else joblib.load(model_path)


def load_metadata(kind: str) -> dict:
    _, meta_path = _paths(kind)
    return json.loads(meta_path.read_text()) if meta_path.exists() else {}


def min_history_years(kind: str) -> int:
    """Rows needed before the forecast start: the RNN's window plus row 0 (no rate)."""
    return load_model(kind).window + 1 if kind == "rnn" else 2


# ------------------------------------------------------------------------ forecasting
def _rows(asset, scenario, years, values, method) -> list[tuple]:
    return [
        ("concrete", asset, scenario, int(y), round(float(v), 4), "mm", "carbonation_depth", method)
        for y, v in zip(years, values)
    ]


def forecast_reference(kind: str) -> pd.DataFrame:
    """The 7 reference piles: workbook history (<= cutoff) + the model's forecast."""
    model = load_model(kind)
    series, _ = ce.load_series(DATA)
    rows: list[tuple] = []
    for s in series:
        pathway, gcm = s["scenario"].split("|")
        if pathway != PATHWAY or gcm not in GCMS:
            continue
        years = ce.series_years(s)
        cut = ce.cut_index(s, ce.CUTOFF_YEAR)
        rows += _rows(s["asset"], s["scenario"], years[:cut], s["depth"][:cut], "observed")
        f_years, f_depth = ce.rollout(model, s, ce.CUTOFF_YEAR)
        rows += _rows(s["asset"], s["scenario"], f_years, f_depth, kind)
    return pd.DataFrame(rows, columns=SCHEMA)


def forecast_imported(series_list: list[dict], kind: str) -> pd.DataFrame:
    """Imported series (``concrete_import.to_model_series``) -> history + forecast rows.

    Only the A1FI pathway's 3 named climate models are forecast (what the dashboard
    shows). Each forecast starts the year after the pile's last inspection.
    """
    model = load_model(kind)
    need = min_history_years(kind)
    rows: list[tuple] = []
    for s in series_list:
        pathway, gcm = s["scenario"].split("|")
        if pathway != PATHWAY or gcm not in GCMS:
            continue
        years = np.asarray(s["years"])
        cut = int(np.searchsorted(years, s["last_observed_year"] + 1))
        if cut < need:
            raise ValueError(
                f"{s['asset']}: needs inspections up to at least {int(years[0]) + need - 1} "
                f"for the {kind.upper()} model (it reads the {need - 1} years before the forecast)."
            )
        if cut >= len(years):
            raise ValueError(
                f"{s['asset']}: no climate years after the last inspection to forecast."
            )
        rows += _rows(s["asset"], s["scenario"], years[:cut], s["depth"][:cut], "observed")
        f_years, f_depth = ce.rollout(model, s, s["last_observed_year"])
        rows += _rows(s["asset"], s["scenario"], f_years, f_depth, kind)
    if not rows:
        raise ValueError(f"No {PATHWAY} climate scenarios to forecast ({', '.join(GCMS)}).")
    return pd.DataFrame(rows, columns=SCHEMA)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--save", action="store_true", help="fit on the reference data and save")
    args = parser.parse_args()
    if not args.save:
        parser.error("pass --save")
    train_and_save()


if __name__ == "__main__":
    main()
