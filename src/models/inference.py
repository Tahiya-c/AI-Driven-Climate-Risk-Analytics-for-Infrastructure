"""
inference.py -- load the deployed concrete carbonation model and project trajectories.

There is ONE trained model, not one per asset/scenario. train_final_model.py fits
it on all 49 concrete series and writes three files to artifacts/:

    rnn_carbonation.pt      state_dict for CarbonationRNN (Elman RNN, 16 hidden)
    rnn_carbonation.json    scaler stats, feature order, hyperparameters
    wc_regression.json      two-parameter fallback: log(m) = a*wc + b, xc = sqrt(m*t + c)

The RNN consumes the TARGET year's climate (temperature, RH, CO2) as an input.
Those are GCM projections known in advance, so this is legitimate -- but the model
cannot run on history alone. This module projects the final ``horizon`` years of
each series in the consolidated CSV, which already carries GCM climate out to 2070.
Projecting past 2070 needs additional forward climate drivers that are not wired in
yet; use ``method="wc_regress"`` for a climate-free (but cruder) extrapolation.

Accuracy for the RNN artefact comes from carbonation_benchmark.py, not this file
(leave-one-scenario-out: 0.048 mm depth MAE over 21 y, beats both physics
baselines on 7/7 scenario folds). This module only rebuilds trajectories.

Public API
----------
    list_assets()                         -> list[str]
    list_scenarios()                      -> list[str]   ("A1FI|CSIRO Mk3.5", ...)
    get_projection(asset, scenario, ...)  -> (years, values)   depth in mm
    get_history(asset, scenario)          -> (years, values)   observed depth in mm
    predict_scenario(asset, scenario, ...) -> DataFrame[year, value]
    predict_batch(assets, scenarios, ...) -> long DataFrame
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

try:  # imported as a package (src.models.inference) ...
    from .rnn_model import SERIES_START, build_step, load_series
except ImportError:  # ... or run as a script from src/models/
    from rnn_model import SERIES_START, build_step, load_series

# --------------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = ROOT / "artifacts"
RNN_WEIGHTS = ARTIFACT_DIR / "rnn_carbonation.pt"
RNN_META = ARTIFACT_DIR / "rnn_carbonation.json"
WC_META = ARTIFACT_DIR / "wc_regression.json"
DATA_PATH = ROOT / "data" / "processed" / "concrete_kembla_v6_consolidated.csv"

# --------------------------------------------------------------- dashboard tags
MATERIAL = "concrete"
METRIC = "carbonation_depth"
UNIT = "mm"
DEFAULT_HORIZON = 21


class CarbonationRNN(nn.Module):
    """Must stay identical to train_final_model.CarbonationRNN so the state_dict loads."""

    def __init__(self, n_feat: int, hidden: int = 16):
        super().__init__()
        self.rnn = nn.RNN(n_feat, hidden, num_layers=1, nonlinearity="tanh", batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):  # noqa: D102
        out, _ = self.rnn(x)
        return self.head(out[:, -1, :]).squeeze(-1)


# ------------------------------------------------------------------- loading
@lru_cache(maxsize=1)
def _rnn():
    """Return (model, x_mean, x_std, y_mean, y_std, window) for the deployed RNN.

    ``window`` comes from the saved artifact's own metadata (``meta["window"]``),
    not a module-level constant -- train_final_model.py may have trained with a
    different window than rnn_model.py's default (e.g. after Optuna tuning changes
    TUNED_RNN_PARAMS), and this way inference can never silently fall out of sync
    with whatever was actually deployed.
    """
    if not RNN_WEIGHTS.exists():
        raise FileNotFoundError(
            f"{RNN_WEIGHTS} not found. Run:\n"
            f"    python src/models/train_final_model.py {DATA_PATH} --out artifacts"
        )
    meta = json.loads(RNN_META.read_text())
    model = CarbonationRNN(meta["n_features"], meta["hidden"])
    model.load_state_dict(torch.load(RNN_WEIGHTS, map_location="cpu", weights_only=True))
    model.eval()
    x_mean = np.asarray(meta["x_mean"], dtype=np.float32)
    x_std = np.asarray(meta["x_std"], dtype=np.float32)
    return model, x_mean, x_std, float(meta["y_mean"]), float(meta["y_std"]), int(meta["window"])


def rnn_window() -> int:
    """The deployed RNN's lookback window -- see ``_rnn``'s docstring for why this
    is read from the artifact instead of being a static constant."""
    return _rnn()[5]


@lru_cache(maxsize=1)
def _wc_params() -> tuple[float, float]:
    meta = json.loads(WC_META.read_text())
    return float(meta["a"]), float(meta["b"])


@lru_cache(maxsize=1)
def _series_index() -> dict[tuple[str, str], dict]:
    """Map (asset, scenario) -> the series dict produced by rnn_model.load_series."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"{DATA_PATH} not found (data/processed/ is gitignored).")
    series, _ = load_series(DATA_PATH)
    return {(s["asset"], s["scenario"]): s for s in series}


@lru_cache(maxsize=1)
def _history_index() -> dict[tuple[str, str], pd.DataFrame]:
    """Observed carbonation depth per series, including the first year (2000)."""
    df = pd.read_csv(DATA_PATH)
    df["scenario"] = df["emissions_pathway"].astype(str) + "|" + df["climate_model"].astype(str)
    out: dict[tuple[str, str], pd.DataFrame] = {}
    cols = ["year", "carb_depth_xc_mm"]
    for (asset, scenario), g in df.groupby(["asset_code", "scenario"]):
        out[(asset, scenario)] = g.sort_values("year")[cols].reset_index(drop=True)
    return out


def list_assets() -> list[str]:
    return sorted({asset for asset, _ in _series_index()})


def list_scenarios() -> list[str]:
    return sorted({scenario for _, scenario in _series_index()})


def _series(asset: str, scenario: str) -> dict:
    try:
        return _series_index()[(asset, scenario)]
    except KeyError:
        raise KeyError(
            f"no series for asset={asset!r} scenario={scenario!r}. "
            f"assets={list_assets()} scenarios={list_scenarios()}"
        ) from None


def _check_horizon(series_len: int, horizon: int) -> None:
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    window = rnn_window()
    if series_len - horizon < window:
        raise ValueError(
            f"horizon={horizon} too long: series has {series_len} usable years and the "
            f"RNN needs a {window}-year lookback, so horizon must be <= {series_len - window}"
        )


def _years_for_tail(series: dict, horizon: int) -> list[int]:
    cut = len(series["y"]) - horizon
    return [int(t) + SERIES_START - 1 for t in series["t"][cut:]]


# --------------------------------------------------------------- projection
def _project_rnn(series: dict, horizon: int) -> np.ndarray:
    """Autoregressive rollout, identical in shape to carbonation_benchmark.roll_rnn."""
    model, x_mean, x_std, y_mean, y_std, window = _rnn()
    X, y = series["X"], series["y"]
    cut = len(y) - horizon
    lag = y.copy()
    depth = float(series["depth"][cut - 1])
    out = []
    with torch.no_grad():
        for i in range(cut, len(y)):
            step = ((build_step(X, lag, i, window) - x_mean) / x_std).astype(np.float32)
            rate = float(model(torch.tensor(step[None]))) * y_std + y_mean
            lag[i] = rate
            depth += rate
            out.append(depth)
    return np.asarray(out)


def _project_wc(series: dict, horizon: int) -> np.ndarray:
    """Climate-free fallback: m from w/c ratio, curve anchored on the last observed depth."""
    a, b = _wc_params()
    t, depth = series["t"], series["depth"]
    cut = len(t) - horizon
    m = float(np.exp(a * series["wc"] + b))
    c = depth[cut - 1] ** 2 - m * t[cut - 1]
    return np.sqrt(np.maximum(m * t[cut:] + c, 1e-12))


_PROJECTORS = {"rnn": _project_rnn, "wc_regress": _project_wc}
METHODS = tuple(_PROJECTORS)


def get_projection(
    asset: str,
    scenario: str,
    horizon: int = DEFAULT_HORIZON,
    method: str = "rnn",
) -> tuple[list[int], list[float]]:
    """Project carbonation depth (mm) for the final ``horizon`` years of a series.

    Returns ``(years, values)`` as plain Python lists, JSON-serialisable and ready
    for ``plotly`` (``x=years, y=values``). With ``horizon=21`` the years are
    2050-2070 -- the held-out tail the benchmark scores against.
    """
    if method not in _PROJECTORS:
        raise ValueError(f"method must be one of {sorted(_PROJECTORS)}, got {method!r}")
    series = _series(asset, scenario)
    _check_horizon(len(series["y"]), horizon)
    values = _PROJECTORS[method](series, horizon)
    return _years_for_tail(series, horizon), [float(v) for v in values]


def get_history(asset: str, scenario: str) -> tuple[list[int], list[float]]:
    """Observed carbonation depth (mm) for every year in the workbook (2000-2070)."""
    _series(asset, scenario)  # validate the key
    g = _history_index()[(asset, scenario)]
    return g["year"].astype(int).tolist(), g["carb_depth_xc_mm"].astype(float).tolist()


def predict_scenario(
    asset: str,
    scenario: str,
    horizon: int = DEFAULT_HORIZON,
    method: str = "rnn",
) -> pd.DataFrame:
    """``get_projection`` as a two-column DataFrame (``year``, ``value``)."""
    years, values = get_projection(asset, scenario, horizon, method)
    return pd.DataFrame({"year": years, "value": values})


def predict_batch(
    assets: Iterable[str] | None = None,
    scenarios: Iterable[str] | None = None,
    horizon: int = DEFAULT_HORIZON,
    method: str = "rnn",
) -> pd.DataFrame:
    """Every (asset, scenario) projection as one long DataFrame in dashboard columns.

    Defaults to all known concrete assets and scenarios.
    """
    assets = list(assets) if assets is not None else list_assets()
    scenarios = list(scenarios) if scenarios is not None else list_scenarios()
    frames = []
    for asset in assets:
        for scenario in scenarios:
            years, values = get_projection(asset, scenario, horizon, method)
            frames.append(
                pd.DataFrame(
                    {
                        "material": MATERIAL,
                        "asset_id": asset,
                        "scenario": scenario,
                        "year": years,
                        "value": values,
                        "unit": UNIT,
                        "metric": METRIC,
                        "method": method,
                    }
                )
            )
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    a, s = list_assets()[0], list_scenarios()[0]
    yrs, vals = get_projection(a, s)
    print(f"{a} / {s}  ({yrs[0]}-{yrs[-1]})")
    for yr, v in zip(yrs, vals):
        print(f"  {yr}  {v:.4f} mm")
