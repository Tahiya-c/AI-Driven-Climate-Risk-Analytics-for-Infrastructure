"""
Carbonation trajectory benchmark -- Port Kembla concrete.

Four methods, evaluated identically, on two axes of generalisation.

METHODS
  rnn          Vanilla Elman RNN (REQ-06). Autoregressive rollout: predicts one
               year's increment, feeds that back, repeats to the horizon.
  persistence  Next increment = last observed increment, held constant.
  sqrt_law     Fit xc^2 = m*t + c on the held-out series' own history and
               extrapolate. The workbook obeys this form to mean R^2 = 0.993.
  wc_regress   Two parameters. Fit log(m) = a*wc_ratio + b on the TRAINING
               series only, predict m for the held-out series from its w/c,
               then anchor c on the last observed depth. Never sees the
               held-out series' trajectory shape.

AXES
  asset     leave-one-asset-out. Can we project a pile we have never modelled?
  scenario  leave-one-scenario-out. Can we project a climate pathway we have
            never trained on? (This is what the dashboard's scenario selector
            actually does, and it was previously untested.)

WHY THESE FOUR
  Fitting xc^2 = m*t + c to all 49 series shows the data factorises: the
  scenario effect is a near-constant multiplier on m (identical across piles to
  +/-0.02), and an SVD of log(m) puts 96% of variance in one component. Effective
  degrees of freedom are ~13, not 3,479. wc_regress exists to show what that
  implies -- if two parameters match a recurrent network, the honest conclusion
  is that the dataset does not support the larger model.

DATA NOTES (both apply to the consolidated CSV)
  * age_years is NOT the exposure clock. The workbook counts from 2000 for every
    asset -- PileF shows 1.58 mm of carbonation ten years before it was built.
  * carb_depth_increment_mm is cumulative growth since 2000, exactly
    xc(t) - xc(2000). It is a linear function of carb_depth_xc_mm and must never
    be a feature or target. The true annual rate is computed here by differencing.
  * cover_h_mm / member_diameter_mm / cement_content_kgm3 are exactly collinear
    (r = +/-1.000), one distinct value per pile. With 7 assets they are
    indistinguishable from an asset label; including them tripled held-out error.

Usage
-----
    python carbonation_benchmark.py concrete_kembla_v6_consolidated.csv
    python carbonation_benchmark.py data.csv --axis scenario --horizon 40
    python carbonation_benchmark.py data.csv --skip-rnn      # fast, no torch
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SERIES_KEYS = ["asset_code", "emissions_pathway", "climate_model"]
CLIMATE = ["mean_temp_c", "mean_rh_pct", "co2_ppm"]
STATIC = ["wc_ratio", "fc_mpa"]
CATS = ["zone"]

WINDOW = 12
HORIZON = 21
HIDDEN = 16
EPOCHS = 300
PATIENCE = 30
LR = 5e-3
SEEDS = (0, 1, 2)
SERIES_START = 2000


# ------------------------------------------------------------------ loading
def load_series(csv_path):
    df = pd.read_csv(csv_path).sort_values(SERIES_KEYS + ["year"])
    df["annual_rate"] = df.groupby(SERIES_KEYS)["carb_depth_xc_mm"].diff()
    df["t_exposure"] = df["year"] - SERIES_START + 1

    cat = pd.get_dummies(df[CATS].astype(str), prefix=CATS)
    df = pd.concat([df, cat], axis=1)
    feat_cols = CLIMATE + ["t_exposure"] + STATIC + list(cat.columns)

    series = []
    for key, g in df.groupby(SERIES_KEYS):
        g = g.sort_values("year").iloc[1:]        # first year has no rate
        t = g["t_exposure"].to_numpy(dtype=np.float64)
        d = g["carb_depth_xc_mm"].to_numpy(dtype=np.float64)
        m, c = np.polyfit(t, d ** 2, 1)           # true slope, for diagnostics
        series.append({
            "asset": key[0],
            "scenario": f"{key[1]}|{key[2]}",
            "X": g[feat_cols].to_numpy(dtype=np.float32),
            "y": g["annual_rate"].to_numpy(dtype=np.float32),
            "t": t, "depth": d,
            "wc": float(g["wc_ratio"].iloc[0]),
            "m_true": float(m),
        })
    return series, feat_cols


def metrics(pred, true):
    err = np.asarray(pred) - np.asarray(true)
    ss = float(np.sum((true - np.mean(true)) ** 2))
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "r2": 1 - float(np.sum(err ** 2)) / ss if ss > 0 else float("nan"),
        "final": float(abs(err[-1])),
    }


# --------------------------------------------------------------- baselines
def roll_persistence(s, horizon):
    cut = len(s["y"]) - horizon
    r, d = float(s["y"][cut - 1]), float(s["depth"][cut - 1])
    out = []
    for _ in range(horizon):
        d += r
        out.append(d)
    return np.array(out), s["depth"][cut:]


def roll_sqrt(s, horizon):
    cut = len(s["t"]) - horizon
    m, c = np.polyfit(s["t"][:cut], s["depth"][:cut] ** 2, 1)
    return np.sqrt(np.maximum(m * s["t"][cut:] + c, 1e-12)), s["depth"][cut:]


def fit_wc_regression(train_series):
    """log(m) = a*wc + b, fitted on training series only."""
    wc = np.array([s["wc"] for s in train_series])
    lm = np.log(np.array([s["m_true"] for s in train_series]))
    a, b = np.polyfit(wc, lm, 1)
    return a, b


def roll_wc(s, horizon, a, b):
    """Predict m from w/c, anchor c on the last observed depth."""
    cut = len(s["t"]) - horizon
    m = float(np.exp(a * s["wc"] + b))
    c = s["depth"][cut - 1] ** 2 - m * s["t"][cut - 1]
    return np.sqrt(np.maximum(m * s["t"][cut:] + c, 1e-12)), s["depth"][cut:]


# --------------------------------------------------------------------- RNN
def _torch():
    import torch
    import torch.nn as nn
    return torch, nn


def build_step(X, lag, i, window):
    """Window ending at i-1 predicting i. Row j carries the rate observed at j
    (never a future value); the target year's climate is appended to each row --
    these are GCM projections, so future forcing is known at prediction time."""
    w = np.concatenate([X[i - window:i], lag[i - window:i, None]], axis=1)
    return np.concatenate([w, np.tile(X[i, :len(CLIMATE)], (window, 1))], axis=1)


def make_windows(series_list, window=WINDOW):
    Xs, ys = [], []
    for s in series_list:
        X, y = s["X"], s["y"]
        for i in range(window, len(y)):
            Xs.append(build_step(X, y, i, window))
            ys.append(y[i])
    return np.stack(Xs), np.array(ys, dtype=np.float32)


def train_rnn(Xtr, ytr, Xva, yva, seed):
    torch, nn = _torch()
    torch.manual_seed(seed)
    np.random.seed(seed)

    mu, sd = Xtr.mean(axis=(0, 1)), Xtr.std(axis=(0, 1))
    sd[sd < 1e-8] = 1.0
    ymu, ysd = ytr.mean(), max(ytr.std(), 1e-8)
    Xtr_t = torch.tensor((Xtr - mu) / sd)
    ytr_t = torch.tensor((ytr - ymu) / ysd)
    Xva_t = torch.tensor((Xva - mu) / sd)
    yva_t = torch.tensor((yva - ymu) / ysd)

    class Net(nn.Module):
        def __init__(self, n):
            super().__init__()
            self.rnn = nn.RNN(n, HIDDEN, 1, nonlinearity="tanh", batch_first=True)
            self.head = nn.Linear(HIDDEN, 1)

        def forward(self, x):
            o, _ = self.rnn(x)
            return self.head(o[:, -1, :]).squeeze(-1)

    model = Net(Xtr.shape[2])
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    lossf = nn.MSELoss()
    best, state, bad = np.inf, None, 0
    n = len(Xtr_t)
    for _ in range(EPOCHS):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, 128):
            idx = perm[i:i + 128]
            opt.zero_grad()
            lossf(model(Xtr_t[idx]), ytr_t[idx]).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            v = float(lossf(model(Xva_t), yva_t))
        if v < best - 1e-7:
            best, bad = v, 0
            state = {k: t.clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict(state)
    return model, mu, sd, ymu, ysd


def roll_rnn(model, s, horizon, mu, sd, ymu, ysd, window=WINDOW):
    torch, _ = _torch()
    X, y = s["X"], s["y"]
    cut = len(y) - horizon
    lag = y.copy()
    d = float(s["depth"][cut - 1])
    out = []
    model.eval()
    with torch.no_grad():
        for i in range(cut, len(y)):
            step = ((build_step(X, lag, i, window) - mu) / sd).astype(np.float32)
            r = float(model(torch.tensor(step[None]))) * ysd + ymu
            lag[i] = r
            d += r
            out.append(d)
    return np.array(out), s["depth"][cut:]


# ------------------------------------------------------------------ driver
def pick_validation(series, key, groups, held, strategy):
    """Choose the validation group for a fold.

    'rotate' takes the next group alphabetically. On the SCENARIO axis that is
    unsafe: alphabetical order puts A1B|CSIRO next to A1B|MIROC, whose growth
    multipliers differ by 0.005 -- early stopping would then be tuned on a
    near-duplicate of the test scenario, biasing model selection optimistically.

    'distant' (default) picks the group whose mean growth slope m is furthest
    from the held-out group's, guaranteeing validation and test are not
    near-duplicates. Deterministic, so runs stay reproducible.
    """
    if strategy == "rotate":
        return groups[(groups.index(held) + 1) % len(groups)]
    mean_m = {g: float(np.mean([s["m_true"] for s in series if s[key] == g]))
              for g in groups}
    others = [g for g in groups if g != held]
    return max(others, key=lambda g: abs(mean_m[g] - mean_m[held]))


def run(series, axis, horizon, use_rnn, val_strategy="distant"):
    key = "asset" if axis == "asset" else "scenario"
    groups = sorted({s[key] for s in series})
    rows = []

    for held in groups:
        val = pick_validation(series, key, groups, held, val_strategy)
        tr = [s for s in series if s[key] not in (held, val)]
        va = [s for s in series if s[key] == val]
        te = [s for s in series if s[key] == held]

        rec = {"held_out": held, "n_test": len(te)}

        for name, fn in (("persistence", roll_persistence), ("sqrt_law", roll_sqrt)):
            p, t = zip(*[fn(s, horizon) for s in te])
            rec[f"{name}_mae"] = metrics(np.concatenate(p), np.concatenate(t))["mae"]
            rec[f"{name}_final"] = np.mean([metrics(*fn(s, horizon))["final"] for s in te])

        a, b = fit_wc_regression(tr)
        p, t = zip(*[roll_wc(s, horizon, a, b) for s in te])
        rec["wc_regress_mae"] = metrics(np.concatenate(p), np.concatenate(t))["mae"]
        rec["wc_regress_final"] = np.mean(
            [metrics(*roll_wc(s, horizon, a, b))["final"] for s in te])

        if use_rnn:
            Xtr, ytr = make_windows(tr)
            Xva, yva = make_windows(va)
            maes, fins = [], []
            for sd_ in SEEDS:
                model, mu, sdv, ymu, ysd = train_rnn(Xtr, ytr, Xva, yva, sd_)
                p, t = zip(*[roll_rnn(model, s, horizon, mu, sdv, ymu, ysd)
                             for s in te])
                mm = metrics(np.concatenate(p), np.concatenate(t))
                maes.append(mm["mae"])
                fins.append(np.mean([metrics(*roll_rnn(model, s, horizon, mu,
                                                       sdv, ymu, ysd))["final"]
                                     for s in te]))
            rec["rnn_mae"] = float(np.mean(maes))
            rec["rnn_sd"] = float(np.std(maes))
            rec["rnn_final"] = float(np.mean(fins))

        rows.append(rec)
        cols = ["rnn", "persistence", "sqrt_law", "wc_regress"] if use_rnn \
            else ["persistence", "sqrt_law", "wc_regress"]
        line = "  ".join(f"{c}={rec[c + '_mae']:.4f}" for c in cols)
        print(f"  {held:<22} {line}")

    return pd.DataFrame(rows)


def main(csv_path, axis, horizon, use_rnn, val_strategy="distant"):
    series, feat_cols = load_series(csv_path)
    assets = sorted({s["asset"] for s in series})
    scens = sorted({s["scenario"] for s in series})
    print(f"{len(series)} series | {len(assets)} assets | {len(scens)} scenarios")
    print(f"horizon={horizon}y  window={WINDOW}  "
          f"rnn={'on' if use_rnn else 'off'}  val={val_strategy}\n")

    axes = ["asset", "scenario"] if axis == "both" else [axis]
    out = {}
    for ax in axes:
        print(f"--- leave-one-{ax}-out  (depth MAE over horizon, mm) ---")
        res = run(series, ax, horizon, use_rnn, val_strategy)
        out[ax] = res
        print()
        summary = res[[c for c in res.columns if c.endswith("_mae")
                       or c.endswith("_final")]].mean()
        print(summary.round(4).to_string())
        if use_rnn:
            for other in ("persistence", "sqrt_law", "wc_regress"):
                w = int((res["rnn_mae"] < res[f"{other}_mae"]).sum())
                print(f"  RNN beats {other}: {w}/{len(res)} folds")
        print()
        p = Path(f"results_benchmark_{ax}.csv")
        res.to_csv(p, index=False)
        print(f"wrote {p}\n")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", type=Path)
    ap.add_argument("--axis", choices=["asset", "scenario", "both"],
                    default="both")
    ap.add_argument("--horizon", type=int, default=HORIZON)
    ap.add_argument("--skip-rnn", dest="use_rnn", action="store_false")
    ap.add_argument("--val-strategy", choices=["distant", "rotate"],
                    default="distant",
                    help="how to pick the early-stopping validation group")
    a = ap.parse_args()
    main(a.csv, a.axis, a.horizon, a.use_rnn, a.val_strategy)