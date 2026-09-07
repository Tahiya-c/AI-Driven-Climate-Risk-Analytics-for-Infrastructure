"""
Train the DEPLOYABLE carbonation model and save it for the dashboard.

Why this is a separate script from carbonation_benchmark.py
-----------------------------------------------------------
The benchmark trains one model per fold, each with an asset or scenario
deliberately withheld. None of those seven models is the one you deploy -- each
is blind to part of the data on purpose, so that we can measure generalisation.
This script trains ONE model on all 49 series and saves it.

Do not report accuracy numbers from this script. It has no held-out data, so any
score it produces is meaningless. Accuracy comes from the benchmark; this
produces the artefact the dashboard loads.

Saves to artifacts/:
    rnn_carbonation.pt        model weights
    rnn_carbonation.json      scaler stats, feature order, hyperparameters
    wc_regression.json        the two-parameter fallback (a, b in log(m)=a*wc+b)

Both models are saved because they answer different questions:
    rnn            -- project a KNOWN pile under a climate scenario
                      (benchmark: 0.048 mm depth MAE over 21 y, beats both
                       physics baselines on 7/7 scenario folds)
    wc_regression  -- project an UNSEEN pile from its w/c ratio alone
                      (benchmark: 0.167 mm at 50 y, beats the RNN on 6/7
                       asset folds)

IMPORTANT for whoever wires this into the dashboard
---------------------------------------------------
The RNN consumes the TARGET year's climate (temperature, RH, CO2) as an input.
These are GCM projections, known in advance, so this is legitimate -- but it
means the model CANNOT run on history alone. The caller must supply forward
climate projections for every year being predicted. If those are unavailable,
use the wc_regression fallback, which needs only w/c and an anchor depth.

Usage
-----
    python train_final_model.py data/processed/concrete_kembla_v6_consolidated.csv
    python train_final_model.py <csv> --out artifacts
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from rnn_model import (
    CLIMATE,
    EPOCHS,
    HIDDEN,
    LR,
    SERIES_START,
    WINDOW,
    fit_wc_regression,
    load_series,
    make_windows,
)

VAL_FRACTION = 0.15      # random windows held out purely for early stopping
SEED = 0


class CarbonationRNN(nn.Module):
    def __init__(self, n_feat, hidden=HIDDEN):
        super().__init__()
        self.rnn = nn.RNN(n_feat, hidden, num_layers=1,
                          nonlinearity="tanh", batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.head(out[:, -1, :]).squeeze(-1)


def main(csv_path, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    series, feat_cols = load_series(csv_path)
    print(f"training on all {len(series)} series "
          f"({len({s['asset'] for s in series})} assets x "
          f"{len({s['scenario'] for s in series})} scenarios)")

    # ---- two-parameter model: log(m) = a*wc + b, fitted on everything ----
    a, b = fit_wc_regression(series)
    wc_path = out_dir / "wc_regression.json"
    wc_path.write_text(json.dumps({
        "form": "m = exp(a * wc_ratio + b);  xc(t) = sqrt(m*t + c)",
        "a": float(a), "b": float(b),
        "t_definition": f"t = year - {SERIES_START} + 1",
        "c_definition": "anchor on last observed depth: c = depth_last**2 - m*t_last",
        "note": "fitted on all 49 series; benchmark numbers come from the "
                "leave-one-asset-out run, not from this fit",
    }, indent=2))
    print(f"wrote {wc_path}   log(m) = {a:.4f}*wc + {b:.4f}")

    # ---- RNN on all series ----
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    X, y = make_windows(series)
    n = len(X)
    idx = np.random.permutation(n)
    n_val = int(n * VAL_FRACTION)
    va_i, tr_i = idx[:n_val], idx[n_val:]
    Xtr, ytr, Xva, yva = X[tr_i], y[tr_i], X[va_i], y[va_i]
    print(f"{n} windows -> {len(Xtr)} train / {len(Xva)} val (early stopping only)")

    mu, sd = Xtr.mean(axis=(0, 1)), Xtr.std(axis=(0, 1))
    sd[sd < 1e-8] = 1.0
    ymu, ysd = float(ytr.mean()), float(max(ytr.std(), 1e-8))

    Xtr_t = torch.tensor((Xtr - mu) / sd)
    ytr_t = torch.tensor((ytr - ymu) / ysd)
    Xva_t = torch.tensor((Xva - mu) / sd)
    yva_t = torch.tensor((yva - ymu) / ysd)

    model = CarbonationRNN(X.shape[2])
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    lossf = nn.MSELoss()
    best, state, bad, best_ep = np.inf, None, 0, 0

    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(Xtr_t))
        for i in range(0, len(Xtr_t), 128):
            j = perm[i:i + 128]
            opt.zero_grad()
            lossf(model(Xtr_t[j]), ytr_t[j]).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            v = float(lossf(model(Xva_t), yva_t))
        if v < best - 1e-7:
            best, bad, best_ep = v, 0, ep
            state = {k: t.clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= 30:
                break

    model.load_state_dict(state)
    pt = out_dir / "rnn_carbonation.pt"
    torch.save(model.state_dict(), pt)

    meta = {
        "architecture": "Elman RNN, 1 layer, tanh",
        "hidden": HIDDEN,
        "window": WINDOW,
        "n_features": int(X.shape[2]),
        "feature_order": feat_cols + ["lag_annual_rate"] +
                         [f"target_year_{c}" for c in CLIMATE],
        "x_mean": mu.tolist(), "x_std": sd.tolist(),
        "y_mean": ymu, "y_std": ysd,
        "target": "annual increment of carb_depth_xc_mm (mm/yr)",
        "t_definition": f"t = year - {SERIES_START} + 1",
        "best_epoch": int(best_ep),
        "requires_future_climate": True,
        "warning": "no held-out data; do not quote accuracy from this fit",
    }
    js = out_dir / "rnn_carbonation.json"
    js.write_text(json.dumps(meta, indent=2))
    print(f"wrote {pt} and {js}   (stopped at epoch {best_ep})")

    print("\nTo project with the RNN, the caller must supply forward climate "
          "projections for every predicted year. Reconstruct depth by "
          "cumulative sum of the predicted increments from the last known depth.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("csv", type=Path)
    p.add_argument("--out", default="artifacts")
    args = p.parse_args()
    main(args.csv, args.out)
