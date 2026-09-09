"""
Shared evaluation harness for the concrete carbonation models.

Every model -- RNN, tabular (Ridge / RandomForest / CatBoost), and the physics
baselines -- is trained and scored *identically* here so "which model wins" has a
real answer. Nothing model-specific about data prep, the forecast rollout, or the
metrics lives outside this file.

THE SPEC
--------
Data      data/processed/concrete_kembla_v6_consolidated.csv (real, in repo)
Series    (asset_code, emissions_pathway, climate_model) -- 49
Target    annual_rate = carb_depth_xc_mm.diff() within each series (mm/yr)
Train     years <= CUTOFF_YEAR (2026)
Forecast  years CUTOFF_YEAR+1 .. 2070, FULLY AUTOREGRESSIVE -- every model feeds
          its own predicted rate forward as next year's lag, never the true one
Features  mean_temp_c, mean_rh_pct, co2_ppm, t_exposure, wc_ratio, fc_mpa, zone(1-hot)
          (+ lag_annual_rate, recursive; the RNN additionally sees a WINDOW-year history)
Excluded  carb_depth_increment_mm, carb_depth_xc_mm as a feature, age_years,
          cover_h_mm, member_diameter_mm, cement_content_kgm3, year, year_built,
          exposure_class, carbonation_class, chloride_class, *_prob, *_icorr, chloride_*
          (load_series already yields exactly the kept 9 columns)
Score     cumsum from the last known (CUTOFF_YEAR) depth -> MAE / RMSE / R2 /
          final-year abs error, vs the real 2027-2070 depth
Runs      primary  : fit on all 49 series' <=2026 history, forecast every series forward
          scenario : leave-one-scenario-out (7 folds), each still 2026-forward
          asset    : leave-one-asset-out   (7 folds), each still 2026-forward

Climate features stay REAL for every year including 2027-2070 -- they are the
workbook's scenario projections, known in advance. Only the lag switches from
observed to self-predicted past the cutoff.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

try:
    from .rnn_model import (
        SERIES_START,
        WINDOW,
        build_step,
        fit_wc_regression,
        load_series,
        train_rnn,
    )
except ImportError:  # run as a script from src/models/
    from rnn_model import (
        SERIES_START,
        WINDOW,
        build_step,
        fit_wc_regression,
        load_series,
        train_rnn,
    )

CUTOFF_YEAR = 2026
FINAL_YEAR = 2070
RNN_SEEDS = (0, 1, 2)
_EPS = 1e-12


# --------------------------------------------------------------------- helpers
def series_years(s: dict) -> np.ndarray:
    """Calendar year for every row of a series (t_exposure = year - 2000 + 1)."""
    return (s["t"] + SERIES_START - 1).astype(int)


def cut_index(s: dict, cutoff: int = CUTOFF_YEAR) -> int:
    """First row index whose year is strictly after the cutoff (start of the forecast)."""
    return int(np.searchsorted(series_years(s), cutoff + 1))


def _refit_slope(s: dict, cut: int) -> float:
    """xc^2 = m*t + c fitted on this series' own history up to the cutoff."""
    t, d = s["t"][:cut], s["depth"][:cut]
    m, _ = np.polyfit(t, d**2, 1)
    return float(m)


# --------------------------------------------------------------------- metrics
def score_forecast(pred_years: np.ndarray, pred_depth: np.ndarray, s: dict) -> dict:
    years = series_years(s)
    true_depth = s["depth"][np.isin(years, pred_years)]
    err = np.asarray(pred_depth) - true_depth
    ss = float(np.sum((true_depth - true_depth.mean()) ** 2))
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "r2": 1.0 - float(np.sum(err**2)) / ss if ss > 0 else float("nan"),
        "final_abs_err": float(abs(err[-1])),
        "n_years": int(len(err)),
    }


# ------------------------------------------------------------- central rollout
def rollout(model: Model, s: dict, cutoff: int = CUTOFF_YEAR) -> tuple[np.ndarray, np.ndarray]:
    """Autoregressive forecast of cumulative depth for one series, cutoff+1 .. end.

    `lag` holds the annual rate for every row: real up to `cut`, then overwritten
    with the model's own predictions as we step forward. Every model type goes
    through this exact loop -- no model runs its own rollout.
    """
    years = series_years(s)
    cut = cut_index(s, cutoff)
    lag = s["y"].astype(np.float64).copy()
    lag[cut:] = np.nan  # so any model that peeks at a future true rate blows up loudly
    depth = float(s["depth"][cut - 1])

    out_years, out_depth = [], []
    for i in range(cut, len(s["y"])):
        rate = float(model.rate(s, i, lag, cut))
        lag[i] = rate
        depth += rate
        out_years.append(int(years[i]))
        out_depth.append(depth)
    return np.asarray(out_years), np.asarray(out_depth)


# ---------------------------------------------------------------- model registry
@dataclass
class Model:
    """fit() sees only training series' <=cutoff history; rate() predicts one year."""

    name: str

    def fit(self, train: list[dict], cutoff: int) -> None:  # noqa: D401
        ...

    def rate(self, s: dict, i: int, lag: np.ndarray, cut: int) -> float:
        raise NotImplementedError

    @property
    def available(self) -> bool:
        return True


class PersistenceModel(Model):
    def __init__(self) -> None:
        super().__init__("persistence")

    def rate(self, s, i, lag, cut):
        return lag[i - 1]  # last observed rate, then held flat (lag[i-1] is our own prev pred)


class MeanModel(Model):
    def __init__(self) -> None:
        super().__init__("mean")
        self._rate = 0.0

    def fit(self, train, cutoff):
        vals = []
        for s in train:
            cut = cut_index(s, cutoff)
            vals.append(s["y"][1:cut])  # row 0 has no rate
        self._rate = float(np.mean(np.concatenate(vals)))

    def rate(self, s, i, lag, cut):
        return self._rate


class SqrtLawModel(Model):
    """Per-series naive extrapolation of xc^2 = m*t + c fitted on <=cutoff history."""

    def __init__(self) -> None:
        super().__init__("sqrt_law")
        self._mc: dict[tuple, tuple[float, float]] = {}

    def _params(self, s, cut):
        key = (s["asset"], s["scenario"])
        if key not in self._mc:
            m, c = np.polyfit(s["t"][:cut], s["depth"][:cut] ** 2, 1)
            self._mc[key] = (float(m), float(c))
        return self._mc[key]

    def rate(self, s, i, lag, cut):
        m, c = self._params(s, cut)
        d_i = np.sqrt(max(m * s["t"][i] + c, _EPS))
        d_prev = np.sqrt(max(m * s["t"][i - 1] + c, _EPS))
        return d_i - d_prev


class WcRegressModel(Model):
    """log(m) = a*wc + b fitted across training series; c anchored at the cutoff depth."""

    def __init__(self) -> None:
        super().__init__("wc_regress")
        self._a = self._b = 0.0

    def fit(self, train, cutoff):
        rebased = []
        for s in train:
            cut = cut_index(s, cutoff)
            rebased.append({"wc": s["wc"], "m_true": _refit_slope(s, cut)})
        self._a, self._b = fit_wc_regression(rebased)

    def rate(self, s, i, lag, cut):
        m = float(np.exp(self._a * s["wc"] + self._b))
        c = s["depth"][cut - 1] ** 2 - m * s["t"][cut - 1]
        d_i = np.sqrt(max(m * s["t"][i] + c, _EPS))
        d_prev = np.sqrt(max(m * s["t"][i - 1] + c, _EPS))
        return d_i - d_prev


class RnnModel(Model):
    """Elman RNN over WINDOW-year windows; identical training code to rnn_model.train_rnn."""

    def __init__(self, seeds: tuple[int, ...] = RNN_SEEDS, val_fraction: float = 0.15) -> None:
        super().__init__("rnn")
        self.seeds = seeds
        self.val_fraction = val_fraction
        self._fits: list[tuple] = []

    def _windows(self, train: list[dict], cutoff: int):
        xs, ys = [], []
        for s in train:
            cut = cut_index(s, cutoff)
            X, y = s["X"], s["y"]
            for i in range(WINDOW, cut):  # target year must be <= cutoff
                xs.append(build_step(X, y, i, WINDOW))
                ys.append(y[i])
        return np.stack(xs), np.asarray(ys, dtype=np.float32)

    def fit(self, train, cutoff):
        X, y = self._windows(train, cutoff)
        rng = np.random.default_rng(0)
        idx = rng.permutation(len(X))
        n_val = max(1, int(len(X) * self.val_fraction))
        va, tr = idx[:n_val], idx[n_val:]
        self._fits = [
            train_rnn(X[tr], y[tr], X[va], y[va], seed) for seed in self.seeds
        ]

    def rate(self, s, i, lag, cut):
        import torch

        step = build_step(s["X"], lag.astype(np.float32), i, WINDOW)
        preds = []
        for model, mu, sd, ymu, ysd in self._fits:
            xn = ((step - mu) / sd).astype(np.float32)
            with torch.no_grad():
                preds.append(float(model(torch.tensor(xn[None]))) * ysd + ymu)
        return float(np.mean(preds))


class _TabularModel(Model):
    """Flat one-step regressor: predict annual_rate[i] from year i's features + lag[i-1]."""

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._est = None
        self._scaler = None

    def _make_estimator(self):
        raise NotImplementedError

    def _needs_scaling(self) -> bool:
        return False

    def _rows(self, train: list[dict], cutoff: int):
        X, y = [], []
        for s in train:
            cut = cut_index(s, cutoff)
            for i in range(1, cut):  # need lag[i-1]; target year <= cutoff
                X.append(np.append(s["X"][i], s["y"][i - 1]))
                y.append(s["y"][i])
        return np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.float64)

    def fit(self, train, cutoff):
        X, y = self._rows(train, cutoff)
        if self._needs_scaling():
            from sklearn.preprocessing import StandardScaler

            self._scaler = StandardScaler().fit(X)
            X = self._scaler.transform(X)
        self._est = self._make_estimator()
        self._est.fit(X, y)

    def rate(self, s, i, lag, cut):
        row = np.append(s["X"][i], lag[i - 1]).reshape(1, -1).astype(np.float64)
        if self._scaler is not None:
            row = self._scaler.transform(row)
        return float(self._est.predict(row)[0])


class RidgeModel(_TabularModel):
    def __init__(self) -> None:
        super().__init__("ridge")

    def _needs_scaling(self) -> bool:
        return True

    def _make_estimator(self):
        from sklearn.linear_model import Ridge

        return Ridge(alpha=1.0)


class RandomForestModel(_TabularModel):
    def __init__(self) -> None:
        super().__init__("random_forest")

    def _make_estimator(self):
        from sklearn.ensemble import RandomForestRegressor

        return RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)


class CatBoostModel(_TabularModel):
    def __init__(self) -> None:
        super().__init__("catboost")

    @property
    def available(self) -> bool:
        try:
            import catboost  # noqa: F401

            return True
        except ImportError:
            return False

    def _make_estimator(self):
        from catboost import CatBoostRegressor

        return CatBoostRegressor(
            iterations=600, learning_rate=0.04, depth=6,
            loss_function="RMSE", random_seed=42, verbose=0,
        )


_REGISTRY = {
    "persistence": PersistenceModel,
    "mean": MeanModel,
    "sqrt_law": SqrtLawModel,
    "wc_regress": WcRegressModel,
    "rnn": RnnModel,
    "ridge": RidgeModel,
    "random_forest": RandomForestModel,
    "catboost": CatBoostModel,
}
ALL_MODELS = tuple(_REGISTRY)


def make_model(name: str) -> Model:
    return _REGISTRY[name]()


# --------------------------------------------------------------------- driver
@dataclass
class FoldResult:
    run: str
    held_out: str
    per_series: list[dict] = field(default_factory=list)


def _fit_and_forecast(model: Model, train: list[dict], test: list[dict], cutoff: int) -> list[dict]:
    model.fit(train, cutoff)
    rows = []
    for s in test:
        yrs, dep = rollout(model, s, cutoff)
        m = score_forecast(yrs, dep, s)
        m.update(asset=s["asset"], scenario=s["scenario"], model=model.name)
        rows.append(m)
    return rows


def run_benchmark(
    csv_path,
    model_names: list[str],
    runs: tuple[str, ...] = ("primary", "scenario", "asset"),
    cutoff: int = CUTOFF_YEAR,
):
    series, _ = load_series(csv_path)
    models = [make_model(n) for n in model_names]
    skipped = [m.name for m in models if not m.available]
    models = [m for m in models if m.available]

    detail: list[dict] = []
    for name in [m.name for m in models]:
        model = make_model(name)
        if "primary" in runs:
            for r in _fit_and_forecast(model, series, series, cutoff):
                detail.append({"run": "primary", "held_out": "ALL", **r})
        for axis, key in (("scenario", "scenario"), ("asset", "asset")):
            if axis not in runs:
                continue
            for group in sorted({s[key] for s in series}):
                tr = [s for s in series if s[key] != group]
                te = [s for s in series if s[key] == group]
                model = make_model(name)
                for r in _fit_and_forecast(model, tr, te, cutoff):
                    detail.append({"run": axis, "held_out": group, **r})

    detail_df = pd.DataFrame(detail)
    summary = (
        detail_df.groupby(["run", "held_out", "model"])
        .agg(
            n_series=("mae", "size"),
            mae=("mae", "mean"),
            mae_std=("mae", "std"),
            rmse=("rmse", "mean"),
            r2=("r2", "mean"),
            final_mae=("final_abs_err", "mean"),
        )
        .reset_index()
    )
    return summary, detail_df, skipped
