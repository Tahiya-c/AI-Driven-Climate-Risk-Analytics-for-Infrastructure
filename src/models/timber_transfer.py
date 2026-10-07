"""Train on PileA, test on PileB: hazard-normalised MLP/GBM for timber material loss.

    python -m src.models.timber_transfer --select-features        # feature set (PileA hold-out)
    python -m src.models.timber_transfer --tune gbm --trials 50   # Optuna on the PileA hold-out
    python -m src.models.timber_transfer --tune mlp --trials 50
    python -m src.models.timber_transfer --evaluate               # final PileB scores + ablation
    python -m src.models.timber_transfer --save --force           # save the models of record

Both models use the same features, the same target and the same techniques:

- **Target:** radial material loss in mm, ``(initial_diameter - max(remaining, 0)) / 2``
  -- the dashboard's metric. Capped at the pile radius, so it is bounded.
- **Hazard normalisation:** the inner estimator learns ``(loss - sapwood loss) / k0``
  (``combined_factor``), and predictions are multiplied back by the test pile's own ``k0``
  before the sapwood loss is added back. AS 5604-style heartwood decay scales with k0, and
  PileA's k0 is a single constant (1.3), so without this a model trained only on PileA has
  no way to learn that PileB (k0 = 3.12) decays ~2.4x faster. The sapwood layer is lost
  whole in year one whatever the hazard (Defect D1), so it is kept out of the scaling.
- **Zero-variance filter:** features constant across the training rows (every pile-level
  k factor, initial diameter) are dropped at fit time. They carry nothing learnable from
  one pile, and the MLP otherwise feeds PileB's out-of-range constants into random weights.
- **Model selection on PileA only, by a chronological hold-out:** each PileA climate series
  trains on its earlier years and validates on its last 20% (2057-2070). The feature set
  (with or without ``thermal_dose``) and all Optuna hyperparameters are chosen on that
  hold-out. It scores forecasting forward in time; Monte Carlo CV (random row splits)
  was dropped because neighbouring years leaked between train and validation, giving
  R2 ~0.999 for almost any model. PileB is touched once, for the final score.

The tuned hyperparameters live with each model: TRANSFER_MLP_PARAMS in
src/models/timber_model.py and TRANSFER_GBM_PARAMS in src/models/timber_model_gbm.py.
Write-up and results: docs/timber_model_updated.md.
"""

from __future__ import annotations

import argparse
import json
import warnings
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import MODEL_DIR
from src.data.timber_data_prep import DATA_PATH, load_timber_data
from src.features.timber_features import FEATURE_COLUMNS, TARGET_COLUMN, add_thermal_dose
from src.models.timber_model import (
    MODEL_KINDS,
    TRANSFER_MLP_PARAMS,
    build_estimator,
    suggest_mlp_params,
)
from src.models.timber_model_gbm import TRANSFER_GBM_PARAMS, suggest_gbm_params

TRAIN_PILE = "PileA"
TEST_PILE = "PileB"
SPLIT_YEAR = 2026  # dashboard history/forecast boundary
LOSS_COLUMN = "material_loss_mm"
HAZARD_COLUMN = "combined_factor"  # k0
# d_sap: a pile attribute (workbook column M), not a model input. Passed alongside
# FEATURE_COLUMNS so the sapwood loss can be taken out before hazard normalisation.
SAPWOOD_COLUMN = "layer_of_sapwood_mm"
MODEL_INPUT_COLUMNS = [*FEATURE_COLUMNS, SAPWOOD_COLUMN]
SHIFT_WINDOW = (2022, SPLIT_YEAR)
LARGE_OFFSET_MM = 15.0  # a shift bigger than this means the model is badly off for that pile
FEATURE_SETS = ("standard", "thermal_dose")
HOLDOUT_FRACTION = 0.2  # last 20% of each PileA climate series' years validate
SERIES_KEY = ["scenario_label", "emissions_pathway"]
RESULTS_PATH = MODEL_DIR / "timber_transfer_results.json"
SAVED_MODEL_KINDS = ("gbm", "mlp")


def saved_model_path(model_kind: str):
    return MODEL_DIR / f"timber_transfer_{model_kind}.joblib"


# Output of ``--select-features``. Tuned hyperparameters: TRANSFER_MLP_PARAMS
# (timber_model.py), TRANSFER_GBM_PARAMS (timber_model_gbm.py), TRANSFER_RIDGE_PARAMS below.
SELECTED_FEATURE_SET = "thermal_dose"
# Ridge regression: the linear baseline compared against the MLP (same features, target,
# hazard normalisation, zero-variance filter and Optuna + hold-out protocol).
TRANSFER_RIDGE_PARAMS: dict[str, Any] = {"alpha": 9.047071957568372}
TRANSFER_KINDS = (*MODEL_KINDS, "ridge")


def material_loss(frame: pd.DataFrame, remaining_diameter=None) -> pd.Series:
    """Radial loss in mm, capped at the pile radius (remaining diameter floored at 0)."""
    remaining = frame[TARGET_COLUMN] if remaining_diameter is None else remaining_diameter
    remaining = pd.Series(np.asarray(remaining, dtype=float), index=frame.index).clip(lower=0)
    return (frame["initial_diameter_mm"] - remaining) / 2


def load_transfer_frame() -> pd.DataFrame:
    frame = load_timber_data(DATA_PATH)
    frame[LOSS_COLUMN] = material_loss(frame)
    return frame


def sapwood_loss(X: pd.DataFrame) -> np.ndarray:
    """The engine's sapwood loss: the whole ``d_sap`` layer once age > 0, else 0.

    Matches ``timber_engine.simulate`` (Defect D1): the full layer is lost in the first
    year after installation, whatever the pile's hazard. Uses only age and the pile's
    sapwood layer, never the target.
    """
    if SAPWOOD_COLUMN not in X:
        raise ValueError(
            f"X needs {SAPWOOD_COLUMN!r} (the pile's sapwood layer) as well as "
            "FEATURE_COLUMNS: pass rows[MODEL_INPUT_COLUMNS]."
        )
    age = X["age_years"].to_numpy(dtype=float)
    return np.where(age > 0, X[SAPWOOD_COLUMN].to_numpy(dtype=float), 0.0)


class HazardNormalisedRegressor(RegressorMixin, BaseEstimator):
    """Learn heartwood loss per unit hazard; predict material loss for any pile.

    ``X`` is the MODEL_INPUT_COLUMNS matrix (FEATURE_COLUMNS plus the sapwood layer).
    The fixed sapwood loss (``sapwood_loss``) is taken out before dividing by k0 and
    added back after multiplying, so it is not scaled by the pile's hazard: the
    spreadsheet removes the same layer whatever the hazard, and scaling it put PileB
    ~28 mm too high. ``feature_set="thermal_dose"`` appends ``age x water_factor``
    (see ``add_thermal_dose``). Columns with zero variance in the training rows are
    dropped. Predictions are clipped to [0, pile radius].
    """

    def __init__(self, estimator: Any, feature_set: str = "standard"):
        self.estimator = estimator
        self.feature_set = feature_set

    def _features(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.drop(columns=SAPWOOD_COLUMN, errors="ignore")
        return add_thermal_dose(X) if self.feature_set == "thermal_dose" else X

    def fit(self, X: pd.DataFrame, y) -> HazardNormalisedRegressor:
        features = self._features(X)
        # nunique, not std() > 0: a repeated float such as 1.3 has a std of ~1e-16, which
        # would wrongly keep pile-constant factors (heartwood_factor, combined_factor).
        self.kept_columns_ = list(features.columns[features.nunique() > 1])
        heartwood = np.asarray(y, dtype=float) - sapwood_loss(X)
        target = heartwood / X[HAZARD_COLUMN].to_numpy(dtype=float)
        self.estimator_ = clone(self.estimator).fit(features[self.kept_columns_], target)
        return self

    def predict_uncapped(self, X: pd.DataFrame) -> np.ndarray:
        """Predicted loss before clipping to [0, radius] (used by the shift/anchor functions)."""
        per_hazard = self.estimator_.predict(self._features(X)[self.kept_columns_])
        heartwood = np.asarray(per_hazard, dtype=float) * X[HAZARD_COLUMN].to_numpy(dtype=float)
        return sapwood_loss(X) + heartwood

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        loss = self.predict_uncapped(X)
        return np.clip(loss, 0, X["initial_diameter_mm"].to_numpy(dtype=float) / 2)


def anchor_forecast(
    rows: pd.DataFrame, uncapped: np.ndarray, split_year: int = SPLIT_YEAR
) -> np.ndarray:
    """Shift each series' forecast so it starts from that series' actual loss at ``split_year``.

    ``rows`` holds one pile's rows; ``uncapped`` is ``predict_uncapped`` for them. Per
    (scenario_label, emissions_pathway) series, offset = actual - predicted at
    ``split_year``. The offset is added *before* clipping to [0, radius], so the
    forecast can still reach the full radius. It corrects whatever constant error the
    model has for this pile using one real observation, the latest history value. The
    result is no longer a pure PileA-only prediction.
    """
    keys = ["scenario_label", "emissions_pathway"]
    frame = rows[[*keys, "year", LOSS_COLUMN, "initial_diameter_mm"]].assign(raw=uncapped)
    at_split = frame.loc[frame["year"] == split_year].set_index(keys)
    offset = (at_split[LOSS_COLUMN] - at_split["raw"]).rename("offset")
    frame = frame.join(offset, on=keys)
    anchored = frame["raw"] + frame["offset"].fillna(0.0)
    return np.clip(anchored, 0, frame["initial_diameter_mm"] / 2).to_numpy()


def flag_large_offsets(offsets: pd.Series, label: str, threshold: float) -> list[dict]:
    """Warn about, and return, every offset whose size exceeds ``threshold`` mm.

    A shift that large means the model is badly off for that pile, so it is reported
    rather than applied silently. ``offsets`` is indexed by series (or asset).
    """
    large = offsets[offsets.abs() > threshold]
    for key, value in large.items():
        warnings.warn(
            f"{label}: shift of {value:+.1f} mm for {key} exceeds {threshold:g} mm; "
            "the model is far from this pile's measured values.",
            stacklevel=3,
        )
    return [
        {"series": key if isinstance(key, str) else " | ".join(map(str, key)),
         "offset_mm": float(value)}
        for key, value in large.items()
    ]


def shift_forecast(
    rows: pd.DataFrame,
    uncapped: np.ndarray,
    window: tuple[int, int] = SHIFT_WINDOW,
    max_offset: float = LARGE_OFFSET_MM,
) -> tuple[np.ndarray, dict]:
    """Shift each series by the model's mean error against the real values in ``window``.

    Like ``anchor_forecast`` but averaged over several years (default 2022-2026), so one
    outlying value can't set the offset. Per (scenario_label, emissions_pathway) series,
    offset = mean(actual - predicted) over the window, added before clipping to
    [0, radius]. Every row is shifted; callers show real values for the history years.
    Returns the shifted values and a record of the offsets, with any above
    ``max_offset`` mm listed (and warned about) under ``"large_offsets"``. These scores
    use the pile's own history, so they are an "after inspection update" figure, not
    model accuracy.
    """
    keys = ["scenario_label", "emissions_pathway"]
    frame = rows[[*keys, "year", LOSS_COLUMN, "initial_diameter_mm"]].assign(raw=uncapped)
    in_window = frame.loc[frame["year"].between(*window)]
    offset = (
        (in_window[LOSS_COLUMN] - in_window["raw"])
        .groupby([in_window[key] for key in keys])
        .mean()
        .rename("offset")
    )
    frame = frame.join(offset, on=keys)
    shifted = frame["raw"] + frame["offset"].fillna(0.0)
    record = {
        "window": list(window),
        "threshold_mm": max_offset,
        "offsets_mm": {" | ".join(map(str, k)): float(v) for k, v in offset.items()},
        "large_offsets": flag_large_offsets(offset, "shift_forecast", max_offset),
    }
    return np.clip(shifted, 0, frame["initial_diameter_mm"] / 2).to_numpy(), record


def suggest_ridge_params(trial) -> dict[str, Any]:
    """Optuna search space for Ridge: the L2 regularisation strength only."""
    return {"alpha": trial.suggest_float("alpha", 1e-4, 1e3, log=True)}


def _inner_estimator(model_kind: str, params: dict[str, Any], random_state: int):
    """Ridge scales X and y like the MLP; MLP/GBM come from timber_model.build_estimator."""
    if model_kind == "ridge":
        return TransformedTargetRegressor(
            regressor=Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(**params))]),
            transformer=StandardScaler(),
        )
    return build_estimator(model_kind, params, random_state)


def build_transfer_model(
    model_kind: str,
    params: dict[str, Any] | None = None,
    feature_set: str | None = None,
    random_state: int = 42,
) -> HazardNormalisedRegressor:
    """Defaults: the stored TRANSFER_*_PARAMS and SELECTED_FEATURE_SET."""
    if model_kind not in TRANSFER_KINDS:
        raise ValueError(f"model_kind must be one of {TRANSFER_KINDS}, got {model_kind!r}")
    if params is None:
        params = {
            "mlp": TRANSFER_MLP_PARAMS,
            "gbm": TRANSFER_GBM_PARAMS,
            "ridge": TRANSFER_RIDGE_PARAMS,
        }[model_kind]
    return HazardNormalisedRegressor(
        _inner_estimator(model_kind, params, random_state),
        feature_set=feature_set or SELECTED_FEATURE_SET,
    )


def metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def _fit(model, X, y):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return model.fit(X, y)


def chronological_holdout(rows: pd.DataFrame) -> tuple[pd.Index, pd.Index]:
    """Per climate series: earliest (1 - HOLDOUT_FRACTION) of years fit, the rest validate."""
    fit_parts, val_parts = [], []
    for _, series in rows.groupby(SERIES_KEY, sort=False):
        years = np.sort(series["year"].unique())
        cut = years[int(round(len(years) * (1 - HOLDOUT_FRACTION)))]
        fit_parts.append(series.index[series["year"] < cut])
        val_parts.append(series.index[series["year"] >= cut])
    return pd.Index(np.concatenate(fit_parts)), pd.Index(np.concatenate(val_parts))


def holdout_metrics(make_model, rows: pd.DataFrame) -> dict[str, float]:
    """Fit on the earlier years of ``rows``; score R2/RMSE/MAE on the held-out later years."""
    fit_idx, val_idx = chronological_holdout(rows)
    X, y = rows[MODEL_INPUT_COLUMNS], rows[LOSS_COLUMN]
    model = _fit(make_model(), X.loc[fit_idx], y.loc[fit_idx])
    return metrics(y.loc[val_idx], model.predict(X.loc[val_idx]))


def _pile_rows(frame: pd.DataFrame, pile: str) -> pd.DataFrame:
    return frame.loc[frame["asset_code"] == pile]


def _pile(frame: pd.DataFrame, pile: str) -> tuple[pd.DataFrame, pd.Series]:
    rows = _pile_rows(frame, pile)
    return rows[MODEL_INPUT_COLUMNS], rows[LOSS_COLUMN]


def select_feature_set(frame: pd.DataFrame) -> dict[str, Any]:
    """Pick the shared feature set by mean hold-out RMSE on PileA across both models."""
    rows = _pile_rows(frame, TRAIN_PILE)
    results = {}
    for feature_set in FEATURE_SETS:
        per_model = {
            kind: holdout_metrics(lambda k=kind: build_transfer_model(k, {}, feature_set), rows)
            for kind in MODEL_KINDS
        }
        results[feature_set] = {
            "per_model": per_model,
            "mean_rmse": float(np.mean([m["rmse"] for m in per_model.values()])),
        }
    best = min(results, key=lambda name: results[name]["mean_rmse"])
    return {"selected": best, "results": results}


def tune(model_kind: str, n_trials: int, frame: pd.DataFrame, seed: int = 42) -> dict[str, Any]:
    """Optuna TPE search; objective = RMSE (mm) on PileA's chronological hold-out."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    rows = _pile_rows(frame, TRAIN_PILE)
    fit_idx, val_idx = chronological_holdout(rows)
    X, y = rows[MODEL_INPUT_COLUMNS], rows[LOSS_COLUMN]
    suggest = {
        "mlp": suggest_mlp_params,
        "gbm": suggest_gbm_params,
        "ridge": suggest_ridge_params,
    }[model_kind]

    def objective(trial) -> float:
        model = _fit(
            build_transfer_model(model_kind, suggest(trial), random_state=seed),
            X.loc[fit_idx],
            y.loc[fit_idx],
        )
        return float(mean_squared_error(y.loc[val_idx], model.predict(X.loc[val_idx])) ** 0.5)

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    validation_years = rows.loc[val_idx, "year"]
    return {
        "model_kind": model_kind,
        "n_trials": n_trials,
        "holdout": f"last {HOLDOUT_FRACTION:.0%} of years per {TRAIN_PILE} climate series",
        "validation_years": [int(validation_years.min()), int(validation_years.max())],
        "holdout_rmse": study.best_value,
        "optuna_params": study.best_params,
        "estimator_params": suggest(optuna.trial.FixedTrial(study.best_params)),
    }


def forecast_mask(frame: pd.DataFrame) -> pd.Series:
    """Rows the dashboard shows as forecasts: after SPLIT_YEAR, the 3 climate scenarios."""
    return (frame["year"] > SPLIT_YEAR) & (frame["scenario_label"] != "Baseline")


def evaluate(frame: pd.DataFrame, seeds: tuple[int, ...] = (0, 1, 2, 3, 42)) -> dict[str, Any]:
    """Ablation, final PileB scores, PileA hold-out scores and seed stability."""
    X_a, y_a = _pile(frame, TRAIN_PILE)
    pile_b = frame.loc[frame["asset_code"] == TEST_PILE]
    X_b, y_b = pile_b[MODEL_INPUT_COLUMNS], pile_b[LOSS_COLUMN]
    forecast = forecast_mask(pile_b)

    from src.models.timber_model import build_pipeline

    class _Plain:
        """Step 1: original untuned estimator on the raw loss target."""

        def __init__(self, kind):
            self.model = (
                Pipeline([("scaler", StandardScaler()), ("ridge", Ridge())])
                if kind == "ridge"
                else build_pipeline(kind)
            )

        def fit(self, X, y):
            self.model.fit(X[FEATURE_COLUMNS], y)
            return self

        def predict(self, X):
            prediction = self.model.predict(X[FEATURE_COLUMNS])
            return np.clip(prediction, 0, X["initial_diameter_mm"].to_numpy() / 2)

    steps = {
        "1_baseline": lambda kind: _Plain(kind),
        "2_hazard_normalised": lambda kind: build_transfer_model(kind, {}),
        "3_tuned": lambda kind: build_transfer_model(kind),
    }
    ablation = []
    for step, make in steps.items():
        for kind in TRANSFER_KINDS:
            model = _fit(make(kind), X_a, y_a)
            prediction = model.predict(X_b)
            ablation.append(
                {
                    "step": step,
                    "model": kind,
                    **{f"pileB_all_{k}": v for k, v in metrics(y_b, prediction).items()},
                    **{
                        f"pileB_forecast_{k}": v
                        for k, v in metrics(y_b[forecast], prediction[forecast.to_numpy()]).items()
                    },
                }
            )

    final = {}
    for kind in TRANSFER_KINDS:
        per_seed = []
        for seed in seeds:
            model = _fit(build_transfer_model(kind, random_state=seed), X_a, y_a)
            per_seed.append(metrics(y_b, model.predict(X_b)))
        final[kind] = {
            "holdout_pileA": holdout_metrics(
                lambda k=kind: build_transfer_model(k), _pile_rows(frame, TRAIN_PILE)
            ),
            "seed_stability_pileB": {
                name: {
                    "mean": float(np.mean([s[name] for s in per_seed])),
                    "std": float(np.std([s[name] for s in per_seed])),
                }
                for name in ("r2", "rmse", "mae")
            },
        }
    return {
        "feature_set": SELECTED_FEATURE_SET,
        "split_year": SPLIT_YEAR,
        "n_train_pileA": int(len(X_a)),
        "n_test_pileB": int(len(X_b)),
        "n_test_pileB_forecast": int(forecast.sum()),
        "ablation": ablation,
        "final": final,
        "seeds": list(seeds),
    }


def library_versions() -> dict[str, str]:
    """Versions a saved model depends on; joblib files are not portable across them."""
    import platform

    import lightgbm
    import sklearn

    return {
        "python": platform.python_version(),
        "scikit-learn": sklearn.__version__,
        "lightgbm": lightgbm.__version__,
        "numpy": np.__version__,
        "joblib": joblib.__version__,
    }


def save_models(
    frame: pd.DataFrame, kinds: tuple[str, ...] = SAVED_MODEL_KINDS, overwrite: bool = False
) -> dict:
    """Fit each model on all PileA rows; save it with joblib plus a metadata JSON.

    The saved files are the models of record: every imported pile is predicted with
    them (src/models/timber_forecast.py), never with a freshly trained model. An existing
    model is therefore never replaced unless ``overwrite=True`` (``--save --force``).
    Load with ``joblib.load(saved_model_path("gbm"))``; ``.predict(frame[MODEL_INPUT_COLUMNS])``
    returns material loss in mm, and ``anchor_forecast`` works with ``.predict_uncapped``.
    """
    existing = [str(saved_model_path(kind)) for kind in kinds if saved_model_path(kind).exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"Saved models already exist: {existing}. Imports must keep using the same "
            "model; pass overwrite=True (--save --force) only to deliberately replace it."
        )
    from datetime import UTC, datetime

    saved_at = datetime.now(UTC)
    X_a, y_a = _pile(frame, TRAIN_PILE)
    pile_b = frame.loc[frame["asset_code"] == TEST_PILE]
    forecast = forecast_mask(pile_b).to_numpy()
    summary = {}
    for kind in kinds:
        model = _fit(build_transfer_model(kind), X_a, y_a)
        path = saved_model_path(kind)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, path)
        prediction = model.predict(pile_b[MODEL_INPUT_COLUMNS])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            shifted, shift_record = shift_forecast(
                pile_b, model.predict_uncapped(pile_b[MODEL_INPUT_COLUMNS])
            )
        meta = {
            "model_version": f"timber_transfer_{kind}_{saved_at:%Y%m%d_%H%M%S}",
            "saved_at_utc": saved_at.isoformat(timespec="seconds"),
            "library_versions": library_versions(),
            "model_kind": kind,
            "trained_on": TRAIN_PILE,
            "n_train_rows": int(len(X_a)),
            "target": f"{LOSS_COLUMN} (radial material loss, mm, capped at pile radius)",
            "input_columns": MODEL_INPUT_COLUMNS,
            "sapwood_column": SAPWOOD_COLUMN,
            "used_columns": model.kept_columns_,
            "feature_set": model.feature_set,
            "hazard_column": HAZARD_COLUMN,
            "params": repr(
                {
                    "mlp": TRANSFER_MLP_PARAMS,
                    "gbm": TRANSFER_GBM_PARAMS,
                    "ridge": TRANSFER_RIDGE_PARAMS,
                }[kind]
            ),
            f"{TEST_PILE}_all_years": metrics(pile_b[LOSS_COLUMN], prediction),
            f"{TEST_PILE}_forecast_{SPLIT_YEAR + 1}_on": metrics(
                pile_b[LOSS_COLUMN][forecast], prediction[forecast]
            ),
            # After-inspection-update figure only: uses PileB's own 2022-2026 values.
            f"{TEST_PILE}_forecast_{SPLIT_YEAR + 1}_on_shifted": metrics(
                pile_b[LOSS_COLUMN][forecast], shifted[forecast]
            ),
            "shift": shift_record,
        }
        path.with_suffix(".json").write_text(json.dumps(meta, indent=2))
        summary[kind] = {"path": str(path), **meta}
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--select-features", action="store_true")
    parser.add_argument("--tune", choices=TRANSFER_KINDS)
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--save", action="store_true", help="fit GBM + MLP on PileA and save them")
    parser.add_argument(
        "--force", action="store_true", help="with --save: replace existing saved models"
    )
    args = parser.parse_args()
    if not (args.select_features or args.tune or args.evaluate or args.save):
        parser.error("pass --select-features, --tune, --evaluate and/or --save")

    frame = load_transfer_frame()
    results = json.loads(RESULTS_PATH.read_text()) if RESULTS_PATH.exists() else {}
    if args.select_features:
        selection = select_feature_set(frame)
        print(json.dumps(selection, indent=2))
        print(f"Set SELECTED_FEATURE_SET = {selection['selected']!r}")
        results["feature_selection"] = selection
    if args.tune:
        best = tune(args.tune, args.trials, frame)
        print(f"Best hold-out RMSE on PileA: {best['holdout_rmse']:.2f} mm")
        where = {"mlp": "timber_model.py", "gbm": "timber_model_gbm.py"}.get(
            args.tune, "timber_transfer.py"
        )
        params = best["estimator_params"]
        print(f"Paste into TRANSFER_{args.tune.upper()}_PARAMS ({where}): {params!r}")
        results[f"tuning_{args.tune}"] = {
            **best,
            "estimator_params": repr(best["estimator_params"]),
        }
    if args.evaluate:
        evaluation = evaluate(frame)
        table = pd.DataFrame(evaluation["ablation"])
        print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        print(json.dumps(evaluation["final"], indent=2))
        results["evaluation"] = evaluation
    if args.save:
        for kind, info in save_models(frame, overwrite=args.force).items():
            print(f"saved {kind} -> {info['path']} (+ .json metadata)")
    if not (args.select_features or args.tune or args.evaluate):
        return
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    print(f"Saved results to {RESULTS_PATH}")


if __name__ == "__main__":
    # Run through the importable module, not __main__, so saved models pickle their
    # class as src.models.timber_transfer.HazardNormalisedRegressor and load anywhere.
    from src.models import timber_transfer

    timber_transfer.main()
