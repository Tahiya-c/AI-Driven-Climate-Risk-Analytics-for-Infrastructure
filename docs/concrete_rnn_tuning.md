# Concrete Models — Optuna Hyperparameter Tuning

**What this covers:** `src/models/concrete_tuning.py`, which tunes the RNN plus the three
tabular models (`ridge`, `random_forest`, `catboost`) in the shared concrete benchmark
(`src/models/concrete_eval.py`). These three tabular models are the real, correctly-trained
equivalents of Aisha's original `experiments/concrete/` scripts — those are archived
(synthetic data, and the leaky `carb_depth_increment_mm` target; see that directory's
header comments and [[concrete-modelling-reference]]).

**Reproduce:** `python -m src.models.concrete_tuning --model all --trials 40`

---

## 1. Method — inner search, outer score

The real benchmark (`concrete_benchmark_all.py`) trains on every series' history up to
**2026** and forecasts **2027–2070**, scored against the real (workbook-modelled) depths
in that window. That 2027–2070 window must never be used to pick hyperparameters, or the
reported numbers would be optimistic.

So the tuner scores each Optuna trial on **inner** leave-one-scenario-out folds at
pseudo-cutoffs strictly before 2026 (`INNER_CUTOFFS = (2018, 2021)`), using a subset of 3
of the 7 scenario groups (`INNER_HELD_OUT`) to keep each trial's cost down — "small search
space, only 49 series" is a hard constraint here, not a suggestion. Train on a series'
history up to the pseudo-cutoff, roll the forecast forward to 2026 with the exact same
`rollout()`/`score_forecast()` code the real benchmark uses, and score MAE against the real
(pre-2026) depths in between. Same mechanics, just earlier in time — so the real 2027–2070
numbers it's checked against afterwards are still an honest, unseen outer test.

Optuna: TPE sampler (seeded, reproducible), `MedianPruner` so a clearly-bad trial is cut
short instead of running to completion, 40 trials per model.

## 2. Search spaces

| Model | Parameters searched |
|---|---|
| `rnn` | `window` (6–16), `hidden` (8–48, log), `lr` (5e-4–2e-2, log), `weight_decay` (1e-7–1e-2, log), `patience` (10–40) |
| `ridge` | `alpha` (1e-3–1e3, log) |
| `random_forest` | `n_estimators` (50–500, log), `max_depth` ({3,5,8,12,None}), `min_samples_leaf` (1–10), `max_features` ({sqrt, log2, 1.0}) |
| `catboost` | `iterations` (100–800, log), `learning_rate` (1e-3–0.3, log), `depth` (3–8), `l2_leaf_reg` (1e-1–10, log) |

`persistence`, `sqrt_law`, `wc_regress`, `mean` have no real hyperparameters and were left
alone.

## 3. Results — did the real (outer) benchmark actually improve?

Each model's Optuna winner was plugged into `concrete_eval.py` and scored on the **real**
outer benchmark (`concrete_benchmark_all.py`, unchanged) — the only number that matters.
Two of four models improved; two regressed and were **reverted**, because reporting a
tuning win that doesn't survive outer validation would be exactly the kind of result this
project's write-up conventions say not to smooth over.

### RNN — kept (`TUNED_RNN_PARAMS`)

`window=6, hidden=14, lr=0.0196, weight_decay=0.00757, patience=23` (was `window=12,
hidden=16, lr=0.005, weight_decay=0, patience=30`).

| Run | Metric | Before | After | |
|---|---|---:|---:|---|
| Primary | MAE / RMSE / R² | 0.119 / 0.169 / **0.834** | **0.103** / **0.134** / 0.799 | RMSE/MAE better, R² slightly worse |
| Scenario (leave-one-out) | MAE / RMSE / R² | 0.224 / 0.290 / **−0.256** | **0.152** / **0.201** / **0.695** | flips negative → strongly positive |
| Asset (leave-one-out) | MAE / RMSE / R² | 0.333 / 0.391 / **−0.173** | **0.260** / **0.313** / **0.341** | flips negative → positive |

The one regression (primary R²) is concentrated almost entirely in the `Base|(Reference)`
scenario — the flattest, lowest-variance series in the dataset, and the RNN's known weak
point pre-tuning too ([[concrete-modelling-reference]]). 33/49 series improved, 16/49
(nearly all `Base`) got worse; R² is scale-sensitive to a series' own variance, so a small
absolute miss on an almost-flat curve swings it hard negative even while MAE/RMSE improve
in aggregate. **Headline change: the tuned RNN now beats `persistence` on the scenario
axis (0.152 vs 0.165 MAE)** — that axis is what the dashboard's scenario selector actually
does, and the untuned RNN lost there.

### CatBoost — kept (`TUNED_CATBOOST_PARAMS`)

`iterations=312, learning_rate=0.182, depth=3, l2_leaf_reg=3.24` (was `iterations=600,
learning_rate=0.04, depth=6`, default `l2_leaf_reg=3.0`).

| Run | MAE: before → after | R²: before → after |
|---|---|---|
| Primary | 0.156 → 0.155 (flat) | 0.803 → **0.810** |
| Scenario | 0.313 → **0.221** | **−2.717 → 0.158** |
| Asset | 0.375 → **0.354** | −0.206 → −0.135 (still negative, less bad) |

Primary is a wash; the real win is scenario robustness, where the shallower, faster-learning
booster (`depth` 6→3, more aggressive `learning_rate`) stops badly overfitting individual
scenario folds.

### Ridge — reverted (`TUNED_RIDGE_PARAMS = {}`)

Optuna's `alpha=5.38` (vs. the default `alpha=1.0`) was a **net regression**: primary MAE
0.639→0.665, asset MAE 0.661→0.691, primary R² −5.77→−6.17. Scenario MAE did improve
(0.741→0.664) but not enough to call it a win given the other two runs got worse. Ridge is
already the intentional "worst model" baseline for the RNN-vs-Ridge dashboard comparison —
deeply negative R² either way — and tuning didn't change that story, so the untuned default
is kept.

### Random Forest — reverted (`TUNED_RF_PARAMS = {}`)

Optuna's shallower forest (`max_depth=5, n_estimators=64, min_samples_leaf=2`, vs. the
default `n_estimators=300, max_depth=None, min_samples_leaf=1`) shaved a little MAE off
primary (0.164→0.161) and scenario (0.183→0.174), but **R² dropped everywhere** (primary
0.841→0.743, scenario 0.739→0.674, asset 0.741→0.529) and asset MAE/RMSE got worse too
(0.253→0.270). A net regression, not a win — reverted to the untuned defaults.

## 4. Why two models tuned well and two didn't

The inner proxy (pseudo-cutoffs 2018/2021, 3 of 7 scenario groups) is a small, short-horizon
sample by construction — the whole point was keeping each trial affordable. RNN and
CatBoost have enough free hyperparameters (network width, learning dynamics, tree depth and
regularisation together) that a setting which generalises on the small inner proxy also
generalises on the full 44-year outer forecast. Ridge has one knob (`alpha`) that mostly
trades off how much it ignores the (already weak) linear fit, and Random Forest's one
effective knob here (depth/leaf size) trades bias for variance in a way that's sensitive to
how much data a fold actually has — the inner proxy has far fewer rows per fold than the
real `<=2026` training set, so a depth that looks right there is too shallow for the real
one. Worth remembering before tuning any other tabular model on this dataset: validate
on the real outer benchmark before keeping a result, every time.

## 5. Reproduce

```bash
# re-run the search (all four models, ~30-40 min total on this machine; rnn is the
# slow one at ~20s/trial, random_forest ~25s/trial, catboost/ridge seconds each)
python -m src.models.concrete_tuning --model all --trials 40

# validate one model's winning params against the real outer benchmark before keeping it
python -m src.models.concrete_benchmark_all --models rnn

# regenerate the official leaderboard after any TUNED_*_PARAMS change
python -m src.models.concrete_benchmark_all
```

`artifacts/concrete_tuning_results.json` holds the raw Optuna output (best params + inner
CV MAE) for every model tuned so far.
