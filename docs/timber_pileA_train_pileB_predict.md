# Timber: Train on PileA, Predict PileB — Data, Method, and Results

**What this covers:** the timber pile deterioration surrogate models (`src/models/timber_model.py`, `src/models/timber_model_gbm.py`), and one specific experiment built on top of them — training exclusively on PileA and using the trained model to predict PileB's entire 2000–2070 trajectory, as a leave-one-pile-out generalization test with Monte Carlo cross-validation.

---

## 1. The dataset

**Source:** `data/processed/timber_kembla_v6_3_consolidated.csv` — 994 rows, output of a deterministic AS 5604-style marine timber borer deterioration model (an Excel spreadsheet, ported/consolidated to CSV), **not real field inspection data**. It simulates two timber piles at Port Kembla:

| Pile | Species | Installed | Initial diameter | Replacement threshold |
|---|---|---|---:|---:|
| PileA | Turpentine Gum | 2005 | 400mm | 200mm |
| PileB | Gum | 2006 | 500mm | 200mm |

Each pile has 71 yearly rows (2000–2070) × 7 climate streams (a shared `Baseline`, plus 3 GCMs — `Hotter & Drier`, `Most Likely`, `Colder & Wetter` — each under 2 emissions pathways, A1B/A1FI) = 497 rows per pile, 994 total.

**Target:** `remaining_diameter_mm` — the deterministic model's own predicted pile diameter at a given year/scenario (again, model output, not a measurement).

**Features (`src/features/timber_features.py::FEATURE_COLUMNS`, 13 columns):**
```
age_years, summer_sea_temp_c, salinity, water_factor, timber_exposure_factor,
salinity_factor, shelter_factor, protection_factor, model_contact_factor,
model_knot_factor, environment_factor, construction_factor, initial_diameter_mm
```

**Important fix already applied:** two columns — `sapwood_rate_mm_year` and `heartwood_rate_mm_year` — used to be in this feature list and were removed. They are the deterministic engine's own internal rate variables; together with `age_years` and `initial_diameter_mm` they let a model reconstruct the target **algebraically, to within 2.3×10⁻¹³mm** — i.e. the model wasn't learning to forecast, it was reading off the answer. Both columns are now in `LEAKY_COLUMNS` instead and are never used as inputs. All numbers in this report are from the fixed, leak-free feature set.

---

## 2. The two models

Both live behind one shared `train()` function in `src/models/timber_model.py`, so they always get identical features/target/split — only the estimator differs:

| | MLP (`timber_model.py`) | GBM (`timber_model_gbm.py`) |
|---|---|---|
| Architecture | `StandardScaler → MLPRegressor(hidden_layer_sizes=(64,32))`, ReLU, Adam, early stopping | LightGBM (falls back to XGBoost if unavailable), 400 trees, depth 5 |
| Needs feature scaling? | Yes (in the pipeline) | No |
| Random state | 42 | 42 |

---

## 3. This experiment: train blind on PileA, predict PileB

**Script:** `scripts/train_pileA_predict_pileB.py` (`python -m scripts.train_pileA_predict_pileB`)

### 3.1 Method

1. **Monte Carlo cross-validation on PileA only.** PileA's 497 rows are repeatedly split into random 80/20 train/validation subsets — 30 times (`sklearn.model_selection.ShuffleSplit`, which is what "Monte Carlo CV" means: many random re-splits, not k-fold's one fixed partition) — and MAE/R² are recorded each time. This estimates how stable each architecture's fit to PileA is, on its own.
2. **Refit on all of PileA.** Monte Carlo CV measures generalization error; it doesn't hand you a deployable model. Each architecture is refit once on the *entire* PileA set (still only PileA — PileB is never touched during training) to produce the model actually used for prediction.
3. **Predict PileB's full trajectory.** The PileA-only model is pointed at PileB via `src/models/timber_scenarios.py::build_pile_trajectories` (unmodified, existing code): a single shared line from 2000 through the **split year**, then three diverging lines afterward — one per climate scenario (`Hotter & Drier`, `Most Likely`, `Colder & Wetter`; the two emissions pathways per scenario are averaged into one line each). The split year is `default_split_year()` — today's real calendar year, clamped into the data's range — which resolved to **2026**.
4. **Every point on every line is a model prediction**, including the pre-2026 segment — there is no "real observed" data in this particular output. This is a leave-one-pile-out generalization test, not a history/forecast split of measured data.
5. Deterioration is enforced non-increasing and floored at 0mm remaining diameter (250mm max deterioration depth, since PileB's initial diameter is 500mm) — same physical-plausibility rule the rest of the app already applies.

### 3.2 Output files

| File | Rows | Columns |
|---|---:|---|
| `data/dashboard/dashboard_data_timber_pileB_from_pileA_mlp.csv` | 213 | `material, asset_id, scenario, year, value, unit, metric, method` |
| `data/dashboard/dashboard_data_timber_pileB_from_pileA_gbm.csv` | 213 | same schema |

`value` = deterioration depth in mm (`(initial_diameter_mm − predicted_remaining_diameter_mm) / 2`), `metric="deterioration_depth"`, `method="mlp"`/`"gbm"` for every row (again — no `"observed"` rows here, unlike the main dashboard files). `scenario` uses plain names (`"Hotter & Drier"`, etc.), **not** the `"pathway|climate_model"` format the main `dashboard_data_timber_*.csv` files use, because the two emissions pathways are averaged together here.

213 rows = 3 scenarios × 71 years (2000–2070), each scenario's line sharing the same values through 2026 and diverging after.

---

## 4. Results

### 4.1 PileA Monte Carlo cross-validation (30 random 80/20 splits, held out from PileA itself)

| Model | MAE (mm) | RMSE (mm) | R² |
|---|---:|---:|---:|
| MLP | 5.26 ± 0.45 | 7.11 ± 0.64 | 0.996 ± 0.001 |
| GBM | 2.37 ± 0.30 | 4.25 ± 0.56 | 0.999 ± 0.000 |

GBM fits PileA's own data slightly tighter than MLP here — consistent with GBM's known behaviour of fitting training-range data very closely (its weakness is specifically *extrapolating past* that range, which is exactly what predicting PileB requires next).

**This table describes fit to PileA, not accuracy on PileB — see 4.2.**

### 4.2 PileA-trained model vs. PileB's real values — the actual accuracy of the dashboard CSVs

This is the number that matters for `dashboard_data_timber_pileB_from_pileA_{mlp,gbm}.csv`: each PileA-trained model, scored directly against all 497 of PileB's own real (deterministic-model) `remaining_diameter_mm` rows — PileB the model never saw during training or validation.

| Model | MAE (mm) | RMSE (mm) | R² | n |
|---|---:|---:|---:|---:|
| MLP | 141.11 | 179.63 | 0.545 | 497 |
| GBM | 141.93 | 179.33 | 0.547 | 497 |

Both models land in roughly the same place — MAE ~141mm, R² ~0.55 — a large step down from the ~2–5mm Monte Carlo CV numbers in §4.1, because §4.1 measures interpolation within PileA's own trajectory while this measures genuine prediction of a structurally different, never-seen pile. **Interesting/notable:** GBM's number here (141.93mm MAE, R² 0.547) is numerically identical to the leave-one-pile-out figure in the older `timber_project_report.md`, which was computed *before* the feature-leakage fix — for this particular split, removing the two leaky rate columns didn't move GBM's result at all. MLP's did move (133.44mm pre-fix → 141.11mm now), meaning MLP was leaning on the leaky columns here and GBM apparently wasn't.

### 4.3 PileB predicted replacement year, by scenario

| Scenario | MLP | GBM |
|---|---:|---:|
| Colder & Wetter | 2039.8 | 2039.3 |
| Hotter & Drier | 2040.4 | 2039.8 |
| Most Likely | 2040.8 | 2041.2 |

Both models land in a tight 2039–2041 band across all three scenarios and agree with each other to within ~1.4 years per scenario — a reasonable cross-check given they were trained the same way on the same (single) pile.

---

## 5. Caveats to carry into any discussion of these numbers

- **Only 2 piles exist in the whole dataset.** Training on PileA and testing on PileB is the only leave-one-pile-out test available — it's a genuine domain-shift test (different species, different exposure/contact/knot codes, different initial diameter), but it's also an N=1 test with no way to know if it's typical.
- **Training on a single pile makes several features constant.** Multiple `*_factor` columns are fixed per asset; with only PileA in the training set they carry zero variance, so effectively only `age_years`, `summer_sea_temp_c`, and `salinity` vary meaningfully during this particular training run.
- **The whole dataset is simulator output, not field measurement** (§1) — both training and "ground truth" trace back to the same deterministic Excel model, not real inspections.
- **The 2000–2026 segment is a prediction, not history** — worth stating explicitly if this ever gets shown next to the main dashboard files, where the pre-2050 segment *is* real simulated ground truth (`method="observed"`).

---

## Reproduce

```bash
python -m scripts.train_pileA_predict_pileB
```

Regenerates both dashboard CSVs and `scripts/train_pileA_predict_pileB_metrics.json`. `random_state=42` is fixed for both the Monte Carlo CV splits and the final refit; exact figures can drift slightly across scikit-learn/LightGBM versions.
