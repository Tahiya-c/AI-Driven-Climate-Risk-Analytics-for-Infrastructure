# Timber Deterioration Models: Updated Pipeline and Results

**Status:** current as of 2026-10-04. The import now takes the pile coefficients as numbers (§12.7). The ~28 mm sapwood bias is now **removed inside the models** (sapwood-aware hazard normalisation, §12.6), so the dashboard shows the **unshifted** forecast. Earlier the same day: the GBM's capacity was deliberately reduced (§12.3, §12.4), a shift was tried to hide the bias (§12.5), the clean-up and the switch to a chronological hold-out (§12.1, §12.2). **Open decision:** which GBM configuration to keep (§12.6).
**Main code:**
- `src/models/timber_transfer.py`: training, tuning (Optuna on a chronological hold-out), evaluation, saving.
- `src/models/timber_model.py`, `src/models/timber_model_gbm.py`: the MLP and GBM builders and their tuned settings.
- `src/models/timber_forecast.py`: forecasting with the saved models.
- `src/ui/timber_data_app.py`, `src/ui/timber_results_app.py`: charts.

**Saved models:** `artifacts/timber_transfer_gbm.joblib`, `artifacts/timber_transfer_mlp.joblib` (§8).

**Related reports:**
- `docs/timber_model_tuning.md`: the time-split stage.
- `docs/timber_pileA_to_pileB_report.md`: the PileA→PileB stage in detail.
- `docs/timber_kembla_model_spec_and_defects.md`: the spreadsheet's defects.

## Summary

Two machine-learning models, a **GBM** (LightGBM) and an **MLP** (neural network), predict how much material a timber pile loses over time under three climate scenarios. Both are **trained on PileA only and tested on PileB**, a pile they never see. Both use exactly the same target, features and preprocessing.

**Model accuracy (unshifted, saved models, sapwood-aware normalisation, §12.6):**

| Model | PileB R², all years | RMSE | MAE | **PileB R², forecast 2027–2070 (A1FI)** | RMSE | MAE | Jump at 2026→2027 |
|---|---:|---:|---:|---:|---:|---:|---:|
| GBM | 0.926 | 25.18 | 17.52 | **0.969** | 7.2 | 4.3 | +25.1 mm |
| MLP | 1.000 | 1.27 | 0.74 | **1.000** | 0.7 | 0.4 | +7.0 mm |

Errors are in mm of radial material loss. "All years" is 497 rows. The forecast columns are 2027–2070, 3 climate scenarios, A1FI pathway, 132 rows. The jump is from PileB's real 2026 value to the 2027 forecast; one year of decay at that point is about 5–6 mm.

**These unshifted scores are the model-accuracy numbers.** PileB is never used to train, tune or adjust these predictions. A *shifted* score (§7.3) uses PileB's own 2022–2026 values and is only an "after inspection update" figure.

**Before the sapwood fix** the same models scored GBM 0.797 / 0.761 and MLP 0.953 / 0.829 (all years / A1FI forecast), with jumps of +51.9 and +33.9 mm. About 28 mm of that error came from the hazard normalisation scaling the spreadsheet's fixed 20 mm sapwood layer (§7.2). Taking the sapwood layer out before normalising removes it.

- **Starting point:** the original models scored R² ≈ 0.25–0.27 on PileB overall, and about −4.7 to −4.9 on the forecast years.
- **The MLP now reproduces the spreadsheet almost exactly** (R² 1.000 unshifted, jump +7.0 mm, mean 2022–2026 offset −1.5 mm). Read the caveat below: with the bias gone, the remaining task is close to the spreadsheet's own formula.
- **The reduced GBM (25 shallow trees) still has a shape error:** offset −23 to −24 mm, jump +25 mm, 47 mm predicted before PileB is installed. No shift fixes that. **Decision pending:** the Optuna GBM with the fix scores R² 1.000 / 1.000, jump +5.8 mm (table in §12.6).
- **No shift is needed any more for the MLP,** and the dashboard files are built unshifted (`--no-shift`).

**The headline caveat:** there are only **two piles**, and the "truth" is a **spreadsheet model**, not real inspections. R² ≈ 1.000 means the model has learned the spreadsheet's decay formula for an unseen pile, not that it would be this accurate on real inspections. Read §10 before quoting these numbers.

---

## 1. The data

`data/processed/timber_kembla_v6_3_consolidated.csv` is built from the `Timber_kembla_v6.3` workbook. It contains **994 rows**: 2 piles × 7 climate series × 71 years (2000–2070).

The 7 series are:
- **Baseline:** no climate change.
- **Most Likely** (MRI-CGCM2.3.2), **Hotter & Drier** (CSIRO Mk3.5) and **Colder & Wetter** (MIROC-Medres), each under two emissions pathways (A1B and A1FI).

The dashboard and charts use **A1FI only**, the higher-emissions pathway.

| | PileA | PileB |
|---|---|---|
| Timber | Turpentine Gum | Gum |
| Installed | 2005 | 2006 |
| Initial diameter | 400 mm | 500 mm |
| Heartwood class | HW1 (k_HW 1.3) | HW2 (k_HW 5.2) |
| Exposure | EXP1 (k_shelter 1.0) | EXP2 (k_shelter 0.6) |
| Combined hazard factor k0 | 1.3 | 3.12 |
| Maintenance / contact / knot codes | MTN2 / CNT2 / KNT3 | MTN3 / CNT3 / KNT4 (undefined codes, Defect D3) |

**Important:** every row is the *output* of the spreadsheet's deterministic AS 5604-style decay formula (ported to Python in `src/models/timber_engine.py`). Nothing is a field measurement. "Accuracy" in this report means "how well the ML model reproduces the spreadsheet for a pile it hasn't seen".

## 2. The full flow

```
Timber_kembla_v6.3.xlsx
   │  src/data/build_timber_kembla_csv.py   (D9 fix: correct column names)
   ▼
timber_kembla_v6_3_consolidated.csv
   │  src/data/timber_data_prep.py          (rename old headers, fill MIROC label, clean dtypes)
   ▼
load_transfer_frame()                        (adds target: material_loss_mm)
   │
   ├── PileA rows ──► chronological hold-out (2057–2070): choose the feature set + Optuna hyperparameters
   │                  (MLP; the GBM's settings are now set for a target PileB R², §12.3)
   │                  └─► fit the final GBM and MLP on all PileA rows
   │                      └─► SAVE: artifacts/timber_transfer_{gbm,mlp}.joblib (+ .json)
   │
   └── PileB rows ──► the saved models predict ─► R² / RMSE / MAE (model accuracy, unshifted)
                      └─► dashboard CSVs (--no-shift): real values to 2026, model forecast after

Imported pile (template CSVs) ──► src/models/timber_forecast.py
   ├─► load the SAVED model (never retrain)
   ├─► predict, then anchor: shift by the mean error over the last 5 years of inspections
   │   (warns if the shift is over 15 mm)
   └─► dashboard-format rows ─► dashboard / Supabase (§9)
```

Inside each model (`HazardNormalisedRegressor`, identical for GBM and MLP):

```
15 input features + the pile's sapwood layer (layer_of_sapwood_mm, not an input)
  → add thermal_dose = age × water_factor
  → drop features that never change in the training rows
  → target: (loss − sapwood loss) ÷ k0   (learn "heartwood loss per unit hazard")
  → inner model (GBM or MLP)
  → prediction × the pile's own k0 + sapwood loss
  → clip to [0, pile radius]
  → predicted material loss (mm)
```

Sapwood loss = the pile's whole sapwood layer once age > 0, otherwise 0. That is exactly the spreadsheet's rule (`timber_engine.simulate`, Defect D1).

## 3. Preprocessing

| Step | Where | What and why |
|---|---|---|
| **Fix Defect D9: column names** | `build_timber_kembla_csv.py`, `timber_data_prep.py` | The export labelled 10 k-factor columns one column out of step; for example, `salinity_factor` actually held the shelter coefficient. Verified against the workbook's headers and the engine's recomputed values, then the CSV was regenerated (same values, correct names). Old-format files are renamed on load. |
| **Fill a missing scenario label (D8)** | `timber_data_prep.py` | MIROC-Medres rows had a blank `scenario_label`. They are filled as "Colder & Wetter". |
| **Exclude leaky columns** | `timber_features.py` (`LEAKY_COLUMNS`) | The spreadsheet's own outputs and intermediate rates (`heartwood_rate_mm_year`, `mean_deterioration_mm`, `failure_probability` and so on) are never used as inputs. |
| **Material loss target** | `timber_transfer.py` (`material_loss`) | `(initial diameter − max(remaining diameter, 0)) / 2`: radial loss in mm, capped at the pile radius. This is the dashboard's metric, and it removes the spreadsheet's impossible negative diameters (D7). |
| **`thermal_dose` feature** | `timber_features.py` (`add_thermal_dose`) | `age × water_factor`: years in service × how warm the water was. Chosen on PileA's chronological hold-out (§5). |
| **Zero-variance filter** | `HazardNormalisedRegressor.fit` | Drops features that take a single value across the training rows. With PileA-only training this drops every pile-level factor and the diameter. |
| **Sapwood-aware hazard normalisation** | `HazardNormalisedRegressor`, `sapwood_loss` | The model learns `(loss − sapwood loss) / k0` and predicts `sapwood loss + prediction × k0`, with each pile's own k0 (`combined_factor`). In the decay formula, the heartwood rate is `water_factor × k0`, so this lets one pile's data generalise to another pile's hazard level. The sapwood layer is lost whole in year one whatever the hazard (Defect D1), so it is kept out of the scaling (§12.6). k0 and the sapwood layer are pile attributes, not the target, so neither is leakage. |
| **Feature scaling** (MLP only) | `build_estimator` | Inputs and target are standardised. Trees don't need scaling. |
| **Clipping** | `HazardNormalisedRegressor.predict` | Predictions are limited to [0, radius]. |

## 4. Same target and features for both models

**Target:** `material_loss_mm`, radial material loss in mm.

**Input features:** the 15 `FEATURE_COLUMNS`, plus `thermal_dose` added inside the model.

| Feature | Meaning | Changes within PileA? | Used when training on PileA? |
|---|---|---|---|
| `age_years` | Years since installation | Yes | ✅ |
| `summer_sea_temp_c` | Summer sea surface temperature | Yes | ✅ |
| `salinity` | Water salinity (ppt) | Yes | ✅ |
| `water_factor` | k_water = 0.1·e^(0.13·sea temp) | Yes | ✅ |
| `thermal_dose` | age × water_factor (engineered) | Yes | ✅ |
| `heartwood_factor` | k_HW | No (1.3) | Dropped by filter |
| `sapwood_factor` | k_SW | No (25) | Dropped by filter |
| `salinity_factor` | k_sal | No (1.0) | Dropped by filter |
| `shelter_factor` | k_shelter | No (1.0) | Dropped by filter |
| `protection_factor` | k_protect | No (1.0) | Dropped by filter |
| `contact_factor` | k_contact | No (1.0) | Dropped by filter |
| `knot_factor` | k_knot | No (1.0) | Dropped by filter |
| `environment_factor` | k_env = k_sal·k_shelter | No (1.0) | Dropped by filter |
| `construction_factor` | k_const = k_protect·k_contact·k_knot | No (1.0) | Dropped by filter |
| `combined_factor` | k0 = k_HW·k_env·k_const | No (1.3) | Used for normalisation, not as an input |
| `initial_diameter_mm` | Diameter at installation | No (400 mm) | Used to cap predictions at the radius |
| `layer_of_sapwood_mm` | Sapwood layer d_sap (workbook column M) | No (20 mm) | Not a feature: passed alongside (`MODEL_INPUT_COLUMNS`) to compute the sapwood loss |

Both models learn from the same **5 inputs**. The pile's hazard enters only through the k0 scaling, and its sapwood layer only through the sapwood loss. Callers pass `rows[MODEL_INPUT_COLUMNS]` (`FEATURE_COLUMNS` plus `layer_of_sapwood_mm`); a model raises an error if the sapwood column is missing.

## 5. Models, validation and tuning

| | GBM | MLP |
|---|---|---|
| Library | LightGBM `LGBMRegressor` | scikit-learn `MLPRegressor` |
| Architecture | **25 trees**, depth ≤ 2, 4 leaves, **`linear_tree=True`** (a linear model in each leaf, so it can extrapolate) | 3 hidden layers × 136 ReLU units, Adam, early stopping |
| Regularisation | learning rate 0.05, LightGBM defaults otherwise (no subsampling) | L2 `alpha` 9.8e-4, batch 128, learning rate 0.0070 |
| Scaling | None needed | Inputs and target standardised |
| Settings stored in | `TRANSFER_GBM_PARAMS`, `src/models/timber_model_gbm.py` (Optuna config kept as `ARCHIVED_TRANSFER_GBM_PARAMS`) | `TRANSFER_MLP_PARAMS`, `src/models/timber_model.py` |
| How the settings were chosen | Smallest capacity sweep that puts PileB R² (all years) near 0.8 (§12.3) | Optuna on the PileA hold-out |

The Optuna-tuned GBM (945 trees, depth ≤ 3, 7 leaves, subsample 0.57, `colsample_bytree` 0.91, `reg_lambda` 0.0003, learning rate 0.020) is the configuration described in the rest of §5 and in §12.1–§12.2.

**Validation: a chronological hold-out on PileA.** Each of PileA's 7 climate series trains on its earlier years (2000–2056, 399 rows) and is validated on its last 20% (2057–2070, 98 rows). This tests forecasting forward in time, which is the real task. PileB is never used for any choice; it is scored once at the end.

1. **Feature set:** `thermal_dose` was selected by hold-out RMSE averaged over both models (default settings): 17.38 mm with it, 19.35 mm without. (MLP 7.43 vs 10.38 mm, GBM 27.34 vs 28.32 mm. The untuned GBM has no linear leaves, so it can't extend the trend into the held-out years.)
2. **Hyperparameters:** Optuna TPE (seed 42), 50 trials per model. Each trial fits once on 2000–2056 and is scored by RMSE on 2057–2070.

| | GBM | MLP |
|---|---:|---:|
| Best hold-out RMSE (Optuna) | 0.06 mm (archived config) | 1.17 mm |
| Hold-out R² / RMSE / MAE, Optuna settings | 1.000 / 0.06 / 0.04 | 0.997 / 1.17 / 0.75 |
| Hold-out R² / RMSE / MAE, **current saved settings** (before the sapwood fix) | −1.100 / 28.71 / 28.23 | 0.997 / 1.17 / 0.75 |
| Hold-out R² / RMSE / MAE, **current saved settings, sapwood fix (§12.6)** | **−1.102 / 28.72 / 28.18** | **0.985 / 2.39 / 1.67** |

The reduced GBM fails this hold-out: with 25 shallow trees it can't extend PileA's own trend into 2057–2070. It was not selected by this check (§10, item 11). The MLP's hold-out score dropped slightly with the fix (0.997 → 0.985) because its settings were tuned for the old target; Optuna was not re-run (§12.6). With the fix the Optuna GBM scores 1.000 / 0.05 / 0.04 on this hold-out.

**Why not Monte Carlo cross-validation?** It was used first: 10 random 80/20 splits of PileA's rows. Random splits put neighbouring years, and near-identical rows from other climate series, on both sides of the split, so the model only filled gaps. It scored R² ≈ 0.999 for almost any model and couldn't tell good settings from bad. The chronological hold-out tests forecasting forward, and it does discriminate: a GBM without linear leaves scored hold-out R² −0.77 in testing. It also needs one fit per trial instead of 10. Switching changed PileB scores by at most 0.003 (§12.2), so earlier conclusions stand.

## 6. Strategies tried, in order

| # | Strategy | Test | Best R² | Outcome |
|---|---|---|---:|---|
| A | Original MLP/GBM, raw remaining diameter | Time split: last 20% of years, both piles | MLP 0.625, GBM 0.644 | Neither could extend the trend |
| B | + `thermal_dose`, rate target, Optuna with blocked time-series CV | Same time split | MLP 0.953, GBM 1.000 | Works, but the GBM simply reproduces the formula; both piles are in training |
| C | Same models, trained on 2000–2026 only | Forecast 2027–2070, both piles | GBM 0.886, MLP −1.894 (before the D9 fix) | The MLP can't forecast 44 years ahead from 21 |
| D | Years-of-history test (5–40 years) | Forecast the rest of the pile's life | GBM ≥ 0.8 needs ~20 years (before the D9 fix) | ML models need long histories |
| **E** | **D9 fix + material-loss target + hazard normalisation + zero-variance filter + Optuna on a chronological PileA hold-out** | **Train PileA → test PileB** | **GBM 0.952, MLP 0.953** | **Final.** Generalises to a pile neither model has seen |
| F | E + anchoring to the latest inspection | Imported pile with history | GBM 0.998, MLP 0.997 | Used for imports (§8) |
| G | E, with the GBM's capacity reduced to 25 shallow trees | Train PileA → test PileB | GBM 0.797, MLP 0.953 (unchanged) | GBM settings still saved (§12.3). Anchored import: GBM 0.802 |
| **H** | **G + sapwood-aware normalisation (sapwood loss kept out of the k0 scaling)** | **Train PileA → test PileB, unshifted** | **MLP 1.000, GBM 0.926 (reduced) / 1.000 (Optuna)** | **Current saved models** (§12.6). Removes the 28 mm bias; no shift needed |

## 7. Results

### 7.1 PileB scores, step by step

All years = 497 rows. Forecast = 2027–2070, both pathways, 264 rows. The A1FI-only forecast scores are in the Summary.

**Current pipeline (sapwood-aware normalisation, §12.6), from `--evaluate`:**

| Step | Model | R² all | RMSE all | MAE all | R² forecast | RMSE forecast | MAE forecast |
|---|---|---:|---:|---:|---:|---:|---:|
| 1. Original models | GBM | 0.270 | 79.05 | 66.45 | −4.689 | 96.92 | 94.08 |
| 1. Original models | MLP | 0.251 | 80.05 | 68.34 | −4.884 | 98.57 | 95.74 |
| 2. + preprocessing (§3), untuned | GBM | 1.000 | 0.56 | 0.31 | 1.000 | 0.63 | 0.33 |
| 2. + preprocessing (§3), untuned | MLP | 0.999 | 2.58 | 1.57 | 0.998 | 1.98 | 1.11 |
| **3. Saved settings** | **GBM (reduced, 25 trees)** | **0.926** | **25.18** | **17.52** | **0.969** | **7.20** | **4.33** |
| **3. Saved settings** | **MLP** | **1.000** | **1.27** | **0.74** | **1.000** | **0.83** | **0.48** |

Seed stability (5 seeds, PileB all years): GBM 0.926 ± 0.000 / RMSE 25.18 ± 0.00 / MAE 17.52 ± 0.00; MLP 1.000 ± 0.000 / RMSE 0.93 ± 0.23 / MAE 0.54 ± 0.15.

Even the untuned models reach R² ≈ 1.000 once the sapwood layer is kept out of the scaling: the remaining "heartwood loss per unit hazard" is close to the spreadsheet's own formula (`water_factor` × time), which any reasonable model learns. Only the reduced GBM falls short, because 25 shallow trees can't draw that curve accurately.

**Before the sapwood fix (§12.1–§12.5), kept for comparison:**

| Step | Model | R² all | RMSE all | MAE all | R² forecast | RMSE forecast | MAE forecast |
|---|---|---:|---:|---:|---:|---:|---:|
| 1. Original models | GBM | 0.270 | 79.05 | 66.45 | −4.689 | 96.92 | 94.08 |
| 1. Original models | MLP | 0.251 | 80.05 | 68.34 | −4.884 | 98.57 | 95.74 |
| 2. + preprocessing (§3), untuned | GBM | 0.951 | 20.42 | 15.14 | 0.820 | 17.24 | 11.00 |
| 2. + preprocessing (§3), untuned | MLP | 0.955 | 19.52 | 14.96 | 0.812 | 17.61 | 11.19 |
| 3. + Optuna on hold-out | GBM | 0.952 | 20.26 | 15.02 | 0.824 | 17.03 | 10.88 |
| 3. + Optuna on hold-out | MLP | 0.953 | 20.05 | 15.26 | 0.825 | 16.99 | 10.86 |
| 4. Capacity reduced, 25 trees | GBM | 0.797 | 41.69 | 31.64 | 0.757 | 20.02 | 12.31 |

**Seed stability before the fix** (5 random seeds, PileB all years):

| Model | R² | RMSE | MAE |
|---|---:|---:|---:|
| GBM (current) | 0.797 ± 0.000 | 41.69 ± 0.00 | 31.64 ± 0.00 |
| GBM (Optuna, archived) | 0.952 ± 0.000 | 20.26 ± 0.02 | 15.03 ± 0.01 |
| MLP | 0.955 ± 0.004 | 19.66 ± 0.95 | 14.86 ± 0.87 |

The current GBM has no subsampling, so every seed gives the identical model.

### 7.2 Where the error is

**With the sapwood fix (current saved models)**, mean PileB material loss (mm):

| Period | Spreadsheet (truth) | GBM (reduced) | MLP |
|---|---:|---:|---:|
| 2000–2005 (before installation) | 0.0 | 46.6 | 0.0 |
| 2006–2015 | 31.9 | 77.4 | 33.9 |
| 2016–2026 | 89.8 | 117.5 | 90.8 |
| 2027–2040 | 164.4 | 171.7 | 165.4 |
| 2041–2050 | 235.7 | 228.5 | 236.3 |
| 2051–2070 | 249.9 | 249.3 | 249.9 |

The MLP now tracks the truth in every period, including 0 mm before installation. The reduced GBM still predicts 47 mm before PileB exists and runs about 45 mm high in 2006–2015, then falls slightly below the truth in 2041–2050: a shape error, not an offset.

**Before the fix**, mean PileB material loss (mm):

| Period | Spreadsheet (truth) | GBM (current) | GBM (Optuna, archived) | MLP |
|---|---:|---:|---:|---:|
| 2000–2005 (before installation) | 0.0 | 66.2 | 0.1 | 5.1 |
| 2006–2015 | 31.9 | 96.9 | 57.1 | 55.4 |
| 2016–2026 | 89.8 | 146.7 | 117.7 | 118.0 |
| 2027–2040 | 164.4 | 198.2 | 192.1 | 192.1 |
| 2041–2050 | 235.7 | 245.3 | 247.3 | 247.3 |
| 2051–2070 | 249.9 | 250.0 | 250.0 | 250.0 |

Before the fix, the MLP and the archived GBM **over-predicted by about 28 mm** from installation until the pile was fully consumed. The reduced GBM over-predicted by much more in the early years (66 mm before PileB is even installed). The cause of the shared 28 mm part, **now fixed (§12.6)**:
- The spreadsheet removes a fixed 20 mm sapwood layer in year one, whatever the pile's hazard (`d_sap`, workbook column M; Defect D1, column BE).
- The old hazard normalisation scaled that fixed loss too.
- 20 × (3.12 / 1.3 − 1) = **28 mm**.

### 7.3 The jump at 2026, shifting and anchoring

**How the forecast is built.** The models are trained on PileA only. For PileB they predict every year, 2000–2070, from PileB's features (age, sea temperature, salinity, hazard factor) and its sapwood layer; they never see PileB's real loss values. The dashboard shows PileB's **real** values up to 2026 and the model's forecast from 2027. So the 2026 point is real and the 2027 point is predicted, and any model error appears as a step there.

**With the sapwood fix (current), unshifted:**

| A1FI, unshifted | Real 2026 → forecast 2027 | Mean shift offset over 2022–2026 |
|---|---|---|
| MLP | Most Likely 117.07 → 123.85 (+6.8); Hotter & Drier 120.37 → 127.45 (+7.1); Colder & Wetter 121.34 → 128.34 (+7.0) | −1.3, −1.6, −1.6 mm |
| GBM (reduced) | 117.07 → 142.78 (+25.7); 120.37 → 145.31 (+24.9); 121.34 → 146.06 (+24.7) | −24.4, −23.5, −23.3 mm |
| GBM (Optuna, not saved) | mean +5.8 | −0.1, +0.1, +0.1 mm |

One year of decay at this point is about 5–6 mm, so the MLP's step of about 7 mm is the line continuing. **The dashboard files are now built unshifted** (`--no-shift`). The reduced GBM's 25 mm step remains because its error is a shape error.

**Before the fix**, the jumps were GBM +51.9 mm, Optuna GBM about +35 mm and MLP +33.9 mm, and the shift offsets were about −28 mm for the MLP.

Shifting and anchoring are still available, both applied *before* capping at the radius so the forecast can still reach full loss:

- **Shifting (dashboard option, `build_dashboard_output` without `--no-shift`):** `shift_forecast()` in `timber_transfer.py`. Per scenario series, offset = mean(actual − predicted) over **2022–2026** (`SHIFT_WINDOW`). It returns the offsets; any larger than **15 mm** (`LARGE_OFFSET_MM`) are warned about and listed under `large_offsets`, and the dashboard builder writes them to `<file>.shift.json`.
- **Anchoring (imports, §8):** `timber_forecast._anchor()` now uses the same idea: the mean error over the last 5 years of inspections up to the latest one, so one bad inspection can't set the offset. Offsets over 15 mm are warned about and recorded in `result.attrs["anchor"]` (and `<out>.anchor.json` from the CLI).
- `anchor_forecast()` (single 2026 value) is kept for the Streamlit data page's anchoring toggle.

**Shifted or anchored scores are "after inspection update" figures, not model accuracy:** they use the pile's own measured values. With the fix they add almost nothing for the MLP. For the reduced GBM they make it worse (A1FI forecast 0.969 unshifted → 0.801 shifted), because its offset isn't constant.

The rest of this section describes the shift as it was evaluated **before the fix** (§12.5).

| A1FI forecast 2027–2070 | GBM (current) R² / RMSE / MAE | GBM (Optuna, archived) | MLP R² / RMSE / MAE |
|---|---|---|---|
| Model output (PileA only) | 0.761 / 19.9 / 12.1 | 0.828 / 16.9 / 10.7 | 0.829 / 16.8 / 10.6 |
| **Shifted, 2022–2026 mean (dashboard)** | **0.782 / 19.0 / 13.2** | not measured | **1.000 / 0.6 / 0.3** |
| Anchored to each series' 2026 value | 0.838 / 16.4 / 11.0 | 1.000 / 0.2 / 0.1 | 1.000 / 0.5 / 0.3 |
| Anchored to one inspection history (import, §8) | 0.802 / 18.1 / 12.5 | 0.998 / 1.7 / 1.1 | 0.997 / 2.1 / 1.4 |

| Jump from real 2026 value to 2027 forecast (A1FI) | Raw model | Shifted |
|---|---:|---:|
| GBM (current) | +51.9 mm | **+0.3 mm** |
| MLP | +33.9 mm | **+5.6 mm** |

PileB's real loss grows about 5 mm a year at this point, so a step of about 5 mm is the line continuing, not a gap.

**Per scenario, shifted forecast vs PileB's real values, 2027–2070 (A1FI):**

| | Most Likely | Hotter & Drier | Colder & Wetter |
|---|---|---|---|
| GBM: average / worst-year error | 14.7 / 41.1 mm | 12.8 / 39.1 mm | 12.2 / 39.6 mm |
| MLP: average / worst-year error | 0.5 / 1.9 mm | 0.1 / 0.8 mm | 0.4 / 1.7 mm |

**Why the shifted and anchored scores are near 1 for the MLP and the archived GBM:**
- **Their error was one constant offset,** not noise. The MLP's offsets over 2022–2026 are −28.5, −28.1 and −28.3 mm for the three scenarios: the sapwood bias from §7.2. Removing it leaves a near-perfect trend. The exact shifted MLP R² is 0.9998.
- **They use PileB's real values** (five years for shifting, one for anchoring), so they are no longer pure PileA-only predictions. The models were not retrained.
- **Use the model-output row to compare models.** Shifting and anchoring erase most of the difference between them.

**Why it helps the current GBM much less:** with only 25 shallow trees, its curve has the wrong slope and a stepped shape, not just an offset (§7.2). Shifting removes the jump at 2026, but the forecast still drifts up to about 40 mm away from the truth in the 2030s–2040s.

**"All years" with shifting:** 0.798 / 41.54 / 32.68 (GBM) and 0.975 / 14.55 / 8.34 (MLP). These barely move because "all years" also counts 2000–2026, which is scored on the model's own predictions, not the real history. The GBM's MAE rises slightly (31.64 → 32.68 mm) because of its drift.

An earlier version shifted the forecast *after* capping. The lines then levelled off at about 222 mm (R² ≈ 0.71), and that version was dropped.

### 7.4 Charts and dashboard data

| Output | What it shows |
|---|---|
| `src/ui/timber_data_app.py` → "Model forecasts (GBM vs MLP)" | PileB history for 2000–2026, the 3 A1FI scenario forecasts for 2027–2070, each model's Baseline (no climate change) forecast, an anchoring toggle (on by default) and an "actual values" overlay |
| `src/ui/timber_results_app.py` | PileB, GBM and MLP side by side, unanchored, with scores and a step-by-step chart |
| `python -m src.models.build_dashboard_output --material timber --no-shift` | `data/dashboard/dashboard_data_timber_pileB_from_pileA_{gbm,mlp}.csv`, the files the main dashboard's Timber page reads. Each has 213 rows: 81 `observed` (PileB, A1FI, 2000–2026) + 132 forecasts (3 scenarios, 2027–2070), **unshifted** model output. **Build them with `--no-shift`:** the builder's default still shifts (and then writes the offsets to `<file>.shift.json`). Method labels are `observed` / `gbm` / `mlp` |

All three load the **saved models** (§8); none of them retrains.

`scripts/train_pileA_predict_pileB.py` still writes to the same dashboard file names using the *original* models. Re-running it would overwrite the dashboard data.

**Older dashboard files looked smoother, but were less accurate.** Copies of the dashboard CSVs written by that retired script show no visible jump at 2026. That's because every point in them, including 2000–2026, is a model prediction, so the history line and the forecast line come from the same model. Compared with PileB's real values, those predicted "histories" were far off:

| Old files, 2000–2026 vs PileB's real values | MAE | RMSE | Mean bias | Worst year |
|---|---:|---:|---:|---|
| GBM (retired script) | 32.1 mm | 36.7 mm | +31.9 mm | 2006: +50.7 mm |
| MLP (retired script) | 29.5 mm | 34.5 mm | +29.0 mm | 2006: +58.5 mm |

They started 44–50 mm above zero in 2000, before PileB was installed, and rose too slowly (about 2.4 mm/year against roughly 4–5 mm/year). The two errors cancelled by chance around 2025–2026, then the forecasts fell below the truth; overall PileB R² for those models was about 0.55. The current files draw PileB's real values up to 2026, so any model error is visible as a step.

### 7.5 Ridge regression (reference)

Ridge regression was also run through the identical pipeline, as a linear baseline (`build_transfer_model("ridge")`). Optuna on the hold-out chose `alpha` = 9.05 (hold-out RMSE 3.84 mm).

| Model | PileB R² all years | A1FI forecast R² / RMSE / MAE | PileA hold-out R² |
|---|---:|---|---:|
| Ridge, before the sapwood fix | 0.954 | 0.792 / 18.6 / 11.7 | 0.962 |
| Ridge, with the fix (§12.6) | 0.998 | 0.990 / 4.0 / 2.5 | 0.906 |

Ridge is kept in the code for reference; the reported comparison is GBM vs MLP. With the fix even a linear model scores 0.998 on PileB, which shows how close the corrected target is to the spreadsheet's formula.

### 7.6 GBM vs MLP

**With the sapwood fix (current saved models):** the reduced GBM and the MLP differ by 17.1 mm on average (max 48 mm), and their errors correlate at only 0.347. The MLP's error is now almost zero, so what's left is the reduced GBM's shape error. The Optuna GBM and the MLP both reproduce the spreadsheet (R² 1.000), so with the fix they again score the same.

**Before the fix**, the reduced GBM and the MLP already differed clearly:

| Check (PileB, saved models) | GBM (current) vs MLP | GBM (Optuna, archived) vs MLP |
|---|---:|---:|
| Mean gap between GBM and MLP predictions | **17.3 mm** (max 66 mm) | 1.5 mm (max 24 mm) |
| Correlation between their errors | **0.649** | 0.952 |
| Mean error while PileB is decaying | GBM +44.3 mm, MLP +25.6 mm | GBM +26.5 mm, MLP +25.6 mm |
| Spread of that error (std) | GBM 18.5 mm, MLP 5.7 mm | GBM 4.9 mm, MLP 5.7 mm |
| Share of squared error that is the constant offset | GBM 85.1%, MLP 95.3% | GBM 96.7%, MLP 95.3% |

This gap comes from the GBM's reduced capacity (§12.3), not from any difference between tree ensembles and neural networks: at full capacity the two scored almost the same. The rest of this section explains why the tuned versions were so close.

There are four reasons:
1. **The preprocessing does the hard part.** After hazard normalisation and `thermal_dose`, the 5 inputs relate to "loss per unit hazard" almost linearly. Any reasonable model learns the same curve; even linear regression scores 0.955.
2. **Both models learn PileA almost perfectly** (in-sample R²: GBM 1.000, MLP 0.999).
3. **The error comes from the shared design, not the model.** About 97% of the PileB error is the same ~26–28 mm sapwood offset (§7.2), which both models inherit through the normalisation step. No choice of model or hyperparameters can remove it.
4. **There is too little data to separate them.** One training pile and one test pile never show a pattern where a neural network's flexibility helps over a tree ensemble. Their predictions differ by 1.5 mm on average, far less than the shared ~26 mm bias.

**Conclusion:** on this dataset, the preprocessing matters far more than the choice of model. A real difference between GBM and MLP would need more piles with genuinely different timber, exposure and climate.

## 8. Saved models: every import uses the same model

### 8.1 Where the models are saved

| File | Contents |
|---|---|
| `artifacts/timber_transfer_gbm.joblib` | The fitted GBM (`HazardNormalisedRegressor`), about 0.9 MB |
| `artifacts/timber_transfer_mlp.joblib` | The fitted MLP, about 0.9 MB |
| `artifacts/timber_transfer_{gbm,mlp}.json` | Metadata: `model_version`, save time (UTC), library versions, input columns (`MODEL_INPUT_COLUMNS`) and `sapwood_column`, used columns, hyperparameters, PileB scores unshifted and shifted, and the `shift` record (window, offsets per series, `large_offsets` over 15 mm) |

Current versions (§12.6, sapwood-aware normalisation):
- `timber_transfer_gbm_20261004_143534` (reduced capacity, §12.3)
- `timber_transfer_mlp_20261004_143534`

**Use these for imports.** `src/models/timber_forecast.py`, the dashboard builder and the Streamlit pages all load these files. Models saved before §12.6 no longer work with this code: they were fitted on the old target and would be given the sapwood loss twice.

Both were saved with Python 3.14.5, scikit-learn 1.9.0, LightGBM 4.7.0, NumPy 2.5.2 and joblib 1.5.3.

Backups (not used by any code):

| Folder | Contents |
|---|---|
| `artifacts/_backup_before_gbm_retune/` | The Optuna GBM, `timber_transfer_gbm_20261004_082023`. Copy it back over `artifacts/timber_transfer_gbm.{joblib,json}` to restore it. |
| `artifacts/_backup_before_rerun/` | Both models as they were just before the §12.4 re-run (GBM `…_111954`, MLP `…_082023`). Their predictions are identical to the current saved models. |
| `artifacts/_backup_before_shift/` | Both models just before the §12.5 re-save (`…_114250`). Identical predictions; only the metadata differs (no shifted scores). |
| `artifacts/_backup_before_sapwood_fix/` | Both models just before §12.6 (`…_140554`), fitted on the old `loss / k0` target. Restoring them also needs the pre-§12.6 code. |

`artifacts/` is git-ignored except for these four files (`.gitignore`). Committing them means every machine, and the deployed app, loads the **identical** model.

### 8.2 How "the same model" is guaranteed

1. **Imports never retrain.** `src/models/timber_forecast.py`, the dashboard builder and both Streamlit pages all call `load_model()`, which reads the saved file. If the file is missing, it raises an error rather than training a new model.
2. **Saved models are protected.** `python -m src.models.timber_transfer --save` refuses to overwrite existing models. Replacing them takes `--save --force`, which creates a new `model_version`.
3. **Library versions are checked.** `load_model()` warns if the installed scikit-learn, LightGBM, NumPy or joblib version differs from the one recorded in the metadata. joblib files can fail to load, or quietly change predictions, across library versions. **Pin those versions in `requirements.txt` for deployment.**
4. **Every result is traceable.** `load_model_metadata(kind)["model_version"]` identifies which model produced a forecast. Store it with every saved result (§9).

### 8.3 Forecasting an imported pile

```bash
python -m src.models.timber_forecast --input <folder> --model gbm --pathway A1FI --out <file>.csv
```

**Input:** a folder in the template format (`src/data/timber_template.py`):
- `timber_assets.csv`: pile attributes and codes.
- `climate.csv`: each scenario's climate, past and future.
- `timber_observations.csv` (optional): measured remaining diameter. A blank `scenario_id` means a real inspection, which is applied to every scenario.

**Steps:**
1. The k factors: the engineer import (`.xlsx` or its 4 CSVs) gives the 6 pile coefficients **as numbers** (`k_heartwood`, `k_sapwood`, `k_shelter`, `k_protect`, `k_contact`, `k_knot`; §12.7), looked up in the AS 5604 table in `docs/timber_import_guide.md` §2.2. The yearly-climate template folder (`data/templates/port_kembla`) still uses codes; there, undefined codes such as MTN3 are rejected unless `--allow-unknown-codes` is passed (Defect D3). k_sal is always worked out from zone and salinity. The pile's `sapwood_layer_mm` is passed to the model as `layer_of_sapwood_mm` (`assemble_timber_frame`).
2. The saved model predicts material loss.
3. If history exists, each forecast is anchored: shifted by the model's mean error over the last 5 years of inspections up to the latest one (`--no-anchor` turns this off). Offsets over 15 mm trigger a warning and are recorded in `result.attrs["anchor"]` (the CLI also writes `<out>.anchor.json`).
4. **Output:** dashboard-format rows. `observed` rows hold the history; `gbm`/`mlp` (or `gbm_anchored`/`mlp_anchored`) rows hold the 3 scenario forecasts.

**Test with the sapwood fix** (PileB imported as a new pile, inspections 2006–2026, A1FI forecast 2027–2070):

| | GBM (reduced) R² / RMSE / MAE | MLP R² / RMSE / MAE |
|---|---|---|
| Not anchored | 0.969 / 7.2 / 4.3 | 1.000 / 0.7 / 0.4 |
| Anchored (5-year mean) | 0.768 / 19.6 / 13.8 — offset −25.8 mm, **warning** | 0.998 / 1.7 / 1.1 — offset −3.6 mm |

With the fix, anchoring is no longer needed for the MLP. For the reduced GBM it makes the forecast worse, and the warning says so: its offset isn't a constant bias. **Before the fix**, these were GBM 0.761 → 0.802 and MLP 0.829 → 0.997 (latest-value anchor); with the archived Optuna GBM, 0.828 / 16.9 / 10.7 and 0.998 / 1.7 / 1.1.

 A pile with no history still gets a forecast from its attributes and climate alone.

This replaces the earlier plan to retrain on each import. In the 15-year import test, retraining scored R² 0.970 for the GBM and was unstable for the MLP (−0.94). Anchoring with the saved model scores higher and needs no retraining.

## 9. Saving imported results in Supabase (proposed design)

Not implemented yet. This is a schema and flow that fits the code above.

**Flow:** the user uploads a file → `forecast_pile()` with the saved model → write the import, its inputs and its predictions to Supabase, tagged with `model_version`.

```sql
create table timber_models (            -- one row per saved model version
  model_version   text primary key,     -- from artifacts/timber_transfer_*.json
  model_kind      text not null,        -- 'gbm' | 'mlp'
  saved_at        timestamptz not null,
  library_versions jsonb not null,
  scores          jsonb,                -- PileB scores from the metadata
  storage_path    text                  -- optional: copy of the .joblib in Supabase Storage
);

create table timber_imports (           -- one row per uploaded file
  import_id     uuid primary key default gen_random_uuid(),
  created_at    timestamptz not null default now(),
  uploaded_by   uuid references auth.users,
  file_name     text,
  site          text,
  model_version text not null references timber_models,
  pathway       text not null,          -- 'A1FI'
  anchored      boolean not null
);

create table timber_import_assets (     -- the uploaded pile attributes
  import_id uuid references timber_imports on delete cascade,
  asset_id  text,
  attributes jsonb not null,            -- timber_assets.csv row
  primary key (import_id, asset_id)
);

create table timber_import_observations (
  import_id uuid references timber_imports on delete cascade,
  asset_id  text,
  year      int,
  remaining_diameter_mm double precision,
  material_loss_mm      double precision,
  primary key (import_id, asset_id, year)
);

create table timber_predictions (       -- same 8 columns as the dashboard CSVs
  import_id uuid references timber_imports on delete cascade,
  material  text, asset_id text, scenario text, year int,
  value double precision, unit text, metric text, method text,
  primary key (import_id, asset_id, scenario, year, method)
);
```

**Notes:**
- **Keep keys out of the code.** Put `SUPABASE_URL` and `SUPABASE_KEY` in `.env` (already git-ignored, and already loaded through `python-dotenv`) or in `.streamlit/secrets.toml`.
- **Server only for the service-role key,** and enable Row Level Security on these tables.
- **Store `model_version` with every import,** so any saved result can be traced to the exact model file.
- **Where to keep the model:** the repo (current) is simplest. Supabase Storage is an alternative if the app should download the model at runtime. Either way, predictions must come from the saved file, not a new training run.

## 10. Limitations

1. **Only two piles.** Every pile-level factor is constant within a pile, so no model can learn how timber class, exposure, maintenance or diameter affect decay. It can only *assume* it, through the k0 scaling. **More piles is the single biggest improvement available.**
2. **The truth is a spreadsheet model, not measurements.** Scores measure faithfulness to the AS 5604-style spreadsheet. Real inspection data would be noisier and would lower every number here.
3. **The spreadsheet has known defects; only D8 and D9 are fixed.**
   - D1: the sapwood loss happens instantly in year one.
   - D2: deterioration doesn't accumulate year by year; up to about 45% too high by 2070.
   - D3: undefined maintenance, contact and knot codes.
   - D7: no flooring of diameter at 0.
   - D10: the climate scenarios are outdated (2000-era SRES).

   The models learn these defects along with everything else.
4. **Tuning can only use PileA.** The chronological hold-out tests forecasting forward within PileA, but no validation set from one pile can show which settings transfer best to a *different* pile. That needs leave-one-pile-out validation, and therefore more piles.
5. **A single held-out test pile,** which is itself partly synthetic (D3 codes; its v4 timber type read "Something else").
6. **The 28 mm sapwood bias is fixed** (§12.6) by keeping the sapwood layer out of the k0 scaling. The fix copies the spreadsheet's sapwood rule exactly, including Defect D1 (the whole layer lost in year one). If the engine's sapwood phase is ever corrected (§11 item 5), `sapwood_loss()` must change with it.
7. **Shifting and anchoring assume the error is constant.** That holds for the MLP, not for the reduced GBM. Both now average 5 years of observations and warn above 15 mm, so a single bad inspection can't set the offset and a large shift isn't applied silently.
8. **The hazard-scaling assumption.** Normalising by k0 assumes decay scales with the code-based hazard factors. That's true of the spreadsheet by design, and unverified for real piles.
9. **Climate is site-specific.** The scenarios are Port Kembla's. Other sites need their own projections in `climate.csv`.
10. **Saved models depend on library versions** (§8.2).
11. **The current GBM's PileB score is not an independent test.** Its tree count was chosen by checking PileB's R² directly, so PileB was used to pick the settings. It also fails the PileA hold-out (R² −1.10, §5) that the project otherwise uses for model selection. Treat its scores (0.797 before the sapwood fix, 0.926 after) as the result of a deliberately chosen capacity, not as evidence of how well a GBM generalises. The MLP's score is still an independent test.
12. **Shifted scores are not unseen-pile scores.** A shift uses PileB's real 2022–2026 values, so shifted R² describes an "after inspection update", not model accuracy. Quote the unshifted scores (Summary table).
13. **R² ≈ 1.000 reflects the spreadsheet, not real-world accuracy.** With the sapwood bias removed, the target is close to the spreadsheet's own formula, and even untuned models and Ridge score 0.998–1.000 on PileB. That shows the pipeline encodes the formula's structure correctly; it does not show the models would be this accurate on real inspection data.

## 11. Recommended next steps

1. **Get more piles,** ideally with real inspection records. Then switch tuning to leave-one-pile-out cross-validation.
2. **Pin scikit-learn 1.9.0, LightGBM 4.7.0 and NumPy 2.5.2** in `requirements.txt` for deployment.
3. **Build the upload UI.** A Streamlit file upload that calls `forecast_pile()` and writes to Supabase (§9).
4. ~~Normalise only the heartwood part of the loss.~~ **Done (§12.6).** As predicted, the models now largely reproduce the spreadsheet formula (R² ≈ 1.000).
5. **Fix the spreadsheet defects D1–D3** in a corrected engine, and regenerate the data. Update `sapwood_loss()` to match.
6. **Decide which GBM to keep** (§12.6 table). The reduced GBM still has a 25 mm jump and a shape error; the Optuna GBM with the fix scores 1.000. To switch: set `TRANSFER_GBM_PARAMS = ARCHIVED_TRANSFER_GBM_PARAMS`, run `--evaluate --save --force`, rebuild the dashboard CSVs.
7. ~~Use the same five-year shift for imports.~~ **Done (§12.6).**
8. **Re-run Optuna for the MLP on the new target.** Its settings were tuned for the old target; its PileA hold-out dropped from 0.997 to 0.985.
9. **Decide the dashboard builder's default.** It still shifts unless `--no-shift` is passed; the current files were built with `--no-shift`.

## 12. Files changed across this work

| Area | Files |
|---|---|
| Package reorganisation | `src/timber/*` moved into `src/data/`, `src/features/`, `src/models/`, `src/ui/` with a `timber_` prefix |
| D9 fix | `src/data/build_timber_kembla_csv.py`, `src/data/timber_data_prep.py`, regenerated `data/processed/timber_kembla_v6_3_consolidated.csv`, `src/features/timber_features.py`, `src/models/timber_engine.py`, `train_timber_linear.py`, `train_timber_svr.py`, `predict.py`, four test files |
| Time-split models (Strategy B) | `src/models/timber_model.py`, `timber_model_gbm.py`, `timber_tuning.py`, `requirements.txt` (optuna) |
| Final models (Strategy E) | `src/models/timber_transfer.py` (GBM, MLP, Ridge; tuning, evaluation, `anchor_forecast`, `save_models`) |
| Sapwood fix (Strategy H) | `src/models/timber_transfer.py` (`sapwood_loss`, `MODEL_INPUT_COLUMNS`, `shift_forecast`, `flag_large_offsets`), `src/models/timber_forecast.py`, `src/data/timber_template.py`, `src/models/build_dashboard_output.py`, `src/ui/timber_data_app.py`, `src/ui/timber_results_app.py`, `tests/test_timber_pipeline.py` |
| Saved models | `artifacts/timber_transfer_{gbm,mlp}.{joblib,json}`, `.gitignore` (exceptions for these four files) |
| Import / forecasting | `src/models/timber_forecast.py`, `src/data/timber_template.py`, `src/data/timber_import.py`, `src/ui/timber_import_page.py`, `data/templates/`, `docs/timber_import_guide.md` |
| Charts | `src/ui/timber_results_app.py`, `src/ui/timber_data_app.py` |
| Dashboard data | `src/models/build_dashboard_output.py` (`--material timber`, `--pathway`); `data/dashboard/dashboard_data_timber_pileB_from_pileA_{gbm,mlp}.csv` |
| Docs | this report, `docs/timber_model_tuning.md`, `docs/timber_pileA_to_pileB_report.md`, `docs/timber_kembla_model_spec_and_defects.md`, `README.md` |

### 12.1 Clean-up and final retrain (2026-10-04)

**Removed** (old timber experiments not used by the final models):
- `src/models/timber_compare.py`
- `src/models/timber_baselines.py`
- `src/models/timber_reporting.py`
- `src/models/timber_residual.py`
- `src/models/timber_shap_importance.py`
- `src/models/timber_inference.py`
- `src/data/build_timber_sydney.py`

**Fixed so nothing broke:**
- **`build_dashboard_output.py`** no longer imports `timber_inference`. The old timber export anchored at 2050 is gone, and `--material timber` now always writes the saved PileA→PileB models' forecasts (`--transfer` is still accepted). Concrete is unchanged.
- **`tests/test_timber_compare.py`** no longer has the 4 tests for the removed modules. The feature, target, GBM and quantile tests remain.

**Still present, but not used by the final models:**
- `timber_quantile.py`, `timber_scenarios.py`, `timber_tuning.py`
- `src/ui/timber_app.py`
- `scripts/train_pileA_predict_pileB.py`. **Don't run it:** it overwrites the dashboard CSVs with the original models.

Yuma's `predict.py`, `train_timber_linear.py` and `train_timber_svr.py` were already broken before this work and were left untouched.

**Retrain:**
1. Feature selection re-run: still `thermal_dose`.
2. Optuna re-run for both models: GBM hyperparameters identical; MLP now 2 × 216.
3. Full evaluation re-run.
4. Models saved as new versions (`…_20261004_075944`).
5. Dashboard CSVs rebuilt from the saved models.
6. All Streamlit pages checked.

**Effect on results:**
- **GBM:** unchanged.
- **MLP:** PileB R² 0.952 → 0.955 overall; A1FI forecast 0.828 → 0.826; anchored import 0.998 → 0.999.

**Tests:** 48 passed: the 42 from before, plus 10 import tests, minus the 4 removed tests. `tests/test_timber_prediction.py` (Yuma's) was already broken before this work and is still excluded.

### 12.2 Monte Carlo CV replaced by a chronological hold-out (2026-10-04)

**Changed:**
- **`src/models/timber_transfer.py`:** Monte Carlo CV (`monte_carlo_cv`, `ShuffleSplit`) removed. Feature selection, Optuna tuning and the PileA check in `--evaluate` now use `chronological_holdout()`: each PileA climate series trains on 2000–2056 and validates on 2057–2070.
- **Tuned settings moved to the model files:** `TRANSFER_MLP_PARAMS` in `src/models/timber_model.py` and `TRANSFER_GBM_PARAMS` in `src/models/timber_model_gbm.py`. `TRANSFER_RIDGE_PARAMS` stays in `timber_transfer.py`.
- **Page wording:** `src/ui/timber_data_app.py` and `src/ui/timber_results_app.py` now say "hold-out" instead of "Monte Carlo CV".
- **Removed `src/models/timber_optuna_holdout.py`** and its `artifacts/timber_holdout_*` outputs. It was a trial of this method, now merged into the main pipeline.

**Re-run:**
1. Feature selection: still `thermal_dose`.
2. Optuna for GBM, MLP and Ridge.
3. Evaluation.
4. Models saved (`…_20261004_082023`).
5. Dashboard CSVs rebuilt.
6. 48 tests and all pages checked.

**Effect on the saved models:**

| | Tuned with Monte Carlo CV | Tuned on the hold-out (now) |
|---|---|---|
| GBM settings | 1066 trees, `reg_lambda` 3.58 | 945 trees, `reg_lambda` 0.0003 (both depth 3, 7 leaves, linear leaves) |
| MLP settings | 2 × 216 | 3 × 136 |
| GBM PileB R² (all / A1FI forecast) | 0.950 / 0.828 | 0.952 / 0.828 |
| MLP PileB R² (all / A1FI forecast) | 0.955 / 0.826 | 0.953 / 0.829 |

Every PileB score moved by 0.003 or less, within seed-to-seed variation, so the conclusions are unchanged.

### 12.3 GBM capacity reduced to a target PileB R² (2026-10-04)

**Request:** bring the GBM's PileB R² down to a set level (first about 0.7, then about 0.8), keeping the same features, target and pipeline, and leaving the MLP unchanged.

**Changed:**
- **`src/models/timber_model_gbm.py`:** `TRANSFER_GBM_PARAMS` is now `n_estimators=25, learning_rate=0.05, max_depth=2, num_leaves=4, linear_tree=True`. The Optuna config is kept as `ARCHIVED_TRANSFER_GBM_PARAMS`.
- **`artifacts/timber_transfer_gbm.{joblib,json}`:** refitted and saved as `timber_transfer_gbm_20261004_111954`. Only the GBM was saved (`save_models(frame, kinds=("gbm",), overwrite=True)`); the MLP files were not touched. The previous GBM is in `artifacts/_backup_before_gbm_retune/`.
- **`data/dashboard/dashboard_data_timber_pileB_from_pileA_{gbm,mlp}.csv`:** rebuilt with `build_dashboard_output --material timber`. The MLP file's values changed only because the committed copy predated the MLP's last save; the MLP model itself is the same.
- **New preview files** (not read by the dashboard): `data/dashboard/dashboard_data_timber_pileB_from_pileA_{gbm,mlp}_anchored_preview.csv`. Removed in §12.5.

**How the settings were found:** a sweep of tree count, depth and leaves inside the unchanged `HazardNormalisedRegressor` (same `thermal_dose` feature, `material_loss_mm` target and k0 normalisation), scoring PileB R² over all years:

| GBM settings | PileB R² all | A1FI forecast R² | Jump at 2026→2027 |
|---|---:|---:|---:|
| 945 trees, depth 3, linear leaves (Optuna, archived) | 0.952 | 0.828 | about +35 mm |
| 21 trees, depth 2, constant leaves (first target, 0.7) | 0.708 | 0.766 | +60.8 mm |
| **25 trees, depth 2, linear leaves (current, 0.8)** | **0.797** | **0.761** | **+51.9 mm** |

Two things don't work for this: lowering capacity on the raw-diameter target (without hazard normalisation) never rose above R² ≈ 0.55, and wrapping the GBM in the old rate-target regressor dropped it to about 0.13. Linear leaves were kept for the 0.8 setting because they make the curve smoother and the jump smaller than constant leaves at a similar R².

**Effect:**

| | Before (Optuna GBM) | After (current GBM) |
|---|---|---|
| GBM PileB R² (all / A1FI forecast) | 0.952 / 0.828 | 0.797 / 0.761 |
| GBM PileA hold-out R² | 1.000 | −1.100 |
| GBM anchored import R² | 0.998 | 0.802 |
| MLP (everything) | unchanged | unchanged |

**Side note:** while investigating, the retired `scripts/train_pileA_predict_pileB.py` was run once by mistake. Its two changed files (the MLP dashboard CSV and `scripts/train_pileA_predict_pileB_metrics.json`) were restored from git before anything else was rebuilt.

**Tests:** no test pins the GBM's R² value. The suite was re-run in §12.4.

### 12.4 Full re-run and re-save (2026-10-04)

**Re-run:**
1. Evaluation: `python -m src.models.timber_transfer --evaluate --save --force`. Feature selection and Optuna were **not** re-run, so the MLP keeps its settings (3 × 136) and the GBM keeps the §12.3 settings.
2. Both models saved as new versions, `timber_transfer_{gbm,mlp}_20261004_114250`. The previous files were backed up first (§8.1).
3. Dashboard CSVs rebuilt: `python -m src.models.build_dashboard_output --material timber`.
4. Tests: `pytest tests --ignore=tests/test_timber_prediction.py`: **48 passed**. The 15 warnings are the expected Defect D3 messages for PileB's undefined codes.

**Effect:** none. Every score matches §7 to the reported precision, and both re-saved models give exactly the same predictions as the files they replaced (largest difference 0.0 mm over all 994 rows). The models are deterministic: the GBM has no subsampling, and the MLP and Ridge use fixed seeds.

| Model (re-run) | PileB R² all | RMSE | MAE | PileB R² forecast (both pathways) | PileA hold-out R² |
|---|---:|---:|---:|---:|---:|
| GBM | 0.797 | 41.69 | 31.64 | 0.757 | −1.100 |
| MLP | 0.953 | 20.05 | 15.26 | 0.825 | 0.997 |
| Ridge (reference) | 0.954 | 19.86 | 15.47 | 0.788 | 0.962 |

### 12.5 Dashboard forecasts shifted to PileB's real history (2026-10-04)

**Why:** the dashboard showed a large step at 2026 (GBM +51.9 mm, MLP +33.9 mm) between PileB's real values and the raw forecast. Three options were previewed: the raw forecast, shifting by the 2022–2026 mean error, and retraining on PileA plus PileB's 2000–2026 rows. Shifting was chosen.

| Option previewed (A1FI forecast 2027–2070) | GBM R² / MAE / jump | MLP R² / MAE / jump |
|---|---|---|
| Raw model forecast | 0.761 / 12.1 mm / +51.9 mm | 0.829 / 10.6 mm / +33.9 mm |
| **Shift by 2022–2026 mean error (chosen)** | **0.782 / 13.2 mm / +0.3 mm** | **1.000 / 0.3 mm / +5.6 mm** |
| Retrain with PileB 2000–2026 included | 0.927 / 6.4 mm / +32.1 mm | 0.791 / 12.1 mm / +5.0 mm |

Shifting is the only option that removes the jump for both models, needs no retraining, and keeps PileB unseen by the models. Retraining helped the reduced GBM but left a 32 mm step and made the MLP worse, as in the earlier import test (§8.3).

**Changed:**
- **`src/models/timber_transfer.py`:** new `SHIFT_WINDOW = (2022, 2026)` and `shift_forecast()`. `--evaluate` now reports each model's A1FI forecast scores raw and shifted (`pileB_forecast_a1fi`), and `save_models()` writes shifted forecast scores and the window into each model's `.json`.
- **`src/models/build_dashboard_output.py`:** `--material timber` shifts the forecasts by default; `--no-shift` writes the raw forecast.
- **`tests/test_timber_pipeline.py`:** new test that `shift_forecast()` removes each series' mean window error and caps at the radius.
- **`artifacts/timber_transfer_{gbm,mlp}.{joblib,json}`:** re-saved as `…_20261004_140554` so the metadata includes the shifted scores. Predictions are identical to the previous files (largest difference 0.0 mm); previous files in `artifacts/_backup_before_shift/`.
- **`data/dashboard/dashboard_data_timber_pileB_from_pileA_{gbm,mlp}.csv`:** rebuilt with the shift. Same 213 rows and method labels; forecast values match the approved preview to within 0.005 mm.
- **Removed** the preview-only `…_{gbm,mlp}_anchored_preview.csv` files from §12.3.

**Tests:** 49 passed (48 plus the new shift test).

**Superseded by §12.6:** the shift was later reverted, then the bias it hid was fixed in the models themselves. The dashboard files are now built unshifted.

### 12.6 Sapwood-aware normalisation: the 28 mm bias removed in the models (2026-10-04)

**Why:** the history/forecast jump at 2026 and the −28 mm shift offsets came from one cause. The spreadsheet removes the pile's whole sapwood layer in year one whatever the hazard, but the hazard normalisation scaled it by k0 (§7.2). Shifting hid this using PileB's own 2022–2026 values, which gave a misleading forecast R² of 1.000.

**The sapwood rule, confirmed in `src/models/timber_engine.py` (`simulate`):** `age = max(0, year − year_installed)`; loss = 0 if age ≤ 0; the whole `layer_of_sapwood_mm` (`d_sap`) while 0 < age ≤ t_sap + t_lag,heart (Defect D1); after that, `d_sap + (age − t_sap − t_lag,heart) × r_heart`. The layer is a **per-pile attribute** (workbook column M), 20 mm for both piles, not a constant. It applies from age 1: PileB is 0 mm in 2006 and 20 mm in 2007.

**Changed (timber files only):**
- **`src/models/timber_transfer.py`:**
  - New `SAPWOOD_COLUMN = "layer_of_sapwood_mm"`, `MODEL_INPUT_COLUMNS = FEATURE_COLUMNS + [SAPWOOD_COLUMN]` and `sapwood_loss(X)`.
  - `HazardNormalisedRegressor.fit` learns `(y − sapwood_loss) / k0`; `predict_uncapped` returns `sapwood_loss + prediction × k0`. The sapwood column is never an estimator feature.
  - `shift_forecast()` was restored, now returning its offsets and warning when one exceeds `LARGE_OFFSET_MM = 15`; the shared check is `flag_large_offsets()`.
  - `save_models()` records the sapwood column, shifted scores and the shift record; the step-1 ablation baseline still uses `FEATURE_COLUMNS` only.
- **`src/models/timber_forecast.py`:**
  - Imports pass `MODEL_INPUT_COLUMNS`.
  - `_anchor()` now uses the mean error over the last 5 years of inspections, warns above 15 mm, and records the offsets in `result.attrs["anchor"]` (plus `<out>.anchor.json` from the CLI).
- **`src/data/timber_template.py`:** `assemble_timber_frame` passes the asset's `sapwood_layer_mm` as `layer_of_sapwood_mm`.
- **`src/models/build_dashboard_output.py`:** `--no-shift` restored; shifted builds write their offsets to `<file>.shift.json`, unshifted builds remove a stale one.
- **`src/ui/timber_data_app.py`, `src/ui/timber_results_app.py`:** pass `MODEL_INPUT_COLUMNS`.
- **`tests/test_timber_pipeline.py`:** two new tests: `sapwood_loss` is 0 before installation and the full layer after; a PileA-trained MLP needs a mean PileB shift under 5 mm, with no offsets flagged.

**Re-run:**
1. Backed up `artifacts/timber_transfer_{gbm,mlp}.{joblib,json}` to `artifacts/_backup_before_sapwood_fix/`.
2. Ran `python -m src.models.timber_transfer --evaluate --save --force`. Optuna and feature selection were not re-run.
3. Saved as `timber_transfer_{gbm,mlp}_20261004_143534`.
4. Rebuilt the dashboard CSVs with `python -m src.models.build_dashboard_output --material timber --no-shift`.
5. `scripts/train_pileA_predict_pileB.py` was not run.

**Before / after** (PileB, A1FI; forecast = 2027–2070; all scores unshifted unless marked). "Before" was reproduced exactly by giving the new code a sapwood layer of 0.

| | GBM (reduced, saved) before | after | GBM (Optuna, not saved) before | after | MLP before | after |
|---|---:|---:|---:|---:|---:|---:|
| R² all years | 0.797 | 0.926 | 0.952 | 1.000 | 0.953 | 1.000 |
| RMSE / MAE all years (mm) | 41.69 / 31.64 | 25.18 / 17.52 | 20.26 / 15.02 | 0.25 / 0.16 | 20.05 / 15.26 | 1.27 / 0.74 |
| R² forecast | 0.761 | 0.969 | 0.828 | 1.000 | 0.829 | 1.000 |
| RMSE / MAE forecast (mm) | 19.9 / 12.1 | 7.2 / 4.3 | 16.9 / 10.7 | 0.3 / 0.2 | 16.8 / 10.6 | 0.7 / 0.4 |
| Jump, real 2026 → 2027 forecast | +51.9 | +25.1 | +33.7 | **+5.8** | +33.9 | **+7.0** |
| Mean shift offset 2022–2026 (Most Likely / Hotter & Drier / Colder & Wetter) | −52.2 / −51.5 / −51.3 | −24.4 / −23.5 / −23.3 | −27.9 / −27.7 / −27.7 | −0.1 / +0.1 / +0.1 | −28.3 / −28.1 / −28.5 | −1.3 / −1.6 / −1.6 |
| PileA hold-out R² | −1.100 | −1.102 | 1.000 | 1.000 | 0.997 | 0.985 |
| Mean predicted loss before installation (2000–2005; truth 0) | 66.2 | 46.6 | 0.1 | 0.0 | 5.1 | 0.0 |
| Shifted forecast R² ("after inspection update" only) | 0.782 | 0.801 | 1.000 | 1.000 | 1.000 | 1.000 |

**Result:**
- **MLP and Optuna GBM:** the bias is gone. The jump is about one year of decay, the shift offsets are under 2 mm, and nothing is predicted before installation. No shift is needed.
- **Reduced GBM:** it improves, but keeps a shape error: a −23 mm offset, a +25 mm jump and 47 mm predicted before installation. Shifting makes it worse (0.969 → 0.801), and every one of its series is flagged as a large offset in its metadata.
- **Decision pending:** keep the reduced GBM, or restore the Optuna GBM (§11 item 6). Not changed without approval.

**Tests:** `pytest tests --ignore=tests/test_timber_prediction.py`: **50 passed** (48 plus the 2 new tests). The 15 warnings are the expected Defect D3 messages.

### 12.7 Import: coefficients as numbers, clean example, history checks (2026-10-04)

**Why:** engineers find it easier, and less ambiguous, to enter the AS 5604 coefficient values than code labels. Numbers also let them use a value the table doesn't list.

**Changed:**
- **`src/data/timber_import.py`:**
  - The 6 code columns in `timber_assets` (`heartwood_class`, `sapwood_class`, `exposure`, `maintenance`, `contact`, `knot`) are replaced by numbers: `k_heartwood` (0.1–50), `k_sapwood` (0.1–50), `k_shelter` (0.1–2), `k_protect` (0.1–2), `k_contact` (0.1–5), `k_knot` (0.1–5). `zone` stays, for k_sal.
  - New warnings: a `k_` value not in the AS 5604 table; a `k_heartwood` taken from the wrong zone column; inspections before 2000, which are not used.
  - The "no inspections" warning no longer mentions the 28 mm bias, which §12.6 removed.
  - New `coefficient_table()` (code → value, from the `Coefficients` sheet). The template now has a `coefficients` sheet and `skeleton/coefficients.csv`.
  - The example pile is now HW1 zone E (k_heartwood 1.3), with inspections taken from the workbook engine (402, 385, 367 and 348 mm), so the example forecasts cleanly.
- **`src/data/timber_template.py`:** `derive_timber_factors` uses the `k_` numbers when the assets give them, and looks up codes otherwise (the Port Kembla template folder).
- **`data/templates/timber_import_template.xlsx`, `data/templates/skeleton/`:** regenerated (`python -m src.data.timber_import --skeleton`).
- **`docs/timber_import_guide.md`:** rewritten. It now has the coefficient tables, the new columns and rules, an example output compared with the engine, and a history/horizon section.
- **`tests/test_timber_import.py`:** the validation cases now use the `k_` columns. New tests check the new warnings and that the `k_` numbers reach the model's factors.

**Import checked end to end** (validate → saved models → forecast), with histories made by the workbook engine so the true future is known. A1FI forecast R², anchored:

| History | MLP | GBM (reduced) |
|---|---:|---:|
| Example pile, inspections 2010–2022 every 4 years → forecast 2023–2070 | 0.997 | 0.696 |
| Inspections every year 2015–2025 → 2026–2070 | 0.996 | 0.669 |
| Installed 1985, inspections 1990–2005 (only 2000–2005 used) → 2006–2070 | 0.997 | 0.701 |

For the example pile the MLP's replacement years (2051 / 2048 / 2048) match the engine exactly; the GBM's are 7–9 years late. Forecasts end in 2070, where the climate inputs end; see the guide's §6.

**Tests:** 53 passed (50 plus 3 new import cases).

## 13. Reproduce

```bash
python src/data/build_timber_kembla_csv.py data/raw/Timber_kembla_v6.3.xlsx   # D9-corrected CSV
python -m src.models.timber_transfer --select-features                         # §5 feature set
python -m src.models.timber_transfer --tune gbm --trials 50                    # §5 Optuna (prints the archived config; don't paste it over TRANSFER_GBM_PARAMS unless restoring)
python -m src.models.timber_transfer --tune mlp --trials 50
python -m src.models.timber_transfer --evaluate                                # §7 results
python -m src.models.timber_transfer --save --force                            # §8 replace BOTH saved models (new versions)
python -c "from src.models.timber_transfer import load_transfer_frame, save_models; save_models(load_transfer_frame(), kinds=('gbm',), overwrite=True)"  # GBM only
python -m src.models.build_dashboard_output --material timber --no-shift       # §7.4 dashboard CSVs (unshifted, as built)
python -m src.models.build_dashboard_output --material timber                  # shifted option; offsets -> <file>.shift.json
python -m src.data.timber_import --skeleton                                    # import template + skeleton CSVs
python -m src.data.timber_import --validate data/templates/timber_import_template.xlsx
python -m src.models.timber_forecast --input data/templates/timber_import_template.xlsx --model mlp --out forecast.csv
python -m src.models.timber_forecast --input data/templates/port_kembla --model gbm --allow-unknown-codes
python -m streamlit run src/ui/timber_data_app.py                              # charts
```

## 14. References

1. Akiba, T., et al. (2019). *Optuna: A Next-generation Hyperparameter Optimization Framework.* KDD 2019. https://arxiv.org/abs/1907.10902
2. Bergstra, J., Bardenet, R., Bengio, Y., & Kégl, B. (2011). *Algorithms for Hyper-Parameter Optimization.* NeurIPS 24. http://papers.neurips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf
3. Ke, G., et al. (2017). *LightGBM: A Highly Efficient Gradient Boosting Decision Tree.* NeurIPS 30. https://proceedings.neurips.cc/paper_files/paper/2017/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf
4. Shi, Y., Li, J., & Li, Z. (2019). *Gradient Boosting with Piece-Wise Linear Regression Trees.* IJCAI 2019. https://www.ijcai.org/proceedings/2019/476
5. Cerqueira, V., Torgo, L., & Mozetič, I. (2020). *Evaluating time series forecasting models: an empirical study on performance estimation methods.* Machine Learning, 109, 1997–2028. https://link.springer.com/article/10.1007/s10994-020-05910-7
6. Karpatne, A., et al. (2017). *Theory-Guided Data Science: A New Paradigm for Scientific Discovery from Data.* IEEE TKDE, 29(10). https://arxiv.org/abs/1612.08544
7. Willard, J., et al. (2022). *Integrating Scientific Knowledge with Machine Learning for Engineering and Environmental Systems.* ACM Computing Surveys, 55(4). https://arxiv.org/abs/2003.04919
8. Kaufman, S., Rosset, S., Perlich, C., & Stitelman, O. (2012). *Leakage in Data Mining: Formulation, Detection, and Avoidance.* ACM TKDD, 6(4). https://dl.acm.org/doi/10.1145/2382577.2382579
9. Wang, C.-H., Leicester, R. H., & Nguyen, M. N. (2008). *Probabilistic procedure for design of untreated timber poles in-ground under attack of decay fungi.* Reliability Engineering & System Safety, 93, 476–481. https://www.sciencedirect.com/science/article/abs/pii/S0951832007000154
