# Timber Models: Train on PileA, Test on PileB

**Code:** `src/models/timber_transfer.py` (models, Monte Carlo CV, Optuna, evaluation), `src/ui/timber_results_app.py` (charts)
**Data:** `data/processed/timber_kembla_v6_3_consolidated.csv`, regenerated with corrected headers (Defect D9 fix)
**Raw results:** `artifacts/timber_transfer_results.json`

## Summary

Both models were trained on **PileA only** and scored on **PileB**, a pile they never saw. Both use the same target (material loss, mm), the same features and the same techniques.

| Model | PileB R² (all years) | PileB R² (forecast 2027–2070) | RMSE (mm) | MAE (mm) |
|---|---:|---:|---:|---:|
| GBM (LightGBM) | **0.950** | **0.825** | 20.64 | 15.44 |
| MLP | **0.952** | **0.825** | 20.20 | 15.34 |

- **Neither model scores 1.** Both over-predict PileB's loss by about 28 mm. That gap has a known physical cause (§5.3).
- **The two models end up almost level.** They were not pushed towards different targets (e.g. 0.70 and 0.85). Both were tuned by the same procedure and scored once on PileB, and these are the scores that came out. Making one model deliberately worse to create a gap would make the comparison meaningless.
- **Over the forecast years, which is what the dashboard shows, both score about 0.82.**

**Before vs. after on PileB (all years):**

| Model | Before (original untuned models) | After |
|---|---:|---:|
| MLP | 0.251 | 0.952 |
| GBM | 0.270 | 0.950 |

---

## 1. Step 1: fixing Defect D9 (mislabelled columns)

### What was wrong

The CSV export gave the workbook's k-factor columns the wrong names. The defect register listed 3 columns, but the problem was larger: **every label from column AQ to AY was one column out of step**. I verified this two ways:

1. Against the workbook's own headers (row 7 of the PileA sheet).
2. Against the k factors recomputed by `src/models/timber_engine.py`. Every value matches.

| Col | Workbook header | Old CSV name | New CSV name |
|---|---|---|---|
| AP | ktimber HW | `timber_exposure_factor` | `heartwood_factor` |
| AQ | ktimber SW | `model_hazard_wood` | `sapwood_factor` |
| AR | ksal | `model_service_water` | `salinity_factor` |
| AS | kshelter | `salinity_factor` | `shelter_factor` |
| AT | kprotect | `shelter_factor` | `protection_factor` |
| AU | kcontact | `protection_factor` | `contact_factor` |
| AV | kknot | `model_contact_factor` | `knot_factor` |
| AW | kenv | `model_knot_factor` | `environment_factor` |
| AX | kconst | `environment_factor` | `construction_factor` |
| AY | k0 | `construction_factor` | `combined_factor` |
| BJ | treplacement | `replacement_time_years` | `design_life_years` |

Several old names reappear as correct names for a *different* column; for example, old `salinity_factor` (really k_shelter) versus new `salinity_factor` (really k_sal). So the rename is applied in a single pass, never one name at a time.

### What was done

1. **Fixed the CSV builder** (`src/data/build_timber_kembla_csv.py`) so it writes the workbook's names.
2. **Regenerated the Kembla CSV** from `data/raw/Timber_kembla_v6.3.xlsx` and compared it with the old file. All 994 rows and all 46 columns have identical values; only the 11 headers changed.
3. **Made the loader rename old exports automatically.** `src/data/timber_data_prep.py` now has `D9_COLUMN_RENAMES` and `fix_d9_column_names()`, which rename any export still using the old headers (e.g. `timber_sydney_v6_3_consolidated.csv`) when it is loaded.
4. **Updated the feature list.** `FEATURE_COLUMNS` now names all 11 k factors correctly.
   - Previously the list held only 8 of them, under the wrong names.
   - The two additions, `sapwood_factor` and `salinity_factor`, are constant for both piles today (25 and 1.0).
   - Feature count: 13 → 15.
5. **Fixed `decay_dose`.** It was meant to use salinity, but was actually reading k_shelter. It now reads the real k_sal.
6. **Marked D9 as fixed** in `docs/timber_kembla_model_spec_and_defects.md`.

**Side effect on earlier results:** the two added (constant) columns change the MLP slightly on the earlier time-based split in `docs/timber_model_tuning.md`. The GBM is unchanged.

| Time-split test R² | Before D9 fix | After D9 fix |
|---|---:|---:|
| MLP baseline | 0.625 | 0.877 |
| MLP tuned | 0.968 | 0.953 |
| GBM baseline | 0.644 | 0.644 |
| GBM tuned | 1.000 | 1.000 |

## 2. Step 2: why train-on-PileA / test-on-PileB was failing

With the original models, PileB R² was about 0.25–0.27, and **negative (−4.7 to −4.9) on the forecast years**.

The cause is structural. Every pile-level factor is a **single constant across all of PileA's rows**:
- `heartwood_factor` = 1.3
- `shelter_factor` = 1.0
- `combined_factor` (k0) = 1.3
- initial diameter = 400 mm

A model trained only on PileA therefore has no way to learn what these factors do. PileB's values are different (k0 = 3.12, diameter 500 mm), so a PileA-trained model predicts PileA-like decay for PileB, which in reality decays about 2.4× faster.

The two models fail in different ways:
- **GBM:** tree models simply ignore features that never change in training.
- **MLP:** it still multiplies those features by weights. Those weights were never trained against varying values, so PileB's different constants push its output in arbitrary directions.

## 3. Step 3: what was changed in the models (same for both)

| Technique | What it does | Why |
|---|---|---|
| **Target: material loss (mm)** | `(initial_diameter − max(remaining_diameter, 0)) / 2`: radial loss, capped at the pile radius | The dashboard's metric. It is bounded, and it ignores the engine's meaningless negative diameters (D7). |
| **Hazard normalisation** | The inner model learns `loss / k0`; predictions are multiplied by the test pile's own k0 | In the AS 5604-style decay model, the heartwood decay rate is proportional to k0 (`r_heart = kwater × k0`). This lets a model trained on one pile scale to another pile's hazard level, which the data alone cannot teach it. |
| **Zero-variance filter** | Drops features that are constant across the training rows | They carry nothing learnable from one pile, and they destabilise the MLP. |
| **`thermal_dose` feature** | `age × water_factor` | Selected on PileA (Step 4). |
| **Clip predictions** | To [0, pile radius] | Loss can't be negative or exceed the radius. |

All of this lives in `HazardNormalisedRegressor` (`src/models/timber_transfer.py`). It wraps either the MLP or the LightGBM estimator, so both models go through exactly the same steps.

**Is the hazard normalisation leakage?** No. k0 (`combined_factor`) is a lookup from the pile's own codes (timber class, exposure, maintenance and so on), and it is known before any deterioration happens. It does not use PileB's target values. It is a physics assumption ("decay scales with hazard"), so the results only hold as long as that assumption holds; see §6.

## 4. Step 4: model selection with Monte Carlo cross-validation (PileA only)

**Monte Carlo CV** here means 10 repeated random 80/20 splits of PileA's 497 rows (`ShuffleSplit`), averaging the validation error. PileB was not used for anything in this step.

### 4.1 Feature set (shared by both models)

| Feature set | MLP MC-CV RMSE | GBM MC-CV RMSE | Mean |
|---|---:|---:|---:|
| Standard (15 columns) | 4.44 mm | 1.70 mm | 3.07 mm |
| **Standard + `thermal_dose`** | 3.29 mm | 0.99 mm | **2.14 mm** (selected) |

### 4.2 Hyperparameters (Optuna)

- **Sampler:** TPE, `seed=42`, with a `MedianPruner`.
- **Trials:** 50 per model.
- **Objective:** mean Monte Carlo CV RMSE on PileA.
- **Search spaces:** the same as `docs/timber_model_tuning.md` (`suggest_mlp_params`, `suggest_gbm_params`, including LightGBM's `linear_tree`).

| Model | Best MC-CV RMSE (PileA) | Chosen hyperparameters |
|---|---:|---|
| MLP | 2.02 mm | 3 × 214 ReLU layers, `alpha` 2.6e-5, learning rate 0.0052, batch 128 |
| GBM | 0.23 mm | 1066 trees, learning rate 0.0168, `max_depth` 3, 7 leaves, `linear_tree=True`, subsample 0.64, `colsample_bytree` 0.70, `reg_lambda` 3.58 |

These are stored as `TRANSFER_MLP_PARAMS` and `TRANSFER_GBM_PARAMS` in `src/models/timber_transfer.py`.

## 5. Results

### 5.1 PileB, after each step (497 rows; forecast = 2027–2070, the 3 climate scenarios, 264 rows)

| Step | Model | R² all | RMSE all | MAE all | R² forecast | RMSE forecast | MAE forecast |
|---|---|---:|---:|---:|---:|---:|---:|
| 1. Original untuned models | MLP | 0.251 | 80.05 | 68.34 | −4.884 | 98.57 | 95.74 |
| 1. Original untuned models | GBM | 0.270 | 79.05 | 66.45 | −4.689 | 96.92 | 94.08 |
| 2. + hazard normalisation, filter, `thermal_dose` | MLP | 0.955 | 19.52 | 14.96 | 0.812 | 17.61 | 11.19 |
| 2. + hazard normalisation, filter, `thermal_dose` | GBM | 0.951 | 20.42 | 15.14 | 0.820 | 17.24 | 11.00 |
| **3. + Optuna (MC CV), final** | **MLP** | **0.952** | **20.20** | **15.34** | **0.825** | **17.02** | **10.87** |
| **3. + Optuna (MC CV), final** | **GBM** | **0.950** | **20.64** | **15.44** | **0.825** | **17.01** | **10.87** |

All errors are in mm of material loss.

**Seed stability** (final models, seeds 0, 1, 2, 3, 42; PileB, all years):

| Model | R² mean ± std | RMSE mean ± std | MAE mean ± std |
|---|---:|---:|---:|
| MLP | 0.951 ± 0.002 | 20.56 ± 0.33 | 15.66 ± 0.30 |
| GBM | 0.951 ± 0.002 | 20.50 ± 0.33 | 15.30 ± 0.30 |

**Monte Carlo CV on PileA** (final models): MLP R² 0.9986 ± 0.0003; GBM R² 0.99998 ± 0.00001.

### 5.2 Tuning barely changes PileB scores (corrected)

> **Correction (2026-10-04):** an earlier version of this report said tuning made the MLP worse on PileB (0.980 → 0.954). That came from a bug: the zero-variance filter used `std() > 0`, and a repeated float like 1.3 has a std of about 1e-16, so the pile-constant `heartwood_factor` and `combined_factor` were kept instead of dropped. The MLP then fed PileB's values into weights never trained on varying inputs, and the untuned MLP happened to land well. The filter now uses `nunique() > 1`. All numbers in this report are from the fixed code. Feature selection was re-run and still picks `thermal_dose`. The Optuna searches were not re-run: within PileA those columns were constant either way, so they should not materially change which hyperparameters win. The MLP's random weight initialisation does change with the number of inputs, though, so a re-run could pick slightly different values.

After the fix, tuning is roughly neutral on PileB:

| Model | Untuned, all years | Tuned, all years | Untuned, forecast | Tuned, forecast |
|---|---:|---:|---:|---:|
| MLP | 0.955 | 0.952 | 0.812 | 0.825 |
| GBM | 0.951 | 0.950 | 0.820 | 0.825 |

This happens because Monte Carlo CV on PileA only measures how well a model fills in gaps *inside one pile's own curve*. Random row splits leave neighbouring years in training, so almost any model scores R² ≈ 0.999 there. Optuna can then improve the within-pile fit (MC-CV RMSE: MLP 3.29 → 2.15 mm), but that says little about another pile. **Within-pile Monte Carlo CV is not a reliable guide to cross-pile performance**; a better tuning objective needs more piles (§7).

### 5.3 Where the remaining error comes from

Mean material loss on PileB (mm), with the final models:

| Period | Engine | GBM | MLP |
|---|---:|---:|---:|
| 2000–2005 (pre-installation) | 0.0 | 0.0 | 5.0 |
| 2006–2015 | 31.9 | 60.1 | 55.2 |
| 2016–2026 | 89.8 | 117.7 | 118.7 |
| 2027–2040 | 164.4 | 192.1 | 192.2 |
| 2041–2050 | 235.7 | 247.3 | 247.3 |
| 2051–2070 | 249.9 | 250.0 | 250.0 |

Both models over-predict by about **28 mm from installation until the loss reaches the radius**. The error comes from the hazard normalisation:
- The engine removes a fixed 20 mm of sapwood in the first year, whatever the pile's hazard.
- Dividing *all* of the loss by k0 also scales that fixed 20 mm.
- The result is 20 × (3.12 / 1.3 − 1) = **28 mm**: exactly the observed gap.

As a consequence, both models predict PileB reaching its full 250 mm radius around 2045, while the engine says around 2050.

**This is why the scores are not 1.** It is also a concrete, fixable next step (§7.2).

### 5.4 The charts

`src/ui/timber_results_app.py` (`streamlit run src/ui/timber_results_app.py`) shows the GBM and MLP charts side by side, in the dashboard's own chart format (`material_forecast_chart`):
- **History:** PileB's material loss from 2000–2026 (engine output).
- **Projections:** each model's forecast for Most Likely, Hotter & Drier and Colder & Wetter, 2027–2070.
- **Optional overlay:** the engine's own values for the forecast years.
- **Metric tiles:** R² for all years and for the forecast years, plus RMSE and MAE.

## 6. Caveats

1. **The truth is the spreadsheet engine, not real inspections.** These scores measure how well the models reproduce the AS 5604-style engine on an unseen pile.
2. **The result rests on one test pile.** PileB is a single held-out asset (497 rows, 7 climate series), and its own data notes flag it as synthetic ("Something else" timber type in v4; Defect D3 codes).
3. **Hazard normalisation assumes decay scales with k0.** That holds in the engine, so it works well here. Real piles may not follow it, and this is the main assumption to test on real data.
4. **Monte Carlo CV within one pile is optimistic** (R² ≈ 0.999) and doesn't predict cross-pile performance (§5.2).
5. **The two models score almost the same.** The comparison shows no meaningful accuracy difference between MLP and GBM in this setup (both R² ≈ 0.95 overall, 0.825 on the forecast years). GBM trains much faster.

## 7. Recommended next steps

1. **Get more piles.** With 3 or more piles, tuning can use leave-one-pile-out CV, which measures cross-pile performance directly instead of within-pile fit (§5.2).
2. **Normalise only the heartwood part of the loss.** Subtract the fixed sapwood layer before dividing by k0, then add it back. This should remove most of the 28 mm bias (§5.3). It is a physics-based change decided in advance, not a tweak tuned against PileB.
3. **Re-tune with that change**, using the same Monte Carlo CV protocol.

## 8. Files touched

| File | Change |
|---|---|
| `src/data/build_timber_kembla_csv.py` | D9: corrected column names (`COLMAP`, numeric and output lists) |
| `data/processed/timber_kembla_v6_3_consolidated.csv` | Regenerated with corrected headers; values identical (gitignored) |
| `src/data/timber_data_prep.py` | Added `D9_COLUMN_RENAMES`, `fix_d9_column_names()`; loader applies it; `NUMERIC_COLUMNS` renamed |
| `src/features/timber_features.py` | `FEATURE_COLUMNS` renamed, 13 → 15 (all 11 k factors); `LEAKY_COLUMNS` uses `design_life_years`; `decay_dose` uses the real k_sal |
| `src/models/timber_transfer.py` | **New:** `HazardNormalisedRegressor`, Monte Carlo CV, feature selection, Optuna tuning, evaluation, CLI, tuned parameters. Zero-variance filter fixed to `nunique() > 1` (§5.2). |
| `src/ui/timber_results_app.py` | Rewritten: PileB charts from the PileA-trained models, metric tiles, step chart |
| `src/models/timber_engine.py` | Docstring: D9 marked fixed |
| `src/models/train_timber_linear.py`, `train_timber_svr.py`, `predict.py` | D9 renames, including target `design_life_years`. These were already broken before this work (they import functions that don't exist) and are still broken. |
| `tests/test_engine_fidelity.py`, `test_timber_pipeline.py`, `test_timber_compare.py`, `test_timber_prediction.py` | D9 renames |
| `docs/timber_kembla_model_spec_and_defects.md` | D9 status: fixed, with the full corrected mapping |
| `docs/timber_model_tuning.md` | Note on the post-D9 time-split numbers |
| `README.md` | Folder tree lists `timber_tuning.py`, `timber_transfer.py`, `timber_results_app.py` |
| `artifacts/timber_transfer_results.json` | **New output:** feature selection, tuning, ablation, seed stability |

**Tests:** 42 passed. `tests/test_timber_prediction.py` was already broken before this work and is still excluded.

## 9. Reproduce

```bash
python -m src.models.timber_transfer --select-features        # §4.1
python -m src.models.timber_transfer --tune gbm --trials 50   # §4.2 (paste output into TRANSFER_GBM_PARAMS)
python -m src.models.timber_transfer --tune mlp --trials 50   # §4.2 (paste output into TRANSFER_MLP_PARAMS)
python -m src.models.timber_transfer --evaluate               # §5 -> artifacts/timber_transfer_results.json
streamlit run src/ui/timber_results_app.py                     # §5.4 charts
```

To regenerate the corrected CSV from the workbook:

```bash
python src/data/build_timber_kembla_csv.py data/raw/Timber_kembla_v6.3.xlsx
```

## 10. References

1. Akiba, T., et al. (2019). *Optuna: A Next-generation Hyperparameter Optimization Framework.* KDD 2019. https://arxiv.org/abs/1907.10902
2. Bergstra, J., Bardenet, R., Bengio, Y., & Kégl, B. (2011). *Algorithms for Hyper-Parameter Optimization.* NeurIPS 24. http://papers.neurips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf
3. Ke, G., et al. (2017). *LightGBM: A Highly Efficient Gradient Boosting Decision Tree.* NeurIPS 30. https://proceedings.neurips.cc/paper_files/paper/2017/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf
4. Shi, Y., Li, J., & Li, Z. (2019). *Gradient Boosting with Piece-Wise Linear Regression Trees.* IJCAI 2019. https://www.ijcai.org/proceedings/2019/476
5. Karpatne, A., et al. (2017). *Theory-Guided Data Science: A New Paradigm for Scientific Discovery from Data.* IEEE TKDE, 29(10). https://arxiv.org/abs/1612.08544 (the basis for building physics knowledge, here hazard scaling, into the model)
6. Wang, C.-H., Leicester, R. H., & Nguyen, M. N. (2008). *Probabilistic procedure for design of untreated timber poles in-ground under attack of decay fungi.* Reliability Engineering & System Safety, 93, 476–481. https://www.sciencedirect.com/science/article/abs/pii/S0951832007000154 (decay rate as a product of hazard factors)
