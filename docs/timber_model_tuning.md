# Timber Surrogate Models: Improvement and Optuna Tuning

**Code:** `src/models/timber_model.py` (MLP + shared rate-target wrapper), `src/models/timber_model_gbm.py` (GBM), `src/models/timber_tuning.py` (Optuna search + evaluation), `src/features/timber_features.py::add_thermal_dose`
**Data:** `data/processed/timber_kembla_v6_3_consolidated.csv` (994 rows = 2 piles x 7 scenario/pathway series x 71 years)
**Raw results:** `artifacts/timber_tuning_results.json`

**Headline (time-based test split, the app's actual use case):**

| Model | Test R² | Test RMSE (mm) | Test MAE (mm) |
|---|---:|---:|---:|
| MLP, before | 0.625 | 108.08 | 77.69 |
| **MLP, after** | **0.968** | **31.62** | **24.38** |
| GBM, before | 0.644 | 105.31 | 82.09 |
| **GBM, after** | **1.000** | **1.50** | **1.04** |

Both models beat the 0.80 R² target. Both use the same features and the same target. Read [§6 Caveats](#6-caveats-read-before-quoting-these-numbers) before quoting these numbers anywhere.

> **Update (2026-10-04): Defect D9 fix.** `FEATURE_COLUMNS` now names all 11 k factors correctly, going from 13 to 15 columns (see `docs/timber_pileA_to_pileB_report.md` §1). This changes the MLP on this split:
>
> | Time-split test R² | This report | After D9 fix |
> |---|---:|---:|
> | MLP baseline | 0.625 | 0.877 |
> | MLP tuned | 0.968 | 0.953 |
> | GBM baseline | 0.644 | 0.644 |
> | GBM tuned | 1.000 | 1.000 |
>
> The tables below are the pre-fix numbers.

---

## 1. Starting point

The leak-free models from `docs/timber_kembla_to_sydney_eval.md` §4.1 scored:

- MLP: test R² 0.625
- GBM: test R² 0.644

Both used the 13 `FEATURE_COLUMNS` and target `remaining_diameter_mm`, evaluated on `time_based_split`. That split holds out the last 20% of years (2057–2070) of every pile/scenario series.

## 2. Diagnosis: why both models were stuck around 0.63

In `src/models/timber_engine.py`, the deterministic engine computes:

```
water_factor  = 0.1 * exp(0.13 * summer_sea_temp_c)
r_heart       = water_factor * k_HW * k_sal * k_shelter * k_protect * k_contact * k_knot
remaining_mm  = D0 - 2 * (sapwood_layer + (age - lag) * r_heart)
```

So remaining diameter falls roughly **linearly with age**. The rate depends on the pile's constant factors and the (time-varying) sea temperature.

The time-based split trains on ages −6…51 and tests on ages 51…65. The test is therefore an **extrapolation** in age, and it failed for two reasons:

- **GBM:** tree ensembles predict a flat constant beyond the largest value seen in training. They cannot continue a downward trend.
- **MLP:** it had to discover the multiplicative `age × temperature` interaction from about 800 rows, and it extrapolated that poorly.

## 3. What was changed (one step at a time)

Each step was added on top of the previous one. Both models always receive identical inputs and identical targets.

### Step 1: `thermal_dose` feature

This step added `thermal_dose = max(age_years, 0) × water_factor` (`add_thermal_dose` in `src/features/timber_features.py`).

- It is "years in service × how hot those years were", a degree-days-style exposure term.
- It is built only from two columns already in `FEATURE_COLUMNS`, so it adds no new information and no leakage. It hands the model an interaction it would otherwise have to learn.

### Step 2: rate target (target reparameterisation)

Instead of learning the diameter directly, both models learn the **average deterioration rate**:

```
rate = (initial_diameter_mm - remaining_diameter_mm) / age_years     (0 before installation)
```

The rate is converted back to a diameter with `D0 − rate × age`.

- The rate is close to constant over time, so extrapolating to later years no longer needs a trend the model can't represent.
- The final target is still `remaining_diameter_mm`: all metrics are computed on diameter.
- Both conversions use only `initial_diameter_mm` and `age_years`, which are already model inputs.

This is implemented as a scikit-learn-compatible wrapper, `DeteriorationRateRegressor` (`src/models/timber_model.py`). It adds `thermal_dose` itself and returns diameters from `predict()`. Callers therefore still pass the plain 13-column `FEATURE_COLUMNS` matrix, and `timber_scenarios.py`, `timber_inference.py`, the Streamlit app and `timber_compare.py` work unchanged.

For the MLP, the inner estimator also standardises the target (`TransformedTargetRegressor`) as well as the features.

### Step 3: Optuna hyperparameter tuning

- **Sampler:** TPE (Bergstra et al., 2011), via Optuna (Akiba et al., 2019), `seed=42`, with a `MedianPruner`.
- **Trials:** 100 for GBM, 60 for MLP.
- **Objective:** mean validation RMSE (mm, on diameter).
- **Validation:** the test years are never used during tuning. `time_based_split` first removes the test years. Tuning then uses **blocked, expanding-window time-series CV inside the training years only** (`blocked_time_folds`: 3 folds, each validating on the next 9-year block after its training years). This mirrors the forecast-forward test and follows the forward-chaining guidance in Cerqueira et al. (2020). Shuffled K-fold would let the model interpolate between neighbouring years, which would overstate forecasting skill.

| Model | Search space | Best CV RMSE |
|---|---|---:|
| MLP | 1–3 hidden layers, 16–256 units, relu/tanh, L2 `alpha` 1e-6…1e-1, learning rate 1e-4…1e-2, batch 16–128 | 18.87 mm |
| GBM (LightGBM) | `n_estimators` 100–2000, `learning_rate` 0.005–0.3, `max_depth` 2–10, `num_leaves` 4–64, `min_child_samples`, row/column subsampling, L1/L2 regularisation, **`linear_tree` on/off** | 1.36 mm |

`linear_tree=True` puts a linear model in every leaf instead of a constant (piece-wise linear GBDT, Shi et al., 2019). Optuna chose it, and it is the main reason the GBM can now extrapolate.

**Chosen hyperparameters** (stored as `TUNED_MLP_PARAMS` / `TUNED_GBM_PARAMS`):

```python
# MLP
{"hidden_layer_sizes": (136, 136, 136), "activation": "relu", "alpha": 0.000976,
 "learning_rate_init": 0.006978, "batch_size": 128}
# GBM (LightGBM)
{"n_estimators": 1414, "learning_rate": 0.0694, "max_depth": 6, "num_leaves": 19,
 "min_child_samples": 19, "subsample": 0.931, "subsample_freq": 1, "colsample_bytree": 0.952,
 "reg_alpha": 0.0030, "reg_lambda": 0.0505, "linear_tree": True}
```

## 4. Results

All metrics are on `remaining_diameter_mm` in millimetres, `random_state=42`.

### 4.1 Time-based split (train 798 rows, test 196 rows; the primary result)

| Step | Model | Test R² | Test RMSE | Test MAE | Train R² |
|---|---|---:|---:|---:|---:|
| 1. Baseline | MLP | 0.625 | 108.08 | 77.69 | 0.997 |
| 1. Baseline | GBM | 0.644 | 105.31 | 82.09 | 1.000 |
| 2. + thermal_dose | MLP | 0.859 | 66.34 | 43.29 | 0.998 |
| 2. + thermal_dose | GBM | 0.668 | 101.76 | 72.08 | 1.000 |
| 3. + rate target (default params) | MLP | 0.452 | 130.71 | 113.68 | 0.998 |
| 3. + rate target (default params) | GBM | 0.978 | 26.25 | 15.79 | 1.000 |
| **4. + Optuna tuning (final)** | **MLP** | **0.968** | **31.62** | **24.38** | 0.991 |
| **4. + Optuna tuning (final)** | **GBM** | **1.000** | **1.50** | **1.04** | 1.000 |

Reading the steps:

- **GBM:** the rate target does most of the work, taking it from 0.668 to 0.978. `linear_tree` then closes the remaining gap.
- **MLP:** with default hyperparameters the rate target alone *hurts* it (0.452). It only pays off once tuned. This is why step 3 and step 4 must be read together.

**Seed stability** (final models, 5 seeds: 0, 1, 2, 3, 42):

| Model | R² mean ± std | RMSE mean ± std | MAE mean ± std |
|---|---:|---:|---:|
| MLP | 0.920 ± 0.056 | 47.39 ± 15.37 | 35.83 ± 11.15 |
| GBM | 1.000 ± 0.000 | 1.55 ± 0.15 | 1.06 ± 0.10 |

The MLP's 0.968 at seed 42 is on the good side of its range. **0.92 is the fairer MLP figure to quote.** GBM is essentially seed-independent.

**Replacement-year check** (PileA, `summarize_pile`, compared with the engine's own first year below the replacement diameter):

| Scenario | Engine (truth) | GBM | MLP |
|---|---:|---:|---:|
| Hotter & Drier | 2039–2040 | 2039 | 2035 |
| Most Likely | 2041 | 2041 | 2035 |
| Colder & Wetter | 2039 | 2039 | 2035 |

GBM matches the engine. The MLP is 4–6 years early, even with an R² of 0.97, so R² alone overstates how usable the MLP is for the app's replacement-year output.

### 4.2 Leave-one-pile-out split (train PileA, test PileB; a diagnostic)

| Step | Model | Test R² | Test RMSE | Test MAE |
|---|---|---:|---:|---:|
| 1. Baseline | MLP | 0.545 | 179.63 | 141.12 |
| 1. Baseline | GBM | 0.547 | 179.33 | 141.93 |
| 4. Final | MLP | 0.165 | 243.37 | 192.19 |
| 4. Final | GBM | 0.132 | 248.09 | 195.60 |

**The improvements do not transfer to an unseen pile, and they make this split worse.** With only 2 piles, every pile-level factor (`timber_exposure_factor`, `salinity_factor`, `construction_factor`, `model_knot_factor`, `initial_diameter_mm`) takes a single value in training. The model cannot learn how the rate depends on them, and the rate target makes it rely on exactly those factors. This is a data limit, not a tuning limit (see §7).

## 5. What changed in the code

| File | Change |
|---|---|
| `src/features/timber_features.py` | Added `THERMAL_DOSE_COLUMN` and `add_thermal_dose()` |
| `src/models/timber_model.py` | Added `DeteriorationRateRegressor`, `suggest_mlp_params`, `build_estimator`, `build_tuned_model`, `TUNED_MLP_PARAMS`. `build_pipeline(..., tuned=False)` keeps the original estimators. `train(..., tuned=True)` now uses the tuned rate-target model by default, and falls back to the original for other targets or feature sets (e.g. `timber_compare --target distance`, `decay_dose`). `main()` now also prints RMSE. |
| `src/models/timber_model_gbm.py` | Added `suggest_gbm_params` (including `linear_tree`) and `TUNED_GBM_PARAMS` |
| `src/models/timber_tuning.py` | New: `blocked_time_folds`, `tune`, `evaluate`, CLI |
| `requirements.txt` | Added `optuna>=4.0,<6.0` |

`build_pipeline()` still defaults to the *original* estimators because `timber_residual.py` and `scripts/train_pileA_predict_pileB.py` call it directly, and the residual experiment trains on a non-diameter target. `train()`, which the Streamlit app, `timber_inference.py`, `timber_compare.py` and `scripts/retrain_and_eval_sydney.py` use, now gets the tuned models. **This means dashboard output and cached artifacts change the next time they are regenerated.** Existing tests pass: 42 passed. The one exception is `tests/test_timber_prediction.py`, which was already broken before this work; see the reorganisation notes.

## 6. Caveats: read before quoting these numbers

1. **This measures surrogate fidelity, not real-world accuracy.** The "ground truth" is the output of a deterministic AS 5604-style spreadsheet engine (see `src/data/timber_data_prep.py`), not field measurements. An R² of 1.000 means the GBM has learned to reproduce that engine's formula, including its known defects (`docs/timber_kembla_model_spec_and_defects.md`). It says nothing about how well the engine, or the model, predicts real pile decay.
2. **GBM ≈ 1.000 is expected here, not suspicious.** The target is an exact function of the inputs. Once the model form (rate target + linear leaves) matches the formula's shape, near-perfect agreement follows. No input is an engine output: `sapwood_rate_mm_year`, `heartwood_rate_mm_year` and the rest of `LEAKY_COLUMNS` stay excluded (see Kaufman et al., 2012 on leakage). `water_factor` is an engine intermediate, but it is a deterministic function of `summer_sea_temp_c` alone and was already in the baseline feature set.
3. **Only 2 piles.** Results describe forecasting PileA and PileB forward in time. They do not show generalisation to other piles (§4.2).
4. **MLP variance.** Quote 0.92 ± 0.06 rather than the single-seed 0.968, and prefer GBM for replacement-year outputs.
5. **The test set is small:** 196 rows, from 14 series, over 14 years each.

## 7. Recommended next steps

1. **Use the tuned GBM as the timber model of record.** It has the best accuracy, is seed-stable, and gets replacement years right.
2. **Get more piles.** This is the only fix for §4.2. With more assets, the pile-level factors vary in training and the rate target can generalise across piles.
3. **Regenerate** `data/dashboard/dashboard_data_timber*.csv`, `artifacts/timber_condition_model*.joblib`, and the older narrative docs (`timber_model_results.md`, `timber_model_comparison.md`, `timber_project_report.md`) against the tuned models.
4. **Consider enforcing physical constraints in the model:** non-negative rate, and non-increasing diameter over time (`timber_scenarios.py` currently applies a running minimum after prediction). This is the "physics-guided loss / architecture" family in Willard et al. (2022).

## 8. Reproduce

```bash
python -m src.models.timber_tuning --model gbm --trials 100   # prints params to paste into TUNED_GBM_PARAMS
python -m src.models.timber_tuning --model mlp --trials 60    # prints params to paste into TUNED_MLP_PARAMS
python -m src.models.timber_tuning --evaluate                 # every table in §4 -> artifacts/timber_tuning_results.json
python -m src.models.timber_model                             # tuned MLP: train, print MAE/RMSE/R², save
```

Tuning took a few minutes on a laptop. Exact figures can drift slightly across scikit-learn/LightGBM versions.

## 9. References

1. Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019). *Optuna: A Next-generation Hyperparameter Optimization Framework.* KDD 2019. https://arxiv.org/abs/1907.10902 (the tuning framework used).
2. Bergstra, J., Bardenet, R., Bengio, Y., & Kégl, B. (2011). *Algorithms for Hyper-Parameter Optimization.* NeurIPS 24. http://papers.neurips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf (the TPE sampler Optuna uses).
3. Ke, G., Meng, Q., Finley, T., et al. (2017). *LightGBM: A Highly Efficient Gradient Boosting Decision Tree.* NeurIPS 30. https://proceedings.neurips.cc/paper_files/paper/2017/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf (the GBM backend).
4. Shi, Y., Li, J., & Li, Z. (2019). *Gradient Boosting with Piece-Wise Linear Regression Trees.* IJCAI 2019. https://www.ijcai.org/proceedings/2019/476 (the basis for `linear_tree`, which lets GBM extrapolate).
5. Cerqueira, V., Torgo, L., & Mozetič, I. (2020). *Evaluating time series forecasting models: an empirical study on performance estimation methods.* Machine Learning, 109, 1997–2028. https://link.springer.com/article/10.1007/s10994-020-05910-7 (why blocked/forward-chaining CV was used for tuning).
6. Karpatne, A., et al. (2017). *Theory-Guided Data Science: A New Paradigm for Scientific Discovery from Data.* IEEE TKDE, 29(10), 2318–2331. https://arxiv.org/abs/1612.08544 (using domain physics to shape features and targets).
7. Willard, J., Jia, X., Xu, S., Steinbach, M., & Kumar, V. (2022). *Integrating Scientific Knowledge with Machine Learning for Engineering and Environmental Systems.* ACM Computing Surveys, 55(4). https://arxiv.org/abs/2003.04919 (a survey of physics-guided ML; the source for the next steps in §7).
8. Kaufman, S., Rosset, S., Perlich, C., & Stitelman, O. (2012). *Leakage in Data Mining: Formulation, Detection, and Avoidance.* ACM TKDD, 6(4). https://dl.acm.org/doi/10.1145/2382577.2382579 (why engine outputs stay out of the feature set).
9. Wang, C.-H., Leicester, R. H., & Nguyen, M. N. (2008). *Probabilistic procedure for design of untreated timber poles in-ground under attack of decay fungi.* Reliability Engineering & System Safety, 93, 476–481. https://www.sciencedirect.com/science/article/abs/pii/S0951832007000154 (the CSIRO decay-rate model family behind the AS 5604-style engine).
