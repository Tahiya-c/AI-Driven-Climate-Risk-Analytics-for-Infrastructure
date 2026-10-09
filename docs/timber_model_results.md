# Timber Pile Surrogate Model — Results

**Model:** `StandardScaler → MLPRegressor(hidden_layer_sizes=(64, 32))` (scikit-learn)
**Target:** `remaining_diameter_mm`
**Trained on:** `data/processed/timber_kembla_v6_3_consolidated.csv` (994 rows, 2 piles)
**Reproduce with:** `python -m src.models.timber_model` (prints the same headline metrics; retrains and overwrites `artifacts/timber_condition_model.joblib`)
**Generated:** run against the timber pipeline (`src/data/`, `src/features/`, `src/models/`) as of this branch; regenerate before quoting these numbers elsewhere, since retraining is stochastic (`random_state=42` is fixed, but re-running after any data or feature change will shift results).

> **Read the [Limitations](#limitations) section before citing any number below.** With 2 piles and a target that goes physically invalid (negative diameter) for over half the held-out rows, these metrics describe fit to two specific assets, not a validated general-purpose model.

---

## 1. Dataset

| | |
|---|---|
| Rows | 994 |
| Piles (`asset_code`) | `PileA` (Turpentine Gum, installed 2005, Ø400 mm), `PileB` (Gum, installed 2006, Ø500 mm) |
| Year range | 2000–2070 (71 years each) |
| Scenarios | `Baseline` (142 rows — 71 × 2 piles), `Hotter & Drier`, `Most Likely`, `Colder & Wetter` (142 rows each — 71 years × 2 emissions pathways × 2 piles) |
| Replacement threshold | 200 mm for both piles (`replacement_diameter_mm`) |
| Features used | 15 numeric columns (see `src/features/timber_features.py::FEATURE_COLUMNS`) — age, sea temp, salinity, and the deterministic model's own engineered hazard/rate factors |
| Target | `remaining_diameter_mm` (deterministic model output, **not** ground-truth field measurement — see caveat below) |

Rows split by `(asset_code, scenario_label)`:

| asset_code | scenario_label | rows |
|---|---|---|
| PileA | Baseline | 71 |
| PileA | Colder & Wetter | 142 |
| PileA | Hotter & Drier | 142 |
| PileA | Most Likely | 142 |
| PileB | Baseline | 71 |
| PileB | Colder & Wetter | 142 |
| PileB | Hotter & Drier | 142 |
| PileB | Most Likely | 142 |

## 2. Train/test split

Default split is **chronological within each `(asset_code, scenario_label)` group**: the earliest 80% of years train, the latest 20% test (`time_based_split` in `src/models/timber_model.py`). This measures whether the model can extrapolate a *known* pile's trajectory into later years — the actual use case for the app — rather than measuring generalization to an unseen pile.

| Split | n_train | n_test |
|---|---|---|
| Time-based (default) | 798 | 196 |

An alternate leave-one-pile-out split (`split_strategy="pile"`) is available as a harder diagnostic; see [§6](#6-alternate-split-leave-one-pile-out).

## 3. Headline accuracy

| Split | MAE (mm) | RMSE (mm) | R² |
|---|---:|---:|---:|
| Train | 4.37 | 6.68 | 0.998 |
| **Test** | **42.65** | **69.59** | **0.845** |

For scale: initial pile diameters are 400 mm (PileA) and 500 mm (PileB), and the replacement threshold is 200 mm — a 42.65 mm test MAE is roughly 10–20% of the diameter scale, concentrated in the hardest (latest-year, most-extrapolated) part of each trajectory. The near-perfect train fit (R² 0.998) against a much weaker test fit (R² 0.845) is expected for a chronological split evaluated on a physically extreme extrapolation region — see [§3.3](#33-error-by-age).

### 3.1 Error by scenario (test set)

| scenario_label | mean abs err (mm) | median abs err (mm) | max abs err (mm) | n |
|---|---:|---:|---:|---:|
| Baseline | 19.24 | 14.13 | 52.81 | 28 |
| Most Likely | 27.87 | 6.62 | 143.31 | 56 |
| Hotter & Drier | 49.14 | 17.59 | 219.44 | 56 |
| Colder & Wetter | 62.65 | 28.78 | 242.64 | 56 |

The large gap between mean and median in every row shows the error is dominated by a minority of far-future rows, not a uniform bias.

### 3.2 Error by pile (test set)

| asset_code | mean abs err (mm) | median abs err (mm) | R² | n |
|---|---:|---:|---:|---:|
| PileA | 7.39 | 4.59 | 0.941 | 98 |
| PileB | 77.92 | 59.65 | 0.107 | 98 |

This is the single most important number in this report: **the model fits PileA's held-out years well and PileB's poorly.** PileB deteriorates much faster in the raw data (its predicted `remaining_diameter_mm` swings from +500 mm to below −1600 mm across the full 2000–2070 window for some pathways), so its test rows sit deeper in the region where the underlying deterministic model itself is extrapolating past physical failure. With only 2 piles, this is a single data point about "hard vs. easy" pile behaviour, not a pattern — see [Limitations](#limitations).

### 3.3 Error by age

| age bucket (years since install) | mean abs err (mm) | n |
|---|---:|---:|
| 51–60 | 30.82 | 133 |
| 61–70 | 67.62 | 63 |

All 196 test rows fall in the tail of each trajectory (this is what a chronological 80/20 split does by construction), and error rises with age/extrapolation distance, as expected for a regressor asked to extrapolate beyond its training range.

**108 of 196 test rows (55%) have a raw ground-truth `remaining_diameter_mm` below zero** — i.e. the deterministic model they were computed from has already extrapolated past physical failure for those rows. The MAE above is therefore partly measuring how well the surrogate matches an already-invalid extrapolation, not how well it predicts real remaining timber.

## 4. Feature importance

Permutation importance (mean drop in negative-MAE score over 20 shuffles per feature, computed on the test split):

| Feature | Importance | Std dev |
|---|---:|---:|
| `heartwood_rate_mm_year` | 28.76 | 1.98 |
| `timber_exposure_factor` | 19.00 | 1.38 |
| `salinity_factor` | 18.85 | 1.37 |
| `model_knot_factor` | 18.50 | 1.38 |
| `sapwood_rate_mm_year` | 16.20 | 0.68 |
| `initial_diameter_mm` | 16.01 | 1.20 |
| `construction_factor` | 14.84 | 1.14 |
| `summer_sea_temp_c` | 11.32 | 0.78 |
| `salinity` | 10.82 | 0.95 |
| `age_years` | 9.55 | 0.88 |
| `water_factor` | 6.85 | 0.45 |
| `model_contact_factor` | 0.00 | 0.00 |
| `environment_factor` | 0.00 | 0.00 |
| `protection_factor` | 0.00 | 0.00 |
| `shelter_factor` | 0.00 | 0.00 |

The deterioration-rate columns (`heartwood_rate_mm_year`, `sapwood_rate_mm_year`) and the species/exposure/salinity/knot factor columns dominate, which lines up with the deterministic model's own physics. The four zero-importance columns (`model_contact_factor`, `environment_factor`, `protection_factor`, `shelter_factor`) are **constant within this 2-pile dataset** (each pile has one fixed value for these), so the MLP had no variation to learn from — this is a data-coverage artefact, not evidence those factors are physically irrelevant.

## 5. Predicted replacement year, per pile per scenario

Produced by `src/models/timber_scenarios.py`: Baseline history through the current year, then each scenario's own trajectory, model-predicted, forced non-increasing (deterioration cannot reverse), floored at 0 mm.

### PileA (Turpentine Gum, installed 2005, Ø400 mm, threshold 200 mm)

| Scenario | Predicted replacement year | Remaining diameter at 2070 (mm) |
|---|---:|---:|
| Colder & Wetter | 2038.7 | 5.3 |
| Hotter & Drier | 2039.1 | 2.4 |
| Most Likely | 2040.9 | 23.3 |

### PileB (Gum, installed 2006, Ø500 mm, threshold 200 mm)

| Scenario | Predicted replacement year | Remaining diameter at 2070 (mm) |
|---|---:|---:|
| Colder & Wetter | 2030.8 | 0.0 |
| Hotter & Drier | 2030.9 | 0.0 |
| Most Likely | 2031.3 | 0.0 |

Two patterns worth flagging rather than hiding:

- **PileB's three scenarios converge to within ~0.5 years of each other**, while PileA's spread across ~2.1 years. Given PileB's much larger test-set error ([§3.2](#32-error-by-pile-test-set)), this convergence should be treated as a property of the model's fit to PileB specifically, not evidence that climate scenario has less effect on faster-deteriorating piles in general.
- **Both piles reach 0 mm well before 2070** under every scenario, and the deterministic model's own `remaining_diameter_mm` output goes negative for both piles from around the 2030s–2040s onward. All replacement-year predictions here are extrapolations into a regime the deterministic model itself was not designed to represent accurately (see [Limitations](#limitations)).

## 6. Alternate split: leave-one-pile-out

`train(df, split_strategy="pile")` trains on one pile's 497 rows and tests on the other's 497 rows entirely unseen:

| Split | MAE (mm) | R² | n_test |
|---|---:|---:|---:|
| Leave-one-pile-out | 133.44 | 0.662 | 497 |

Substantially worse than the time-based split (42.65 mm / 0.845), as expected: PileA and PileB differ in species, exposure, and deterioration rate, so training on one and testing on the other is a genuine domain-shift test, not a variance estimate for "the next pile at this port."

## 7. Validation against the deterministic model (not used as a feature)

`failure_probability` and `replacement_time_years` are outputs of the original AS 5604-style deterministic model and were **excluded from training** to avoid leakage; they're used here only as an external sanity check. At year 2070, every row in the dataset reports `failure_probability = 1.0` for both piles across all scenarios — i.e. the deterministic model itself considers both piles certain to have failed well before the end of the projection window, consistent with the surrogate's replacement-year predictions all landing in 2031–2041.

The deterministic model's own `replacement_time_years` (design-margin based, measured from `year_installed`) ranges 26.9–39.9 years for PileA and 12.1–18.0 years for PileB across scenarios — i.e. design-year 2032–2045 for PileA and 2018–2024 for PileB. These are **not directly comparable** to the surrogate's replacement-year predictions above, because `replacement_time_years` is computed against a `design_deterioration_mm` margin (a stricter, design-code threshold), not against the raw 200 mm `replacement_diameter_mm` crossing the surrogate targets. Both point in the same direction (PileB fails much sooner than PileA), which is the useful cross-check here, not exact agreement.

## Limitations

- **Two piles.** Every metric above describes fit to `PileA` and `PileB` specifically. There is no basis in this dataset for claiming the model generalizes to a different species, exposure class, or port — see [§6](#6-alternate-split-leave-one-pile-out) for how much accuracy drops under a genuine unseen-pile test.
- **The target itself is a deterministic model's output, not measured field data.** Both the training labels and the "ground truth" used to compute test MAE/R² are `remaining_diameter_mm` values produced by the original AS 5604-style spreadsheet model, which is known to over-extrapolate past physical failure (negative diameters) for far-future rows — 55% of this report's test rows fall in that regime. The surrogate model is therefore validated against another model's extrapolation, not against reality.
- **MIROC-Medres → "Colder & Wetter" is an inferred label**, not confirmed against the original workbook (see `README.md` and `src/data/timber_data_prep.py` for the reasoning). If that inference is wrong, the "Colder & Wetter" numbers throughout this report would actually belong to a different scenario name.
- **Monotonicity is enforced, not learned.** The MLP's raw predictions are not guaranteed non-increasing over time; `src/models/timber_scenarios.py` applies a running minimum before reporting replacement years. Without that step, replacement-year detection would be ambiguous (a trajectory could cross the threshold, bounce back above it, and cross again).
- **Retraining is not fully deterministic across environments.** `random_state=42` fixes scikit-learn's own RNG, but results can still shift slightly with different scikit-learn versions or BLAS backends. Treat the numbers in this report as indicative, and regenerate them (`python -m src.models.timber_model`, plus the analysis snippet referenced above) before relying on exact figures.
