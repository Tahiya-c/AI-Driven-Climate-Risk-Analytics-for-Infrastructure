# Timber Pile Deterioration Forecasting — Project Report

**Dataset:** `data/processed/timber_kembla_v6_3_consolidated.csv`
**Pipeline:** `src/data/timber_data_prep.py`, `src/features/` (`timber_features.py`, `timber_targets.py`), `src/models/` (`timber_model.py`, `timber_model_gbm.py`, `timber_residual.py`, `timber_quantile.py`, `timber_compare.py`)
**Reproduce:** `python -m src.models.timber_model` (MLP baseline) · `python -m src.models.timber_compare --model gbm` (full comparison)

This report summarises the timber pile replacement-forecasting work: how the dataset was prepared, what the model is trained to predict, which model architectures were built and how they were evaluated, their measured accuracy, and the limitations that should qualify every number below.

---

## 1. Dataset Consolidation

The dataset is the **output of an existing deterministic engineering model**, built in the style of AS 5604, not raw field measurements. It is a spreadsheet-derived export covering **994 rows** across **2 timber piles**, each with 71 years of projections (2000–2070) under 4 climate scenarios.

| | |
|---|---|
| Piles | `PileA` — Turpentine Gum, installed 2005, initial diameter 400 mm · `PileB` — Gum, installed 2006, initial diameter 500 mm |
| Replacement threshold | 200 mm for both piles |
| Year range | 2000–2070 (71 years per pile per scenario) |
| Scenario labels | `Baseline`, `Hotter & Drier`, `Most Likely`, `Colder & Wetter` |
| Rows | 994 total |

Row counts by pile and scenario:

| asset_code | scenario_label | rows |
|---|---|---:|
| PileA / PileB | Baseline | 71 each |
| PileA / PileB | Hotter & Drier | 142 each |
| PileA / PileB | Most Likely | 142 each |
| PileA / PileB | Colder & Wetter | 142 each |

(Non-baseline scenarios carry two `emissions_pathway` rows per pile-year — A1B and A1FI — which is why they're double the Baseline count.)

**Cleaning step required:** the source export left `scenario_label` blank for all 284 rows where `climate_model == "MIROC-Medres"`. These were relabeled `"Colder & Wetter"`, inferred from the project's three-GCM naming convention (CSIRO Mk3.5 = Hotter & Drier, MRI-CGCM2.3.2 = Most Likely) and cross-checked against the sibling concrete-pile dataset's documented GCM provenance. This is an **assumption, not a confirmed fact** — see [§8](#8-limitations-of-the-model). All other numeric columns were coerced to consistent dtypes; no other missing data was found.

---

## 2. Target and Features

**Primary target:** `remaining_diameter_mm` — the deterministic model's own predicted pile diameter at a given pile/year/scenario. This is a regression target, and it is itself model output, not a physical measurement (see [§8](#8-limitations-of-the-model)).

**Alternate target tested:** `distance_to_threshold_mm = remaining_diameter_mm − replacement_diameter_mm`, reframing the problem as "how far from the replacement decision" rather than raw diameter.

**Feature set (15 numeric columns)**, all already engineered by the deterministic model — no target-leaking columns included:

`age_years`, `summer_sea_temp_c`, `salinity`, `water_factor`, `timber_exposure_factor`, `salinity_factor`, `shelter_factor`, `protection_factor`, `model_contact_factor`, `model_knot_factor`, `environment_factor`, `construction_factor`, `sapwood_rate_mm_year`, `heartwood_rate_mm_year`, `initial_diameter_mm`

**Explicitly excluded ("leaky") columns:** `failure_probability`, `replacement_time_years`, `mean_deterioration_mm`, `design_deterioration_mm`, `design_diameter_mm`. These are *outputs* of the deterministic model being replicated — using them as inputs would let the model see the answer. They are retained only for post-hoc validation (§7 of `docs/timber_model_results.md`), never as training features.

**Categorical columns** (`timber_type`, `exposure`, `heartwood_class`, `contact`, `knot_presence`, `maintenance_measure`) are available but excluded by default: with only 2 piles, each is constant per asset, so one-hot encoding them mostly re-encodes "which pile is this" — information the numeric `*_factor` columns already carry through the deterministic model's own physics.

**Engineered alternative — `decay_dose`:** a cumulative sum, ordered by `age_years` within each pile/scenario/pathway trajectory, of `salinity_factor × summer_sea_temp_c × timber_exposure_factor`. Tested as a single non-linear "cumulative environmental stress" feature in place of `age_years`, `summer_sea_temp_c`, `salinity_factor`, and `timber_exposure_factor` individually. Result: it **underperformed** the standard feature set (see [§4](#4-results)) and is not recommended.

---

## 3. Models and Evaluation

Two surrogate architectures were built and compared, plus two derived variants:

| Model | Description |
|---|---|
| **MLP** | `StandardScaler → MLPRegressor` (scikit-learn), 2 hidden layers (64, 32 units), Adam solver, early stopping. Existing baseline. |
| **GBM** | LightGBM gradient-boosted tree ensemble (400 trees, depth 5), falling back to XGBoost if LightGBM is unavailable. Added for this comparison. |
| Residual-correction | Either model, retrained to predict *the correction* needed to bring the deterministic model's raw output into a physically valid (≥ 0 mm) range, rather than the diameter directly. |
| Quantile (GBM only) | Three separate LightGBM models with a native quantile loss, fit at the 10th/50th/90th percentiles, giving an uncertainty band around each prediction. |

**A naive, unfitted baseline** — `initial_diameter_mm − mean_deterioration_mm`, pure arithmetic on the deterministic model's own numbers — is scored alongside both trained models to confirm they are earning their added complexity.

**Evaluation splits:**

- **Time-based (default).** For each pile/scenario series, the earliest 80% of years train and the latest 20% test. This evaluates whether a model can extrapolate a *known* pile's own trajectory forward in time — the task the forecasting app actually performs.
- **Leave-one-pile-out (diagnostic).** Train on one pile's 497 rows, test on the other's 497 rows entirely unseen. A much harder generalization stress test, included because only 2 piles exist and this is the only way to probe cross-asset generalization at all.

**Metrics:** MAE and RMSE (mm, lower is better) and R² (higher is better), computed identically for every model/split combination so results are directly comparable.

---

## 4. Results

### Headline comparison — time-based split (the split that matters for the app)

| Model | MAE (mm) | RMSE (mm) | R² | n test |
|---|---:|---:|---:|---:|
| **MLP** | **42.65** | **69.59** | **0.845** | 196 |
| GBM | 81.77 | 104.21 | 0.652 | 196 |
| Naive baseline | 276.32 | 297.86 | −1.845 | 196 |

### Headline comparison — leave-one-pile-out split (harder diagnostic)

| Model | MAE (mm) | RMSE (mm) | R² | n test |
|---|---:|---:|---:|---:|
| **MLP** | **133.44** | 154.77 | **0.662** | 497 |
| GBM | 141.93 | 179.33 | 0.547 | 497 |
| Naive baseline | 185.54 | 228.37 | 0.265 | 497 |

### Error by pile (time-based split test set) — the largest single effect found

| Model | PileA MAE | PileA R² | PileB MAE | PileB R² |
|---|---:|---:|---:|---:|
| MLP | 7.39 | 0.941 | 77.92 | 0.107 |
| GBM | 44.98 | — | 118.55 | — |

### Residual-correction vs. direct prediction (physically-valid test rows only, n = 88)

| Base model | Approach | MAE (mm) | R² |
|---|---|---:|---:|
| GBM | Residual-correction | **4.45** | **0.977** |
| GBM | Direct | 39.09 | −0.494 |
| MLP | Residual-correction | 7.43 | 0.895 |
| MLP | Direct | **5.44** | **0.962** |

### Feature/target reframing experiments

| Variant | MLP MAE | GBM MAE |
|---|---:|---:|
| Standard features, diameter target *(baseline)* | 42.65 | 81.77 |
| `decay_dose` features, diameter target | 61.07 | 84.58 |
| Standard features, distance-to-threshold target | 51.01 | 81.77 |

---

## 5. Findings

1. **The MLP beats GBM on every metric, on both splits.** The test set is almost entirely extrapolation into years neither model trained on. Gradient-boosted trees cannot extrapolate past the range of values seen during training — a tree's prediction for a far-future row flattens instead of continuing the trend — while the MLP's smooth activation functions keep extrapolating, imperfectly but usefully. On a more interpolation-heavy task, GBM would likely be far more competitive.

2. **Pile-to-pile variation dwarfs the model-architecture choice.** Both models fit PileA well (MLP R² 0.94) and PileB poorly (MLP R² 0.11). PileB deteriorates far faster in the underlying data (predicted diameter swings from +500 mm to below −1600 mm across the projection window for some pathways), pushing its test rows deep into the region where the deterministic model itself is extrapolating past physical failure.

3. **Residual-correction helps or hurts depending on the base model, and the direction makes sense.** It is a large win for GBM (MAE 39.1 → 4.5 mm) because GBM's direct extrapolation was genuinely poor — predicting *how much correction is needed* is an easier, more localized problem. It is a small loss for MLP, whose direct prediction was already good, so the extra layer just adds noise. **Practical rule: pair GBM with residual-correction; use the MLP directly.**

4. **SHAP feature importance exposes a real 2-pile confounding artifact.** Several top-ranked features (`timber_exposure_factor`, `salinity_factor`, `model_knot_factor`, `initial_diameter_mm`, `construction_factor`) are constant *within* each pile's own rows — they only differ *between* PileA and PileB, so they can only ever encode "which pile is this," not real deterioration-over-time signal. The MLP spreads importance across several of these redundant columns; GBM collapses almost entirely onto one (`timber_exposure_factor`) plus the genuinely time-varying `heartwood_rate_mm_year`, and assigns literally zero importance to 9 other columns including `age_years`. Neither pattern is "more correct" — it is a direct, explainable consequence of having only 2 piles to learn from.

5. **Neither feature/target reframing beat the baseline.** `decay_dose` made the MLP notably worse (42.7 → 61.1 mm) by collapsing four informative columns into one. The distance-to-threshold target left GBM's error numerically unchanged (a useful sanity check — since `distance = diameter − 200`, a well-fit tree model's absolute error is mathematically invariant to that shift) and made the MLP modestly worse. Standard features and the raw diameter target remain the recommended configuration.

6. **Quantile uncertainty bands are usable but imperfect.** 10th/50th/90th-percentile GBM models were run through the same replacement-year logic as the point-estimate models. For PileB's harder-to-fit "Colder & Wetter" scenario, the bands cross (p90 replacement year earlier than p50) — a known artifact of fitting independent quantile models rather than a jointly-constrained one. Bands should currently be read as directional, not as strict bounds.

---

## 7. Summary of Accuracy

| | MLP | GBM | Naive baseline |
|---|---:|---:|---:|
| Train MAE (mm) | 4.37 | 1.72 | — |
| **Test MAE (mm), time split** | **42.65** | 81.77 | 276.32 |
| Test R², time split | **0.845** | 0.652 | −1.845 |
| Test MAE (mm), leave-one-pile-out | **133.44** | 141.93 | 185.54 |
| Test R², leave-one-pile-out | **0.662** | 0.547 | 0.265 |

**Headline:** the MLP is the more accurate surrogate on every split and every metric tested, at roughly **half the test-set error of GBM** on the split that matters for the deployed app. Both trained models comfortably outperform the naive arithmetic baseline, confirming the added model complexity is earning its keep. GBM's one clear strength is not as a stand-alone predictor but as a **residual-correction layer**, where it substantially outperforms its own direct-prediction mode.

---

## 8. Limitations of the Model

- **Only 2 piles exist in the dataset.** Every metric in this report describes how well these two specific models fit `PileA` and `PileB`. The leave-one-pile-out split is the only available stress test for generalization to an unseen pile, and accuracy drops substantially under it for both architectures.
- **The training target is itself a model's output, not a field measurement.** `remaining_diameter_mm` comes from the deterministic AS 5604-style spreadsheet model, which is known to over-extrapolate past physical failure — producing negative "remaining diameter" values for far-future rows. 55% of the MLP's time-split test rows fall in that already-invalid region, so part of the reported error reflects disagreement with an extrapolation that was already physically wrong, not with ground truth.
- **The MIROC-Medres → "Colder & Wetter" relabeling is an inferred assumption** (§1), not yet confirmed against the original workbook. If wrong, every "Colder & Wetter" figure in this report belongs to a different scenario name.
- **Monotonicity in the scenario/replacement-year logic is enforced, not learned.** Neither model guarantees non-increasing diameter predictions over time on its own; a running-minimum is applied afterward, since deterioration cannot physically reverse.
- **Quantile bands can cross** for the harder-to-fit pile (§5.6), and should be treated as directional rather than as calibrated statistical bounds until a monotonic quantile method is adopted.
- **Results are reproducible but not bit-identical across environments.** All figures use a fixed `random_state=42`, but different library versions or BLAS backends can shift results slightly. Re-run `python -m src.models.timber_compare` before quoting exact numbers elsewhere.

---

*Full technical detail behind these results: `docs/timber_model_results.md` (MLP baseline, full breakdown) and `docs/timber_model_comparison.md` (MLP vs. GBM, every experiment, reproduction commands).*
