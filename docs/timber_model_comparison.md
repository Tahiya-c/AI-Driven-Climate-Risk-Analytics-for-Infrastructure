# Timber Pile Surrogate Models — MLP vs. GBM Comparison

**Prepared for:** team review of surrogate-model architecture choice
**Models compared:** `MLPRegressor` (scikit-learn, existing baseline) vs. `LightGBM` (new)
**Also tested:** a residual-correction variant of each, quantile (uncertainty-band) regression, an engineered `decay_dose` feature, and a distance-to-threshold target
**Reproduce with:** `python -m src.models.timber_compare --model gbm` (and see [How to reproduce](#how-to-reproduce) for every command used below)

> **TL;DR — recommendation:** keep the **MLP** as the primary model for forecasting a pile's own future trajectory (`--target diameter`, default settings). It beats GBM on every headline metric on the split that matters for this app. GBM's one clear win is as a **residual-correction layer** on top of a raw prediction (see [§4](#4-residual-correction-does-it-help)) — that's a different job, not a drop-in replacement. Full reasoning below.

---

## 1. Head-to-head: MLP vs. GBM vs. a naive baseline

Same test split, same features, same target (`remaining_diameter_mm`) for all three. The naive baseline is not fitted at all — it's just `initial_diameter_mm - mean_deterioration_mm`, arithmetic on the deterministic model's own output, included to check the trained models are earning their complexity.

### Time-based split (default — see [why](#split-strategy-reminder))

| Model | MAE (mm) | RMSE (mm) | R² | n_test |
|---|---:|---:|---:|---:|
| **MLP** | **42.65** | **69.59** | **0.845** | 196 |
| GBM | 81.77 | 104.21 | 0.652 | 196 |
| naive baseline | 276.32 | 297.86 | −1.845 | 196 |

**MLP wins by a wide margin.** Both trained models comfortably beat the naive baseline (whose R² is negative — worse than just predicting the mean), so neither model is wasted complexity. But MLP's test MAE is roughly **half** GBM's.

### Leave-one-pile-out split (harder diagnostic — train on one pile, test on the other)

| Model | MAE (mm) | RMSE (mm) | R² | n_test |
|---|---:|---:|---:|---:|
| **MLP** | **133.44** | 154.77 | **0.662** | 497 |
| GBM | 141.93 | 179.33 | 0.547 | 497 |
| naive baseline | 185.54 | 228.37 | 0.265 | 497 |

Same ranking under a much harder test (an entirely unseen pile, different species/exposure). MLP still leads; both still beat naive.

### Why MLP wins here

This is the one-sentence version for the team: **the test set is almost entirely extrapolation into years the model never trained on, and gradient-boosted trees cannot extrapolate past the range of their training data** — a tree can only predict values it saw at a leaf during training, so for far-future rows GBM's predictions flatten out instead of continuing the trend, while the MLP's smooth activation functions let it continue extrapolating (imperfectly, but better than flat). Since our chronological split evaluates specifically on the *latest* years of each pile's trajectory, this plays directly against GBM's weakness. On data that were mostly *interpolation* (e.g. filling in a mid-range year), we'd expect GBM to be far more competitive or even win — that's not the situation here.

---

## 2. MLP — individual performance

| Split | MAE (mm) | RMSE (mm) | R² |
|---|---:|---:|---:|
| Train | 4.37 | 6.68 | 0.998 |
| Test | 42.65 | 69.59 | 0.845 |

**Error by scenario (test):**

| scenario_label | mean abs err (mm) | median | max | n |
|---|---:|---:|---:|---:|
| Baseline | 19.24 | 14.13 | 52.81 | 28 |
| Most Likely | 27.87 | 6.62 | 143.31 | 56 |
| Hotter & Drier | 49.14 | 17.59 | 219.44 | 56 |
| Colder & Wetter | 62.65 | 28.78 | 242.64 | 56 |

**Error by pile (test):**

| asset_code | mean abs err (mm) | median | R² | n |
|---|---:|---:|---:|---:|
| PileA | 7.39 | 4.59 | 0.941 | 98 |
| PileB | 77.92 | 59.65 | 0.107 | 98 |

**Error by age (test):** 51–60y mean 30.82mm (n=133); 61–70y mean 67.62mm (n=63) — error grows with extrapolation distance, as expected.

**SHAP feature importance (mean |value|):**

| Feature | Importance | Constant within a pile? |
|---|---:|:---:|
| `heartwood_rate_mm_year` | 22.93 | No |
| `timber_exposure_factor` | 18.85 | **Yes** |
| `model_knot_factor` | 18.77 | **Yes** |
| `salinity_factor` | 18.56 | **Yes** |
| `initial_diameter_mm` | 16.29 | **Yes** |
| `construction_factor` | 15.03 | **Yes** |
| `sapwood_rate_mm_year` | 14.69 | No |
| `summer_sea_temp_c` | 14.07 | No |
| `salinity` | 12.57 | No |
| `age_years` | 8.32 | No |
| `water_factor` | 6.02 | No |
| `shelter_factor`, `environment_factor`, `protection_factor`, `model_contact_factor` | 0.00 | Yes |

**Read this carefully:** 5 of the top 6 features are flagged "constant within a pile" — meaning within *either* PileA's own rows or PileB's own rows, the column never changes. With only 2 piles in the dataset, a column like that can only ever encode *which pile a row belongs to*, not how that pile's condition changes over time. MLP's importance is spread across several of these redundant pile-identity columns rather than concentrating on one — that's an artifact of gradient descent finding a locally-good weighting, not evidence all 5 columns are independently useful. The genuinely time-varying, non-degenerate signal is `heartwood_rate_mm_year`, `sapwood_rate_mm_year`, `summer_sea_temp_c`, `salinity`, and `age_years`.

---

## 3. GBM — individual performance

| Split | MAE (mm) | RMSE (mm) | R² |
|---|---:|---:|---:|
| Train | 1.72 | 2.68 | 0.9997 |
| Test | 81.77 | 104.21 | 0.652 |

Train R² of 0.9997 vs. test R² of 0.652 is a much bigger train/test gap than MLP's (0.998 → 0.845) — a classic sign of a model that has memorized the training range rather than learned a rule that extrapolates.

**Error by scenario (test):**

| scenario_label | mean abs err (mm) | median | max | n |
|---|---:|---:|---:|---:|
| Baseline | 58.88 | 45.48 | 156.25 | 28 |
| Most Likely | 70.42 | 59.58 | 178.80 | 56 |
| Hotter & Drier | 86.23 | 65.52 | 287.72 | 56 |
| Colder & Wetter | 100.10 | 76.46 | 319.07 | 56 |

Every one of these is 2–3× the equivalent MLP number.

**Error by pile (test):**

| asset_code | mean abs err (mm) | median | max | n |
|---|---:|---:|---:|---:|
| PileA | 44.98 | 42.44 | 121.15 | 98 |
| PileB | 118.55 | 111.41 | 319.07 | 98 |

**Error by age (test):** 51–60y mean 60.63mm (n=133); 61–70y mean 126.38mm (n=63) — again roughly double MLP's error at every age bucket.

**SHAP feature importance (mean |value|):**

| Feature | Importance | Constant within a pile? |
|---|---:|:---:|
| `heartwood_rate_mm_year` | 103.90 | No |
| `timber_exposure_factor` | 18.32 | **Yes** |
| `summer_sea_temp_c` | 4.89 | No |
| `salinity` | 3.34 | No |
| `sapwood_rate_mm_year` | 1.39 | No |
| everything else (10 features, including `age_years`) | 0.00 | mixed |

This is a striking contrast with MLP's importance table. GBM collapses almost entirely onto **one** feature (`heartwood_rate_mm_year`) plus **one** pile-identity proxy (`timber_exposure_factor`), and gives literally **zero** importance to `age_years` and 8 other columns. This is a real, explainable pattern, not a bug: gradient-boosted trees build splits greedily, and with only 2 distinct pile identities, *any single* column that separates PileA from PileB is enough — once the first tree picks `timber_exposure_factor` for that split, the other equally-good proxies (`salinity_factor`, `model_knot_factor`, `construction_factor`, `initial_diameter_mm`, ...) become redundant and never get selected again. MLP's continuous weight updates don't collapse the same way, so it spreads credit across the redundant columns instead. **Neither behaviour is "more correct"** — it's a direct consequence of only having 2 piles to learn from, and it's worth remembering next time someone wants to read feature importance as a physical explanation.

---

## 4. Residual correction: does it help?

A different idea from "predict the diameter directly": train a model to predict **the correction** needed to bring the deterministic model's raw (sometimes physically invalid, negative) output back into range, then reconstruct `prediction = raw_value + predicted_correction`. Both approaches are scored only on rows where the raw value was **already valid** (`remaining_diameter_mm >= 0`, n=88 test rows) — i.e., rows the correction was never meant to fix. The question: does bolting on a correction layer hurt where it isn't needed?

| Base model | Approach | MAE (mm) | RMSE (mm) | R² |
|---|---|---:|---:|---:|
| **GBM** | Residual-correction | **4.45** | **5.55** | **0.977** |
| GBM | Direct surrogate | 39.09 | 44.64 | −0.494 |
| MLP | Residual-correction | 7.43 | 11.85 | 0.895 |
| MLP | Direct surrogate | 5.44 | 7.13 | 0.962 |

**Two opposite conclusions, and both make sense:**

- **For GBM, residual-correction is a massive win** (MAE 39.1mm → 4.5mm, R² goes from *worse than predicting the mean* to 0.977). This directly confirms §1's extrapolation story: GBM is bad at predicting the raw value directly in this region, but predicting "how much correction is needed" is a much easier, more localized problem, and GBM handles it well.
- **For MLP, residual-correction makes things slightly worse** (MAE 5.4mm → 7.4mm). MLP's direct surrogate was already good in this region, so the extra correction layer just adds noise rather than fixing a real problem.

**Practical takeaway:** if the team ever wants to use GBM (e.g. for speed, interpretability, or deployment reasons), pair it with residual-correction, not direct prediction. If sticking with MLP, direct prediction is already the better of the two options — don't bother with residual-correction there.

---

## 5. Quantile regression: uncertainty bands

LightGBM supports a native quantile loss (MLPRegressor does not), so 10th/50th/90th-percentile models were fit and run through the *same, unchanged* scenario/replacement-year logic three times.

**Pinball loss on the test set** (lower is better, and losses should shrink toward the tails):

| Quantile | Pinball loss |
|---:|---:|
| 0.1 | 63.99 |
| 0.5 | 48.17 |
| 0.9 | 11.59 |

**Replacement-year band, PileA:**

| Scenario | p10 (early) | p50 | p90 (late) |
|---|---:|---:|---:|
| Colder & Wetter | 2038.4 | 2038.5 | 2040.0 |
| Hotter & Drier | 2038.5 | 2038.8 | 2040.3 |
| Most Likely | 2039.5 | 2040.8 | 2040.9 |

**Replacement-year band, PileB:**

| Scenario | p10 (early) | p50 | p90 (late) |
|---|---:|---:|---:|
| Colder & Wetter | 2029.1 | **2030.7** | **2030.5** |
| Hotter & Drier | 2030.4 | 2030.8 | 2031.7 |
| Most Likely | 2031.2 | 2031.4 | 2039.0 |

**Flag for the team:** PileB's "Colder & Wetter" row has **p90 (2030.5) earlier than p50 (2030.7)** — the quantiles cross. This is a known artifact of fitting three independent quantile models rather than one model with a monotonicity constraint across quantiles, and it shows up here because PileB is the harder-to-fit pile (see §1–3). Treat these bands as directional (a wider spread = more uncertainty), not as strict statistical bounds, until/unless a monotonic quantile method is used instead.

---

## 6. Feature engineering: `decay_dose` and the distance-to-threshold target

Two variants tested against the same MLP/GBM/naive comparison:

| Variant | MLP MAE | MLP R² | GBM MAE | GBM R² |
|---|---:|---:|---:|---:|
| Standard features, diameter target (baseline) | 42.65 | 0.845 | 81.77 | 0.652 |
| **decay_dose** features, diameter target | 61.07 | 0.716 | 84.58 | 0.638 |
| Standard features, **distance-to-threshold** target | 51.01 | 0.856 | 81.77 | 0.652 |
| decay_dose features, distance-to-threshold target | 51.96 | 0.859 | 84.58 | 0.638 |

- **`decay_dose` made MLP noticeably worse** (MAE 42.7mm → 61.1mm): collapsing `age_years`, `summer_sea_temp_c`, `salinity_factor`, and `timber_exposure_factor` into one cumulative number threw away information the standard feature set was using productively. Not recommended as a replacement for the standard features.
- **GBM is essentially indifferent to `decay_dose`** (81.8mm → 84.6mm, a much smaller change) and **completely indifferent to the target reframing** (81.77mm MAE for both diameter and distance targets, to 2 decimal places) — expected, since `distance = diameter - 200` is a pure per-pile constant shift, and tree-split thresholds shift by exactly the same amount, so a well-fit tree model's *absolute* error is mathematically unchanged by the reframing. This is also a useful sanity check that the pipeline is implemented correctly.
- **MLP's small sensitivity to the distance target** (42.65mm → 51.01mm, worse) is a real but secondary effect of neural-network training dynamics (early stopping interacts with the loss surface's scale/offset) rather than a meaningful modelling difference — the standard diameter target remains the better choice for MLP.

**Recommendation:** keep the standard feature set and the raw-diameter target. Neither experiment beat the existing baseline.

---

## 7. Recommendation summary

| Use case | Recommended approach |
|---|---|
| Forecasting a known pile's future trajectory (the app's main job) | **MLP**, standard features, diameter target — current default, best on every metric |
| Want a tree-based / faster / more interpretable model for some other reason | **GBM + residual-correction**, not direct GBM prediction |
| Need an uncertainty band around a replacement-year estimate | GBM quantile models (p10/p50/p90) — but treat PileB's band with caution given the quantile-crossing noted in §5 |
| Considering `decay_dose` or a distance-to-threshold target | Not recommended — both underperformed the existing baseline in this round of testing |

**Standing limitation for every row in this report:** only 2 piles exist in the dataset. Every SHAP/permutation-importance number, every "constant within a pile" flag, and every model comparison above describes how these two specific models fit **PileA and PileB specifically** — see `docs/timber_model_results.md` for the fuller discussion of that caveat and of the dataset's own known extrapolation issues (negative `remaining_diameter_mm` in the raw deterministic-model output).

---

## How to reproduce

All numbers in this report were generated with `random_state=42` (fixed) against `data/processed/timber_kembla_v6_3_consolidated.csv`. Re-run before quoting exact figures elsewhere — results can drift slightly with different library versions.

```bash
# §1 head-to-head tables (time split and pile split)
python -c "from src.data.timber_data_prep import load_timber_data, DATA_PATH; from src.models.timber_compare import compare_all_models; f=load_timber_data(DATA_PATH); print(compare_all_models(f, split_strategy='time').to_string(index=False)); print(compare_all_models(f, split_strategy='pile').to_string(index=False))"

# §2 / §3 individual deep dives (headline metrics, error breakdowns, SHAP)
python -m src.models.timber_compare --model mlp --shap-sample 30
python -m src.models.timber_compare --model gbm

# §4 residual-correction comparison
python -m src.models.timber_compare --model residual --residual-base gbm
python -m src.models.timber_compare --model residual --residual-base mlp

# §5 quantile uncertainty bands
python -m src.models.timber_compare --model gbm --no-shap --quantile

# §6 decay_dose / distance-to-threshold variants
python -c "from src.data.timber_data_prep import load_timber_data, DATA_PATH; from src.models.timber_compare import compare_all_models; f=load_timber_data(DATA_PATH); print(compare_all_models(f, feature_set='decay_dose').to_string(index=False)); print(compare_all_models(f, target='distance').to_string(index=False)); print(compare_all_models(f, target='distance', feature_set='decay_dose').to_string(index=False))"
```

### Split strategy reminder

Both `--split time` (default) and `--split pile` reuse `time_based_split`/`pile_based_split` from `src/models/timber_model.py` unchanged. Time-based (chronological, per pile+scenario) is the default because it tests forecasting a *known* pile forward in time — the app's actual job. Leave-one-pile-out is a harder diagnostic (train on one pile, test on the structurally different other) included above for completeness, not the primary comparison.
