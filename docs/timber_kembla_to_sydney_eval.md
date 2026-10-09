# Kembla → Sydney Evaluation Run — What Was Done and What It Means

**Script:** `scripts/retrain_and_eval_sydney.py` (`python -m scripts.retrain_and_eval_sydney`)
**Inputs:** `data/processed/timber_kembla_v6_3_consolidated.csv` (training), `data/processed/timber_sydney_v6_3_consolidated.csv` (evaluation)
**Outputs:** `artifacts/timber_condition_model_mlp.joblib`, `artifacts/timber_condition_model_gbm.joblib`, `data/dashboard/dashboard_data_timber_sydney.csv`, `scripts/retrain_and_eval_sydney_metrics.json`

> **Read this before the numbers below.** This run produced a working, reproducible pipeline end to end, and the mechanics are sound — but the headline Sydney numbers are **not evidence the model generalizes to a different port**. Two specific, verified reasons why are in [§1](#1-the-critical-caveat-sydneys-data-is-kemblas-data) and [§2](#2-the-second-caveat-the-feature-set-leaks-the-target). Read both before quoting any accuracy figure from this report elsewhere.

---

## 1. The critical caveat: Sydney's data is Kembla's data

Before evaluating anything, I diffed `timber_sydney_v6_3_consolidated.csv` against `timber_kembla_v6_3_consolidated.csv` row-for-row (matched on `asset_code`/`emissions_pathway`/`climate_model`/`year`):

| Column | Max absolute difference |
|---|---:|
| `remaining_diameter_mm` | 1.1×10⁻¹³ mm |
| `summer_sea_temp_c` | 3.6×10⁻¹⁵ °C |
| `heartwood_rate_mm_year` | 1.8×10⁻¹⁵ mm/yr |
| `replacement_time_years` | 1.4×10⁻¹⁴ yr |
| `salinity` | 0.17 ppt (negligible) |

Those are floating-point noise, not real differences. The asset register is identical too — same pile descriptions ("Pile 32 @ Berth 201/202"), same species, same diameters (400mm/500mm), same install years (2005/2006), same baseline sea temperature (21.5°C) and salinity (35.5 ppt). The Sydney source workbook's `Baseline` sheet even reads site name "Botany Bay," but every simulated number is Kembla's number.

**What this means:** `timber_sydney_v6_3_consolidated.csv` is not real, distinct Botany Bay data — it's a copy of the Kembla workbook (or an unfinished template never populated with Sydney-specific values). Evaluating a Kembla-trained model against it is testing the model against a near-duplicate of its own training data, not against an unseen port. Any accuracy number below that looks "good" is at least partly (see §3) explained by this, not by genuine cross-site generalization.

**Before this becomes a real demo:** whoever supplied `Timber_sydney_v6.3.xlsm` needs to confirm it's meant to hold Botany Bay's actual asset register and climate baseline, and re-supply it if this was the wrong export.

## 2. The second caveat: the feature set leaked the target — now fixed

**Status: fixed, as of this update.** `src/features/timber_features.py::FEATURE_COLUMNS` used to include `sapwood_rate_mm_year` and `heartwood_rate_mm_year` — the deterministic engine's own internal rate variables. Verified: together with `age_years` and `initial_diameter_mm` (also in the feature set), they reconstructed `remaining_diameter_mm` **exactly**:

```
max abs reconstruction error: 2.27e-13 mm   (i.e. exact, not approximate)
```

This happened because `layer_of_sapwood_mm` (needed for the full formula) is a constant 20mm for every pile in both datasets, so the model didn't need it as an explicit input — the rest of the formula was fully invertible from what was already given. This was already flagged as a known issue in the Phase-4 rebuild spec (`docs/timber_kembla_model_spec_and_defects.md`) but had not previously been confirmed against the *current* `src/models/timber_model.py` — it was confirmed empirically, then **removed** from `FEATURE_COLUMNS` and added to `LEAKY_COLUMNS`. All numbers in §4 below are from the retrained, leak-free models.

**Before vs. after, Kembla test split (the split that matters for the app):**

| Model | Metric | Leaky (before) | Fixed (after) | Change |
|---|---|---:|---:|---:|
| MLP | Test MAE (mm) | 42.65 | **77.69** | +35.04mm worse |
| MLP | Test R² | 0.845 | **0.625** | −0.220 |
| GBM | Test MAE (mm) | 81.77 | **82.09** | ~unchanged |
| GBM | Test R² | 0.652 | **0.644** | ~unchanged |

**MLP's test error nearly doubled** once it could no longer look up the answer from the rate columns — this is the honest number, and it's a much more consequential drop than a rounding-level change. **GBM barely moved**, which makes sense in hindsight: GBM's own SHAP importance (`docs/timber_model_comparison.md` §3) was already collapsed almost entirely onto `heartwood_rate_mm_year` as a single split variable — losing it forces GBM onto a different, weaker proxy, landing at a similar error rather than a much worse one.

**The most important consequence: MLP's large lead over GBM was mostly an artifact of the leak, not real.** With the leak removed, MLP (77.69mm) and GBM (82.09mm) are now close to each other rather than MLP beating GBM by roughly 2×. This doesn't necessarily flip the recommendation in `docs/timber_model_comparison.md` (MLP still edges out GBM), but the *margin* reported there was substantially inflated — the honest headline is closer to "both extrapolate poorly on this task, MLP marginally less poorly," not "MLP is clearly the better architecture."

**Net effect of §1 + §2 together:** §1 (Sydney data being a duplicate of Kembla's) is still unresolved and still means the Sydney numbers below aren't evidence of cross-port generalization. §2 is now fixed on the Kembla side, so the *feature leakage* axis of distortion is gone — the Sydney numbers below are honest with respect to leakage, just not yet honest with respect to being a genuinely different site.

---

## 3. What was actually done

1. **Retrained** MLP (`src/models/timber_model.py`) and GBM (`src/models/timber_model_gbm.py`) on Kembla, using `remaining_diameter_mm` as target, `time_based_split`, `random_state=42`, and the **leak-free `FEATURE_COLUMNS`** (§2 — `sapwood_rate_mm_year`/`heartwood_rate_mm_year` removed). Numbers below therefore no longer match `docs/timber_model_results.md`/`docs/timber_model_comparison.md`/`docs/timber_project_report.md`, which still report the pre-fix, leaky numbers — those three docs need a full regeneration pass (SHAP, quantile, decay_dose experiments all need re-running against the new feature set) before they can be trusted again; that hasn't been done yet, this run only covers the plain MLP/GBM point-estimate comparison.
2. **Saved** both fitted pipelines to `artifacts/timber_condition_model_mlp.joblib` and `artifacts/timber_condition_model_gbm.joblib` (previously only one model at a time could be saved, to a single shared path — this run keeps both).
3. **Loaded** Sydney's CSV. Its `scenario_label` for the MIROC-Medres scenario reads `"Cooler & Wetter"`; Kembla's pipeline expects `"Colder & Wetter"` (the two builder scripts independently chose different spellings for the same scenario — flagged in an earlier session too). Harmonized in memory before loading; the source CSV on disk was not modified.
4. **Scored** both trained models against all 994 Sydney rows, then again split at `age_years <= 15` ("observed window," meant to emulate a port manager who only has ~15 years of real history) vs `age_years > 15` ("forecast beyond the window").
5. **Built a dashboard-schema CSV** for Sydney (`data/dashboard/dashboard_data_timber_sydney.csv`), same 8 columns as `src/models/build_dashboard_output.py` produces for Kembla: `age<=15` rows carry `method="observed"` (Sydney's own simulated ground truth, converted to `deterioration_depth`), `age>15` rows carry `method="mlp"`/`"gbm"` (the Kembla-trained model's forecast). This is a **separate small script**, not a modification of the shared `build_dashboard_output.py` — that script's Kembla/concrete pipeline is calendar-year-anchored (2050), while this Sydney run needed an install-relative age anchor (15 years), and I didn't want to bolt a second anchoring concept onto shared infrastructure other teammates rely on without a team decision first.

---

## 4. Results

### 4.1 Kembla retrain (leak-free — no longer matches the older docs, see §3)

| Model | Train MAE | Train R² | Test MAE | Test R² | n train / test |
|---|---:|---:|---:|---:|---:|
| MLP | 5.87 | 0.997 | **77.69** | **0.625** | 798 / 196 |
| GBM | 1.19 | 1.000 | 82.09 | 0.644 | 798 / 196 |

This is the honest headline number now: with the leaky rate columns gone, MLP's test MAE nearly doubles (42.65mm → 77.69mm) and its lead over GBM almost disappears.

### 4.2 Sydney evaluation (read §1 before interpreting these — still a duplicate dataset)

| Model | Split | MAE (mm) | RMSE (mm) | R² | n |
|---|---|---:|---:|---:|---:|
| MLP | Full Sydney (994 rows) | 28.23 | 51.97 | 0.939 | 994 |
| MLP | age ≤ 15 ("observed") | 7.79 | 10.67 | 0.971 | 301 |
| MLP | age > 15 ("forecast") | 37.10 | 61.84 | 0.880 | 693 |
| GBM | Full Sydney (994 rows) | 37.89 | 74.11 | 0.875 | 994 |
| GBM | age ≤ 15 ("observed") | 1.35 | 2.12 | 0.999 | 301 |
| GBM | age > 15 ("forecast") | 53.76 | 88.74 | 0.753 | 693 |

**§1 still applies in full** — Sydney's data is still a duplicate of Kembla's (unresolved, needs real data supplied), so "Full Sydney" still looks better than the honest Kembla test MAE for the same reason as before (roughly 80% of these rows are rows the model was literally trained on). What's changed is that this inflation is no longer *compounded* by feature leakage on top of it — MLP's Sydney-full MAE (28.23mm) is still an optimistic number, but it's optimistic for one reason (duplicate data) instead of two.

**GBM now looks *worse* than MLP again on Sydney** (37.89mm vs 28.23mm, and notably worse on the age>15 forecast split: 53.76mm vs 37.10mm) — consistent with the Kembla-side finding that GBM's extrapolation weakness was always real; leakage had been masking it on both datasets before, not just on Kembla.

---

## 5. Dashboard output

`data/dashboard/dashboard_data_timber_sydney.csv` — 1,687 rows:

```
method
gbm         693
mlp         693
observed    301
```

Same 8-column schema as `dashboard_data_timber.csv`/`dashboard_data_concrete.csv` (`material, asset_id, scenario, year, value, unit, metric, method`), `metric="deterioration_depth"`, `unit="mm"`, so it can sit alongside the existing dashboard files without any frontend change. It should **not** be presented next to the Kembla dashboard file as if it demonstrates something Kembla's own file doesn't, given §1.

---

## 6. What would make this a real, presentable result

1. **Get genuine Sydney/Botany Bay data** — real asset register (species, diameters, install years, zone/exposure/maintenance codes) and real local baseline sea temperature/salinity, not a copy of Kembla's. Confirm with whoever owns `Timber_sydney_v6.3.xlsm`. **Still outstanding.**
2. ~~Fix the feature leakage~~ **Done.** `sapwood_rate_mm_year`/`heartwood_rate_mm_year` removed from `FEATURE_COLUMNS`, moved to `LEAKY_COLUMNS`; models retrained; §2/§4 above show the before/after. Kembla test MAE got meaningfully worse (42.65mm → 77.69mm for MLP) — that worse number is the honest one.
3. **Regenerate the three narrative docs** (`docs/timber_model_results.md`, `docs/timber_model_comparison.md`, `docs/timber_project_report.md`) against the fixed feature set — they still report the pre-fix numbers as current. This needs re-running `src/models/timber_compare.py`'s full suite (SHAP importance, quantile bands, residual-correction, decay_dose/distance-target experiments), not just the plain MLP/GBM comparison covered here.
4. **Re-run this script once real Sydney data lands** — nothing about the pipeline needs to change, since it already separates "observed window" from "forecast beyond the window" and already produces the dashboard-schema output. At that point, the age>15 MAE becomes a legitimate answer to "how well does a Kembla-trained model forecast an unseen port's pile past 15 years of real history" — which is the actual question worth presenting.

---

## 7. Reproduce

```bash
python -m scripts.retrain_and_eval_sydney
```

Regenerates `artifacts/timber_condition_model_{mlp,gbm}.joblib`, `data/dashboard/dashboard_data_timber_sydney.csv`, and `scripts/retrain_and_eval_sydney_metrics.json`. `random_state=42` is fixed for the MLP/GBM training step; exact figures can still drift slightly across scikit-learn/LightGBM versions.
