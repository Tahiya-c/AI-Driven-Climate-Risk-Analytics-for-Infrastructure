# Concrete: Model Comparison (post-Optuna)

**Summary:** Five models forecast how much carbonation depth a concrete pile gains over
time under climate scenarios, all trained and scored identically by
`src/models/concrete_eval.py` (`python -m src.models.concrete_benchmark_all`) — same real
data (49 series: 7 piles × 7 scenarios), same target (`annual_rate`, the true year-over-year
depth change), same autoregressive 2027–2070 rollout. RNN and CatBoost are now
**Optuna-tuned** (`python -m src.models.concrete_tuning`; see
[concrete_rnn_tuning.md](concrete_rnn_tuning.md) for the full before/after and why Ridge
and Random Forest's tuned params were tried and **reverted**, not kept).

Two conditions, because they answer different questions:
- **Known pile** — forecast a pile the model has already seen trained on the other 48
  series+itself. This is what the dashboard's scenario selector actually does.
- **Unseen pile** (leave-one-pile-out, 7 folds) — forecast a pile the model never saw at
  all during training, the closer analogue to Wen Shern's PileA→PileB timber test.

| Model | R² (known pile) | R² (unseen pile) | RMSE (mm, known pile) | MAE (mm, known pile) |
|---|---:|---:|---:|---:|
| **RNN** (tuned) | 0.799 | **0.341** | **0.134** | **0.103** |
| CatBoost (tuned) | **0.810** | −0.135 | 0.194 | 0.155 |
| Random Forest | **0.841** | 0.741 | 0.215 | 0.164 |
| Persistence (no-model baseline) | 0.813 | 0.813 | 0.205 | 0.165 |
| Ridge (deliberate worst-case) | −5.77 | −7.89 | 0.829 | 0.639 |

*(sqrt_law / wc_regress — two physics-closed-form baselines — and mean sit between Ridge
and the pack; full 8-model table in `results_concrete_benchmark.csv`.)*

## Reading this table honestly

- **No single model wins both columns.** Random Forest has the best known-pile R² (0.841,
  narrowly ahead of persistence 0.813 and CatBoost 0.810) but on the unseen-pile axis
  **persistence is actually the best** (0.813, unchanged from the known-pile column — it
  has no model fit to go stale), with Random Forest a clear second (0.741). The tuned
  CatBoost, strong on known piles, goes **negative** on unseen piles (−0.135, worse than
  just repeating last year's rate) — it has learned something pile-specific that doesn't
  transfer.
- **RNN is the one that matters for the dashboard**, because the dashboard never asks a
  model to forecast a pile it has zero data for — all 7 piles are already in the training
  set. On that known-pile axis RNN has the best MAE/RMSE (0.103mm / 0.134mm) by a clear
  margin, even though its R² (0.799) sits a little below CatBoost/RF/persistence — see
  [concrete_rnn_tuning.md §3](concrete_rnn_tuning.md#rnn--kept-tuned_rnn_params) for why
  (one low-variance scenario, `Base|(Reference)`, pulls the mean R² down while MAE/RMSE
  still improve).
- **Persistence (carry forward last year's rate) is a genuinely hard baseline to beat on
  the unseen-pile axis** — nothing here beats it outright; tuning closed most of the RNN's
  gap to it (R² −0.173→0.341) without closing it fully, and the two tree-based models
  trade places (RF close behind, CatBoost well behind).
- **Ridge is intentionally the "worst model"** shown opposite RNN on the dashboard's
  comparison chart (`src/ui/dashboard.py`) — deeply negative R² on both axes regardless of
  tuning, included here for contrast, not as a real contender.

## Caveat carried from every concrete report in this project

The underlying 71 years (2000–2070) are the workbook's **modelled climate projections**,
not real inspection records — high R²/low MAE here means a model reproduces the workbook's
formulas well, not that it has discovered real carbonation physics. See
[[concrete-modelling-reference]] and [[portshield-writeup-conventions]].

## Reproduce

```bash
python -m src.models.concrete_benchmark_all              # regenerates the table above
python -m src.models.concrete_tuning --model all --trials 40   # re-run the Optuna search
```
