# General Timber Replacement Model Performance Report

**Evaluation date:** 22 August 2026  
**Models:** Ordinary least-squares linear regression and RBF support vector regression (SVR)  
**Target:** `replacement_time_years`

## Executive summary

The linear regression model is the stronger model for the current user-facing
forecast. It achieved an R² of **0.9878** and an MAE of **0.122 years** on the
pooled scenario outputs. The SVR achieved an R² of **0.6658** and an MAE of
**0.477 years** on the same outputs.

The principal reason for this difference is that the evaluation is
chronological. Both models train on earlier years and predict later years. A
linear model can extend a learned trend beyond the training period, whereas an
RBF-kernel SVR is primarily an interpolation method and commonly flattens when
inputs move beyond the region represented during training. The test results
show this behavior, particularly for the Colder & Wetter and Hotter & Drier
scenarios.

Based on the current data and evaluation design, the linear model should be the
preferred production model. The SVR should not replace it unless its temporal
generalization improves after tuning or redesign.

## Model and output definition

Both models use the same preprocessing:

- Missing numerical values are replaced with the training median.
- Numerical columns are standardized.
- Missing categorical values are replaced with the most frequent training value.
- Categorical columns are one-hot encoded.

Both PileA and PileB contribute to one general timber model. `asset_code` is not
an input feature. At inference, raw predictions are pooled across both piles and
both emissions pathways. The application presents exactly one result for each
year and scenario:

- Most Likely
- Hotter & Drier
- Colder & Wetter

Consequently, the performance metrics in this report are measured on the same
values presented to users, not on the underlying individual pile/pathway rows.

## Evaluation design

The data is divided chronologically rather than randomly:

| Partition | Years/rows |
|---|---:|
| Training rows | 672 |
| Test rows | 180 |
| First test year | 2056 |
| Test period | 2056–2070 |
| Pooled test outputs | 45 (15 years × 3 scenarios) |

For each test year and scenario, the four underlying actual values and four raw
predictions (two piles × two emissions pathways) are averaged separately. MAE,
RMSE, and R² are then calculated between the 45 pooled actual values and the 45
pooled predictions.

R² measures the reduction in squared error relative to always predicting the
mean actual replacement time:

```text
R² = 1 - (model squared error / mean-baseline squared error)
```

## Overall performance

| Model | MAE (years) | RMSE (years) | R² |
|---|---:|---:|---:|
| Linear regression | **0.1224** | **0.1309** | **0.9878** |
| RBF SVR | 0.4769 | 0.6863 | 0.6658 |

The linear model has both lower typical error and fewer large errors. Its RMSE
is only slightly above its MAE, indicating consistent errors. The larger gap
between SVR's MAE and RMSE indicates that some SVR predictions have notably
larger errors.

## Performance by scenario

| Model | Scenario | MAE (years) | Mean bias (years) | R² |
|---|---|---:|---:|---:|
| Linear | Colder & Wetter | 0.140 | -0.140 | 0.922 |
| Linear | Hotter & Drier | 0.082 | +0.082 | 0.972 |
| Linear | Most Likely | 0.145 | +0.145 | 0.838 |
| SVR | Colder & Wetter | 0.713 | +0.713 | -1.903 |
| SVR | Hotter & Drier | 0.582 | +0.582 | -1.254 |
| SVR | Most Likely | 0.136 | +0.123 | 0.732 |

A positive bias means overprediction. SVR performs reasonably for Most Likely,
but it systematically overpredicts Colder & Wetter and Hotter & Drier. Negative
scenario-level R² means its within-scenario trend is worse than using that
scenario's test-period mean. Its combined R² remains positive because it still
distinguishes the overall levels of the three scenarios.

## Feature influence

Feature influence was measured with grouped permutation importance. One feature
was randomly shuffled in the test rows, predictions were recalculated and
pooled into the user-facing outputs, and the reduction in pooled R² was
recorded. This was repeated 20 times with a fixed random seed. A larger R² drop
means that the model depends more strongly on that feature for this test set.

### Linear regression

| Rank | Feature | Mean R² reduction |
|---:|---|---:|
| 1 | `summer_sea_temp_c` | 2.039 |
| 2 | `salinity_factor` | 0.222 |
| 3 | `initial_diameter_mm` | 0.215 |
| 4 | `year_installed` | 0.209 |
| 5 | `construction_factor` | 0.207 |
| 6 | `model_knot_factor` | 0.206 |
| 7 | `timber_exposure_factor` | 0.204 |
| 8 | `water_factor` | 0.169 |

`summer_sea_temp_c` is the most influential column by a wide margin. This is
reasonable because projected sea temperature varies over time and between
climate scenarios and is associated with the deterioration environment. The
linear model uses its continuous trend effectively when extrapolating into the
later test years. An importance reduction greater than 1 is possible because a
permuted model can perform substantially worse than the mean baseline.

### RBF SVR

| Rank | Feature | Mean R² reduction |
|---:|---|---:|
| 1 | `water_factor` | 0.375 |
| 2 | `model_knot_factor` | 0.351 |
| 3 | `year_installed` | 0.325 |
| 4 | `initial_diameter_mm` | 0.323 |
| 5 | `construction_factor` | 0.322 |
| 6 | `timber_exposure_factor` | 0.314 |
| 7 | `salinity_factor` | 0.311 |
| 8 | `summer_sea_temp_c` | 0.259 |

SVR spreads its dependence across multiple timber and deterioration factors.
`water_factor` is its most influential individual column. Sea temperature is
still important but is not dominant as it is for the linear model. This broad,
nonlinear dependence helps SVR fit individual rows but does not capture the
later pooled temporal trend as effectively.

Permutation importance is predictive, not causal. Several inputs are correlated
or encode overlapping information—for example, `scenario_label`,
`climate_model`, and projected environmental values. Importance can therefore
be shared between columns, and rankings should not be interpreted as physical
causation.

## Why SVR's pooled R² is 0.6658

The evidence supports four explanations:

1. **Limited extrapolation by the RBF kernel.** The model is evaluated on years
   later than every training year. RBF similarity decreases as future inputs
   move away from training examples, causing predictions to flatten or move
   toward familiar training-region values.
2. **Systematic scenario bias.** SVR overpredicts Colder & Wetter by an average
   of 0.713 years and Hotter & Drier by 0.582 years. These errors are directional,
   not random, and therefore reduce R² substantially.
3. **Reduced variance after pooling.** Averaging across piles and pathways
   removes much of the easy-to-explain variation between raw rows. The remaining
   evaluation focuses on relatively small changes across year and scenario, so
   a half-year error is meaningful.
4. **Untuned nonlinear behavior.** The current values (`C=100`, `epsilon=0.1`,
   `gamma="scale"`) have not been selected using a time-aware validation search.
   They may fit raw relationships without providing the best future-year trend.

For illustration, the final three test years show the flattening clearly:

| Year | Scenario | Actual | Linear | SVR |
|---:|---|---:|---:|---:|
| 2068 | Colder & Wetter | 20.822 | 20.627 | 22.200 |
| 2068 | Hotter & Drier | 21.363 | 21.459 | 22.530 |
| 2068 | Most Likely | 23.521 | 23.713 | 23.825 |
| 2070 | Colder & Wetter | 20.587 | 20.365 | 22.366 |
| 2070 | Hotter & Drier | 21.138 | 21.236 | 22.675 |
| 2070 | Most Likely | 23.352 | 23.560 | 23.793 |

The actual Colder & Wetter and Hotter & Drier values decrease between 2068 and
2070. Linear regression follows that direction, whereas SVR increases slightly.

## Interpretation and limitations

- The target appears to come from an existing timber deterioration calculation,
  and several model inputs are factors used by that calculation. The models are
  therefore approximating a modeled relationship rather than being validated
  against independent observed replacement events. High R² should not be read
  as proof of equivalent real-world accuracy.
- There are only 45 pooled outputs in the test evaluation. Additional ports,
  assets, climate models, and independently observed outcomes would provide a
  stronger generalization test.
- PileA/PileB and A1B/A1FI are averaged to match the agreed user-facing output.
  This intentionally hides variation between those underlying cases.
- Feature importance was measured only on the 2056–2070 test period and can
  change with a different time range or dataset.

## Recommendation

Use linear regression as the current general timber projection model. Report
the pooled metrics as the primary performance measures because they evaluate
the exact three scenario values shown to users. Retain raw-row metrics only as
diagnostics.

If SVR is revisited, use expanding-window time-series validation to tune `C`,
`epsilon`, and `gamma`, and compare it against other models that can represent
smooth temporal trends without losing extrapolation behavior.
