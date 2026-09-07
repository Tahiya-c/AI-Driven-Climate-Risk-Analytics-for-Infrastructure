# Timber Data and Model Development Progress Report

**Report date:** 28 August 2026  
**Scope:** `build_timber_kembla_csv.py`, `train_timber_linear.py`, and
`train_timber_svr.py`

## 1. Purpose

This report records the timber data-processing and model-development work
completed to date. The work forms a pipeline that extracts timber pile
projections from the Kembla workbook, converts them into a machine-learning
dataset, and trains two models to predict timber replacement time under future
climate scenarios.

The prediction target used by both models is `replacement_time_years`.

## 2. Work completed

### 2.1 Timber workbook consolidation

The script `src/data/build_timber_kembla_csv.py` was created to transform the
source workbook into one tidy CSV suitable for analysis and model training.

The script currently:

- reads the `PileA` and `PileB` worksheets with `openpyxl` in read-only,
  calculated-value mode;
- identifies projection records by accepting only rows with a numeric value in
  Excel column D (`year`), which excludes banners, headers, and separator rows;
- extracts selected values from Excel columns B to BJ, covering asset
  properties, environmental projections, deterioration factors, calculated
  deterioration results, and replacement time;
- adds `asset_code` and `source_row` fields so every output record can be traced
  to its worksheet and source row;
- corrects the known climate-model typo `CSIRO Mk3.6` to `CSIRO Mk3.5`;
- assigns readable scenario labels from the climate-model name;
- calculates pile age as `age_years = year - year_installed`;
- converts year and measurement fields to appropriate numeric types and trims
  whitespace from descriptive fields;
- sorts the result consistently by asset, emissions pathway, climate model, and
  year; and
- writes the consolidated dataset to
  `data/processed/timber_kembla_v6_3_consolidated.csv` by default.

The script also prints a validation summary after export, including the number
of rows and columns, assets, pathways, models, year range, model-name fixes, and
non-zero null counts. It raises clear errors if a required worksheet is absent
or no projection rows are found.

### 2.2 Current consolidated dataset

The generated CSV was inspected on 28 August 2026 and has the following
characteristics:

| Check | Result |
|---|---:|
| Rows | 994 |
| Columns | 46 |
| Rows for PileA | 497 |
| Rows for PileB | 497 |
| Year range | 2000-2070 |
| Unique projection years | 71 |
| Fully duplicated rows | 0 |
| Missing replacement-time targets | 0 |

The dataset contains the Base, A1B, and A1FI emissions pathways. The Base
pathway has 142 records using the `(Reference)` climate model. Each combination
of A1B/A1FI and the three projected climate models has 142 records, giving a
balanced set across the two piles.

### 2.3 Time-aware linear regression

The script `src/models/train_timber_linear.py` was created as the baseline
time-series linear model (TSLM). It trains one pooled ordinary least-squares
model using data from both piles.

The implementation currently:

- loads and prepares the consolidated timber data;
- checks that all required input and target columns are present;
- uses 31 source, asset, scenario, time, environmental, and deterioration-factor
  features;
- intentionally excludes calculated deterioration outputs that are downstream
  of the inputs, reducing direct target leakage;
- excludes `asset_code`, so the output represents a general timber model rather
  than a separate model for each pile;
- splits the data chronologically, reserving the latest fraction of unique years
  for testing (20% by default);
- imputes missing numerical values with the training median and standardizes
  them;
- imputes missing categorical values with the most frequent training value and
  one-hot encodes categories, while safely ignoring unseen categories;
- fits `LinearRegression` within a single scikit-learn pipeline;
- calculates MAE, RMSE, and R-squared for both individual rows and pooled
  year-by-scenario predictions; and
- saves the fitted pipeline, metadata, metrics, split details, and test scenario
  predictions to `artifacts/timber_replacement_linear.joblib` by default.

The chronological split is appropriate for the intended forecasting use: the
model is evaluated on later years rather than on a random sample of years it
has already seen.

### 2.4 Support vector regression

The script `src/models/train_timber_svr.py` was created to compare the linear
baseline with a nonlinear model. It reuses the same feature list, target,
chronological split, and output approach as the linear model, making the
comparison consistent.

Its preprocessing is equivalent to the linear pipeline, followed by an RBF
support vector regressor. The default hyperparameters are:

| Parameter | Default |
|---|---:|
| `C` | 100.0 |
| `epsilon` | 0.1 |
| `gamma` | `scale` |

All three values can be changed from the command line. The fitted model and its
metadata are saved to `artifacts/timber_replacement_svr.joblib` by default.

### 2.5 Evaluation results obtained so far

The existing performance analysis reports the following results for pooled
year-by-scenario outputs on the chronological test period beginning in 2056:

| Model | MAE (years) | RMSE (years) | R-squared |
|---|---:|---:|---:|
| Linear regression | 0.1224 | 0.1309 | 0.9878 |
| RBF SVR | 0.4769 | 0.6863 | 0.6658 |

On this evaluation, linear regression is the stronger model. It follows the
future trend more accurately, while the RBF SVR tends to flatten or
overpredict when extrapolating beyond the training period. The current
recommendation is therefore to use the linear model for the user-facing timber
forecast and retain SVR as an experimental comparison.

These results describe the evaluation already documented in
`docs/timber_model_performance_report.md`. They should be regenerated after the
current reproducibility issues below are resolved.

## 3. Current workflow

The intended workflow is:

1. Run `build_timber_kembla_csv.py` on the source Kembla workbook.
2. Use the resulting consolidated CSV as the common input to both training
   scripts.
3. Train with a chronological holdout of the latest projection years.
4. Compare the linear and SVR models using the same pooled scenario-level
   metrics.
5. Save the selected fitted model and its evaluation metadata as a Joblib
   artifact for the prediction application.

## 4. Current limitations and outstanding work

The following items remain unresolved as of the report date:

1. **Missing shared scenario module.** Both training scripts import
   `src.models.timber_scenarios`, but that file is not currently present in the
   repository. The test suite consequently stops during collection with
   `ModuleNotFoundError`, and neither training script is currently runnable from
   the checked-out source.
2. **Incomplete scenario mapping.** The CSV contains the climate-model value
   `MIROC-Medres`, while `build_timber_kembla_csv.py` maps
   `MIROC(hires)` to `Colder & Wetter`. As a result, the 142 `MIROC-Medres`
   records currently have a missing `scenario_label`. The source spelling and
   desired normalization need to be confirmed and added to the mapping.
3. **Model artifacts are absent.** The `artifacts` directory currently contains
   only `.gitkeep`; the linear and SVR Joblib files are not present. They must be
   regenerated once the missing scenario helper and label mapping are fixed.
4. **SVR hyperparameters are not time-series tuned.** The present SVR uses
   manually selected defaults. A future comparison should use expanding-window
   validation to tune `C`, `epsilon`, and `gamma` without introducing future
   information into training.
5. **The target is model-derived.** `replacement_time_years` comes from the
   workbook's deterioration calculations rather than independently observed
   replacement events. The reported accuracy therefore measures how well the
   machine-learning models reproduce the workbook relationship, not confirmed
   real-world service-life accuracy.
