# Concrete Import Guide: What Data a Pile Needs

Mirrors [timber_import_guide.md](timber_import_guide.md)'s shape, scoped to concrete's
much smaller feature set. Both import formats share one validation engine
(`src/data/import_schema.py`) — see that module and
[concrete_rnn_tuning.md](concrete_rnn_tuning.md) for the modelling side.

## 1. What you provide, and what the model works out

You give three things per pile: its **static properties** (exposure zone, water/cement
ratio, concrete grade), **climate** for however many years/scenarios you want forecast
(temperature, CO₂ — humidity is optional, see §3), and **optionally, measured
carbonation depths** so the forecast anchors to the pile's real history instead of
starting from zero. The model (the same RNN the live dashboard uses, never retrained on
your import) works out the carbonation trajectory.

## 2. The three sheets

One `.xlsx` with these sheet names, or a folder of same-named `.csv` files
(`python -m src.data.concrete_import --skeleton` writes blank + example versions of
both into `data/templates/`).

### 2.1 `concrete_assets`: one row per pile (required)

| Column | Type | Required | Notes |
|---|---|---|---|
| `asset_id` | text | yes | Unique pile ID |
| `zone` | code | yes | `atmospheric`, `splash`, or `tidal` — never guessed; this and the next two columns are what make a pile a pile |
| `wc_ratio` | float | yes | Water/cement ratio (reference range 0.33–0.60) |
| `fc_mpa` | float | yes | Concrete grade, MPa (reference range 40–120) |

### 2.2 `concrete_climate`: one row per scenario/year (required)

| Column | Type | Required | Notes |
|---|---|---|---|
| `site`, `scenario_label`, `climate_model`, `emissions_pathway` | text/code | yes | Identify the scenario; `emissions_pathway` must be `A1FI`, `A1B`, or `Base` |
| `year` | int | yes | |
| `mean_temp_c` | float | yes | Required, never guessed — this is the dashboard's actual climate driver |
| `co2_ppm` | float | yes | Required, never guessed |
| `mean_rh_pct` | float | **no** | The one column this format will fill in for you — see §3 |

### 2.3 `concrete_observations`: inspections (optional, recommended)

| Column | Type | Required | Notes |
|---|---|---|---|
| `asset_id` | text | yes | Must match `concrete_assets` |
| `year` | int | yes | |
| `carb_depth_mm` | float | yes | Measured carbonation depth (0–100mm) |

Without this sheet, the forecast starts from 0mm at the pile's earliest climate year —
a theoretical "brand new pile" trajectory, not anchored to anything measured.

## 3. The fill policy: why `mean_rh_pct` is special

Every other required column either comes from you or blocks validation if it's missing
— concrete grade, exposure zone and climate drivers are never guessed, because getting
them wrong changes the physics, not just the presentation. `mean_rh_pct` is the one
exception: if you have temperature and CO₂ records but not humidity for some years,
those rows get filled with **that calendar year's typical humidity** across the
reference Port Kembla series (median, computed fresh from
`data/processed/concrete_kembla_v6_consolidated.csv` each run) instead of being
blocked or defaulted to a constant. Every filled row shows up as a `warning` `Issue`
naming the value used, so you can see exactly which numbers were guessed.

This is deliberately the *only* field with this treatment right now — see
`src/data/import_schema.py`'s `Field.fill` options (`year_mean`/`year_mode` are built
and ready; nothing else currently opts in) if the team decides more columns should.

## 4. Validation rules

Every problem is collected and reported together (not fail-on-first):
- Missing required column/sheet, blank required value, wrong type, non-whole `int`.
- Out-of-range value (e.g. `wc_ratio` outside 0.2–0.8).
- Not an allowed code (`zone` outside atmospheric/splash/tidal, `emissions_pathway`
  outside A1FI/A1B/Base).
- Duplicate key row (same `asset_id`, or same `site`+`scenario_label`+`emissions_pathway`+`year`).
- An observation's `asset_id` not present in `concrete_assets`.
- (Warning only) a pile with no observations at all — forecast will run unanchored.

Run `python -m src.data.concrete_import --validate my_piles.xlsx` to see the full report.

## 5. From a valid file to a forecast

```bash
python -m src.models.concrete_forecast --input my_piles.xlsx --method rnn \
    --out data/dashboard/dashboard_data_concrete_import.csv
```

Output is the same schema as every other dashboard CSV:
`material, asset_id, scenario, year, value, unit, metric, method` — history rows get
`method="observed"`, forecast rows get `method="rnn"` (or `"wc_regress"`).

## 6. How much history you need

The deployed RNN looks back a fixed number of years (`inference.rnn_window()`, read
from `artifacts/rnn_carbonation.json` — currently 6, after the Optuna tuning in
`docs/concrete_rnn_tuning.md`) to make each prediction, so it needs at least that many
years of real-or-filled depth immediately before the forecast starts. An import with
less raises a clear error telling you exactly how many years are short, rather than
silently producing a bad forecast. `--method wc_regress` has no such requirement — it
only needs `wc_ratio` and one anchor depth — but is the cruder of the two methods (see
[[concrete-modelling-reference]]).

There's also no way to forecast past the last year your `concrete_climate` sheet
covers — climate rows are the actual forecast horizon, not just context. Add future
years there (your own scenario projections) to extend it.

## 7. Known limitation, stated plainly

Observations between your first and last inspection year, and all years before the
first inspection, are currently **held flat** at the nearest known value (not
interpolated or modelled) when building the RNN's lookback window. A pile with sparse
or late-starting inspections will have an artificially flat recent history feeding the
model, which can understate the true recent rate. This matches the honesty convention
the rest of this project follows — flagged here rather than smoothed over — and is a
reasonable next improvement (timber's `forecast_pile` anchoring in
`src/models/timber_transfer.py` is a more sophisticated version of the same idea,
worth porting here if real pile imports become common).
