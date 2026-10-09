# Timber Import Guide: What Data a Pile Needs

**For:** engineers preparing timber pile data for the deterioration models.
**Template:** `data/templates/timber_import_template.xlsx`. It has a README sheet, example-filled sheets, a `coefficients` lookup sheet and an `allowed_codes` sheet. Blank CSV versions are in `data/templates/skeleton/`, with `*.example.csv` copies and `coefficients.csv`.
**Code:** `src/data/timber_import.py` (validation and conversion), `src/models/timber_forecast.py` (forecast), `src/ui/timber_import_page.py` (upload page).

## 1. What you provide, and what the model works out

The models need **15 inputs per year** (`FEATURE_COLUMNS`) plus the pile's sapwood layer. The **target** is the measured remaining diameter. You only type a few values; the rest is calculated:

| You provide (template sheet) | Where to find it | Used by the model as |
|---|---|---|
| Pile attributes (`timber_assets`) | `InputData` sheet, one row per pile | `initial_diameter_mm`, `layer_of_sapwood_mm` (the sapwood loss) |
| 6 pile coefficients as **numbers** (`timber_assets`: `k_heartwood`, `k_sapwood`, `k_shelter`, `k_protect`, `k_contact`, `k_knot`) | Look up each code in the coefficient table (§2.2) | The heartwood, sapwood, shelter, protection, contact and knot factors, then environment, construction and combined k0 |
| Hazard zone (`timber_assets`: `zone`) | `InputData` column M | `salinity_factor` (k_sal) each year, from zone and salinity |
| Site baseline climate (`site_climate`) | `Baseline` sheet: summer (DJF) sea surface temperature and salinity | The starting point for each year's `summer_sea_temp_c` and `salinity` |
| Climate change at 5 anchor years (`climate_changes`) | `MostLikely` / `Hotter&Drier` / `Cooler&Wetter` sheets | `summer_sea_temp_c` and `salinity` for every year 2000–2070, interpolated between anchor years as the workbook does; then `water_factor` = 0.1·e^(0.13·SST) and `thermal_dose` = age × water_factor |
| Year installed (`timber_assets`) | `InputData` column B | `age_years` = year − year_installed |
| Inspections (`timber_observations`, optional) | Field inspections (in the workbook: pile sheet column BF, `Dt`) | Not a model input. Used to anchor the forecast and to check the model against the pile |

**The models never use these:** the workbook's calculated outputs (`rsap`, `rheart`, `dmean`, `Prob`, `ddesign`, `treplacement`). They would leak the answer.

## 2. The skeleton dataset

The template has **4 data sheets**. Each sheet's first row is the header, exactly as listed below. Sheet and column names must match.

### 2.1 `timber_assets`: one row per pile (required)

| Column | Required | Type | Unit | Allowed values / range | Where to find it | Description |
|---|---|---|---|---|---|---|
| `asset_id` | yes | text | | unique | InputData A (Asset Code) | Pile ID, e.g. PileA |
| `site` | yes | text | | must exist in `site_climate` | Baseline B1 | e.g. Port Kembla |
| `description` | no | text | | | InputData C | e.g. Pile 32 @ Berth 201 |
| `timber_type` | no | text | | | InputData D (Type) | e.g. Turpentine Gum |
| `year_installed` | yes | integer | year | 1900–2070 | InputData B | Installation year |
| `zone` | yes | code | | A–G | InputData M (Zone) | AS 5604 hazard zone. Sets k_sal each year, and which `k_heartwood` column to read |
| `initial_diameter_mm` | yes | number | mm | 50–2000 | InputData G (Do) | Diameter when installed |
| `sapwood_layer_mm` | yes | number | mm | 0–200, and less than the radius | InputData H (dsap) | Sapwood layer thickness |
| `replacement_diameter_mm` | yes | number | mm | 0–2000, and less than `initial_diameter_mm` | InputData I (Dreplace) | Replace the pile below this diameter |
| `k_heartwood` | yes | number | | 0.1–50 | §2.2, heartwood class (InputData E) | Heartwood factor k_HW |
| `k_sapwood` | yes | number | | 0.1–50 | §2.2, sapwood code (InputData F) | Sapwood factor k_SW |
| `k_shelter` | yes | number | | 0.1–2 | §2.2, exposure code (InputData N) | Water zone factor |
| `k_protect` | yes | number | | 0.1–2 | §2.2, maintenance code (InputData J) | Maintenance factor |
| `k_contact` | yes | number | | 0.1–5 | §2.2, contact code (InputData K) | Contact factor |
| `k_knot` | yes | number | | 0.1–5 | §2.2, knot code (InputData L) | Knot factor |

### 2.2 Coefficient values: what to type in the `k_` columns

From the workbook's `Coefficients` sheet (AS 5604-2005). The same table is in the template's `coefficients` sheet and `skeleton/coefficients.csv`.

**`k_heartwood`: the only coefficient that depends on the zone.** Use the column for the pile's `zone`.

| Code | Heartwood | Zones A–C | Zones D–G |
|---|---|---:|---:|
| HW1 | Class 1 | 1.1 | 1.3 |
| HW2 | Class 2 | 1.7 | 5.2 |
| HW3 | Class 3 | 3.4 | 8.8 |
| HW4 | Class 4 | 17.0 | 25.0 |

**`k_sapwood`** (same in every zone):

| Code | Sapwood of softwood | k_sapwood | | Code | Sapwood of hardwood | k_sapwood |
|---|---|---:|---|---|---|---:|
| SPSU | Untreated | 40.0 | | SPHU | Untreated | 25.0 |
| SPSCR1 | Creosote-treated 24%kg/kg | 1.7 | | SPHCR1 | Creosote-treated 13%kg/kg | 0.8 |
| SPSCR2 | Creosote-treated 40%kg/kg | 1.0 | | SPHCR2 | Creosote-treated 22%kg/kg | 0.6 |
| SPSCC1 | CCA-treated 0.6%kg/kg | 4.7 | | SPHCC1 | CCA-treated 0.7%kg/kg | 3.2 |
| SPSCC2 | CCA-treated 1%kg/kg | 3.0 | | SPHCC2 | CCA-treated 1.2%kg/kg | 1.8 |
| SPSCC3 | CCA-treated 2%kg/kg | 1.8 | | SPHCC3 | CCA-treated 2.4%kg/kg | 1.1 |
| SPSCC4 | CCA-treated 5%kg/kg | 1.0 | | SPHDBT | Double-treated CC2+CR2 | 0.5 |
| SPSDBT | Double-treated CC2+CR2 | 0.9 | | | | |

**The other four:**

| Column | Code | Meaning | Value |
|---|---|---|---:|
| `k_shelter` | EXP1 | Calm or sheltered from wave | 1.0 |
| `k_shelter` | EXP2 | Exposed to strong wave or surf | 0.6 |
| `k_protect` | MTN1 | Floating collar | 0.5 |
| `k_protect` | MTN2 | None | 1.0 |
| `k_contact` | CNT1 | Contact with other timber member | 2.0 |
| `k_contact` | CNT2 | None | 1.0 |
| `k_knot` | KNT1 | Having knots without protective plate | 2.0 |
| `k_knot` | KNT2 | Having knots with protective plate | 1.0 |
| `k_knot` | KNT3 | None | 1.0 |

**k_sal is worked out for you,** from `zone` and each year's salinity. You don't enter it:

| Salinity (ppt) | Zones A–D | Zones E–G |
|---|---:|---:|
| 1–10 | 0.7 | 1.0 |
| 11–25 | 0.8 | 1.0 |
| 26–35 | 1.0 | 1.0 |

**Values outside these tables are accepted with a warning,** for example a treatment AS 5604 doesn't list. That also covers the workbook's undefined codes MTN3, CNT3 and KNT4 (Defect D3): enter the value you've agreed with the asset owner. The workbook itself silently used 1.0 for all three.

### 2.3 `site_climate`: one row per site (required)

| Column | Required | Type | Unit | Range | Where to find it | Description |
|---|---|---|---|---|---|---|
| `site` | yes | text | | unique | Baseline B1 | Site name |
| `baseline_summer_sst_c` | yes | number | °C | −2 to 40 | Baseline C9 (SST, DJF) | Port Kembla: 21.5 |
| `baseline_salinity_ppt` | yes | number | ppt | 0 to 45 | Baseline C8 (SSS, DJF) | Port Kembla: 35.5 |

### 2.4 `climate_changes`: change from baseline at each anchor year (required)

There is one row per site × scenario × pathway × anchor year: **3 scenarios × 2 pathways × 5 anchor years = 30 rows per site**. A1FI rows are required. A1B rows are optional, but without them the A1B charts are unavailable. Baseline rows are optional, since no change is assumed.

| Column | Required | Type | Unit | Allowed / range | Where to find it | Description |
|---|---|---|---|---|---|---|
| `site` | yes | text | | in `site_climate` | | |
| `scenario_label` | yes | code | | Most Likely, Hotter & Drier, Colder & Wetter, Baseline | scenario sheet name | Cooler&Wetter sheet → "Colder & Wetter" |
| `climate_model` | yes | text | | | scenario sheet B1 | e.g. MRI-CGCM2.3.2 |
| `emissions_pathway` | yes | code | | A1FI, A1B | scenario sheet row 4 | |
| `anchor_year` | yes | integer | year | 1990, 2010, 2030, 2050, 2070 | scenario sheet A5:A9 | |
| `summer_sst_change_c` | yes | number | °C | −5 to 10 | scenario sheet columns L (A1FI) / M (A1B), "Summer SST change" | Change from `baseline_summer_sst_c` |
| `salinity_change_ppt` | no (blank = 0) | number | ppt | −10 to 10 | see the note below | Change from `baseline_salinity_ppt` |

**Note on salinity (a workbook quirk).** Only the MIROC (Cooler&Wetter) sheet has salinity projections. The workbook's yearly salinity column (pile sheet column E), which the models were trained on, borrows MIROC's **annual** SSS change (Cooler&Wetter columns J–K) for **every** scenario:
- the A1FI values for all scenarios and pathways, except Colder & Wetter A1B
- the A1B values for Colder & Wetter A1B

The workbook does **not** use the summer SSS columns (N–O). The example sheet already contains the right values for Port Kembla:
- A1FI: 0, 0.0167, 0.05, 0.10, 0.17
- Colder & Wetter A1B: 0, 0.0167, 0.05, 0.09, 0.12

At zone E, salinity doesn't change the decay factor (k_sal = 1.0), but it is still one of the models' inputs.

### 2.5 `timber_observations`: inspections (optional, strongly recommended)

| Column | Required | Type | Unit | Range | Description |
|---|---|---|---|---|---|
| `asset_id` | yes | text | | in `timber_assets` | |
| `inspection_year` | yes | integer | year | ≥ `year_installed`, not in the future; 2000 or later to be used | |
| `remaining_diameter_mm` | yes | number | mm | 0 to `initial_diameter_mm` | Measured remaining sound diameter |
| `notes` | no | text | | | |

These are real inspections, one per pile per year, not one per scenario.
- **The model doesn't need them to forecast.** It predicts from the pile's attributes, coefficients and climate alone.
- **With inspections,** the forecast is **anchored**: shifted by the model's average error over the last 5 years of inspections, so it continues from the measured history and one bad reading can't set the shift. If that shift is more than **15 mm**, the app warns that the model doesn't match this pile well.
- **Without inspections,** the forecast is the model alone, with no check against the pile.
- **Recommended history:** at least **15 years** after installation, from several inspections. The validator warns below that.
- **Inspections before 2000 are not used,** because the climate series starts in 2000. The validator warns about them.

### 2.6 Example rows (from the template)

The example pile's inspections are the workbook engine's own values for that pile (A1FI, Most Likely), so it forecasts cleanly. Replace every row with real data.

`timber_assets`

| asset_id | site | description | timber_type | year_installed | zone | initial_diameter_mm | sapwood_layer_mm | replacement_diameter_mm | k_heartwood | k_sapwood | k_shelter | k_protect | k_contact | k_knot |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| EXAMPLE_01 | Port Kembla | Pile 12 @ Berth 105 (example - replace) | Gum | 2006 | E | 450 | 20 | 200 | 1.3 | 25.0 | 1.0 | 1.0 | 1.0 | 1.0 |

The coefficients come from: HW1 in zone E (zones D–G column) → 1.3; SPHU → 25.0; EXP1, MTN2, CNT2 and KNT3 → 1.0.

`site_climate`

| site | baseline_summer_sst_c | baseline_salinity_ppt |
|---|---|---|
| Port Kembla | 21.5 | 35.5 |

`climate_changes` (first rows of 30)

| site | scenario_label | climate_model | emissions_pathway | anchor_year | summer_sst_change_c | salinity_change_ppt |
|---|---|---|---|---|---|---|
| Port Kembla | Most Likely | MRI-CGCM2.3.2 | A1FI | 1990 | 0 | 0 |
| Port Kembla | Most Likely | MRI-CGCM2.3.2 | A1FI | 2010 | 0.1978128 | 0.0166667 |
| Port Kembla | Most Likely | MRI-CGCM2.3.2 | A1FI | 2030 | 0.5934385 | 0.05 |
| Port Kembla | Most Likely | MRI-CGCM2.3.2 | A1FI | 2050 | 1.2327421 | 0.1 |
| Port Kembla | Most Likely | MRI-CGCM2.3.2 | A1FI | 2070 | 1.9907689 | 0.17 |

`timber_observations`

| asset_id | inspection_year | remaining_diameter_mm | notes |
|---|---|---|---|
| EXAMPLE_01 | 2010 | 402 | example - replace with real inspections |
| EXAMPLE_01 | 2014 | 385 | |
| EXAMPLE_01 | 2018 | 367 | |
| EXAMPLE_01 | 2022 | 348 | |

## 3. Validation rules

`validate_import()` reports **every** problem at once, with its sheet, row and column. **Errors** block the forecast; **warnings** don't.

| Check | Level |
|---|---|
| A required sheet (`timber_assets`, `site_climate`, `climate_changes`) or required column is missing | error |
| A required value is blank, a number is not numeric, or a year is not a whole number | error |
| A value is outside its allowed range (including a `k_` value outside its range), or a code (`zone`, scenario, pathway, anchor year) is not in the allowed list | error |
| A duplicate key (two rows for the same pile, site, anchor or inspection year) | error |
| `replacement_diameter_mm` ≥ `initial_diameter_mm`, or `sapwood_layer_mm` ≥ the radius | error |
| A pile's `site` is not in `site_climate` | error |
| A site is missing any A1FI anchor year for any of the 3 scenarios | error |
| The same, for A1B | warning (A1B charts unavailable) |
| An inspection is for an unknown pile, before installation, in the future, or larger than the initial diameter | error |
| A `k_` value is not in the AS 5604 table (§2.2) | warning |
| `k_heartwood` comes from the wrong zone column (e.g. 1.1 for a zone E pile) | warning |
| An inspection is before 2000, so it isn't used | warning |
| Remaining diameter increases between inspections | warning |
| Less than 15 years of inspection history, or no inspections at all | warning |

To check a file from the command line:

```bash
python -m src.data.timber_import --validate my_piles.xlsx
```

## 4. From a valid file to the 2 graphs

1. **Climate:** yearly values for 2000–2070 are built from baseline + interpolated changes. This matches the training data to within 1e-6, checked in `tests/test_timber_import.py`.
2. **Features:** the `k_` numbers become the model's factors; k_sal comes from zone and salinity; then `water_factor`, `age_years` and `thermal_dose` are calculated.
3. **Prediction:** the **saved** GBM and MLP models predict material loss to 2070. They are never retrained; see `docs/timber_model_updated.md` §8.
4. **Anchoring:** if there are inspections, each forecast is shifted by the average error over the last 5 years of inspections. A shift over 15 mm triggers a warning.
5. **Charts:** the output is two charts, GBM and MLP side by side, each showing the inspection history and then Most Likely, Hotter & Drier and Colder & Wetter to 2070 for the chosen pathway. They come with a CSV download in the dashboard format.

**In the app:**

```bash
python -m streamlit run src/ui/timber_data_app.py     # sidebar → "Import a pile"
```

**From the command line,** using the template workbook or a folder of the 4 CSVs:

```bash
python -m src.models.timber_forecast --input data/templates/timber_import_template.xlsx --model mlp --out forecast.csv
```

## 5. Example output

**Validation** (`python -m src.data.timber_import --validate data/templates/timber_import_template.xlsx`):

```
No issues.
VALID
```

**Forecast,** saved models `timber_transfer_{gbm,mlp}_20261004_143534`, A1FI pathway. Each model returns 156 rows: 12 `observed` (4 inspections × 3 scenarios) and 144 forecast rows (2023–2070 × 3 scenarios). The CSV has the dashboard columns:

```
material,asset_id,scenario,year,value,unit,metric,method
timber,EXAMPLE_01,A1FI|CSIRO Mk3.5,2023,54.0179,mm,deterioration_depth,mlp_anchored
timber,EXAMPLE_01,A1FI|CSIRO Mk3.5,2024,56.4818,mm,deterioration_depth,mlp_anchored
timber,EXAMPLE_01,A1FI|CSIRO Mk3.5,2025,58.8464,mm,deterioration_depth,mlp_anchored
```

`value` is radial material loss in mm. The inspections convert as (450 − remaining diameter) / 2: 24.0, 32.5, 41.5 and 51.0 mm.

**Anchor record** (`result.attrs["anchor"]`, and `forecast.anchor.json` next to `--out`):

| Model | Offset applied | Warning? |
|---|---:|---|
| MLP | −1.1 mm | no |
| GBM | −12.2 mm | no (under 15 mm) |

**Material loss forecast (mm), compared with the workbook engine for the same pile:**

| Year | Most Likely: MLP / GBM / engine | Hotter & Drier: MLP / GBM / engine | Colder & Wetter: MLP / GBM / engine |
|---|---|---|---|
| 2030 | 69.2 / 64.0 / 70.0 | 71.4 / 65.3 / 71.9 | 71.9 / 65.7 / 72.5 |
| 2040 | 95.6 / 82.0 / 96.2 | 100.3 / 86.9 / 100.7 | 101.2 / 87.8 / 102.0 |
| 2050 | 123.6 / 102.9 / 124.5 | 132.4 / 108.8 / 132.9 | 134.4 / 110.5 / 135.3 |
| 2060 | 155.9 / 127.4 / 156.0 | 169.9 / 137.1 / 170.4 | 173.0 / 139.5 / 174.5 |
| 2070 | 189.4 / 148.4 / 190.6 | 204.1 / 161.4 / 213.0 | 206.6 / 165.2 / 219.4 |
| **Replacement year** (diameter reaches 200 mm, i.e. 125 mm loss) | **2051 / 2060 / 2051** | **2048 / 2056 / 2048** | **2048 / 2055 / 2048** |

The MLP reproduces the engine's replacement year in every scenario. The saved GBM (deliberately reduced to 25 trees; see `docs/timber_model_updated.md` §12.3) forecasts too slowly and puts replacement 7–9 years late.

## 6. How much history, and how far ahead

Tested with histories made by the workbook engine for a PileA-like pile (heartwood class 1, sheltered), so the true future is known. Forecast accuracy over the forecast years, A1FI, 3 scenarios:

| History supplied | Forecast | MLP R² (anchored) | GBM R² (anchored) |
|---|---|---:|---:|
| Installed 2006; inspections 2010, 2014, 2018, 2022 | 2023–2070 | 0.997 | 0.696 |
| Installed 2006; inspections every year 2015–2025 | 2026–2070 | 0.996 | 0.669 |
| Installed 1985; inspections every year 1990–2005 (only 2000–2005 used) | 2006–2070 | 0.997 | 0.701 |

Unanchored, the MLP scores 0.996–0.998 and the GBM 0.84–0.87; anchoring makes the reduced GBM worse, because its error isn't a constant offset.

**Limits:**
- **The horizon ends in 2070,** where the climate inputs end. A 50-year forecast from 2026 would need climate projections, and the import's year range, extended past 2070.
- **Piles older than about 65 years** in the forecast are beyond the ages the models were trained on (PileA: up to 65). Predictions are capped at the pile radius, but treat them with care.
- **Summer sea temperatures outside about 21.5–24.7 °C** (the trained Port Kembla range) are extrapolation.
- **Inspections before 2000** are not used (§2.5).

## 7. Things to check before importing

- **Non-Port Kembla sites** need their own baseline and anchor changes from regional climate projections. Copying Port Kembla's values would forecast the wrong climate.
- **Pick `k_heartwood` from the right zone column.** It is the only coefficient that depends on the zone, and the validator warns if it looks like the other column's value.
- **Accuracy:** the models were trained on one spreadsheet-generated pile (PileA), so the "truth" behind every score here is the spreadsheet, not real inspections. Treat forecasts for very different piles (other timber classes, zones or diameters) as indicative only. See the limitations in `docs/timber_model_updated.md` §10.
