# Data Dictionary — `concrete_kembla_v6_consolidated.csv`

**Source:** `Concrete_kembla_v6.xlsm` (Port Kembla, concrete assets)
**Grain:** one row per `asset_code` × `emissions_pathway` × `climate_model` × `year`
**Rows:** 3,479 (7 assets × 7 scenario series × 71 years, 2000–2070)
**Columns:** 35

---

## Provenance — which sheets were used

| Sheet | Used? | Why |
|---|---|---|
| `PileA`–`PileG` | **Yes — the data** | 7 per-asset sheets, identical layout. Headers rows 22–24, data rows 26–528, split into 7 scenario blocks by banner rows. |
| `InputData` | Verified only | Asset register. Confirmed to match row 26 of each pile sheet exactly, so it is already denormalised into the pile sheets. No join needed. |
| `Baseline` | No | 1990 reference climate for Wollongong University station (068188). Already propagated into pile columns E/F/G. |
| `MRI-CGCM2.3.2 (Likely)`, `CSIRO Mk3.5 (Drier)`, `MIROC(hires) (Wetter)` | No | GCM projections. Already propagated into pile columns E/F/G. Retained as provenance for `scenario_label`. |
| `Coefficients` | No | Lookup: corrosion rates by exposure class, CO₂ diffusion coefficients, activation energies. Explains the physics; not row data. |
| `Chloride Model` | No | Lookup: `ke`/`n` by exposure zone, surface chloride `Co` by environment. |
| `Design service life` | No | Small reference curve table. |
| `Port Kembla Cement` | No | PKPC mix designs (Mix A–D), exposure classes, strength/diffusion specs. |
| `Mastersheet` | **Excluded** | Template/scratch sheet, not a real asset — see Known Issues. |

---

## Columns

### Identifiers

| Column | Type | Source | Notes |
|---|---|---|---|
| `asset_code` | str | sheet name | PileA–PileG |
| `description` | str | col I | e.g. "Pile No 243 @ Berth 107" |
| `source_row` | int | — | Row number in the original pile sheet. Kept for traceability; drop before training. |

### Asset attributes (constant within an asset)

| Column | Unit | Source | Notes |
|---|---|---|---|
| `year_built` | year | col H | Ranges 1902–2010 |
| `zone` | — | col J | splash / atmospheric / tidal |
| `dist_from_coast_km` | km | col K | 0.4–2.0 |
| `exposure_class` | class | col L | B1, B2, C1, C2 (AS 3600 exposure classification) |
| `carbonation_class` | class | col M | CB1–CB4; maps to `Coefficients` sheet corrosion rates |
| `chloride_class` | class | col N | CL1–CL5; maps to `Coefficients` sheet corrosion rates |
| `protection` | — | col O | "NONE" for all 7 assets |
| `cover_h_mm` | mm | col P | Concrete cover to reinforcement, 50–110 |
| `member_diameter_mm` | mm | col Q | Pile diameter, 1200–1800 |
| `fc_mpa` | MPa | col R | Characteristic compressive strength, 40–120 |
| `wc_ratio` | — | col S | Water/cement ratio, 0.33–0.60 |
| `cement_content_kgm3` | kg/m³ | col T | 250–310 |
| `rebar_diameter_mm` | mm | col U | 12 for all assets (blank in `InputData`) |

### Scenario

| Column | Source | Notes |
|---|---|---|
| `emissions_pathway` | col B | `Base`, `A1FI` (high emissions), `A1B` (mid) |
| `climate_model` | col C | `(Reference)`, `MRI-CGCM2.3.2`, `CSIRO Mk3.5`, `MIROC(hires)` |
| `scenario_label` | derived | From the workbook's own sheet names: MRI → **Most Likely**, CSIRO → **Hotter & Drier**, MIROC → **Colder & Wetter**, Reference → **Baseline** |

### Climate drivers (vary by year × scenario)

| Column | Unit | Source |
|---|---|---|
| `year` | year | col D — 2000–2070 |
| `age_years` | years | derived: `year − year_built` |
| `mean_temp_c` | °C | col E |
| `mean_rh_pct` | % | col F |
| `co2_ppm` | ppm | col G |

### Chloride model inputs (vary by asset/zone)

| Column | Unit | Source |
|---|---|---|
| `surface_chloride_co_kgm3` | kg/m³ | col CK — surface chloride concentration |
| `chloride_diffusion_m2yr` | m²/yr | col CM — diffusion coefficient `Dc` |

### Outputs — carbonation pathway

| Column | Unit | Source | Notes |
|---|---|---|---|
| `carb_depth_xc_mm` | mm | col BD | Carbonation depth `xc(t')`. Corrosion initiates when this reaches `cover_h_mm`. |
| `carb_depth_increment_mm` | mm/yr | col BF | Annual change in `xc`. **This is a deterioration rate** — the most direct candidate for Dr. Daniel's per-material rate projections. |
| `carb_initiation_prob` | 0–1 | col BE | Probability carbonation has initiated corrosion |
| `carb_icorr_ua_cm2` | µA/cm² | col BJ | Carbonation-driven corrosion current density |

### Outputs — chloride pathway

| Column | Unit | Source | Notes |
|---|---|---|---|
| `chloride_c_at_xc_kgm3` | kg/m³ | col CP | `C(xc,t)` chloride concentration at carbonation front |
| `chloride_c_at_cover_kgm3` | kg/m³ | col CQ | `C(h,t)` chloride concentration at rebar depth |
| `chloride_initiation_prob` | 0–1 | col CS | Probability chloride ingress has initiated corrosion; triggers when `C(h,t)` exceeds `Cr` = 3.35 kg/m³ |
| `chloride_icorr_ua_cm2` | µA/cm² | col CV | Chloride-driven corrosion current density |

### Combined

| Column | Unit | Source | Notes |
|---|---|---|---|
| `combined_initiation_prob` | 0–1 | col EC ("Combo") | Combined carbonation + chloride initiation probability. **728 nulls** — see Known Issues. |

---

## Known issues

1. **`combined_initiation_prob` has 728 nulls (21% of rows).** Composed of:
   - 497 — the entire `Base` block; the Combo formula was never applied to the reference scenario.
   - 217 (A1FI) + 14 (A1B) — the formula was not filled down for A1FI CSIRO Mk3.5 years 2000–2029, plus scattered single rows.

   Identical in all 7 pile sheets, so it is a template defect, not per-asset data loss. If you use this as the target, either restrict to complete blocks or recompute it from the two component probabilities.

2. **Model-name typo, corrected.** One row per pile sheet (row 403, A1B, year 2017) was labelled `CSIRO Mk3.6`. Normalised to `CSIRO Mk3.5` — 7 fixes. Would otherwise have created a phantom fourth model on any groupby.

3. **`Mastersheet` excluded.** It carries PileA's description ("Pile No 243 @ Berth 107") but different inputs: year built 2007 vs 1997, exposure C1 vs C2, f'c 60 vs 40 MPa, w/c 0.37 vs 0.41. It is the template the pile sheets were copied from, not an eighth asset. Including it would produce a phantom asset duplicating PileA's identity with contradictory physics.

4. **Salinity is not in the row data.** Only the `MIROC(hires)` sheet carries a salinity projection, and it was never propagated into the pile sheets. The proposal's Climate Exposure Index cites ocean salinity deviation (Hunting et al., 2013) — that variable is not available from this file and must come from elsewhere.

5. **Scenario labels describe moisture, not temperature.** At 2070 under A1FI, MIROC ("Colder & Wetter") reaches 25.44 °C, *above* CSIRO ("Hotter & Drier") at 25.27 °C. The distinction is relative humidity (CSIRO −13.79%, MIROC −0.45%). Worth clarifying with Dr. Daniel before labelling three lines on a chart.

6. **Six series, not three.** Each GCM appears under both A1FI and A1B. Dr. Daniel's brief asks for three lines; choosing which pathway (or averaging them) is an open decision, deliberately left unresolved here.

7. **PileC `year_built` = 1902.** Plausible for Port Kembla but an outlier worth confirming.

8. **Per-block summary cells not carried over.** Each pile sheet holds single-cell values for `ti` (initiation time), `tp` (propagation time), `tcr` (cracking time), `Ec`, `ft`, `Ψcp`, `kR` — populated only in the first row of the Base block. These are scalar summaries, not per-year series, so they do not fit this grain. Extract separately if needed.
