# Timber_kembla_v6_3.xlsm — Model Specification and Defect Register

Reverse-engineered from the workbook supplied by supervisor. Reference standard cited
in the workbook: **AS 5604-2005** (Timber — Natural durability ratings).

---

## 1. Workbook structure

| Sheet | Role |
|---|---|
| `Baseline` | 1975–2004 reference climate for a 1° grid centred on Port Kembla |
| `MostLikely` | GCM = MRI-CGCM2.3.2 |
| `Hotter&Drier` | GCM = CSIRO Mk3.5 |
| `Cooler&Wetter` | GCM = MIROC-Medres |
| `Coefficients` | k-factor lookup tables |
| `InputData` | Asset register (2 piles) |
| `Mastersheet` | Calculation engine, ~502 rows |
| `vbaProject.bin` | 30 KB of VBA (drives sheet population) |

Each scenario sheet carries two SRES emissions pathways: **A1FI** and **A1B**.

### Baseline climate (Port Kembla, 1975–2004)

| Variable | Value |
|---|---|
| Air temperature | 16.4 °C |
| Rainfall | 1094 mm |
| Relative humidity | 74.1 % |
| Sea surface salinity (annual / DJF) | 35.5 / 35.5 ppt |
| Sea surface temperature (annual / DJF) | 19.3 / 21.5 °C |
| CO₂ | 352.5 ppm |

### Climate projection method

Anchor deltas are given at **1990, 2010, 2030, 2050, 2070**. Intermediate years are
**piecewise-linear ramps**:

```
delta(y) = delta(y-1) + (anchor[n+1] - anchor[n]) / COUNT(segment_years)
```

Applied as `absolute = baseline + delta` for temperature, and
`absolute = baseline × (1 + delta/100)` for rainfall and relative humidity.

---

## 2. Asset register (`InputData`)

| Field | PileA | PileB |
|---|---|---|
| Year installed | 2005 | 2006 |
| Description | Pile 32 @ Berth 201 | Pile 32 @ Berth 202 |
| Type | Turpentine Gum | Gum |
| Heartwood class (HW) | HW1 | HW2 |
| Sapwood class (SW) | SPHU | SPHU |
| D₀ (mm) | 400 | 500 |
| d_sap (mm) | 20 | 20 |
| D_replace (mm) | 200 | 200 |
| Maintenance | MTN2 | **MTN3 †** |
| Contact | CNT2 | **CNT3 †** |
| Knot | KNT3 | **KNT4 †** |
| Zone | E | E |
| Exposure | EXP1 | EXP2 |

† These codes **do not exist** in the `Coefficients` sheet. See Defect 3.

---

## 3. Model equations (`Mastersheet`, columns AM–BJ)

Global constants (`Mastersheet` BE1:BF3): `Vd = 1.5`, `α = 0.8`, `D_crit = 200 mm`.

```
age        = MAX(0, year − year_installed)

kwater     = 0.1 · exp(0.13 · T_summer_sea)          # only climate-sensitive term

k_HW       = f(heartwood class, zone group)          # Zones A–C vs D–G columns
k_SW       = f(sapwood class,  zone group)
k_sal      = f(salinity band,  zone group)           # bands: 1–10, 11–25, 26–35 ppt
k_shelter  = f(exposure)                             # EXP1 = 1.0, EXP2 = 0.6
k_protect  = f(maintenance)                          # MTN1 = 0.5, MTN2 = 1.0
k_contact  = f(contact)                              # CNT1 = 2.0, CNT2 = 1.0
k_knot     = f(knot)                                 # KNT1 = 2.0, KNT2 = 1.0, KNT3 = 1.0

k_env      = k_sal · k_shelter
k_const    = k_protect · k_contact · k_knot
k_0        = k_HW · k_sal · k_shelter · k_protect · k_contact · k_knot

r_sap      = kwater · k_SW · k_env · k_const                    # mm/yr
r_heart    = kwater · k_HW · k_sal · k_shelter · k_protect
                    · k_contact · k_knot                        # mm/yr

t_lag,sap  = 0 if r_sap   ≥ 20 else 2 − 0.1·r_sap               # yr
t_lag,heart= 0 if r_heart ≥ 20 else 2 − 0.1·r_heart             # yr
t_sap      = d_sap / r_sap + t_lag,sap                          # yr

d_mean     = 0                                    if age ≤ 0
           = d_sap + (age − t_sap − t_lag,heart) · r_heart
                                                  if age > t_sap + t_lag,heart
           = d_sap                                otherwise      # ← see Defect 1

D_t        = D₀ − 2·d_mean
Prob       = 1  if 2·d_mean > (D₀ − D_replace)
           = d_mean / ((D₀ − D_replace)/2)  otherwise
d_design   = d_mean · (1 + α·Vd)          = 2.2 · d_mean
D_design   = D₀ − 2·d_design
t_replace  = (D_crit/2 − d_sap)/r_heart + t_sap + t_lag,heart
```

### Selected k-factor tables

**k_HW / k_SW** (`Coefficients` R4:T22) — column S for Zones A–C, column T for Zones D–G:

| Code | Material | A–C | D–G |
|---|---|---|---|
| HW1 | Heartwood Class 1 | 1.1 | 1.3 |
| HW2 | Heartwood Class 2 | 1.7 | 5.2 |
| HW3 | Heartwood Class 3 | 3.4 | 8.8 |
| HW4 | Heartwood Class 4 | 17 | 25 |
| SPHU | Sapwood of hardwood, untreated | 25 | 25 |
| SPSU | Sapwood of softwood, untreated | 40 | 40 |

(Treated variants SPHCR1/2, SPHCC1–3, SPHDBT, SPSCR1/2, SPSCC1–4, SPSDBT also present.)

**k_sal** — Zones A–D: 0.7 / 0.8 / 1.0 for 1–10 / 11–25 / 26–35 ppt. Zones E–G: 1.0 for all bands.

---

## 4. Defect register

### D1 — Sapwood phase is a step function (**High**)
`Mastersheet` BE column. The fallback branch returns `d_sap` unconditionally once
`age > 0`, so the full 20 mm sapwood layer is lost in year one. `r_sap`, `t_lag,sap`
and `t_sap` are computed but only used to set a timing threshold; they are never
integrated.

**Fix:** ramp deterioration linearly over the sapwood phase —
`d(t) = r_sap · (age − t_lag,sap)` for `t_lag,sap < age ≤ t_sap`.

### D2 — Deterioration is not cumulative (**High**)
`d_mean` multiplies the *current year's* `r_heart` by the *entire* elapsed age. Under a
warming trajectory this retroactively applies late-century rates to early years.
Measured effect: implied annual increment runs at 1.00× the stated rate near
installation and 1.45× by 2070 — a systematic **overstatement of climate-driven
deterioration of up to ~45%**.

**Fix:** `d_mean(t) = d_sap + Σ_τ r_heart(τ)` accumulated year by year.

### D3 — Silent lookup failures (**High**)
All `MATCH()` calls (AP, AQ, AS, AT, AU, AV) omit the third argument, defaulting to
approximate match. `Coefficients` defines only MTN1–2, CNT1–2, KNT1–3, but `InputData`
assigns PileB **MTN3 / CNT3 / KNT4**. These approximate-match to the last table row and
all silently return 1.0. **PileB's maintenance, contact and knot conditions are being
discarded.** If CNT3 was intended as contact with another member (k = 2.0), PileB's
deterioration rate is understated by a factor of 2.

**Fix:** add `, 0` to every `MATCH()`, and either extend the coefficient tables to cover
MTN3/CNT3/KNT4 or correct the `InputData` codes.

### D4 — `t_replacement` uses a global constant (**Medium**)
Column BJ uses `$BF$3/2` (D_crit = 200) rather than the row's own `Dreplace` (column N).
Correct only while every asset shares a 200 mm threshold.

**Fix:** replace `$BF$3` with `N{row}`.

### D5 — `Prob` is not a probability (**Medium**)
Column BG is a utilisation ratio capped at 1. Meanwhile `Vd = 1.5` (a 150% coefficient
of variation) is collapsed into a single deterministic 2.2× multiplier in `d_design`.
All uncertainty information is discarded.

**Fix:** treat `r_heart` as a random variable (lognormal, CV = Vd) and Monte Carlo the
trajectory. Failure probability then becomes `P(D_t ≤ D_replace)` — an actual probability.

### D6 — Engine handles one asset at a time (**Medium**)
`Mastersheet` holds ~502 rows = 1 asset × 7 climate streams × 71 years, and contains only
PileA. The consolidated CSV was produced by running the workbook twice and concatenating.
Does not scale to a portfolio.

### D7 — No flooring of physical quantities (**Low**)
`D_t` reaches −500 mm for PileB under A1FI. Should be floored at 0 with a
post-failure flag.

### D8 — Missing scenario label in the CSV export (**Low**)
All 284 MIROC-Medres rows have a blank `scenario_label`. Correct value: **Cooler & Wetter**.

### D9 — CSV column names do not match the workbook (**Low**)

| CSV column | Actually contains |
|---|---|
| `timber_exposure_factor` | `k_HW` |
| `model_knot_factor` | `k_shelter` |
| `construction_factor` | `k_HW × k_shelter` (not `k_protect·k_contact·k_knot`) |
| `replacement_time_years` | Re-estimated total design life under that year's conditions — **not** a countdown or a replacement year |

**Fix:** regenerate the export from the workbook with corrected headers.

### D10 — Climate scenarios are of superseded vintage (**Medium, methodological**)
A1B and A1FI are IPCC **SRES** pathways (2000), superseded by RCPs (AR5, 2013) and SSPs
(AR6, 2021). GCMs used (CSIRO Mk3.5, MIROC-Medres, MRI-CGCM2.3.2) are CMIP3-era.

**Fix / opportunity:** substitute CMIP6 SSP1-2.6 / SSP2-4.5 / SSP5-8.5 sea surface
temperature projections for the Port Kembla grid cell.

---

## 5. Note on scenario labelling

"Cooler & Wetter" (MIROC-Medres) produces the **fastest** deterioration of the three
GCMs — summer SST of 22.97 °C in 2040, versus 22.85 (CSIRO) and 22.41 (MRI). The scenario
labels describe the *atmospheric* projection (air temperature, rainfall), whereas the
marine borer model responds only to *sea surface* temperature. This should be stated
explicitly in any reporting, or readers will assume a labelling error.

---

## 6. Version history (v4 → v5 → v6.2 → v6.3)

Established by cell-by-cell diff of the four supplied workbooks.

### v4 → v5 — 4 cells changed

| Cell | v4 | v5 |
|---|---|---|
| `MRI-CGCM2.3.2 (Likely)!C58` | `=(C$7-C$6)/COUNT($A$38:$A$57)+C57` | `=C$7` |
| `CSIRO Mk3.5 (Drier)!C58` | same | `=C$7` |
| `MIROC(hires) (Wetter)!C58` | same | `=C$7` |
| `InputData!D6` | **"Something else"** | "Gum" |

The first three snap the end of the 1990–2050 interpolation segment exactly onto the
2050 anchor rather than letting the accumulated ramp drift.

The fourth is significant: **PileB's timber type was a placeholder ("Something else")
in v4.** PileB should be treated as a synthetic test asset, not a real Port Kembla
record. The rename was applied only to `InputData`; the `PileB` calculation sheet in v5
still reads "Something else".

### v5 → v6.2 — major restructure

| Change | v4 / v5 | v6.2 / v6.3 |
|---|---|---|
| Sheet naming | GCM names (`MRI-CGCM2.3.2 (Likely)`) | Scenario names (`MostLikely`) |
| Baseline sheet | absent | added (1975–2004 Port Kembla) |
| Climate variables | 3 (temp, RH, summer sea) | 7 (+ rainfall, annual SST, annual SSS, summer SSS, CO₂) |
| "Wetter" GCM | MIROC(**hires**) | MIROC-**Medres** |
| Projection anchors | 1990–2100 (2090/2100 rows present but empty) | 1990–2070 only |
| Per-asset sheets | `PileA`, `PileB` present | **deleted** → see Defect D6 |
| `tdesign` | `=D10-G10` (can go negative) | `=MAX(0, D10-G10)` |
| `ktimber SW` zone test | `IF(OR(S10="A"…))` — tests **Exposure** | (fixed in v6.3, see below) |
| Baseline sea temp | **hardcoded `F10 = 23`** | `=Baseline!$C$9` = **21.5 °C** |

**The baseline sea temperature change is the largest numerical effect in the entire
version history.** Because `kwater = 0.1·exp(0.13·T)`, dropping 23 °C to 21.5 °C scales
every deterioration rate by 1/1.215:

| Quantity (PileA, baseline scenario) | v5 | v6.3 |
|---|---|---|
| `r_heart` | 2.585 mm/yr | 2.127 mm/yr |
| `t_replacement` | 33.1 yr | 39.9 yr |

A 1.5 °C change in a single input cell moves predicted service life by **6.8 years**.

Note also that the summer-SST anchor series is otherwise carried over unchanged between
v5 and v6 (e.g. CSIRO: 0.865 / 1.815 / 2.926 °C at 2030 / 2050 / 2070), and the
MIROC anchors differ only by rounding (0.947 → 0.95) despite the model name changing
from *hires* to *Medres*. **Query whether the GCM was genuinely re-run or only
re-labelled.**

### v6.2 → v6.3 — 497 cells, one fix

All 497 differing cells are `Mastersheet` column **AQ** (`ktimber SW`), changing
`S{row}` to `R{row}`:

```
v6.2:  =IF((OR(S10="A",S10="B",S10="C")), INDEX(Coefficients!$S$4:$S$22, …
v6.3:  =IF((OR(R10="A",R10="B",R10="C")), INDEX(Coefficients!$S$4:$S$22, …
```

The zone-group branch was reading the **Exposure** column (S) instead of the **Zone**
column (R). Correct fix.

**However, it changed no results.** All 11,408 cached numeric cells in the Mastersheet
are identical between v6.2 and v6.3. Both assets are Zone E with Exposure EXP1/EXP2, so
neither the old nor the new test matches "A"/"B"/"C" and both route to the Zones D–G
column. The fix is latent: it only affects assets in Zones A–C, and no such asset exists
in the register to validate it against.

### Defect provenance

| Defect | v4 | v5 | v6.2 | v6.3 |
|---|---|---|---|---|
| D1 sapwood step function | present | present | present | present |
| D2 non-cumulative deterioration | present | present | present | present |
| D3 undefined MTN3/CNT3/KNT4 codes | present | present | present | present |
| D4 `t_replacement` uses global `Dcrit` | present | present | present | present |
| D5 `Prob` is a utilisation ratio | present | present | present | present |
| D6 single-asset engine | — | — | **introduced** | present |
| `ktimber SW` reads Exposure not Zone | present | present | present | **fixed** |
| `tdesign` can go negative | present | present | **fixed** | fixed |

**The `Coefficients` sheet is byte-identical across all four versions.** MTN3, CNT3 and
KNT4 have never been defined; PileB's maintenance, contact and knot settings have been
silently discarded since v4.

D1, D2, D4 and D5 are unchanged across the entire history — they are original, not
deliberate later simplifications.

---

## 7. Proposed rebuild architecture

1. **Port** the workbook to Python verbatim; validate by reproducing all 994 CSV rows.
2. **Fork** into a corrected engine fixing D1–D4, D7. Report the delta between legacy and
   corrected as a result in its own right.
3. **Monte Carlo** using `Vd` (D5) to yield genuine failure probabilities and replacement-year
   distributions.
4. **Generate a training corpus** by Latin-hypercube sampling the input space × scenarios
   × MC draws.
5. **Train a surrogate emulator** (gradient boosting / MLP) for sub-millisecond dashboard
   inference, and a **survival model** for replacement-year hazard curves.
6. **Sensitivity analysis** — SHAP on the surrogate plus Sobol indices, to rank
   `k_HW` / exposure / maintenance against climate scenario.
7. **Modernise the climate layer** to CMIP6 SSPs (D10).

**Required caveat:** the surrogate is validated against the mechanistic simulator, not
against field observation. Any reported accuracy is emulation fidelity, not predictive
accuracy. Field validation requires measured residual-diameter inspection records, which
are not present in the supplied data.
