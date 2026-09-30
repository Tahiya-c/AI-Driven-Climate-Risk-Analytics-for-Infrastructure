"""Verbatim Python port of the `Timber_kembla_v6.3.xlsm` Mastersheet calculation.

This is a faithful port, bugs included -- see
`docs/timber_kembla_model_spec_and_defects.md` for the full defect register (D1-D10).
Nothing here is "improved"; the corrected version lives in `engine_corrected.py` so the
legacy-vs-corrected delta can be reported as a result in its own right. In particular:

- D1: the sapwood phase is a step function -- the full `d_sap` layer is lost the instant
  `age > 0`; `r_sap`, `t_lag_sap` and `t_sap` are computed but only used as a timing
  threshold, never integrated.
- D2: `d_mean` multiplies the CURRENT year's `r_heart` by the ENTIRE elapsed age, so a
  warming trajectory retroactively applies late-century rates to early years.
- D3: unknown maintenance/contact/knot codes (PileB's MTN3/CNT3/KNT4) approximate-match
  to the last table entry instead of raising -- see `coefficients.lookup_approx`.
- D4: `t_replace` uses the global `D_CRIT` constant instead of the asset's own
  `D_replace`.
- D7: nothing is floored -- `D_t` can go negative.

Column names in `data/processed/timber_kembla_v6_3_consolidated.csv` do not match the
workbook (Defect D9): `timber_exposure_factor` actually holds `k_HW`, `model_knot_factor`
holds `k_shelter`, and `construction_factor` holds `k_HW * k_shelter`. This module uses
the workbook's own semantics throughout; the CSV's (mislabelled) names are only used for
comparison, in `tests/test_engine_fidelity.py`.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field

import pandas as pd

from src.timber.coefficients import (
    K_CONTACT,
    K_KNOT,
    K_PROTECT,
    K_SHELTER,
    lookup_approx,
    lookup_hw_sw,
    lookup_salinity,
)

# Mastersheet BE1:BF3
VD = 1.5
ALPHA = 0.8
D_CRIT = 200.0

# Baseline sheet, Port Kembla 1975-2004 reference climate
BASELINE_SUMMER_SST_C = 21.5
BASELINE_SALINITY_PPT = 35.5

ANCHOR_YEARS = (1990, 2010, 2030, 2050, 2070)


def _zero_anchors() -> dict[int, float]:
    return dict.fromkeys(ANCHOR_YEARS, 0.0)


def _anchors(anchor_2030: float, anchor_2050: float, anchor_2070: float) -> dict[int, float]:
    """Build a 1990-2070 anchor series from its 2030/2050/2070 points.

    The workbook's 2010 anchor is not an independent input: for every climate stream
    checked against the CSV export, `anchor[2010] == anchor[2030] / 3` exactly. Deriving
    it here (rather than hardcoding the spec document's 8-decimal rounding of that value)
    avoids a ~1e-8 mm drift by 2070 that Defect D2's age-multiplied deterioration
    otherwise amplifies from the rounding error.
    """
    return {
        1990: 0.0,
        2010: anchor_2030 / 3,
        2030: anchor_2030,
        2050: anchor_2050,
        2070: anchor_2070,
    }


def _interpolate_anchors(anchors: dict[int, float], year: int) -> float:
    """Piecewise-linear ramp between anchor years, per the workbook's climate sheets.

    Equivalent to the workbook's recursive `delta(y) = delta(y-1) + step`, since each
    year's step is a fixed fraction of the segment's total change.
    """
    years = sorted(anchors)
    if year <= years[0]:
        return anchors[years[0]]
    if year >= years[-1]:
        return anchors[years[-1]]
    for lo, hi in zip(years, years[1:], strict=False):
        if lo <= year <= hi:
            fraction = (year - lo) / (hi - lo)
            return anchors[lo] + fraction * (anchors[hi] - anchors[lo])
    raise AssertionError(f"year {year} not covered by anchors {years}")


@dataclass(frozen=True)
class ClimateScenario:
    """One `Mastersheet` climate stream (a GCM sheet, or the Baseline reference)."""

    name: str
    climate_model: str
    sst_anchors: dict[str, dict[int, float]]
    salinity_anchors: dict[str, dict[int, float]]

    def resolve(self, year: int, pathway: str) -> tuple[float, float]:
        """Return (summer_sea_temp_c, salinity_ppt) for this scenario/pathway/year."""
        sst_delta = _interpolate_anchors(self.sst_anchors[pathway], year)
        sal_delta = _interpolate_anchors(self.salinity_anchors[pathway], year)
        return BASELINE_SUMMER_SST_C + sst_delta, BASELINE_SALINITY_PPT + sal_delta


BASELINE = ClimateScenario(
    name="Baseline",
    climate_model="(Reference)",
    sst_anchors={"A1FI": _zero_anchors(), "A1B": _zero_anchors()},
    salinity_anchors={"A1FI": _zero_anchors(), "A1B": _zero_anchors()},
)

MOST_LIKELY = ClimateScenario(
    name="Most Likely",
    climate_model="MRI-CGCM2.3.2",
    sst_anchors={
        "A1FI": _anchors(0.5934385, 1.2327421, 1.9907689),
        "A1B": _anchors(0.6264631, 1.0540588, 1.4705744),
    },
    salinity_anchors={"A1FI": _zero_anchors(), "A1B": _zero_anchors()},
)

HOTTER_DRIER = ClimateScenario(
    name="Hotter & Drier",
    climate_model="CSIRO Mk3.5",
    sst_anchors={
        "A1FI": _anchors(0.87, 1.82, 2.93),
        "A1B": _anchors(0.90, 1.53, 2.15),
    },
    salinity_anchors={"A1FI": _zero_anchors(), "A1B": _zero_anchors()},
)

COOLER_WETTER = ClimateScenario(
    name="Cooler & Wetter",
    climate_model="MIROC-Medres",
    sst_anchors={
        "A1FI": _anchors(0.95, 1.98, 3.18),
        "A1B": _anchors(0.98, 1.67, 2.33),
    },
    salinity_anchors={
        "A1FI": _anchors(0.02, 0.04, 0.07),
        "A1B": _anchors(0.02, 0.04, 0.05),
    },
)

SCENARIOS: dict[str, ClimateScenario] = {
    scenario.name: scenario for scenario in (BASELINE, MOST_LIKELY, HOTTER_DRIER, COOLER_WETTER)
}


@dataclass(frozen=True)
class Asset:
    """One row of the `InputData` asset register."""

    code: str
    description: str
    timber_type: str
    year_installed: int
    zone: str
    exposure: str
    heartwood_class: str
    sapwood_class: str
    maintenance: str
    contact: str
    knot: str
    initial_diameter_mm: float  # D_0
    layer_of_sapwood_mm: float  # d_sap
    replacement_diameter_mm: float  # D_replace
    synthetic: bool = field(default=False)
    """Flags a placeholder/test asset never validated as a real Port Kembla record."""


ASSETS: dict[str, Asset] = {
    "PileA": Asset(
        code="PileA",
        description="Pile 32 @ Berth 201",
        timber_type="Turpentine Gum",
        year_installed=2005,
        zone="E",
        exposure="EXP1",
        heartwood_class="HW1",
        sapwood_class="SPHU",
        maintenance="MTN2",
        contact="CNT2",
        knot="KNT3",
        initial_diameter_mm=400.0,
        layer_of_sapwood_mm=20.0,
        replacement_diameter_mm=200.0,
        synthetic=False,
    ),
    "PileB": Asset(
        code="PileB",
        description="Pile 32 @ Berth 202",
        timber_type="Gum",
        year_installed=2006,
        zone="E",
        exposure="EXP2",
        heartwood_class="HW2",
        sapwood_class="SPHU",
        maintenance="MTN3",
        contact="CNT3",
        knot="KNT4",
        initial_diameter_mm=500.0,
        layer_of_sapwood_mm=20.0,
        replacement_diameter_mm=200.0,
        # v4 InputData literally read "Something else" for this asset's timber type
        # (docs/timber_kembla_model_spec_and_defects.md section 6). Never present
        # PileB's results as a real Port Kembla record.
        synthetic=True,
    ),
}


def simulate(
    asset: Asset, scenario: ClimateScenario, pathway: str, years: Iterable[int]
) -> pd.DataFrame:
    """Reproduce the Mastersheet row-by-row for one asset/scenario/pathway across `years`.

    Coefficients that depend only on the asset (k_HW, k_SW, k_shelter, k_protect,
    k_contact, k_knot) are resolved once; k_sal and kwater are resolved per year since
    they depend on that year's climate.
    """
    k_HW = lookup_hw_sw(asset.heartwood_class, asset.zone)
    k_SW = lookup_hw_sw(asset.sapwood_class, asset.zone)
    k_shelter = lookup_approx(asset.exposure, K_SHELTER, "K_SHELTER")
    k_protect = lookup_approx(asset.maintenance, K_PROTECT, "K_PROTECT")
    k_contact = lookup_approx(asset.contact, K_CONTACT, "K_CONTACT")
    k_knot = lookup_approx(asset.knot, K_KNOT, "K_KNOT")

    rows = []
    for year in years:
        age = max(0, year - asset.year_installed)
        summer_sst_c, salinity_ppt = scenario.resolve(year, pathway)
        kwater = 0.1 * math.exp(0.13 * summer_sst_c)
        k_sal = lookup_salinity(salinity_ppt, asset.zone)

        k_env = k_sal * k_shelter
        k_const = k_protect * k_contact * k_knot
        k_0 = k_HW * k_sal * k_shelter * k_protect * k_contact * k_knot

        r_sap = kwater * k_SW * k_env * k_const
        r_heart = kwater * k_HW * k_sal * k_shelter * k_protect * k_contact * k_knot

        t_lag_sap = 0.0 if r_sap >= 20 else 2 - 0.1 * r_sap
        t_lag_heart = 0.0 if r_heart >= 20 else 2 - 0.1 * r_heart
        t_sap = asset.layer_of_sapwood_mm / r_sap + t_lag_sap

        if age <= 0:
            d_mean = 0.0
        elif age > (t_sap + t_lag_heart):
            d_mean = asset.layer_of_sapwood_mm + (age - t_sap - t_lag_heart) * r_heart
        else:
            # DEFECT D1: the else-branch returns d_sap unconditionally -- the sapwood
            # ramp (r_sap, t_lag_sap, t_sap) is computed above but never integrated.
            d_mean = asset.layer_of_sapwood_mm

        D_t = asset.initial_diameter_mm - 2 * d_mean
        threshold = asset.initial_diameter_mm - asset.replacement_diameter_mm
        prob = 1.0 if 2 * d_mean > threshold else d_mean / (threshold / 2)
        d_design = d_mean * (1 + ALPHA * VD)
        D_design = asset.initial_diameter_mm - 2 * d_design
        # DEFECT D4: uses the global D_CRIT rather than asset.replacement_diameter_mm.
        t_replace = (D_CRIT / 2 - asset.layer_of_sapwood_mm) / r_heart + t_sap + t_lag_heart

        rows.append(
            {
                "asset_code": asset.code,
                "scenario_label": scenario.name,
                "climate_model": scenario.climate_model,
                "emissions_pathway": pathway,
                "year": year,
                "age_years": age,
                "summer_sea_temp_c": summer_sst_c,
                "salinity_ppt": salinity_ppt,
                "kwater": kwater,
                "k_HW": k_HW,
                "k_SW": k_SW,
                "k_sal": k_sal,
                "k_shelter": k_shelter,
                "k_protect": k_protect,
                "k_contact": k_contact,
                "k_knot": k_knot,
                "k_env": k_env,
                "k_const": k_const,
                "k_0": k_0,
                "r_sap": r_sap,
                "r_heart": r_heart,
                "t_lag_sap": t_lag_sap,
                "t_lag_heart": t_lag_heart,
                "t_sap": t_sap,
                "d_mean": d_mean,
                "D_t": D_t,
                "prob": prob,
                "d_design": d_design,
                "D_design": D_design,
                "t_replace": t_replace,
            }
        )

    return pd.DataFrame(rows)
