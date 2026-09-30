"""Verbatim Python port of the Timber_*_v6_3.xlsm marine borer deterioration model.

Reproduces the Mastersheet calculation (columns AM-BJ) exactly, defects included.
Reads all site/asset/climate parameters from the workbook so it works for any port.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from openpyxl import load_workbook

VD, ALPHA, D_CRIT = 1.5, 0.8, 200.0
ANCHORS = [1990, 2010, 2030, 2050, 2070]


# ---------------------------------------------------------------- coefficients
def read_coefficients(path: str) -> dict:
    ws = load_workbook(path, data_only=True)["Coefficients"]
    kwood = {}
    for r in range(4, 23):                      # G=code, H=Zones A-C, I=Zones D-G
        code = ws.cell(r, 7).value
        if code:
            kwood[str(code).strip()] = (ws.cell(r, 8).value, ws.cell(r, 9).value)
    ksal = {}
    for r in range(4, 7):                       # L=band, M=Zones A-D, N=Zones E-G
        band = str(ws.cell(r, 12).value).strip()
        ksal[band] = (ws.cell(r, 13).value, ws.cell(r, 14).value)
    def lookup(r0, r1):
        return {str(ws.cell(r, 23).value).strip(): ws.cell(r, 24).value
                for r in range(r0, r1 + 1)}
    return {
        "kwood": kwood, "ksal": ksal,
        "kshelter": lookup(3, 4), "kprotect": lookup(7, 8),
        "kcontact": lookup(11, 12), "kknot": lookup(15, 17),
    }


def _match(code: str, table: dict) -> float:
    """Excel MATCH() without the exact-match argument: unknown codes silently
    resolve to the last table entry. DEFECT D3 -- reproduced deliberately."""
    code = str(code).strip()
    if code in table:
        return float(table[code])
    return float(list(table.values())[-1])


# ------------------------------------------------------------------- climate
SHEETS = {
    "Base": ("Baseline", "(Reference)"),
    "MostLikely": ("MostLikely", None),
    "Hotter&Drier": ("Hotter&Drier", None),
    "Cooler&Wetter": ("Cooler&Wetter", None),
}


def read_site(path: str) -> dict:
    wb = load_workbook(path, data_only=True)
    b = wb["Baseline"]
    site = {"name": b["B1"].value,
            "sst_summer": float(b["C9"].value), "sss_summer": float(b["C8"].value),
            "scenarios": {}}
    for sheet in ("MostLikely", "Hotter&Drier", "Cooler&Wetter"):
        ws = wb[sheet]
        gcm = str(ws["B1"].value).strip()
        anc = {}
        for i, r in enumerate(range(5, 10)):
            anc[ANCHORS[i]] = {
                "A1FI": (ws.cell(r, 12).value or 0.0, ws.cell(r, 14).value or 0.0),
                "A1B":  (ws.cell(r, 13).value or 0.0, ws.cell(r, 15).value or 0.0),
            }
        site["scenarios"][gcm] = anc
    return site


def ramp(anchors: dict, pathway: str, idx: int, years: np.ndarray) -> np.ndarray:
    """Piecewise-linear cumulative ramp between decadal anchors, as the workbook
    does it: delta(y) = delta(y-1) + (a[n+1]-a[n]) / n_years_in_segment."""
    grid, val = {}, 0.0
    grid[ANCHORS[0]] = 0.0
    for a, b in zip(ANCHORS[:-1], ANCHORS[1:]):
        step = (anchors[b][pathway][idx] - anchors[a][pathway][idx]) / (b - a)
        for y in range(a + 1, b + 1):
            val += step
            grid[y] = val
        val = anchors[b][pathway][idx]
        grid[b] = val
    return np.array([grid.get(int(y), 0.0) for y in years])


# -------------------------------------------------------------------- engine
@dataclass
class Asset:
    code: str; year_installed: int; description: str; timber_type: str
    hw: str; sw: str; d0: float; dsap: float; dreplace: float
    maintn: str; contact: str; knot: str; zone: str; exposure: str


def read_assets(path: str) -> list[Asset]:
    ws = load_workbook(path, data_only=True)["InputData"]
    out = []
    for r in range(5, ws.max_row + 1):
        if not ws.cell(r, 1).value:
            continue
        v = [ws.cell(r, c).value for c in range(1, 15)]
        out.append(Asset(str(v[0]).strip(), int(v[1]), str(v[2]).strip(),
                         str(v[3]).strip(), str(v[4]).strip(), str(v[5]).strip(),
                         float(v[6]), float(v[7]), float(v[8]), str(v[9]).strip(),
                         str(v[10]).strip(), str(v[11]).strip(), str(v[12]).strip(),
                         str(v[13]).strip()))
    return out


def simulate(asset: Asset, coef: dict, site: dict, pathway: str, gcm: str,
             years: np.ndarray) -> pd.DataFrame:
    zone_wood = 0 if asset.zone in "ABC" else 1
    zone_sal = 0 if asset.zone in "ABCD" else 1

    if gcm == "(Reference)":
        d_sst = np.zeros(len(years)); d_sss = np.zeros(len(years))
    else:
        anc = site["scenarios"][gcm]
        d_sst = ramp(anc, pathway, 0, years)
        d_sss = ramp(anc, pathway, 1, years)

    sst = site["sst_summer"] + d_sst
    sss = site["sss_summer"] + d_sss

    kwater = 0.1 * np.exp(0.13 * sst)
    k_hw = float(coef["kwood"][asset.hw][zone_wood])
    k_sw = float(coef["kwood"][asset.sw][zone_wood])
    bands = list(coef["ksal"].keys())          # '1 - 10', '11 - 25', '26 - 35'
    s = float(np.mean(sss))
    idx = 0 if s <= 10 else (1 if s <= 25 else 2)
    k_sal = float(coef["ksal"][bands[idx]][zone_sal])
    k_shelter = _match(asset.exposure, coef["kshelter"])
    k_protect = _match(asset.maintn, coef["kprotect"])
    k_contact = _match(asset.contact, coef["kcontact"])
    k_knot = _match(asset.knot, coef["kknot"])

    k_env = k_sal * k_shelter
    k_const = k_protect * k_contact * k_knot
    k0 = k_hw * k_sal * k_shelter * k_protect * k_contact * k_knot

    r_sap = kwater * k_sw * k_env * k_const
    r_heart = kwater * k_hw * k_sal * k_shelter * k_protect * k_contact * k_knot

    tlag_s = np.where(r_sap >= 20, 0.0, 2 - 0.1 * r_sap)
    tlag_h = np.where(r_heart >= 20, 0.0, 2 - 0.1 * r_heart)
    t_sap = asset.dsap / r_sap + tlag_s

    age = np.maximum(0.0, years - asset.year_installed)
    raw_age = years - asset.year_installed
    d_mean = np.where(raw_age <= 0, 0.0,
                      np.where(age > (t_sap + tlag_h),
                               asset.dsap + (age - t_sap - tlag_h) * r_heart,
                               asset.dsap))
    d_t = asset.d0 - 2 * d_mean
    prob = np.where(2 * d_mean > (asset.d0 - asset.dreplace), 1.0,
                    d_mean / ((asset.d0 - asset.dreplace) / 2))
    d_design = d_mean * (1 + ALPHA * VD)
    dd = asset.d0 - 2 * d_design
    t_repl = (D_CRIT / 2 - asset.dsap) / r_heart + t_sap + tlag_h

    return pd.DataFrame({
        "asset_code": asset.code, "description": asset.description,
        "timber_type": asset.timber_type, "year_installed": asset.year_installed,
        "zone": asset.zone, "exposure": asset.exposure,
        "heartwood_class": asset.hw, "sapwood_of_hardwood": asset.sw,
        "maintenance_measure": asset.maintn, "contact": asset.contact,
        "knot_presence": asset.knot, "initial_diameter_mm": asset.d0,
        "layer_of_sapwood_mm": asset.dsap, "replacement_diameter_mm": asset.dreplace,
        "emissions_pathway": pathway, "climate_model": gcm,
        "year": years.astype(int), "age_years": raw_age.astype(int),
        "salinity": sss, "summer_sea_temp_c": sst,
        "design_time_years": np.maximum(0.0, raw_age),
        "model_salinity_ppt": sss,
        "water_factor": kwater, "timber_exposure_factor": k_hw,
        "model_hazard_wood": k_sw, "model_service_water": k_sal,
        "salinity_factor": k_shelter, "shelter_factor": k_protect,
        "protection_factor": k_contact, "model_contact_factor": k_knot,
        "model_knot_factor": k_env, "environment_factor": k_const,
        "construction_factor": k0,
        "sapwood_rate_mm_year": r_sap, "sapwood_lag_years": tlag_s,
        "heartwood_rate_mm_year": r_heart, "heartwood_lag_years": tlag_h,
        "sapwood_duration_years": t_sap, "mean_deterioration_mm": d_mean,
        "remaining_diameter_mm": d_t, "failure_probability": prob,
        "design_deterioration_mm": d_design, "design_diameter_mm": dd,
        "replacement_time_years": t_repl,
    })


SCENARIO_LABEL = {"(Reference)": "Baseline", "MRI-CGCM2.3.2": "Most Likely",
                  "CSIRO Mk3.5": "Hotter & Drier", "MIROC-Medres": "Cooler & Wetter"}


def build(path: str, y0: int = 2000, y1: int = 2070) -> pd.DataFrame:
    coef, site, assets = read_coefficients(path), read_site(path), read_assets(path)
    years = np.arange(y0, y1 + 1)
    streams = [("Base", "(Reference)")] + [(p, g) for p in ("A1B", "A1FI")
                                           for g in site["scenarios"]]
    frames = [simulate(a, coef, site, p, g, years)
              for a in assets for p, g in streams]
    df = pd.concat(frames, ignore_index=True)
    df["scenario_label"] = df["climate_model"].map(SCENARIO_LABEL)
    return df.sort_values(["asset_code", "emissions_pathway", "climate_model", "year"]
                          ).reset_index(drop=True)
