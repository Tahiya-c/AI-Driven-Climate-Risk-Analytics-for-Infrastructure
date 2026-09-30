"""Lookup tables from the `Timber_kembla_v6.3.xlsm` workbook's `Coefficients` sheet.

These tables are byte-identical across workbook versions v4-v6.3 (see
`docs/timber_kembla_model_spec_and_defects.md`, section 6). Dict insertion order
matters: Excel's `MATCH()` calls in the Mastersheet omit the exact-match argument,
so an unknown code approximate-matches to the LAST row of its table (Defect D3).
`lookup_approx` below reproduces that behaviour deliberately.
"""

from __future__ import annotations

import warnings

# code -> (k for zones A-C, k for zones D-G)
K_HW_SW: dict[str, tuple[float, float]] = {
    "HW1": (1.1, 1.3),
    "HW2": (1.7, 5.2),
    "HW3": (3.4, 8.8),
    "HW4": (17.0, 25.0),
    "SPSU": (40.0, 40.0),
    "SPSCR1": (1.7, 1.7),
    "SPSCR2": (1.0, 1.0),
    "SPSCC1": (4.7, 4.7),
    "SPSCC2": (3.0, 3.0),
    "SPSCC3": (1.8, 1.8),
    "SPSCC4": (1.0, 1.0),
    "SPSDBT": (0.9, 0.9),
    "SPHU": (25.0, 25.0),
    "SPHCR1": (0.8, 0.8),
    "SPHCR2": (0.6, 0.6),
    "SPHCC1": (3.2, 3.2),
    "SPHCC2": (1.8, 1.8),
    "SPHCC3": (1.1, 1.1),
    "SPHDBT": (0.5, 0.5),
}

# salinity band -> k_sal, for zones A-D only (zones E-G are 1.0 for every band)
K_SAL_ZONES_A_D: dict[str, float] = {
    "1-10": 0.7,
    "11-25": 0.8,
    "26-35": 1.0,
}

# Table order is significant for Defect D3 (approximate-match fallback).
K_SHELTER: dict[str, float] = {"EXP1": 1.0, "EXP2": 0.6}
K_PROTECT: dict[str, float] = {"MTN1": 0.5, "MTN2": 1.0}
K_CONTACT: dict[str, float] = {"CNT1": 2.0, "CNT2": 1.0}
K_KNOT: dict[str, float] = {"KNT1": 2.0, "KNT2": 1.0, "KNT3": 1.0}

ZONES_A_C = {"A", "B", "C"}
ZONES_A_D = {"A", "B", "C", "D"}


def lookup_hw_sw(code: str, zone: str) -> float:
    """k_HW or k_SW: exact-match on `code`, then split by zone group A-C vs D-G."""
    ac, dg = K_HW_SW[code]
    return ac if zone in ZONES_A_C else dg


def salinity_band(salinity_ppt: float) -> str:
    if salinity_ppt <= 10:
        return "1-10"
    if salinity_ppt <= 25:
        return "11-25"
    return "26-35"


def lookup_salinity(salinity_ppt: float, zone: str) -> float:
    """k_sal: zone group here splits A-D vs E-G (different split from k_HW/k_SW)."""
    if zone not in ZONES_A_D:
        return 1.0
    return K_SAL_ZONES_A_D[salinity_band(salinity_ppt)]


def lookup_approx(code: str, table: dict[str, float], table_name: str) -> float:
    """Reproduce Excel MATCH() without the exact-match flag (Defect D3).

    An unknown code approximate-matches to the LAST entry of the table instead of
    raising. `InputData` assigns PileB the undefined codes MTN3/CNT3/KNT4, which
    all fall back this way; this port keeps that behaviour verbatim in
    `timber_engine.py` and only fixes it in `engine_corrected.py`.
    """
    if code in table:
        return table[code]
    fallback_key = next(reversed(table))
    warnings.warn(
        f"DEFECT D3: unknown code {code!r} not found in {table_name!r}; "
        f"approximate-matching to last table entry {fallback_key!r}={table[fallback_key]} "
        "per Excel MATCH() semantics",
        stacklevel=2,
    )
    return table[fallback_key]
