"""Generic import-validation engine shared by every material's user-upload pipeline.

Material-specific code (``src/data/concrete_import.py``, ``src/data/timber_import.py``)
declares its own ``Field`` list per sheet and calls into the generic pieces here --
``Field``/``Issue``/``ImportResult``, ``read_import``, ``check_sheet``, and the
missing-value fill helpers (``build_year_lookup``/``apply_fill``). Everything
material-specific -- cross-field checks, unit conversions, the actual model call --
stays in that material's own module; nothing here knows "concrete" or "timber".

``check_sheet`` and ``read_import`` are exact extractions of what used to be
``timber_import.py``'s private ``_check_sheet``/``read_import`` -- same behaviour,
just parameterised on ``fields``/``key``/``sheets`` instead of reading timber's
module-level ``SHEETS``/``KEYS`` globals, so both materials share one implementation.

``build_year_lookup``/``apply_fill`` are new: a column a material marks ``fill=`` with
something other than ``"required"``/``"blank_ok"`` gets filled from a per-year lookup
built from the existing training data, instead of failing or defaulting to blank --
every row that gets a filled-in value is recorded as a warning ``Issue`` so a
prediction built on it is never presented with the same confidence as one from
fully-supplied input.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Field:
    name: str
    dtype: str  # "text" | "int" | "float" | "code"
    required: bool
    unit: str = ""
    source: str = ""
    description: str = ""
    allowed: tuple[str, ...] = ()
    minimum: float | None = None
    maximum: float | None = None
    # Only consulted when `required` is False and the column is absent or blank:
    #   "blank_ok"      -> leave it blank/NaN, no warning (current timber behaviour)
    #   "year_mean"     -> numeric: that year's median across the reference data
    #   "year_mode"     -> categorical: that year's most common value
    #   "scenario_table" -> must come from the material's own scenario/climate table,
    #                        never guessed; the material module handles this itself,
    #                        this tag just documents that apply_fill should skip it
    fill: str = "blank_ok"


@dataclass
class Issue:
    level: str  # "error" blocks downstream use (validation/forecasting); "warning" does not
    sheet: str
    message: str
    row: int | None = None  # 1-based data row (row 1 = first row under the header)
    column: str | None = None


@dataclass
class ImportResult:
    tables: dict[str, pd.DataFrame]
    issues: list[Issue] = field(default_factory=list)

    @property
    def errors(self) -> list[Issue]:
        return [issue for issue in self.issues if issue.level == "error"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def issues_frame(self) -> pd.DataFrame:
        frame = pd.DataFrame(
            [vars(issue) for issue in self.issues],
            columns=["level", "sheet", "row", "column", "message"],
        )
        frame["row"] = frame["row"].astype("Int64")
        return frame


# --------------------------------------------------------------------------- reading


def read_import(source: str | Path | Any, sheets: dict[str, tuple[Field, ...]]) -> dict[str, pd.DataFrame]:
    """Read an .xlsx (path or file-like, e.g. a Streamlit upload) or a folder of CSVs.

    ``sheets`` is the material's ``{sheet_name: field_tuple}`` map -- only sheets named
    in it are kept; anything else in the workbook/folder is ignored.
    """
    if isinstance(source, (str, Path)) and Path(source).is_dir():
        directory = Path(source)
        return {
            sheet: pd.read_csv(directory / f"{sheet}.csv")
            for sheet in sheets
            if (directory / f"{sheet}.csv").exists()
        }
    workbook = pd.read_excel(source, sheet_name=None)
    return {sheet: frame for sheet, frame in workbook.items() if sheet in sheets}


# --------------------------------------------------------------------------- validation


def check_sheet(
    sheet: str,
    frame: pd.DataFrame,
    fields: tuple[Field, ...],
    key: list[str],
    issues: list[Issue],
) -> pd.DataFrame:
    """Column presence, types, ranges and codes. Returns the frame with coerced dtypes.

    Generic across materials: every material-specific fact lives in ``fields``
    (the column list) and ``key`` (the duplicate-row key), not in this function.
    """
    frame = frame.copy()
    frame.columns = [str(column).strip() for column in frame.columns]
    frame = frame.dropna(how="all")
    missing = [f.name for f in fields if f.required and f.name not in frame.columns]
    for name in missing:
        issues.append(Issue("error", sheet, f"Missing required column '{name}'", column=name))
    unknown = sorted(set(frame.columns) - {f.name for f in fields})
    if unknown:
        issues.append(Issue("warning", sheet, f"Ignoring unknown columns: {unknown}"))
    if frame.empty:
        issues.append(Issue("error", sheet, "Sheet has no data rows"))
        return frame

    for f in fields:
        if f.name not in frame.columns:
            continue
        column = frame[f.name]
        blank = column.isna() | (column.astype(str).str.strip() == "")
        if f.required:
            for row in np.flatnonzero(blank.to_numpy()):
                issues.append(Issue("error", sheet, "Required value is blank", row + 1, f.name))
        if f.dtype in ("int", "float"):
            numeric = pd.to_numeric(column, errors="coerce")
            for row in np.flatnonzero((numeric.isna() & ~blank).to_numpy()):
                issues.append(
                    Issue("error", sheet, f"Not a number: {column.iloc[row]!r}", row + 1, f.name)
                )
            if f.dtype == "int":
                fractional = numeric.notna() & (numeric % 1 != 0)
                for row in np.flatnonzero(fractional.to_numpy()):
                    issues.append(Issue("error", sheet, "Must be a whole number", row + 1, f.name))
            unit = f" {f.unit}" if f.unit else ""
            for bound, outside, limit in (
                (f.minimum, lambda b: numeric < b, "minimum"),
                (f.maximum, lambda b: numeric > b, "maximum"),
            ):
                if bound is None:
                    continue
                for row in np.flatnonzero(outside(bound).fillna(False).to_numpy()):
                    message = f"{numeric.iloc[row]} is outside the allowed {limit} {bound}{unit}"
                    issues.append(Issue("error", sheet, message, row + 1, f.name))
            frame[f.name] = numeric
        else:
            frame[f.name] = column.where(blank, column.astype(str).str.strip())
        if f.allowed:
            values = frame[f.name].astype("string")
            if f.dtype == "int":
                values = frame[f.name].map(lambda v: None if pd.isna(v) else str(int(v)))
            bad = values.notna() & ~values.isin(f.allowed)
            for row in np.flatnonzero(bad.fillna(False).to_numpy()):
                issues.append(
                    Issue(
                        "error",
                        sheet,
                        f"{values.iloc[row]!r} is not an allowed value {list(f.allowed)}",
                        row + 1,
                        f.name,
                    )
                )
    if key and all(column in frame.columns for column in key):
        duplicated = frame.duplicated(key, keep=False)
        for row in np.flatnonzero(duplicated.to_numpy()):
            issues.append(Issue("error", sheet, f"Duplicate key {key}", row + 1))
    return frame


# ------------------------------------------------------------- missing-value fill policy


def build_year_lookup(
    reference: pd.DataFrame, column: str, year_column: str, categorical: bool = False
) -> dict[int, Any]:
    """Per-year typical value from a material's own reference/training data.

    Mode for categoricals (``categorical=True``), median for numeric columns. This is
    the "most common, based on the year" policy: a year with e.g. unusually few series
    still gets a sensible value, and different years can genuinely differ (a climate
    driver's typical value in 2030 isn't its typical value in 2070).
    """
    if categorical:
        def _mode(s: pd.Series) -> Any:
            m = s.mode()
            return m.iat[0] if not m.empty else None

        return reference.groupby(year_column)[column].agg(_mode).to_dict()
    return reference.groupby(year_column)[column].median().to_dict()


def apply_fill(
    frame: pd.DataFrame,
    field_obj: Field,
    year_column: str,
    lookup: dict[int, Any],
    issues: list[Issue],
    sheet: str,
) -> pd.DataFrame:
    """Fill ``field_obj``'s missing/blank values from ``lookup`` (see ``build_year_lookup``).

    Only touches rows where the value is actually missing -- supplied values are never
    overwritten. Every filled row gets a warning ``Issue`` naming the value used, so the
    caller can show (or block on, if it wants to be stricter) which inputs were guessed
    rather than given. If a row's year isn't in ``lookup`` at all, falls back to the
    nearest year that is, so a request for an out-of-range year still gets *something*
    rather than silently staying blank.
    """
    if field_obj.name not in frame.columns:
        frame = frame.copy()
        frame[field_obj.name] = np.nan
    column = frame[field_obj.name]
    missing = column.isna() | (column.astype(str).str.strip() == "")
    if not missing.any() or not lookup:
        return frame
    years = sorted(lookup)

    def _value_for(year: Any) -> Any:
        if year in lookup:
            return lookup[year]
        nearest = min(years, key=lambda y: abs(y - year))
        return lookup[nearest]

    frame = frame.copy()
    filled = frame.loc[missing, year_column].map(_value_for)
    frame.loc[missing, field_obj.name] = filled
    for row in np.flatnonzero(missing.to_numpy()):
        value = frame.loc[frame.index[row], field_obj.name]
        issues.append(
            Issue(
                "warning",
                sheet,
                f"{field_obj.name} was not supplied; filled with the typical value for "
                f"that year ({value}) instead of a measured one",
                row + 1,
                field_obj.name,
            )
        )
    return frame
