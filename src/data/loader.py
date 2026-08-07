"""Dataset loading helpers."""

from pathlib import Path

import pandas as pd


def load_csv(path: str | Path) -> pd.DataFrame:
    """Load a concrete-pile CSV and reject an empty dataset."""
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError("The selected CSV contains no records.")
    return frame


def numeric_columns(frame: pd.DataFrame) -> list[str]:
    """Return columns suitable for numeric dashboard parameters."""
    return frame.select_dtypes(include="number").columns.tolist()

