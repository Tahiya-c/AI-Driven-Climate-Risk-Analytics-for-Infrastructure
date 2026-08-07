import pandas as pd
import pytest

from src.data.loader import numeric_columns
from src.features.build_features import select_model_features


def test_numeric_columns_filters_text() -> None:
    frame = pd.DataFrame({"age": [2, 4], "port": ["A", "B"], "chloride": [0.2, 0.4]})
    assert numeric_columns(frame) == ["age", "chloride"]


def test_select_model_features_reports_missing_columns() -> None:
    with pytest.raises(ValueError, match="chloride"):
        select_model_features(pd.DataFrame({"age": [1]}), ["age", "chloride"])

