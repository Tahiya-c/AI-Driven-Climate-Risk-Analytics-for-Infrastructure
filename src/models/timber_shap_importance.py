"""SHAP-based feature importance, in place of permutation importance.

Uses shap's model-agnostic ``Explainer``, which picks the exact
``TreeExplainer`` for tree-based estimators (fast) and falls back to a
sampling-based explainer (e.g. permutation) for anything else, such as
the MLP pipeline -- slower, hence the capped sample size.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import shap

from src.timber.features import constant_within_pile_features


def compute_shap_importance(
    model: Any,
    frame: pd.DataFrame,
    feature_columns: list[str],
    sample_size: int = 100,
    random_state: int = 42,
) -> pd.DataFrame:
    """Mean |SHAP value| per feature, computed on a row sub-sample.

    Flags features that are constant within every asset_code (see
    ``constant_within_pile_features``): with only 2 piles in this
    dataset, a non-zero SHAP value on one of those columns reflects
    "which pile is this", not a real deterioration-over-time signal.
    """
    sample = frame.sample(min(sample_size, len(frame)), random_state=random_state)
    background = sample.loc[:, feature_columns]

    try:
        explainer = shap.Explainer(model, background)
    except TypeError:
        # Pipelines/estimators shap doesn't recognise directly (e.g. an
        # sklearn Pipeline wrapping MLPRegressor) still work through their
        # predict function.
        explainer = shap.Explainer(model.predict, background)

    shap_values = explainer(background)
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    constant_columns = set(constant_within_pile_features(frame, feature_columns))

    importance = pd.DataFrame(
        {
            "feature": feature_columns,
            "mean_abs_shap": mean_abs_shap,
            "constant_within_pile": [column in constant_columns for column in feature_columns],
        }
    ).sort_values("mean_abs_shap", ascending=False, ignore_index=True)
    return importance
