"""Feature assembly for the timber deterioration surrogate model."""

from __future__ import annotations

import pandas as pd

TARGET_COLUMN = "remaining_diameter_mm"

# Numeric, already-engineered columns from the deterministic model output.
FEATURE_COLUMNS = [
    "age_years",
    "summer_sea_temp_c",
    "salinity",
    "water_factor",
    "timber_exposure_factor",
    "salinity_factor",
    "shelter_factor",
    "protection_factor",
    "model_contact_factor",
    "model_knot_factor",
    "environment_factor",
    "construction_factor",
    "sapwood_rate_mm_year",
    "heartwood_rate_mm_year",
    "initial_diameter_mm",
]

CATEGORICAL_COLUMNS = [
    "timber_type",
    "exposure",
    "heartwood_class",
    "contact",
    "knot_presence",
    "maintenance_measure",
]

# These are OUTPUTS of the deterministic model this project replicates.
# Using them as inputs would leak the answer -- keep them for validation
# and comparison only, never as model features.
LEAKY_COLUMNS = [
    "failure_probability",
    "replacement_time_years",
    "mean_deterioration_mm",
    "design_deterioration_mm",
    "design_diameter_mm",
]


def build_feature_matrix(
    frame: pd.DataFrame,
    feature_columns: list[str] | None = None,
    include_categorical: bool = False,
) -> pd.DataFrame:
    """Assemble the model input matrix.

    ``feature_columns`` defaults to ``FEATURE_COLUMNS``; pass
    ``DECAY_DOSE_FEATURE_COLUMNS`` (after calling ``add_decay_dose``) to
    use the engineered dose feature instead. ``include_categorical``
    defaults to False: with only two piles in the dataset,
    timber_type/exposure/heartwood_class/contact/knot_presence/
    maintenance_measure are each constant per asset, so one-hot encoding
    them mostly just re-encodes "which pile is this" -- the numeric
    ``*_factor`` columns already carry that information through the
    deterministic model's own physics.
    """
    feature_columns = feature_columns if feature_columns is not None else FEATURE_COLUMNS
    missing = sorted(set(feature_columns) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required feature columns: {', '.join(missing)}")

    matrix = frame.loc[:, feature_columns].copy()
    if include_categorical:
        dummies = pd.get_dummies(frame.loc[:, CATEGORICAL_COLUMNS], drop_first=True)
        matrix = pd.concat([matrix, dummies], axis=1)
    return matrix


# ---------------------------------------------------------------------------
# decay_dose: an engineered alternative to feeding age_years and the rate
# factors in as separate linear inputs.
# ---------------------------------------------------------------------------

DECAY_DOSE_COLUMN = "decay_dose"
DECAY_DOSE_SOURCE_COLUMNS = ["salinity_factor", "summer_sea_temp_c", "timber_exposure_factor"]

# A "trajectory" here means one continuous timeline for one pile under one
# climate/emissions combination -- the same grain scenarios.py already
# treats as a single line. Grouping by asset_code alone (ignoring
# scenario_label/emissions_pathway) would sum dose across unrelated
# parallel climate futures for the same calendar year, which has no
# physical meaning.
DECAY_DOSE_TRAJECTORY_KEY = ["asset_code", "scenario_label", "emissions_pathway"]

DECAY_DOSE_FEATURE_COLUMNS = [
    column
    for column in FEATURE_COLUMNS
    if column not in {"age_years", "summer_sea_temp_c", "salinity_factor", "timber_exposure_factor"}
] + [DECAY_DOSE_COLUMN]


def add_decay_dose(frame: pd.DataFrame) -> pd.DataFrame:
    """Add a cumulative environmental-stress-over-time feature.

    ``decay_dose`` is the cumulative sum, ordered by ``age_years`` within
    each ``(asset_code, scenario_label, emissions_pathway)`` trajectory,
    of ``salinity_factor * summer_sea_temp_c * timber_exposure_factor``.
    It bundles age, salinity, sea temperature, and exposure into a single
    non-linear "cumulative dose" term, as an alternative to
    ``DECAY_DOSE_FEATURE_COLUMNS`` feeding those in as separate linear
    inputs (see ``FEATURE_COLUMNS`` for the standard set).
    """
    required = [*DECAY_DOSE_SOURCE_COLUMNS, *DECAY_DOSE_TRAJECTORY_KEY, "age_years"]
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing columns required for decay_dose: {', '.join(missing)}")

    frame = frame.copy()
    dose_term = (
        frame["salinity_factor"] * frame["summer_sea_temp_c"] * frame["timber_exposure_factor"]
    )
    sort_columns = [*DECAY_DOSE_TRAJECTORY_KEY, "age_years"]
    frame = frame.assign(_dose_term=dose_term).sort_values(sort_columns)
    frame[DECAY_DOSE_COLUMN] = frame.groupby(DECAY_DOSE_TRAJECTORY_KEY)["_dose_term"].cumsum()
    return frame.drop(columns="_dose_term")


def constant_within_pile_features(frame: pd.DataFrame, feature_columns: list[str]) -> list[str]:
    """Feature columns that never vary within a single asset_code's own rows.

    Such columns are confounded with pile identity rather than carrying
    real deterioration-over-time signal: with only 2 piles in this
    dataset, a column that is constant within a pile's own rows can only
    ever help distinguish "which pile is this", not describe how that
    pile's condition changes.
    """
    constant_columns = []
    for column in feature_columns:
        if column not in frame.columns:
            continue
        distinct_within_pile = frame.groupby("asset_code")[column].nunique()
        if (distinct_within_pile <= 1).all():
            constant_columns.append(column)
    return constant_columns
