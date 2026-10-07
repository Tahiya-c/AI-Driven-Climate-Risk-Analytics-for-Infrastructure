"""Feature assembly for the timber deterioration surrogate model."""

from __future__ import annotations

import pandas as pd

TARGET_COLUMN = "remaining_diameter_mm"

# Numeric, already-engineered columns from the deterministic model output.
#
# sapwood_rate_mm_year and heartwood_rate_mm_year are deliberately EXCLUDED:
# together with age_years and initial_diameter_mm (both still included below),
# they let the target be reconstructed algebraically to within floating-point
# noise (max abs error 2.27e-13mm, verified against the full Kembla dataset --
# see docs/timber_kembla_to_sydney_eval.md, section 2). They are the deterministic
# engine's own internal rate variables, not upstream inputs, so they belong in
# LEAKY_COLUMNS alongside mean_deterioration_mm et al., not in the feature set.
#
# The k-factor names follow the workbook's own headers (Defect D9 fix, see
# D9_COLUMN_RENAMES in src/data/timber_data_prep.py). Before the fix this list
# held 8 of the 11 k-factor columns under shifted labels; it now holds all 11
# under their real names. sapwood_factor (k_SW) and salinity_factor (k_sal) are
# the two additions -- both are constant (25 and 1.0) for both piles today.
FEATURE_COLUMNS = [
    "age_years",
    "summer_sea_temp_c",
    "salinity",
    "water_factor",  # kwater = 0.1 * exp(0.13 * summer_sea_temp_c)
    "heartwood_factor",  # k_HW
    "sapwood_factor",  # k_SW
    "salinity_factor",  # k_sal
    "shelter_factor",  # k_shelter
    "protection_factor",  # k_protect
    "contact_factor",  # k_contact
    "knot_factor",  # k_knot
    "environment_factor",  # k_env = k_sal * k_shelter
    "construction_factor",  # k_const = k_protect * k_contact * k_knot
    "combined_factor",  # k0 = k_HW * k_env * k_const
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
# and comparison only, never as model features. sapwood_rate_mm_year and
# heartwood_rate_mm_year are included here (rather than in FEATURE_COLUMNS)
# for the same reason: they are the engine's own internal rate variables, and
# together with age_years/initial_diameter_mm they reconstruct the target
# almost exactly -- see the note above FEATURE_COLUMNS.
LEAKY_COLUMNS = [
    "failure_probability",
    "design_life_years",
    "mean_deterioration_mm",
    "design_deterioration_mm",
    "design_diameter_mm",
    "sapwood_rate_mm_year",
    "heartwood_rate_mm_year",
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
# thermal_dose: the one engineered feature the tuned MLP/GBM surrogates add on
# top of FEATURE_COLUMNS (see DeteriorationRateRegressor in
# src/models/timber_model.py, which calls add_thermal_dose itself so callers
# keep passing the plain FEATURE_COLUMNS matrix).
# ---------------------------------------------------------------------------

THERMAL_DOSE_COLUMN = "thermal_dose"


def add_thermal_dose(features: pd.DataFrame) -> pd.DataFrame:
    """Append ``thermal_dose = max(age_years, 0) * water_factor``.

    ``water_factor`` is the temperature-driven decay-rate multiplier
    (``0.1 * exp(0.13 * summer_sea_temp_c)`` in the AS 5604-style engine), so
    this product is "years in service x how hot those years were" -- a
    degree-days style exposure term. It is built only from two columns already
    in FEATURE_COLUMNS, so it adds no information a model couldn't in
    principle learn; it just hands the age x temperature interaction to the
    model directly instead of making a tree ensemble or a small MLP discover a
    multiplicative interaction from 800 rows. Operates on a feature matrix
    (not the full frame) and works per row, so it is safe on any subset.
    """
    missing = sorted({"age_years", "water_factor"} - set(features.columns))
    if missing:
        raise ValueError(f"Missing columns required for thermal_dose: {', '.join(missing)}")
    features = features.copy()
    features[THERMAL_DOSE_COLUMN] = features["age_years"].clip(lower=0) * features["water_factor"]
    return features


# ---------------------------------------------------------------------------
# decay_dose: an engineered alternative to feeding age_years and the rate
# factors in as separate linear inputs.
# ---------------------------------------------------------------------------

DECAY_DOSE_COLUMN = "decay_dose"
# Before the D9 fix "salinity_factor" here silently read k_shelter; it now reads the
# real salinity coefficient k_sal, matching what this feature was meant to use.
DECAY_DOSE_SOURCE_COLUMNS = ["salinity_factor", "summer_sea_temp_c", "heartwood_factor"]

# A "trajectory" here means one continuous timeline for one pile under one
# climate/emissions combination -- the same grain timber_scenarios.py already
# treats as a single line. Grouping by asset_code alone (ignoring
# scenario_label/emissions_pathway) would sum dose across unrelated
# parallel climate futures for the same calendar year, which has no
# physical meaning.
DECAY_DOSE_TRAJECTORY_KEY = ["asset_code", "scenario_label", "emissions_pathway"]

DECAY_DOSE_FEATURE_COLUMNS = [
    column
    for column in FEATURE_COLUMNS
    if column not in {"age_years", "summer_sea_temp_c", "salinity_factor", "heartwood_factor"}
] + [DECAY_DOSE_COLUMN]


def add_decay_dose(frame: pd.DataFrame) -> pd.DataFrame:
    """Add a cumulative environmental-stress-over-time feature.

    ``decay_dose`` is the cumulative sum, ordered by ``age_years`` within
    each ``(asset_code, scenario_label, emissions_pathway)`` trajectory,
    of ``salinity_factor * summer_sea_temp_c * heartwood_factor``.
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
        frame["salinity_factor"] * frame["summer_sea_temp_c"] * frame["heartwood_factor"]
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
