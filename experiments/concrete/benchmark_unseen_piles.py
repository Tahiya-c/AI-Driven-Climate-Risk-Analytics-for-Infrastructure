from pathlib import Path

import numpy as np
import pandas as pd

from catboost import CatBoostRegressor, Pool

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


# ---------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------

cwd = Path.cwd()
project_root = cwd if (cwd / "data").exists() else cwd.parents[1]

csv_path = (
    project_root
    / "data"
    / "processed"
    / "concrete_kembla_v6_consolidated.csv"
)

df = pd.read_csv(csv_path)

target_col = "carb_depth_increment_mm"


# ---------------------------------------------------------
# 2. FEATURES
# ---------------------------------------------------------

drop_cols = [
    "source_row",
    "description",
    "protection",
    "rebar_diameter_mm",
    "carb_depth_xc_mm",
    "carb_initiation_prob",
    "carb_icorr_ua_cm2",
    "chloride_c_at_xc_kgm3",
    "chloride_c_at_cover_kgm3",
    "chloride_initiation_prob",
    "chloride_icorr_ua_cm2",
    "combined_initiation_prob",
    target_col,
]

cat_features = [
    "zone",
    "exposure_class",
    "carbonation_class",
    "chloride_class",
    "emissions_pathway",
    "climate_model",
    "scenario_label",
]

feature_cols = [
    col
    for col in df.columns
    if col not in drop_cols
    and col != "asset_code"
]

numeric_features = [
    col
    for col in feature_cols
    if col not in cat_features
]


for col in cat_features:
    df[col] = df[col].astype(str)


# ---------------------------------------------------------
# 3. PREPROCESSOR
# ---------------------------------------------------------

def make_preprocessor():

    return ColumnTransformer(
        transformers=[
            (
                "num",
                MinMaxScaler(),
                numeric_features,
            ),
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                cat_features,
            ),
        ]
    )


# ---------------------------------------------------------
# 4. LEAVE-ONE-PILE-OUT TEST
# ---------------------------------------------------------

all_results = []

assets = sorted(df["asset_code"].unique())

print("\nAssets:", assets)


for test_asset in assets:

    print("\n" + "=" * 70)
    print(f"TESTING ON UNSEEN ASSET: {test_asset}")
    print("=" * 70)

    # Every other pile is training data
    train_df = df[
        df["asset_code"] != test_asset
    ].copy()

    # This pile is completely unseen
    test_df = df[
        df["asset_code"] == test_asset
    ].copy()

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]

    X_test = test_df[feature_cols]
    y_test = test_df[target_col]


    # -----------------------------------------------------
    # MEAN BASELINE
    # -----------------------------------------------------

    # Simple benchmark:
    # predict the mean deterioration rate from training data
    baseline_preds = np.full(
        len(y_test),
        y_train.mean()
    )


    # -----------------------------------------------------
    # RIDGE REGRESSION
    # -----------------------------------------------------

    ridge = Pipeline(
        steps=[
            (
                "preprocessor",
                make_preprocessor(),
            ),
            (
                "model",
                Ridge(alpha=1.0),
            ),
        ]
    )

    ridge.fit(
        X_train,
        y_train
    )

    ridge_preds = ridge.predict(
        X_test
    )


    # -----------------------------------------------------
    # RANDOM FOREST
    # -----------------------------------------------------

    rf = Pipeline(
        steps=[
            (
                "preprocessor",
                make_preprocessor(),
            ),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=200,
                    random_state=42,
                ),
            ),
        ]
    )

    rf.fit(
        X_train,
        y_train
    )

    rf_preds = rf.predict(
        X_test
    )


    # -----------------------------------------------------
    # CATBOOST
    # -----------------------------------------------------

    train_pool = Pool(
        X_train,
        y_train,
        cat_features=cat_features,
    )

    test_pool = Pool(
        X_test,
        cat_features=cat_features,
    )

    catboost = CatBoostRegressor(
        iterations=600,
        learning_rate=0.04,
        depth=6,
        loss_function="RMSE",
        random_seed=42,
        verbose=0,
    )

    catboost.fit(
        train_pool
    )

    catboost_preds = catboost.predict(
        test_pool
    )


    # -----------------------------------------------------
    # COMPARE MODELS
    # -----------------------------------------------------

    models = {
        "Mean Baseline": baseline_preds,
        "Ridge Regression": ridge_preds,
        "Random Forest": rf_preds,
        "CatBoost": catboost_preds,
    }


    # -----------------------------------------------------
    # EVALUATION
    # -----------------------------------------------------

    for model_name, predictions in models.items():

        r2 = r2_score(
            y_test,
            predictions,
        )

        rmse = root_mean_squared_error(
            y_test,
            predictions,
        )

        mae = mean_absolute_error(
            y_test,
            predictions,
        )

        # -------------------------------------------------
        # NRMSE
        # -------------------------------------------------

        actual_range = (
            y_test.max()
            - y_test.min()
        )

        if actual_range > 0:
            nrmse = rmse / actual_range
        else:
            nrmse = np.nan


        all_results.append(
            {
                "Test Pile": test_asset,
                "Model": model_name,
                "R2": r2,
                "RMSE": rmse,
                "MAE": mae,
                "NRMSE": nrmse,
            }
        )


        print(
            f"{model_name:20s} "
            f"R2={r2:8.4f} "
            f"RMSE={rmse:8.4f} "
            f"MAE={mae:8.4f} "
            f"NRMSE={nrmse:8.4f}"
        )


# ---------------------------------------------------------
# 5. FULL RESULTS
# ---------------------------------------------------------

results_df = pd.DataFrame(
    all_results
)

print("\n")
print("=" * 95)
print("RESULTS FOR ALL UNSEEN PILES")
print("=" * 95)

print(
    results_df.to_string(
        index=False
    )
)


# ---------------------------------------------------------
# 6. AVERAGE PERFORMANCE
# ---------------------------------------------------------

summary = (
    results_df
    .groupby("Model")
    .agg(
        Mean_R2=(
            "R2",
            "mean"
        ),
        Std_R2=(
            "R2",
            "std"
        ),
        Mean_RMSE=(
            "RMSE",
            "mean"
        ),
        Mean_MAE=(
            "MAE",
            "mean"
        ),
        Mean_NRMSE=(
            "NRMSE",
            "mean"
        ),
    )
    .reset_index()
)


print("\n")
print("=" * 95)
print("AVERAGE UNSEEN-PILE PERFORMANCE")
print("=" * 95)

print(
    summary.to_string(
        index=False
    )
)


# ---------------------------------------------------------
# 7. SAVE RESULTS
# ---------------------------------------------------------

artifacts_dir = (
    project_root
    / "artifacts"
)

artifacts_dir.mkdir(
    exist_ok=True
)


results_path = (
    artifacts_dir
    / "unseen_pile_results.csv"
)

summary_path = (
    artifacts_dir
    / "unseen_pile_summary.csv"
)


results_df.to_csv(
    results_path,
    index=False,
)

summary.to_csv(
    summary_path,
    index=False,
)


print("\nResults saved:")

print(
    f" - {results_path}"
)

print(
    f" - {summary_path}"
)