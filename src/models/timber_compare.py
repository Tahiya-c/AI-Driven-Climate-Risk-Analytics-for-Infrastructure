"""Compare surrogate model architectures for the timber deterioration target.

    python -m src.models.timber_compare --model gbm
    python -m src.models.timber_compare --model mlp --target distance --features decay_dose
    python -m src.models.timber_compare --model residual
    python -m src.models.timber_compare --model gbm --quantile

Every mode reuses ``time_based_split``/``pile_based_split`` from
``src/models/timber_model.py`` unchanged, so results here are directly
comparable to the MLP baseline documented in
``docs/timber_model_results.md``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.models.timber_baselines import naive_diameter_baseline
from src.data.timber_data_prep import DATA_PATH, load_timber_data
from src.features.timber_features import (
    DECAY_DOSE_FEATURE_COLUMNS,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    add_decay_dose,
    build_feature_matrix,
)
from src.models.timber_model import pile_based_split, time_based_split, train
from src.models.timber_quantile import pinball_loss, replacement_year_bands, train_quantile_models
from src.models.timber_reporting import add_predictions, error_by_age, error_by_pile, error_by_scenario
from src.models.timber_residual import train_residual_vs_direct
from src.models.timber_shap_importance import compute_shap_importance
from src.features.timber_targets import DISTANCE_TARGET, add_distance_to_threshold

FEATURE_SETS = {"standard": FEATURE_COLUMNS, "decay_dose": DECAY_DOSE_FEATURE_COLUMNS}


@dataclass
class ComparisonReport:
    model_kind: str
    target_column: str
    feature_set: str
    split_strategy: str
    headline: pd.DataFrame
    by_scenario: pd.DataFrame
    by_pile: pd.DataFrame
    by_age: pd.DataFrame
    pile_split_mae: float
    pile_split_r2: float
    shap_importance: pd.DataFrame | None = None


def _resolve_frame_and_features(
    frame: pd.DataFrame, target: str, feature_set: str
) -> tuple[pd.DataFrame, str, list[str]]:
    if feature_set not in FEATURE_SETS:
        raise ValueError(
            f"Unknown feature set: {feature_set!r} (expected one of {list(FEATURE_SETS)})"
        )
    feature_columns = FEATURE_SETS[feature_set]
    if feature_set == "decay_dose":
        frame = add_decay_dose(frame)

    if target == "diameter":
        target_column = TARGET_COLUMN
    elif target == "distance":
        frame = add_distance_to_threshold(frame)
        target_column = DISTANCE_TARGET
    else:
        raise ValueError(f"Unknown target: {target!r} (expected 'diameter' or 'distance')")

    return frame, target_column, feature_columns


def _test_split(frame: pd.DataFrame, split_strategy: str) -> tuple[pd.Index, pd.Index]:
    if split_strategy == "time":
        return time_based_split(frame)
    if split_strategy == "pile":
        return pile_based_split(frame)
    raise ValueError(f"Unknown split_strategy: {split_strategy!r}")


def _naive_prediction(frame: pd.DataFrame, index: pd.Index, target_column: str) -> pd.Series:
    prediction = naive_diameter_baseline(frame.loc[index])
    if target_column == DISTANCE_TARGET:
        prediction = prediction - frame.loc[index, "replacement_diameter_mm"]
    return prediction


def compare_all_models(
    frame: pd.DataFrame,
    target: str = "diameter",
    feature_set: str = "standard",
    split_strategy: str = "time",
) -> pd.DataFrame:
    """MLP vs GBM vs a trivial linear baseline, scored on one shared test split."""
    frame, target_column, feature_columns = _resolve_frame_and_features(frame, target, feature_set)
    _train_idx, test_idx = _test_split(frame, split_strategy)
    truth = frame.loc[test_idx, target_column]

    rows = []
    for model_kind in ("mlp", "gbm"):
        result = train(
            frame,
            split_strategy=split_strategy,
            model_kind=model_kind,
            target_column=target_column,
            feature_columns=feature_columns,
        )
        rows.append(
            {
                "model": model_kind,
                "mae": result.test_mae,
                "rmse": result.test_rmse,
                "r2": result.test_r2,
                "n_test": result.n_test,
            }
        )

    naive_prediction = _naive_prediction(frame, test_idx, target_column)
    rows.append(
        {
            "model": "naive (initial_diameter - mean_deterioration)",
            "mae": mean_absolute_error(truth, naive_prediction),
            "rmse": mean_squared_error(truth, naive_prediction) ** 0.5,
            "r2": r2_score(truth, naive_prediction),
            "n_test": len(test_idx),
        }
    )
    return pd.DataFrame(rows)


def run_comparison(
    frame: pd.DataFrame,
    model_kind: str = "gbm",
    target: str = "diameter",
    feature_set: str = "standard",
    split_strategy: str = "time",
    compute_shap: bool = True,
    shap_sample: int | None = None,
) -> ComparisonReport:
    """Deep-dive report for a single model: headline metrics, error breakdowns,
    the leave-one-pile-out diagnostic, and SHAP feature importance."""
    frame, target_column, feature_columns = _resolve_frame_and_features(frame, target, feature_set)

    result = train(
        frame,
        split_strategy=split_strategy,
        model_kind=model_kind,
        target_column=target_column,
        feature_columns=feature_columns,
    )

    features = build_feature_matrix(frame, feature_columns=feature_columns)
    _train_idx, test_idx = _test_split(frame, split_strategy)
    predictions = result.pipeline.predict(features.loc[test_idx])
    scored = add_predictions(frame, test_idx, predictions, target_column)

    headline = pd.DataFrame(
        [
            {
                "split": "train",
                "mae": result.train_mae,
                "rmse": result.train_rmse,
                "r2": result.train_r2,
            },
            {
                "split": "test",
                "mae": result.test_mae,
                "rmse": result.test_rmse,
                "r2": result.test_r2,
            },
        ]
    )

    pile_result = train(
        frame,
        split_strategy="pile",
        model_kind=model_kind,
        target_column=target_column,
        feature_columns=feature_columns,
    )

    shap_table = None
    if compute_shap:
        default_sample_size = 150 if model_kind == "gbm" else 30
        sample_size = shap_sample if shap_sample is not None else default_sample_size
        shap_table = compute_shap_importance(
            result.pipeline, frame.loc[test_idx], feature_columns, sample_size=sample_size
        )

    return ComparisonReport(
        model_kind=model_kind,
        target_column=target_column,
        feature_set=feature_set,
        split_strategy=split_strategy,
        headline=headline,
        by_scenario=error_by_scenario(scored),
        by_pile=error_by_pile(scored),
        by_age=error_by_age(scored),
        pile_split_mae=pile_result.test_mae,
        pile_split_r2=pile_result.test_r2,
        shap_importance=shap_table,
    )


def _print_report(report: ComparisonReport) -> None:
    print(
        f"\n=== {report.model_kind.upper()} | target={report.target_column} | "
        f"features={report.feature_set} | split={report.split_strategy} ==="
    )
    print(report.headline.to_string(index=False))
    print("\n-- error by scenario_label (test) --")
    print(report.by_scenario)
    print("\n-- error by asset_code (test) --")
    print(report.by_pile)
    print("\n-- error by age bucket (test) --")
    print(report.by_age)
    print(
        f"\n-- leave-one-pile-out diagnostic -- "
        f"MAE={report.pile_split_mae:.2f}  R2={report.pile_split_r2:.3f}"
    )
    if report.shap_importance is not None:
        print("\n-- SHAP feature importance (mean |value|, test sample) --")
        print(report.shap_importance.to_string(index=False))
        flagged = report.shap_importance.loc[
            report.shap_importance["constant_within_pile"], "feature"
        ].tolist()
        if flagged:
            print(
                "\nNOTE: constant-within-pile features (confounded with pile "
                f"identity, not real trend signal): {', '.join(flagged)}"
            )


def _run_residual_mode(frame: pd.DataFrame, model_kind: str, split_strategy: str) -> None:
    result = train_residual_vs_direct(frame, model_kind=model_kind, split_strategy=split_strategy)
    print(
        f"\n=== Residual-correction ({model_kind}) vs direct surrogate, "
        f"physically-valid test rows only (n={result.n_valid_test}) ==="
    )
    print(
        pd.DataFrame(
            [
                {
                    "approach": "residual-correction",
                    "mae": result.residual_test_mae,
                    "rmse": result.residual_test_rmse,
                    "r2": result.residual_test_r2,
                },
                {
                    "approach": "direct surrogate",
                    "mae": result.direct_test_mae,
                    "rmse": result.direct_test_rmse,
                    "r2": result.direct_test_r2,
                },
            ]
        ).to_string(index=False)
    )


def _run_quantile_mode(frame: pd.DataFrame, feature_set: str, split_strategy: str) -> None:
    feature_columns = FEATURE_SETS[feature_set]
    if feature_set == "decay_dose":
        frame = add_decay_dose(frame)

    quantile_models = train_quantile_models(
        frame, split_strategy=split_strategy, feature_columns=feature_columns
    )
    features = build_feature_matrix(frame, feature_columns=feature_columns)
    _train_idx, test_idx = _test_split(frame, split_strategy)

    print("\n-- quantile pinball loss (test) --")
    for alpha, model in sorted(quantile_models.models.items()):
        truth = frame.loc[test_idx, TARGET_COLUMN].to_numpy()
        prediction = model.predict(features.loc[test_idx])
        loss = pinball_loss(truth, prediction, alpha)
        print(f"  alpha={alpha:.1f}  pinball_loss={loss:.2f}")

    print("\n-- replacement-year uncertainty band, per pile --")
    for pile in sorted(frame["asset_code"].unique()):
        print(f"\n{pile}:")
        print(replacement_year_bands(frame, quantile_models, pile).to_string())


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", choices=["mlp", "gbm", "residual"], default="gbm")
    parser.add_argument("--target", choices=["diameter", "distance"], default="diameter")
    parser.add_argument("--features", choices=list(FEATURE_SETS), default="standard")
    parser.add_argument("--split", choices=["time", "pile"], default="time")
    parser.add_argument(
        "--no-shap", action="store_true", help="skip SHAP importance (slow for --model mlp)"
    )
    parser.add_argument("--shap-sample", type=int, default=None)
    parser.add_argument(
        "--quantile", action="store_true", help="also fit 10/50/90th percentile GBM models"
    )
    parser.add_argument(
        "--residual-base",
        choices=["mlp", "gbm"],
        default="gbm",
        help="base model for --model residual",
    )
    args = parser.parse_args()

    frame = load_timber_data(DATA_PATH)

    print("=== MLP vs GBM vs naive baseline (shared test split) ===")
    print(
        compare_all_models(
            frame, target=args.target, feature_set=args.features, split_strategy=args.split
        ).to_string(index=False)
    )

    if args.model == "residual":
        _run_residual_mode(frame, model_kind=args.residual_base, split_strategy=args.split)
        return

    report = run_comparison(
        frame,
        model_kind=args.model,
        target=args.target,
        feature_set=args.features,
        split_strategy=args.split,
        compute_shap=not args.no_shap,
        shap_sample=args.shap_sample,
    )
    _print_report(report)

    if args.quantile:
        if args.model != "gbm":
            print(
                "\n--quantile requires --model gbm (LightGBM/XGBoost quantile objective); "
                "skipping."
            )
        else:
            _run_quantile_mode(frame, feature_set=args.features, split_strategy=args.split)


if __name__ == "__main__":
    main()
