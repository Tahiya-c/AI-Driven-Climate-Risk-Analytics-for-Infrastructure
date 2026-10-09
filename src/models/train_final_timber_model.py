"""Train deployment timber models on all history, without test scores.

Counterpart to train_final_model.py for concrete.
python -m src.models.train_final_timber_model --models ridge,svr
Accuracy comes from timber_benchmark_all, not this all-data fit.
"""

import argparse
import json
from pathlib import Path

import joblib

from src.config import MODEL_DIR
from src.models.timber_eval import (
    ALL_MODELS,
    DATA_PATH,
    DEPTH_COLUMN,
    FEATURE_COLUMNS,
    TARGET,
    fit_model,
    load_series,
    make_model,
)


def train_final(csv_path, out_dir, model_names=ALL_MODELS):
    series = load_series(csv_path)
    ends = {int(group.year.max()) for group in series}
    if len(ends) != 1:
        raise ValueError("Final training requires a common last history year across series.")
    last_year = ends.pop()
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    for name in model_names:
        pipeline = fit_model(make_model(name), series, last_year)
        metadata = {
            "artifact_version": 2, "model_type": name, "features": FEATURE_COLUMNS,
            "target": TARGET, "depth_column": DEPTH_COLUMN,
            "metric": "deterioration_depth", "unit": "mm", "cutoff_year": last_year,
            "training_scope": "all_available_history", "requires_future_inputs": True,
            "train_rows": sum(len(group) - 2 for group in series),
            "parameters": pipeline["model"].get_params(),
            "evaluation": "No held-out score. Use timber_benchmark_all for accuracy.",
        }
        path = output / f"timber_depth_{name}.joblib"
        joblib.dump({"model": pipeline, **metadata}, path)
        path.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        paths.append(path)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="?", type=Path, default=DATA_PATH)
    parser.add_argument("--out", type=Path, default=MODEL_DIR)
    parser.add_argument("--models", default=",".join(ALL_MODELS))
    args = parser.parse_args()
    for path in train_final(args.csv, args.out, args.models.split(",")):
        print(f"Saved {path} (all-history fit; no held-out accuracy)")


if __name__ == "__main__":
    main()
