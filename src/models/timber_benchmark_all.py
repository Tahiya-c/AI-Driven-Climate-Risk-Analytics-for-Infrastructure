"""Compare timber models using the concrete_benchmark_all CLI convention.

python -m src.models.timber_benchmark_all --models linear,ridge,svr --runs primary
Benchmark models are not saved as deployment artifacts.
"""

import argparse
from pathlib import Path

from src.config import ROOT_DIR
from src.models.timber_eval import ALL_MODELS, CUTOFF_YEAR, DATA_PATH, run_benchmark


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DATA_PATH)
    parser.add_argument("--models", default=",".join(ALL_MODELS))
    parser.add_argument("--runs", default="primary,scenario,asset")
    parser.add_argument("--cutoff", type=int, default=CUTOFF_YEAR)
    parser.add_argument("--out", type=Path, default=ROOT_DIR / "results_timber_benchmark.csv")
    args = parser.parse_args()
    summary, detail, _ = run_benchmark(
        args.csv, args.models.split(","), tuple(args.runs.split(",")), args.cutoff,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out, index=False)
    detail.to_csv(args.out.with_name(args.out.stem + "_detail.csv"), index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
