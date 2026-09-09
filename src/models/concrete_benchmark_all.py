"""
One benchmark for every concrete model -- same data, target, features, splits and
forecast mechanics. Run this to get the "which model wins" table.

    python -m src.models.concrete_benchmark_all
    python -m src.models.concrete_benchmark_all --models rnn,ridge,random_forest,persistence
    python -m src.models.concrete_benchmark_all --runs primary
    python -m src.models.concrete_benchmark_all --skip-rnn        # fast, no torch

Writes:
    results_concrete_benchmark.csv          one row per (run, held_out_group, model)
    results_concrete_benchmark_detail.csv   one row per (run, held_out_group, model, series)

See concrete_eval.py for the full spec. Headline number = the `primary` run:
train on every series' history up to 2026, forecast 2027-2070 autoregressively,
score against the real depths.
"""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from . import concrete_eval as ce
except ImportError:
    import concrete_eval as ce

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV = ROOT / "data" / "processed" / "concrete_kembla_v6_consolidated.csv"


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    p.add_argument("--models", default=",".join(ce.ALL_MODELS),
                   help=f"comma list from: {', '.join(ce.ALL_MODELS)}")
    p.add_argument("--runs", default="primary,scenario,asset")
    p.add_argument("--cutoff", type=int, default=ce.CUTOFF_YEAR)
    p.add_argument("--skip-rnn", action="store_true")
    p.add_argument("--out", type=Path, default=ROOT / "results_concrete_benchmark.csv")
    args = p.parse_args()

    if not args.csv.exists():
        raise SystemExit(
            f"{args.csv} not found. It should be in the repo now; pull main, "
            f"or regenerate with src/data/build_concrete_kembla_csv.py."
        )

    models = [m for m in args.models.split(",") if m]
    if args.skip_rnn:
        models = [m for m in models if m != "rnn"]
    runs = tuple(r for r in args.runs.split(",") if r)

    span = f"{args.cutoff + 1}-{ce.FINAL_YEAR}"
    print(f"data   {args.csv.name}")
    print(f"cutoff train <= {args.cutoff}   forecast {span}  (autoregressive)")
    print(f"models {', '.join(models)}")
    print(f"runs   {', '.join(runs)}\n")

    summary, detail, skipped = ce.run_benchmark(args.csv, models, runs, args.cutoff)
    if skipped:
        print(f"skipped (not installed): {', '.join(skipped)}\n")

    summary.to_csv(args.out, index=False)
    detail_path = args.out.with_name(args.out.stem + "_detail.csv")
    detail.to_csv(detail_path, index=False)

    for run in [r for r in ("primary", "scenario", "asset") if r in runs]:
        block = summary[summary["run"] == run]
        if run == "primary":
            table = block[["model", "n_series", "mae", "mae_std", "rmse", "r2", "final_mae"]]
            title = f"PRIMARY  (train <= {args.cutoff}, forecast {span}, all 49 series)"
        else:
            table = (
                block.groupby("model")
                .agg(folds=("held_out", "nunique"), mae=("mae", "mean"),
                     mae_std=("mae", "std"), rmse=("rmse", "mean"),
                     r2=("r2", "mean"), final_mae=("final_mae", "mean"))
                .reset_index()
            )
            title = f"ROBUSTNESS: leave-one-{run}-out (still 2026-forward)"
        print("=" * 78)
        print(title)
        print("=" * 78)
        print(table.round(4).to_string(index=False))
        print()

    print(f"wrote {args.out}")
    print(f"wrote {detail_path}")


if __name__ == "__main__":
    main()
