"""
scripts/run_pipeline.py
========================
Command-line entry point.

Examples
--------
    # one dataset, real data if present under data/raw/<name>/, else synthetic
    python scripts/run_pipeline.py --dataset cicids2017

    # every dataset that has a split strategy defined (skips cve_nvd)
    python scripts/run_pipeline.py --dataset all

    # fast wiring check, ~1 minute, synthetic data, tiny grids
    python scripts/run_pipeline.py --dataset cicids2017 --quick --force-synthetic
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import DATASETS
from src.pipeline import run_all_datasets, run_full_pipeline


def main():
    parser = argparse.ArgumentParser(description="Run the XAI cyber-risk pipeline.")
    parser.add_argument(
        "--dataset",
        default="cicids2017",
        choices=list(DATASETS.keys()) + ["all"],
        help="Which data-layer source to run (or 'all').",
    )
    parser.add_argument("--quick", action="store_true", help="1-point hyperparameter grids, 2-fold CV.")
    parser.add_argument(
        "--force-synthetic", action="store_true",
        help="Ignore any real files under data/raw/ and use the synthetic stand-in.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--n-rows", type=int, default=20000,
        help="Row count for the synthetic fallback (ignored when real files are found).",
    )
    args = parser.parse_args()

    if args.dataset == "all":
        table = run_all_datasets(
            quick_mode=args.quick, force_synthetic=args.force_synthetic, seed=args.seed,
            n_synthetic_rows=args.n_rows,
        )
    else:
        result = run_full_pipeline(
            args.dataset, quick_mode=args.quick, force_synthetic=args.force_synthetic,
            seed=args.seed, n_synthetic_rows=args.n_rows,
        )
        table = result["comparison_table"]

    print("\n=== Comparison table ===")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
