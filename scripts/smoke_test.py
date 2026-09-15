"""
scripts/smoke_test.py
======================
Runs every layer of the pipeline on a small synthetic dataset with 1-point
hyperparameter grids and 2-fold CV. Takes well under a minute. This is a
wiring check, not a results run -- it exists so you can confirm the repo
works right after cloning it, before pointing it at real data or spending
GPU/CPU time on the full grids in Table III.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline import run_full_pipeline


def main():
    print("Running smoke test on synthetic CICIDS2017-shaped data (quick_mode=True)...\n")
    result = run_full_pipeline("cicids2017", quick_mode=True, force_synthetic=True, seed=42)

    assert len(result["interpretable_models"]) == 3, "expected 3 interpretable models"
    assert len(result["high_performance_models"]) == 3, "expected 3 high-performance models"
    assert not result["comparison_table"].empty, "comparison table is empty"

    print("\n=== Comparison table (synthetic, quick mode) ===")
    print(result["comparison_table"].to_string(index=False))
    print(f"\nArtifacts written to: {result['out_dir']}")
    print("\nSmoke test passed.")


if __name__ == "__main__":
    main()
