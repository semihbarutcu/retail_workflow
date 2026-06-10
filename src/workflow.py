"""End-to-end orchestrator.

Runs the full pipeline in order:
    1. generate artificial data
    2. engineer features + build labels
    3. exploratory data analysis (figures)
    4. train + evaluate models (metrics + figures)

Usage:
    python src/workflow.py
"""
from __future__ import annotations

import time

import eda
import features
import generate_data
import train


def main() -> None:
    steps = [
        ("Generate artificial data", generate_data.generate),
        ("Engineer features & labels", features.build_features),
        ("Exploratory data analysis", eda.run_eda),
        ("Train & evaluate models", train.train),
    ]
    t0 = time.time()
    for i, (title, fn) in enumerate(steps, 1):
        print("\n" + "=" * 70)
        print(f"STEP {i}/{len(steps)}: {title}")
        print("=" * 70)
        fn()
    print("\n" + "=" * 70)
    print(f"Pipeline finished in {time.time() - t0:.1f}s")
    print("Artifacts: data/*.csv, reports/metrics.json, reports/figures/*.png")
    print("=" * 70)


if __name__ == "__main__":
    main()
