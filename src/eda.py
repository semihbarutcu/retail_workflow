"""Step 3 — Exploratory data analysis.

Produces a handful of figures that a data scientist would actually look at
before modelling: the target balance, RFM distributions split by churn, and a
numeric-feature correlation heatmap. Figures are written to reports/figures/.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless backend — we only save files
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import FEATURES_CSV, FIGURES_DIR, TARGET, ensure_dirs


def _save(fig, name: str) -> None:
    path = FIGURES_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"  wrote {path}")


def run_eda() -> None:
    ensure_dirs()
    df = pd.read_csv(FEATURES_CSV)
    print("Running EDA...")

    # 1) Target balance -----------------------------------------------------
    fig, ax = plt.subplots(figsize=(4, 4))
    counts = df[TARGET].value_counts().sort_index()
    ax.bar(["retained (0)", "churned (1)"], counts.values, color=["#4c72b0", "#c44e52"])
    ax.set_title("Churn class balance")
    ax.set_ylabel("customers")
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom")
    _save(fig, "01_target_balance.png")

    # 2) RFM distributions by churn ----------------------------------------
    rfm = ["recency_days", "frequency", "monetary_total"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, col in zip(axes, rfm):
        for label, color in [(0, "#4c72b0"), (1, "#c44e52")]:
            vals = df.loc[df[TARGET] == label, col].clip(
                upper=df[col].quantile(0.99)
            )
            ax.hist(vals, bins=30, alpha=0.6, color=color,
                    label=("churned" if label else "retained"), density=True)
        ax.set_title(col)
        ax.legend()
    fig.suptitle("RFM distributions by churn status")
    _save(fig, "02_rfm_by_churn.png")

    # 3) Correlation heatmap of numeric features ---------------------------
    num = df.select_dtypes(include=[np.number]).drop(columns=["customer_id"])
    corr = num.corr()
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)))
    ax.set_yticks(range(len(corr)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=8)
    ax.set_yticklabels(corr.columns, fontsize=8)
    fig.colorbar(im, fraction=0.046, pad=0.04)
    ax.set_title("Numeric feature correlation")
    _save(fig, "03_correlation_heatmap.png")

    print("EDA complete.")


if __name__ == "__main__":
    run_eda()
