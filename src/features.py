"""Step 2 — Feature engineering and label construction.

From raw customers + transactions we build a modelling table with one row per
customer. Features are the classic RFM trio (Recency, Frequency, Monetary)
plus tenure, basket statistics and channel mix — all computed *as of* the
snapshot date so there is no leakage from the future.

The churn label is simulated here: a customer churns when they make no purchase
in the 90 days following the snapshot. We draw that outcome from a logistic
model whose risk increases with recency (days since last order) and decreases
with frequency, monetary value and the customer's latent loyalty, plus noise.
This guarantees the label is genuinely predictable from the features without
being a deterministic function of them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import (
    CUSTOMERS_CSV,
    FEATURES_CSV,
    RANDOM_SEED,
    SNAPSHOT_DATE,
    TRANSACTIONS_CSV,
    ensure_dirs,
)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def build_features() -> pd.DataFrame:
    ensure_dirs()
    snapshot = pd.Timestamp(SNAPSHOT_DATE)

    customers = pd.read_csv(CUSTOMERS_CSV, parse_dates=["signup_date"])
    txns = pd.read_csv(TRANSACTIONS_CSV, parse_dates=["order_date"])

    # --- RFM and basket aggregates per customer ---------------------------
    grp = txns.groupby("customer_id")
    agg = grp.agg(
        frequency=("amount", "size"),
        monetary_total=("amount", "sum"),
        monetary_avg=("amount", "mean"),
        monetary_std=("amount", "std"),
        last_order=("order_date", "max"),
        first_order=("order_date", "min"),
        n_channels=("channel", "nunique"),
        n_categories=("category", "nunique"),
    )
    agg["recency_days"] = (snapshot - agg["last_order"]).dt.days
    agg["active_span_days"] = (agg["last_order"] - agg["first_order"]).dt.days
    agg = agg.drop(columns=["last_order", "first_order"])

    df = customers.merge(agg, on="customer_id", how="left")

    # Customers with zero transactions: fill sensible defaults.
    df["frequency"] = df["frequency"].fillna(0)
    df["monetary_total"] = df["monetary_total"].fillna(0.0)
    df["monetary_avg"] = df["monetary_avg"].fillna(0.0)
    df["monetary_std"] = df["monetary_std"].fillna(0.0)
    df["n_channels"] = df["n_channels"].fillna(0)
    df["n_categories"] = df["n_categories"].fillna(0)
    df["active_span_days"] = df["active_span_days"].fillna(0)
    # No purchase => maximally "stale" recency (tenure since signup).
    tenure_days = (snapshot - df["signup_date"]).dt.days
    df["recency_days"] = df["recency_days"].fillna(tenure_days)
    df["tenure_days"] = tenure_days

    # --- Simulate the churn label from a logistic risk model --------------
    rng = np.random.default_rng(RANDOM_SEED)

    def z(col: str) -> np.ndarray:
        v = df[col].to_numpy(dtype=float)
        return (v - v.mean()) / (v.std() + 1e-9)

    logit = (
        -0.4                          # base rate
        + 1.6 * z("recency_days")     # stale customers churn more
        - 0.9 * z("frequency")        # frequent buyers stick around
        - 0.5 * z("monetary_total")   # high spenders stick around
        - 2.0 * (df["_loyalty"].to_numpy() - 0.5)  # latent loyalty
    )
    churn_prob = _sigmoid(logit + rng.normal(0, 0.6, size=len(df)))
    df["churned"] = (rng.random(len(df)) < churn_prob).astype(int)

    # --- Drop latent / leakage columns ------------------------------------
    df = df.drop(columns=["_loyalty", "_basket_mean", "signup_date"])

    df.to_csv(FEATURES_CSV, index=False)
    print(f"Built feature table: {df.shape[0]:,} rows x {df.shape[1]} cols -> {FEATURES_CSV}")
    print(f"Churn rate: {df['churned'].mean():.1%}")
    return df


if __name__ == "__main__":
    build_features()
