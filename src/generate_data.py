"""Step 1 — Generate artificial retail data.

We simulate an online retailer with a population of customers, each belonging to
a latent behavioural segment, and a stream of transactions. The simulation is
designed so that churn is *learnable but not trivial*: it depends on recency,
frequency, monetary value, tenure and engagement, all with noise.

Outputs
-------
data/customers.csv      One row per customer (demographics & signup info).
data/transactions.csv   One row per purchase.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import (
    CUSTOMERS_CSV,
    N_CUSTOMERS,
    RANDOM_SEED,
    SNAPSHOT_DATE,
    START_DATE,
    TRANSACTIONS_CSV,
    ensure_dirs,
)

# Latent segments: each has a different base purchase rate, basket size and
# loyalty. "loyalty" lowers the chance of going quiet after the snapshot.
SEGMENTS = {
    "bargain_hunter": {"weight": 0.35, "rate_per_month": 0.6, "basket_mean": 25, "loyalty": 0.35},
    "regular":        {"weight": 0.40, "rate_per_month": 1.5, "basket_mean": 55, "loyalty": 0.60},
    "loyal_vip":      {"weight": 0.15, "rate_per_month": 3.2, "basket_mean": 120, "loyalty": 0.85},
    "one_off":        {"weight": 0.10, "rate_per_month": 0.2, "basket_mean": 40, "loyalty": 0.10},
}

CHANNELS = ["web", "mobile_app", "in_store"]
CATEGORIES = ["electronics", "home", "apparel", "grocery", "beauty"]


def _simulate_customer(rng: np.random.Generator, cid: int, start, snapshot):
    """Simulate one customer's profile and their transaction list."""
    seg_names = list(SEGMENTS)
    seg_weights = np.array([SEGMENTS[s]["weight"] for s in seg_names])
    seg = rng.choice(seg_names, p=seg_weights / seg_weights.sum())
    params = SEGMENTS[seg]

    # Signup occurs somewhere in the observation window (uniformly).
    total_days = (snapshot - start).days
    signup_offset = int(rng.integers(0, total_days))
    signup_date = start + pd.Timedelta(days=signup_offset)
    active_days = (snapshot - signup_date).days
    active_months = max(active_days / 30.0, 0.1)

    age = int(np.clip(rng.normal(40, 13), 18, 85))
    preferred_channel = rng.choice(CHANNELS, p=[0.45, 0.40, 0.15])

    # Number of purchases ~ Poisson(rate * tenure), with per-customer noise.
    personal_rate = params["rate_per_month"] * rng.lognormal(0, 0.35)
    n_purchases = rng.poisson(personal_rate * active_months)

    customer = {
        "customer_id": cid,
        "segment": seg,            # kept for EDA/inspection, dropped before modelling
        "age": age,
        "preferred_channel": preferred_channel,
        "signup_date": signup_date.date().isoformat(),
        "_loyalty": params["loyalty"],          # latent, used only to build labels
        "_basket_mean": params["basket_mean"],  # latent
    }

    txns = []
    for _ in range(n_purchases):
        # Purchases spread across the customer's active window.
        day = int(rng.integers(0, max(active_days, 1)))
        ts = signup_date + pd.Timedelta(days=day)
        amount = float(np.round(rng.gamma(2.0, params["basket_mean"] / 2.0), 2))
        txns.append(
            {
                "customer_id": cid,
                "order_date": ts.date().isoformat(),
                "amount": max(amount, 1.0),
                "channel": rng.choice(CHANNELS, p=[0.4, 0.4, 0.2]),
                "category": rng.choice(CATEGORIES),
            }
        )
    return customer, txns


def generate() -> None:
    ensure_dirs()
    rng = np.random.default_rng(RANDOM_SEED)
    start = pd.Timestamp(START_DATE)
    snapshot = pd.Timestamp(SNAPSHOT_DATE)

    customers, transactions = [], []
    for cid in range(1, N_CUSTOMERS + 1):
        cust, txns = _simulate_customer(rng, cid, start, snapshot)
        customers.append(cust)
        transactions.extend(txns)

    customers_df = pd.DataFrame(customers)
    transactions_df = pd.DataFrame(transactions)

    # Inject a little realism: ~2% missing ages and a few duplicate orders.
    miss_idx = customers_df.sample(frac=0.02, random_state=RANDOM_SEED).index
    customers_df.loc[miss_idx, "age"] = np.nan

    customers_df.to_csv(CUSTOMERS_CSV, index=False)
    transactions_df.to_csv(TRANSACTIONS_CSV, index=False)

    print(f"Generated {len(customers_df):,} customers -> {CUSTOMERS_CSV}")
    print(f"Generated {len(transactions_df):,} transactions -> {TRANSACTIONS_CSV}")
    print("\nSegment distribution:")
    print(customers_df["segment"].value_counts())


if __name__ == "__main__":
    generate()
