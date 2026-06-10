"""Central configuration for the retail churn workflow.

Keeping paths and constants in one place makes the pipeline reproducible and
easy to tweak without hunting through every script.
"""
from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

CUSTOMERS_CSV = DATA_DIR / "customers.csv"
TRANSACTIONS_CSV = DATA_DIR / "transactions.csv"
FEATURES_CSV = DATA_DIR / "features.csv"
METRICS_JSON = REPORTS_DIR / "metrics.json"

# ---------------------------------------------------------------------------
# Synthetic data settings
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
N_CUSTOMERS = 5_000

# The observation window. We generate transactions in [START, SNAPSHOT) and
# define churn over [SNAPSHOT, SNAPSHOT + CHURN_HORIZON_DAYS).
START_DATE = "2024-01-01"
SNAPSHOT_DATE = "2025-01-01"   # "today" from the model's point of view
CHURN_HORIZON_DAYS = 90        # no purchase within 90 days of snapshot => churn

# ---------------------------------------------------------------------------
# Modelling settings
# ---------------------------------------------------------------------------
TEST_SIZE = 0.25
TARGET = "churned"

# Columns that identify a row but must never be fed to the model.
ID_COLUMNS = ["customer_id"]


def ensure_dirs() -> None:
    """Create output directories if they do not already exist."""
    for d in (DATA_DIR, REPORTS_DIR, FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)
