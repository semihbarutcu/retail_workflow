# Retail Churn — End-to-End Data Science Workflow

A self-contained, reproducible data science pipeline built entirely on
**artificial data**. It simulates an online retailer, engineers features,
explores the data, and trains models to predict **customer churn** (no purchase
within 90 days of a snapshot date).

The point is to demonstrate a realistic ML workflow end to end — data
generation → feature engineering → EDA → modelling → evaluation — without
depending on any external dataset.

## The problem

> Given everything we know about a customer *as of a snapshot date*, will they
> churn (make no purchase) in the next 90 days?

This is a binary classification problem. The signal is built around the classic
**RFM** framework (Recency, Frequency, Monetary value) plus tenure, basket
statistics and channel/category engagement.

## Pipeline

| Step | Script | What it does | Output |
|------|--------|--------------|--------|
| 1 | `src/generate_data.py` | Simulate customers (4 latent behavioural segments) and their transactions | `data/customers.csv`, `data/transactions.csv` |
| 2 | `src/features.py` | Build a one-row-per-customer feature table and the churn label (no leakage from the future) | `data/features.csv` |
| 3 | `src/eda.py` | Class balance, RFM-by-churn distributions, correlation heatmap | `reports/figures/01–03` |
| 4 | `src/train.py` | Train Logistic Regression vs. Random Forest in a leak-proof `Pipeline`, evaluate, plot ROC / confusion / permutation importance | `reports/metrics.json`, `reports/figures/04–06` |

`src/config.py` holds all paths, seeds and constants in one place.

## How the synthetic data is designed

- Each customer is drawn from one of four latent segments
  (`bargain_hunter`, `regular`, `loyal_vip`, `one_off`) with different purchase
  rates, basket sizes and loyalty.
- Transaction counts follow a Poisson process scaled by tenure, with
  per-customer noise.
- The churn label is sampled from a **logistic risk model**: risk rises with
  recency and falls with frequency, spend and latent loyalty, plus Gaussian
  noise. This makes churn genuinely *predictable from the features* but not a
  deterministic function of them — so model metrics are meaningful.
- A little realism is injected (missing ages) to exercise the imputation steps.

## Quickstart

```bash
pip install -r requirements.txt
python src/workflow.py        # runs all four steps in order
```

Or run any single step, e.g. `python src/generate_data.py`.
Everything is seeded (`RANDOM_SEED = 42`), so results are reproducible.

## Results (seed 42)

Both models land around **ROC-AUC ≈ 0.84**. See `reports/metrics.json` for the
full breakdown; the plots are written to `reports/figures/` when you run the
pipeline (they regenerate from the seed, so they're not committed). As expected, permutation
importance is dominated by **recency, frequency and monetary** value — the RFM
trio — which is the sanity check that the pipeline learned the intended signal.

## Repository layout

```
src/
  config.py         # paths, seeds, constants
  generate_data.py  # step 1 — synthetic data
  features.py       # step 2 — feature engineering + labels
  eda.py            # step 3 — exploratory plots
  train.py          # step 4 — modelling + evaluation
  workflow.py       # orchestrates steps 1–4
reports/
  metrics.json      # model metrics
  figures/          # EDA + evaluation plots
data/               # generated CSVs (git-ignored, regenerable from seed)
                    # figures/ is also git-ignored and regenerated on each run
```
