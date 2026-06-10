"""Step 4 — Model training and evaluation.

We compare a Logistic Regression baseline against a Random Forest using a
scikit-learn ``Pipeline`` that bundles imputation, scaling and one-hot encoding
so the exact same preprocessing is applied at train and inference time.

Outputs
-------
reports/metrics.json            All metrics for every model.
reports/figures/04_roc.png      ROC curves.
reports/figures/05_confusion.png  Confusion matrix for the best model.
reports/figures/06_importance.png Permutation importance for the best model.
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import (
    FEATURES_CSV,
    FIGURES_DIR,
    ID_COLUMNS,
    METRICS_JSON,
    RANDOM_SEED,
    TARGET,
    TEST_SIZE,
    ensure_dirs,
)


def _build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical = X.select_dtypes(exclude=[np.number]).columns.tolist()

    numeric_pipe = Pipeline(
        [("impute", SimpleImputer(strategy="median")),
         ("scale", StandardScaler())]
    )
    categorical_pipe = Pipeline(
        [("impute", SimpleImputer(strategy="most_frequent")),
         ("onehot", OneHotEncoder(handle_unknown="ignore"))]
    )
    return ColumnTransformer(
        [("num", numeric_pipe, numeric),
         ("cat", categorical_pipe, categorical)]
    )


def train() -> dict:
    ensure_dirs()
    df = pd.read_csv(FEATURES_CSV)

    # 'segment' is a latent label leaked into the raw data for inspection; it
    # would not be known for a real prospect, so we exclude it from features.
    drop_cols = ID_COLUMNS + [TARGET, "segment"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )

    pre = _build_preprocessor(X)
    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(
            n_estimators=300, max_depth=12, min_samples_leaf=5,
            class_weight="balanced", random_state=RANDOM_SEED, n_jobs=-1
        ),
    }

    results, fitted, roc_data = {}, {}, {}
    for name, est in models.items():
        pipe = Pipeline([("pre", pre), ("model", est)])
        pipe.fit(X_train, y_train)
        proba = pipe.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)

        auc = roc_auc_score(y_test, proba)
        report = classification_report(y_test, pred, output_dict=True, zero_division=0)
        results[name] = {
            "roc_auc": round(float(auc), 4),
            "accuracy": round(float(report["accuracy"]), 4),
            "precision_churn": round(float(report["1"]["precision"]), 4),
            "recall_churn": round(float(report["1"]["recall"]), 4),
            "f1_churn": round(float(report["1"]["f1-score"]), 4),
        }
        fitted[name] = pipe
        roc_data[name] = roc_curve(y_test, proba)
        print(f"{name:>22}: AUC={auc:.3f}  F1(churn)={results[name]['f1_churn']:.3f}")

    best_name = max(results, key=lambda k: results[k]["roc_auc"])
    print(f"\nBest model by ROC-AUC: {best_name}")

    # --- Figures ----------------------------------------------------------
    _plot_roc(roc_data, results)
    _plot_confusion(fitted[best_name], X_test, y_test, best_name)
    _plot_importance(fitted[best_name], X_test, y_test, best_name)

    summary = {
        "best_model": best_name,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "churn_rate": round(float(y.mean()), 4),
        "models": results,
    }
    METRICS_JSON.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote metrics -> {METRICS_JSON}")
    return summary


def _plot_roc(roc_data, results) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, (fpr, tpr, _) in roc_data.items():
        ax.plot(fpr, tpr, label=f"{name} (AUC={results[name]['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "04_roc.png", dpi=110)
    plt.close(fig)


def _plot_confusion(pipe, X_test, y_test, name) -> None:
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_estimator(
        pipe, X_test, y_test, display_labels=["retained", "churned"],
        cmap="Blues", ax=ax, colorbar=False
    )
    ax.set_title(f"Confusion matrix — {name}")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "05_confusion.png", dpi=110)
    plt.close(fig)


def _plot_importance(pipe, X_test, y_test, name) -> None:
    result = permutation_importance(
        pipe, X_test, y_test, n_repeats=10,
        random_state=RANDOM_SEED, scoring="roc_auc", n_jobs=-1
    )
    order = result.importances_mean.argsort()[::-1][:12][::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(np.array(X_test.columns)[order], result.importances_mean[order],
            xerr=result.importances_std[order], color="#55a868")
    ax.set_xlabel("Mean ROC-AUC drop when shuffled")
    ax.set_title(f"Permutation importance — {name}")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "06_importance.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    train()
