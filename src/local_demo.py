from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DATA = ROOT / "data"
RESULTS.mkdir(exist_ok=True)
DATA.mkdir(exist_ok=True)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate_customers(n: int = 80000, seed: int = RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    age = rng.integers(18, 75, n)
    tenure_months = rng.integers(1, 73, n)
    monthly_spend = np.clip(rng.normal(55, 22, n), 5, 180)
    usage_hours = np.clip(rng.normal(42, 18, n), 1, 120)
    support_calls = np.clip(rng.poisson(1.6, n), 0, 10)
    late_payments = np.clip(rng.poisson(0.8, n), 0, 8)
    satisfaction_score = np.clip(rng.normal(3.5, 0.9, n), 1, 5)
    contract_type = rng.choice(["month-to-month", "one-year", "two-year"], n, p=[0.55, 0.28, 0.17])
    region = rng.choice(["Almaty", "Astana", "Shymkent", "Other"], n, p=[0.32, 0.25, 0.16, 0.27])
    has_internet = rng.choice([0, 1], n, p=[0.12, 0.88])

    # Churn ықтималдығын жасанды, бірақ мағыналы факторлармен құрамыз.
    z = (
        -1.5
        + 0.55 * (contract_type == "month-to-month")
        - 0.55 * (contract_type == "two-year")
        - 0.022 * tenure_months
        + 0.18 * support_calls
        + 0.28 * late_payments
        - 0.75 * (satisfaction_score - 3.0)
        + 0.009 * (monthly_spend - 55)
        - 0.015 * (usage_hours - 42)
        + 0.25 * (1 - has_internet)
        + 0.90 * ((support_calls >= 3) & (satisfaction_score < 3))
        + 0.70 * ((monthly_spend > 80) & (contract_type == "month-to-month"))
        + 0.65 * (late_payments >= 2)
        + rng.normal(0, 0.35, n)
    )
    churn_prob = sigmoid(z)
    churn = rng.binomial(1, churn_prob)

    df = pd.DataFrame(
        {
            "age": age,
            "tenure_months": tenure_months,
            "monthly_spend": monthly_spend.round(2),
            "usage_hours": usage_hours.round(2),
            "support_calls": support_calls,
            "late_payments": late_payments,
            "satisfaction_score": satisfaction_score.round(2),
            "contract_type": contract_type,
            "region": region,
            "has_internet": has_internet,
            "churn": churn,
        }
    )
    return df


def metric_row(name: str, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict[str, float | str]:
    return {
        "model": name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auc": roc_auc_score(y_true, y_prob),
    }


def main() -> None:
    df = generate_customers()
    df.sample(5000, random_state=RANDOM_STATE).to_csv(DATA / "sample_customers.csv", index=False)

    X = df.drop(columns="churn")
    y = df["churn"].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )

    numeric = [
        "age",
        "tenure_months",
        "monthly_spend",
        "usage_hours",
        "support_calls",
        "late_payments",
        "satisfaction_score",
        "has_internet",
    ]
    categorical = ["contract_type", "region"]

    pre = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical,
            ),
        ]
    )

    X_train_t = pre.fit_transform(X_train)
    X_test_t = pre.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=220,
            max_depth=10,
            min_samples_leaf=8,
            n_jobs=-1,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Gradient Boosted Trees": HistGradientBoostingClassifier(
            max_iter=180,
            learning_rate=0.08,
            max_leaf_nodes=25,
            l2_regularization=0.4,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
    }

    rows: list[dict[str, float | str]] = []
    fitted = {}
    predictions = {}
    for name, model in models.items():
        model.fit(X_train_t, y_train)
        y_pred = model.predict(X_test_t)
        y_prob = model.predict_proba(X_test_t)[:, 1]
        rows.append(metric_row(name, y_test, y_pred, y_prob))
        fitted[name] = model
        predictions[name] = (y_pred, y_prob)

    metrics = pd.DataFrame(rows).sort_values("auc", ascending=False).reset_index(drop=True)
    metrics.to_csv(RESULTS / "metrics.csv", index=False)
    with open(RESULTS / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics.to_dict(orient="records"), f, ensure_ascii=False, indent=2)

    # Model comparison chart
    fig, ax = plt.subplots(figsize=(9, 5))
    chart = metrics.set_index("model")[["accuracy", "f1", "auc"]]
    chart.plot(kind="bar", ax=ax)
    ax.set_ylim(0.45, 1.0)
    ax.set_ylabel("Метрика мәні")
    ax.set_xlabel("")
    ax.set_title("Модельдерді салыстыру")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(["Accuracy", "F1", "ROC-AUC"], loc="lower right")
    plt.xticks(rotation=0)
    plt.tight_layout()
    fig.savefig(RESULTS / "model_comparison.png", dpi=180)
    plt.close(fig)

    best_name = metrics.iloc[0]["model"]
    best_model = fitted[str(best_name)]
    best_pred, _ = predictions[str(best_name)]

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ConfusionMatrixDisplay.from_predictions(y_test, best_pred, cmap="Blues", ax=ax, colorbar=False)
    ax.set_title(f"Confusion matrix — {best_name}")
    ax.set_xlabel("Болжанған класс")
    ax.set_ylabel("Нақты класс")
    plt.tight_layout()
    fig.savefig(RESULTS / "confusion_matrix_best.png", dpi=180)
    plt.close(fig)

    # Feature importance / coefficient-based importance
    feature_names = pre.get_feature_names_out()
    if hasattr(best_model, "feature_importances_"):
        importance = np.asarray(best_model.feature_importances_)
    elif hasattr(best_model, "coef_"):
        importance = np.abs(np.asarray(best_model.coef_)[0])
    else:
        # HistGradientBoosting-де feature_importances_ жоқ, сондықтан permutation importance қолданбай,
        # түсінікті болу үшін Random Forest feature importance көрсетеміз.
        rf = fitted["Random Forest"]
        importance = np.asarray(rf.feature_importances_)
        best_name_for_importance = "Random Forest"
    if 'best_name_for_importance' not in locals():
        best_name_for_importance = str(best_name)

    top_idx = np.argsort(importance)[-10:][::-1]
    labels = [feature_names[i].replace("num__", "").replace("cat__", "") for i in top_idx][::-1]
    vals = importance[top_idx][::-1]
    fig, ax = plt.subplots(figsize=(8, 5.2))
    ax.barh(labels, vals)
    ax.set_title(f"Ең маңызды белгілер — {best_name_for_importance}")
    ax.set_xlabel("Маңыздылық")
    ax.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    fig.savefig(RESULTS / "feature_importance.png", dpi=180)
    plt.close(fig)

    summary = {
        "rows": int(len(df)),
        "churn_rate": float(df["churn"].mean()),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "best_model": str(metrics.iloc[0]["model"]),
        "best_auc": float(metrics.iloc[0]["auc"]),
        "best_f1": float(metrics.iloc[0]["f1"]),
    }
    with open(RESULTS / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(metrics.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\nSummary:", json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
