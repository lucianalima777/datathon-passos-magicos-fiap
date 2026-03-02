from __future__ import annotations

from typing import Any, Iterable

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_classification(model: Any, X, y_true) -> dict:
    y_pred = model.predict(X)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

    if hasattr(model, "predict_proba"):
        try:
            y_proba = model.predict_proba(X)[:, 1]
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
            metrics["auc_pr"] = float(average_precision_score(y_true, y_proba))
        except ValueError:
            metrics["roc_auc"] = float("nan")
            metrics["auc_pr"] = float("nan")
    else:
        metrics["roc_auc"] = float("nan")
        metrics["auc_pr"] = float("nan")

    return metrics


def evaluate_thresholds(
    y_true, y_proba, thresholds: Iterable[float]
) -> list[dict[str, float]]:
    results = []
    for thr in thresholds:
        y_pred = (y_proba >= thr).astype(int)
        results.append(
            {
                "threshold": float(thr),
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, zero_division=0)),
                "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            }
        )
    return results


def evaluate_with_threshold(y_true, y_proba, threshold: float) -> dict[str, float]:
    return evaluate_thresholds(y_true, y_proba, [threshold])[0]


def select_threshold_max_recall(
    threshold_results: list[dict[str, float]], min_precision: float = 0.5
) -> dict[str, float]:
    valid = [row for row in threshold_results if row["precision"] >= min_precision]
    pool = valid if valid else threshold_results
    best = max(pool, key=lambda row: (row["recall"], row["precision"], row["f1"]))
    best = dict(best)
    best["min_precision"] = float(min_precision)
    best["rule"] = "max_recall_with_min_precision"
    if not valid:
        best["rule"] = "max_recall_no_precision_constraint"
    return best
