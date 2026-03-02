from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from backend.src.config import (
    MODELS_DIR,
    RANDOM_STATE,
    TARGET_COLUMN,
    THRESHOLD_GRID,
    THRESHOLD_MIN_PRECISION,
)
from backend.src.models.evaluator import (
    evaluate_classification,
    evaluate_thresholds,
    evaluate_with_threshold,
    select_threshold_max_recall,
)


def build_preprocess_pipeline(categorical_features: list[str], numeric_features: list[str]):
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )

    return preprocess


def build_model(
    categorical_features: list[str],
    numeric_features: list[str],
    model_type: str = "logreg",
) -> Pipeline:
    preprocess = build_preprocess_pipeline(categorical_features, numeric_features)
    if model_type == "random_forest":
        classifier = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )
    else:
        classifier = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver="liblinear",
            random_state=RANDOM_STATE,
        )

    model = Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("classifier", classifier),
        ]
    )

    return model


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    return X, y


def cross_validate_pipeline(
    X: pd.DataFrame,
    y: pd.Series,
    categorical_features: list[str],
    numeric_features: list[str],
    model_type: str = "logreg",
    n_splits: int = 5,
    random_state: int = RANDOM_STATE,
) -> tuple[list[dict], np.ndarray, np.ndarray]:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_metrics: list[dict] = []
    oof_probas = np.zeros(len(y))

    for fold_train_idx, fold_val_idx in skf.split(X, y):
        X_fold_train = X.iloc[fold_train_idx]
        y_fold_train = y.iloc[fold_train_idx]
        X_fold_val = X.iloc[fold_val_idx]
        y_fold_val = y.iloc[fold_val_idx]

        fold_model = build_model(categorical_features, numeric_features, model_type=model_type)
        fold_model.fit(X_fold_train, y_fold_train)

        oof_probas[fold_val_idx] = fold_model.predict_proba(X_fold_val)[:, 1]
        fold_metrics.append(evaluate_classification(fold_model, X_fold_val, y_fold_val))

    return fold_metrics, oof_probas, y.to_numpy()


def train_and_evaluate(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
    model_type: str = "logreg",
    val_size: float = 0.2,
    threshold_min_precision: float = THRESHOLD_MIN_PRECISION,
) -> tuple[Pipeline, dict]:
    X, y = split_features_target(df)

    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
    numeric_features = [col for col in X.columns if col not in categorical_features]

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    val_relative_size = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=val_relative_size,
        stratify=y_train_val,
        random_state=random_state,
    )

    model = build_model(categorical_features, numeric_features, model_type=model_type)
    model.fit(X_train, y_train)

    metrics: dict = {
        "train": evaluate_classification(model, X_train, y_train),
        "val": evaluate_classification(model, X_val, y_val),
        "test": evaluate_classification(model, X_test, y_test),
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "n_test": int(len(X_test)),
        "categorical_features": categorical_features,
        "numeric_features": numeric_features,
        "model_type": model_type,
        "threshold_min_precision": float(threshold_min_precision),
    }

    if not hasattr(model, "predict_proba"):
        return model, metrics

    min_class_count = int(min(y_train_val.value_counts()))
    n_cv_splits = min(5, min_class_count // 2)
    cv_summary: dict | None = None

    if n_cv_splits >= 2:
        fold_metrics, oof_probas, oof_true = cross_validate_pipeline(
            X_train_val,
            y_train_val,
            categorical_features,
            numeric_features,
            model_type=model_type,
            n_splits=n_cv_splits,
            random_state=random_state,
        )
        oof_threshold_metrics = evaluate_thresholds(oof_true, oof_probas, THRESHOLD_GRID)
        best_threshold = select_threshold_max_recall(
            oof_threshold_metrics, min_precision=threshold_min_precision
        )
        cv_summary = {
            "n_splits": n_cv_splits,
            "recall_mean": float(np.mean([m["recall"] for m in fold_metrics])),
            "recall_std": float(np.std([m["recall"] for m in fold_metrics])),
            "precision_mean": float(np.mean([m["precision"] for m in fold_metrics])),
            "precision_std": float(np.std([m["precision"] for m in fold_metrics])),
            "f1_mean": float(np.mean([m["f1"] for m in fold_metrics])),
            "threshold_source": "oof",
        }
        metrics["thresholds_oof"] = oof_threshold_metrics
        metrics["best_threshold"] = best_threshold
    else:
        y_proba_val = model.predict_proba(X_val)[:, 1]
        threshold_metrics = evaluate_thresholds(y_val, y_proba_val, THRESHOLD_GRID)
        best_threshold = select_threshold_max_recall(
            threshold_metrics, min_precision=threshold_min_precision
        )
        metrics["thresholds_val"] = threshold_metrics
        metrics["best_threshold"] = best_threshold

    metrics["cv"] = cv_summary

    y_proba_val = model.predict_proba(X_val)[:, 1]
    metrics["thresholds_val"] = evaluate_thresholds(y_val, y_proba_val, THRESHOLD_GRID)

    y_proba_test = model.predict_proba(X_test)[:, 1]
    metrics["test_at_threshold"] = evaluate_with_threshold(
        y_test, y_proba_test, best_threshold["threshold"]
    )

    return model, metrics


def save_artifacts(
    model: Pipeline, metrics: dict, model_name: str, output_dir: Path | None = None
) -> dict[str, Path]:
    output_dir = output_dir or MODELS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / f"{model_name}.joblib"
    metrics_path = output_dir / f"{model_name}_metrics.json"

    joblib.dump(model, model_path)
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    return {"model": model_path, "metrics": metrics_path}


def train_save_report(
    df: pd.DataFrame,
    model_name: str = "baseline_logreg",
    model_type: str = "logreg",
    output_dir: Path | None = None,
    test_size: float = 0.2,
    val_size: float = 0.2,
    threshold_min_precision: float = 0.7,
) -> dict[str, Any]:
    model, metrics = train_and_evaluate(
        df,
        model_type=model_type,
        test_size=test_size,
        val_size=val_size,
        threshold_min_precision=threshold_min_precision,
    )
    paths = save_artifacts(model, metrics, model_name=model_name, output_dir=output_dir)
    return {"model": model, "metrics": metrics, "paths": paths}
