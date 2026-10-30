"""Train and evaluate the MachineGuard machine-failure classifier."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.config import (
    CATEGORICAL_FEATURES,
    DEFAULT_ALERT_THRESHOLD,
    DEFAULT_DATA_PATH,
    FEATURE_COLUMNS,
    METRICS_PATH,
    MODEL_PATH,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    REPORT_DIR,
    MODEL_DIR,
)
from src.data_utils import load_training_data


def build_pipeline() -> Pipeline:
    numeric_pipeline = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    classifier = RandomForestClassifier(
        n_estimators=350,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    return Pipeline(
        steps=[("preprocessor", preprocessor), ("classifier", classifier)]
    )


def train(data_path: Path, threshold: float = DEFAULT_ALERT_THRESHOLD) -> dict:
    if not 0.01 <= threshold <= 0.99:
        raise ValueError("Alert threshold must be between 0.01 and 0.99.")

    X, y = load_training_data(data_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model = build_pipeline()
    model.fit(X_train, y_train)
    failure_scores = model.predict_proba(X_test)[:, 1]
    predictions = (failure_scores >= threshold).astype(int)

    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
    report = classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["No failure", "Failure"],
        output_dict=True,
        zero_division=0,
    )
    average_precision = average_precision_score(y_test, failure_scores)
    roc_auc = roc_auc_score(y_test, failure_scores)

    # Permutation importance is computed on untouched holdout data, using
    # average precision so the ranking is more informative for rare failures.
    importance = permutation_importance(
        model,
        X_test,
        y_test,
        scoring="average_precision",
        n_repeats=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    feature_importance = [
        {"feature": name, "importance": float(value)}
        for name, value in sorted(
            zip(FEATURE_COLUMNS, importance.importances_mean),
            key=lambda item: item[1],
            reverse=True,
        )
    ]

    metrics = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_name": "RandomForestClassifier",
        "dataset_file": data_path.name,
        "dataset_rows": int(len(X)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "failure_rate": float(y.mean()),
        "test_failure_rate": float(y_test.mean()),
        "default_threshold": float(threshold),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "average_precision": float(average_precision),
        "roc_auc": float(roc_auc),
        "confusion_matrix": matrix.tolist(),
        "classification_report": report,
        "feature_importance": feature_importance,
        "feature_columns": FEATURE_COLUMNS,
        "target_column": "Machine failure",
        "important_note": (
            "Scores are model outputs and may not be calibrated probabilities. "
            "The dataset target is a failure label for a recorded observation; "
            "this model does not predict time-to-failure."
        ),
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("\nTraining complete.")
    print(f"Model saved:   {MODEL_PATH}")
    print(f"Metrics saved: {METRICS_PATH}")
    print(f"Rows:          {len(X):,} (train {len(X_train):,} / test {len(X_test):,})")
    print(f"Failure rate:  {y.mean():.2%}")
    print(f"Precision:     {metrics['precision']:.3f}")
    print(f"Recall:        {metrics['recall']:.3f}")
    print(f"F1 score:      {metrics['f1']:.3f}")
    print(f"Average prec.: {metrics['average_precision']:.3f}")
    print(f"ROC-AUC:       {metrics['roc_auc']:.3f}")
    print(f"Alert threshold used: {threshold:.2f}")
    print("Remember: review recall and precision, not accuracy alone.")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the MachineGuard AI model.")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help=(
            "Path to a labelled CSV. Default: data/ai4i2020.csv. "
            "For offline practice use data/demo_machine_data.csv."
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_ALERT_THRESHOLD,
        help="Failure-score threshold used for evaluation (default: 0.35).",
    )
    args = parser.parse_args()
    train(args.data_path, threshold=args.threshold)


if __name__ == "__main__":
    main()
