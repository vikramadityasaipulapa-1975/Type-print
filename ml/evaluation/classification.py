"""Classification metrics for closed-set Random Forest and LSTM baselines.

These are model classification metrics, not calibrated biometric match scores.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score


def evaluate_classifier(model: Any, features: np.ndarray, labels: np.ndarray, classes: np.ndarray) -> dict[str, Any]:
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)
    return classification_metrics(labels, predictions, classes, probabilities)


def classification_metrics(labels: np.ndarray, predictions: np.ndarray, classes: np.ndarray, probabilities: np.ndarray | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "sample_count": int(len(labels)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_precision": float(precision_score(labels, predictions, labels=classes, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(labels, predictions, labels=classes, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(labels, predictions, labels=classes, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=classes).tolist(),
        "confusion_matrix_labels": [str(value) for value in classes],
        "classification_report": classification_report(
            labels, predictions, labels=classes, target_names=[str(value) for value in classes], output_dict=True, zero_division=0
        ),
    }

    if probabilities is not None and len(classes) > 1 and set(labels.tolist()) == set(classes.tolist()):
        if len(classes) == 2:
            binary_truth = (labels == classes[1]).astype(np.int8)
            result["classification_roc_auc_ovr_weighted"] = float(roc_auc_score(binary_truth, probabilities[:, 1]))
        else:
            result["classification_roc_auc_ovr_weighted"] = float(
                roc_auc_score(labels, probabilities, labels=classes, multi_class="ovr", average="weighted")
            )
    else:
        result["classification_roc_auc_ovr_weighted"] = None
        result["roc_auc_note"] = "ROC-AUC omitted because this evaluation split does not contain every training class."
    return result
