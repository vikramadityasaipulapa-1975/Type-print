"""Train and evaluate the Random Forest baseline on prepared session splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from ml.evaluation.classification import evaluate_classifier
from ml.data.prepared import load_prepared_dataset
from ml.models.random_forest import build_random_forest

IDENTITY_COLUMNS = {"sample_id", "session_id", "user_id"}


def train_random_forest(
    features_csv: str | Path,
    sequences_npz: str | Path,
    model_path: str | Path,
    n_estimators: int = 300,
    random_state: int = 42,
) -> dict[str, Any]:
    prepared = load_prepared_dataset(features_csv, sequences_npz)
    frame = prepared.frame
    feature_columns = [column for column in frame.columns if column not in IDENTITY_COLUMNS]
    if not feature_columns:
        raise ValueError("feature CSV contains no model features")
    try:
        features = frame[feature_columns].to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("all model feature columns must be numeric") from exc
    if np.isinf(features).any():
        raise ValueError("model features cannot contain positive or negative infinity")
    train_indices = prepared.train_indices
    validation_indices = prepared.validation_indices
    test_indices = prepared.test_indices
    labels = frame["user_id"].astype(str).to_numpy()
    train_labels = labels[train_indices]
    classes = np.unique(train_labels)
    if len(classes) < 2:
        raise ValueError("Random Forest identity classification requires at least two users in the training split")
    unknown_labels = set(labels[np.concatenate((validation_indices, test_indices))]) - set(classes)
    if unknown_labels:
        raise ValueError(f"validation/test contains users missing from training: {sorted(unknown_labels)}")
    model = build_random_forest(n_estimators=n_estimators, random_state=random_state)
    model.fit(features[train_indices], train_labels)
    validation_metrics = evaluate_classifier(model, features[validation_indices], labels[validation_indices], classes)
    test_metrics = evaluate_classifier(model, features[test_indices], labels[test_indices], classes)

    artifact = {
        "model": model,
        "feature_names": [column for column in frame.columns if column not in IDENTITY_COLUMNS],
        "classes": classes.tolist(),
        "model_name": "random_forest_identity_baseline",
        "model_version": "0.1.0",
        "random_state": random_state,
        "n_estimators": n_estimators,
        "score_semantics": "closed_set_classification_only; not a calibrated biometric match score",
    }
    destination = Path(model_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, destination)
    metrics = {
        "model_name": artifact["model_name"],
        "model_version": artifact["model_version"],
        "class_count": int(len(classes)),
        "classes": [str(value) for value in classes],
        "split_sample_counts": {"train": int(len(train_indices)), "validation": int(len(validation_indices)), "test": int(len(test_indices))},
        "validation": validation_metrics,
        "test": test_metrics,
        "model_artifact": str(destination),
        "limitations": [
            "Metrics require real consented user samples and are only as representative as the collection sessions.",
            "Closed-set classification metrics are not FAR, FRR, EER, or calibrated authentication scores.",
            "This baseline cannot recognize a user class that was absent from its training split.",
        ],
    }
    metrics_path = destination.with_suffix(destination.suffix + ".metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2, allow_nan=False), encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("features_csv", type=Path)
    parser.add_argument("sequences_npz", type=Path)
    parser.add_argument("--model-path", type=Path, default=Path("models/random_forest.joblib"))
    parser.add_argument("--estimators", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    metrics = train_random_forest(args.features_csv, args.sequences_npz, args.model_path, args.estimators, args.seed)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
