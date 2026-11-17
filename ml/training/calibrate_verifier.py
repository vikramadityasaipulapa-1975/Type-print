"""Evaluate held-out LSTM embedding enrollment and score calibration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import joblib

from ml.biometrics.calibration import build_verification_pairs, fit_distance_calibrator, verification_metrics
from ml.biometrics.profile import create_profiles
from ml.data.prepared import load_prepared_dataset
from ml.inference.lstm_encoder import LSTMEncoder


def calibrate_verifier(
    features_csv: str | Path,
    sequences_npz: str | Path,
    lstm_checkpoint: str | Path,
    output_path: str | Path,
    calibrator_path: str | Path | None = None,
    *,
    min_enrollment_samples: int = 5,
    threshold_override: float | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    if threshold_override is not None and not 0 <= threshold_override <= 1:
        raise ValueError("threshold_override must be between 0 and 1")
    dataset = load_prepared_dataset(features_csv, sequences_npz)
    encoder = LSTMEncoder.from_checkpoint(lstm_checkpoint, device=device)
    if tuple(dataset.sequence_feature_names) != encoder.sequence_feature_names:
        raise ValueError("prepared sequence features do not match the LSTM checkpoint")
    embeddings = np.asarray(
        [encoder.encode_sequence(sequence[:length]) for sequence, length in zip(dataset.sequences, dataset.lengths)],
        dtype=np.float32,
    )
    subjects = dataset.frame["user_id"].astype(str).to_numpy()
    profiles = create_profiles(
        embeddings[dataset.train_indices], subjects[dataset.train_indices], encoder.model_version, min_enrollment_samples
    )
    calibrator, validation_metrics = fit_distance_calibrator(
        profiles,
        embeddings[dataset.validation_indices],
        subjects[dataset.validation_indices],
        encoder.model_version,
    )
    test_distances, test_labels = build_verification_pairs(
        profiles, embeddings[dataset.test_indices], subjects[dataset.test_indices]
    )
    test_scores = np.asarray([calibrator.score(distance) for distance in test_distances[:, 0]], dtype=np.float64)
    decision_threshold = calibrator.threshold if threshold_override is None else threshold_override
    test_metrics = verification_metrics(test_scores, test_labels, decision_threshold)
    output: dict[str, Any] = {
        "model_version": encoder.model_version,
        "enrolled_user_count": len(profiles),
        "enrollment_sample_min": min(profile.enrollment_samples for profile in profiles.values()),
        "enrollment_sample_max": max(profile.enrollment_samples for profile in profiles.values()),
        "validation_calibration": validation_metrics,
        "test_verification": test_metrics,
        "threshold_source": "validation_eer" if threshold_override is None else "configured_override",
        "threshold": float(decision_threshold),
        "calibrator_artifact": str(calibrator_path) if calibrator_path else None,
        "limitations": [
            "Scores are calibrated on validation genuine/impostor pairs and may shift with different users or collection conditions.",
            "Profiles are held in memory by this command and are not persisted; database integration and protected storage are later work.",
            "The evaluation requires multiple consented sessions per user for enrollment, calibration, and test sets.",
        ],
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, indent=2, allow_nan=False), encoding="utf-8")
    if calibrator_path:
        calibrator_destination = Path(calibrator_path)
        calibrator_destination.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(calibrator, calibrator_destination)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("features_csv", type=Path)
    parser.add_argument("sequences_npz", type=Path)
    parser.add_argument("lstm_checkpoint", type=Path)
    parser.add_argument("--output", type=Path, default=Path("models/verification.metrics.json"))
    parser.add_argument("--calibrator-output", type=Path, default=Path("models/distance_calibrator.joblib"))
    parser.add_argument("--min-enrollment-samples", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    metrics = calibrate_verifier(
        args.features_csv,
        args.sequences_npz,
        args.lstm_checkpoint,
        args.output,
        calibrator_path=args.calibrator_output,
        min_enrollment_samples=args.min_enrollment_samples,
        threshold_override=args.threshold,
        device=args.device,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
