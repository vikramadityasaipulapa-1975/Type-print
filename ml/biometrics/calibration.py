"""Held-out genuine/impostor distance calibration and verification metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, confusion_matrix, roc_auc_score

from ml.biometrics.profile import EnrollmentProfile


@dataclass
class DistanceCalibrator:
    model: LogisticRegression
    threshold: float
    validation_eer: float
    model_version: str

    def score(self, distance: float) -> float:
        if not np.isfinite(distance) or distance < 0:
            raise ValueError("distance must be a finite non-negative value")
        return float(self.model.predict_proba(np.asarray([[-distance]], dtype=np.float64))[0, 1])


def build_verification_pairs(
    profiles: dict[str, EnrollmentProfile],
    embeddings: np.ndarray,
    subject_ids: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(embeddings, dtype=np.float64)
    subjects = np.asarray(subject_ids, dtype=str)
    if values.ndim != 2 or subjects.shape != (len(values),):
        raise ValueError("embeddings and subject IDs have incompatible shapes")
    if any(subject not in profiles for subject in subjects):
        raise ValueError("calibration/evaluation contains a user without an enrolled profile")
    distances: list[float] = []
    labels: list[int] = []
    for embedding, subject in zip(values, subjects):
        for target_user, profile in profiles.items():
            distances.append(profile.distance(embedding))
            labels.append(int(target_user == subject))
    if not distances or len(set(labels)) < 2:
        raise ValueError("genuine and impostor pairs are both required")
    return np.asarray(distances, dtype=np.float64).reshape(-1, 1), np.asarray(labels, dtype=np.int8)


def _rates(scores: np.ndarray, labels: np.ndarray, threshold: float) -> tuple[float, float]:
    accepted = scores >= threshold
    impostor = labels == 0
    genuine = labels == 1
    far = float(np.mean(accepted[impostor]))
    frr = float(np.mean(~accepted[genuine]))
    return far, frr


def select_eer_threshold(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    values = np.asarray(scores, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.int8)
    if values.ndim != 1 or targets.shape != values.shape or not np.isfinite(values).all() or set(targets.tolist()) != {0, 1}:
        raise ValueError("EER selection requires finite scores and both genuine/impostor labels")
    unique_scores = np.unique(values)
    midpoints = (unique_scores[:-1] + unique_scores[1:]) / 2
    candidates = np.unique(np.concatenate(([0.0], unique_scores, midpoints, [np.nextafter(1.0, 2.0)])))
    points: list[tuple[float, float, float]] = []
    for candidate in candidates:
        far, frr = _rates(values, targets, float(candidate))
        points.append((abs(far - frr), (far + frr) / 2, float(candidate)))
    minimum_gap = min(point[0] for point in points)
    tied = [point for point in points if np.isclose(point[0], minimum_gap, atol=1e-12)]
    threshold = float(np.median([point[2] for point in tied]))
    far, frr = _rates(values, targets, threshold)
    eer = (far + frr) / 2
    return float(threshold), float(eer)


def fit_distance_calibrator(
    profiles: dict[str, EnrollmentProfile],
    validation_embeddings: np.ndarray,
    validation_subject_ids: Sequence[str],
    model_version: str,
    minimum_validation_auc: float = 0.55,
) -> tuple[DistanceCalibrator, dict[str, float | int | list[list[int]] | None]]:
    if not 0.5 < minimum_validation_auc <= 1.0:
        raise ValueError("minimum_validation_auc must be greater than 0.5 and at most 1.0")
    if not profiles or any(profile.model_version != model_version for profile in profiles.values()):
        raise ValueError("all enrollment profiles must use the calibrator model version")
    distances, labels = build_verification_pairs(profiles, validation_embeddings, validation_subject_ids)
    model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=42)
    model.fit(-distances, labels)
    scores = model.predict_proba(-distances)[:, 1]
    threshold, eer = select_eer_threshold(scores, labels)
    calibration_metrics = verification_metrics(scores, labels, threshold)
    if calibration_metrics["roc_auc"] is None or calibration_metrics["roc_auc"] < minimum_validation_auc:
        raise ValueError("held-out calibration data does not separate genuine and impostor distances reliably")
    calibrator = DistanceCalibrator(model=model, threshold=threshold, validation_eer=eer, model_version=model_version)
    return calibrator, calibration_metrics


def verification_metrics(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict[str, float | int | list[list[int]] | None]:
    values = np.asarray(scores, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.int8)
    if values.ndim != 1 or targets.shape != values.shape or set(targets.tolist()) != {0, 1}:
        raise ValueError("verification metrics require aligned genuine and impostor scores")
    far, frr = _rates(values, targets, threshold)
    eer_threshold, eer = select_eer_threshold(values, targets)
    predicted = (values >= threshold).astype(np.int8)
    return {
        "pair_count": int(len(targets)),
        "threshold": float(threshold),
        "far": far,
        "frr": frr,
        "eer": eer,
        "eer_threshold": eer_threshold,
        "roc_auc": float(roc_auc_score(targets, values)),
        "brier_score": float(brier_score_loss(targets, values)),
        "confusion_matrix": confusion_matrix(targets, predicted, labels=[0, 1]).tolist(),
    }
