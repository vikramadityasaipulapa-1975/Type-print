"""Compact in-memory biometric profiles derived from LSTM embeddings."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EnrollmentProfile:
    user_id: str
    model_version: str
    centroid: np.ndarray
    scale: np.ndarray
    enrollment_samples: int

    def distance(self, embedding: np.ndarray) -> float:
        value = np.asarray(embedding, dtype=np.float64)
        if value.shape != self.centroid.shape or not np.isfinite(value).all():
            raise ValueError("embedding shape does not match the enrolled profile")
        normalized = (value - self.centroid) / self.scale
        return float(np.sqrt(np.mean(np.square(normalized))))


def create_profile(user_id: str, embeddings: np.ndarray, model_version: str, min_samples: int = 5) -> EnrollmentProfile:
    values = np.asarray(embeddings, dtype=np.float64)
    if not user_id or not model_version:
        raise ValueError("user_id and model_version are required")
    if min_samples < 2 or values.ndim != 2 or len(values) < min_samples:
        raise ValueError(f"at least {min_samples} enrollment embeddings are required")
    if values.shape[1] == 0 or not np.isfinite(values).all():
        raise ValueError("enrollment embeddings must be finite vectors")
    center = np.median(values, axis=0)
    dispersion = 1.4826 * np.median(np.abs(values - center), axis=0)
    scale = np.maximum(dispersion, 1e-3)
    return EnrollmentProfile(
        user_id=user_id,
        model_version=model_version,
        centroid=center.astype(np.float32),
        scale=scale.astype(np.float32),
        enrollment_samples=len(values),
    )


def create_profiles(
    embeddings: np.ndarray,
    user_ids: np.ndarray,
    model_version: str,
    min_samples: int = 5,
) -> dict[str, EnrollmentProfile]:
    values = np.asarray(embeddings)
    labels = np.asarray(user_ids, dtype=str)
    if values.ndim != 2 or labels.shape != (len(values),):
        raise ValueError("embeddings and user IDs have incompatible shapes")
    profiles: dict[str, EnrollmentProfile] = {}
    for user_id in np.unique(labels):
        profiles[user_id] = create_profile(user_id, values[labels == user_id], model_version, min_samples)
    if len(profiles) < 2:
        raise ValueError("verification calibration requires profiles for at least two users")
    return profiles
