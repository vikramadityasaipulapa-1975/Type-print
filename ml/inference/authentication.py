"""In-memory user verification using held-out calibrated embedding distances."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ml.biometrics.calibration import DistanceCalibrator
from ml.biometrics.profile import EnrollmentProfile


@dataclass(frozen=True)
class AuthenticationResult:
    target_user_id: str
    status: str
    accepted: bool | None
    similarity_score: float | None
    threshold: float | None
    distance: float | None
    reason: str | None = None


class AuthenticationEngine:
    def __init__(
        self,
        encoder: object,
        profiles: dict[str, EnrollmentProfile],
        calibrator: DistanceCalibrator | None,
        threshold: float | None = None,
        minimum_sequence_length: int = 10,
        uncertainty_margin: float = 0.05,
    ) -> None:
        if minimum_sequence_length < 1:
            raise ValueError("minimum_sequence_length must be positive")
        if not 0 <= uncertainty_margin < 0.5:
            raise ValueError("uncertainty_margin must be in [0, 0.5)")
        selected_threshold = calibrator.threshold if threshold is None and calibrator else threshold
        if selected_threshold is not None and not 0 <= selected_threshold <= 1:
            raise ValueError("threshold must be between 0 and 1")
        self.encoder = encoder
        self.profiles = profiles
        self.calibrator = calibrator
        self.threshold = selected_threshold
        self.minimum_sequence_length = minimum_sequence_length
        self.uncertainty_margin = uncertainty_margin

    def verify(self, target_user_id: str, sequence: np.ndarray) -> AuthenticationResult:
        if target_user_id not in self.profiles:
            return AuthenticationResult(target_user_id, "unknown_user", None, None, self.threshold, None, "profile_not_found")
        values = np.asarray(sequence, dtype=np.float32)
        if values.ndim != 2 or len(values) < self.minimum_sequence_length or not np.isfinite(values).all():
            return AuthenticationResult(target_user_id, "insufficient_data", None, None, self.threshold, None, "sequence_invalid_or_too_short")
        profile = self.profiles[target_user_id]
        if self.calibrator is None or self.threshold is None:
            return AuthenticationResult(target_user_id, "low_confidence", None, None, None, None, "calibration_unavailable")
        if getattr(self.encoder, "model_version", None) != profile.model_version or self.calibrator.model_version != profile.model_version:
            return AuthenticationResult(target_user_id, "low_confidence", None, None, self.threshold, None, "model_version_mismatch")
        try:
            embedding = self.encoder.encode_sequence(values)
            distance = profile.distance(embedding)
            score = self.calibrator.score(distance)
        except ValueError:
            return AuthenticationResult(target_user_id, "insufficient_data", None, None, self.threshold, None, "sequence_incompatible_with_profile")
        difference = score - self.threshold
        if abs(difference) < self.uncertainty_margin:
            return AuthenticationResult(
                target_user_id=target_user_id,
                status="low_confidence",
                accepted=None,
                similarity_score=score * 100.0,
                threshold=self.threshold * 100.0,
                distance=distance,
                reason="score_near_decision_threshold",
            )
        accepted = difference >= 0
        return AuthenticationResult(
            target_user_id=target_user_id,
            status="authenticated" if accepted else "rejected",
            accepted=accepted,
            similarity_score=score * 100.0,
            threshold=self.threshold * 100.0,
            distance=distance,
        )
