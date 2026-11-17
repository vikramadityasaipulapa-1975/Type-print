from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.security import decrypt_json, encrypt_json
from app.models.entities import BiometricProfile, TypingSession
from ml.biometrics.calibration import DistanceCalibrator
from ml.biometrics.profile import EnrollmentProfile, create_profile
from ml.features.keystroke_features import FeatureResult, extract_sample
from ml.inference.authentication import AuthenticationEngine
from ml.inference.lstm_encoder import LSTMEncoder


class ModelService:
    def __init__(self) -> None:
        self.encoder: LSTMEncoder | None = None
        self.calibrator: DistanceCalibrator | None = None
        checkpoint_path = Path(settings.lstm_checkpoint_path)
        calibrator_path = Path(settings.verification_calibrator_path)
        if checkpoint_path.is_file():
            try:
                self.encoder = LSTMEncoder.from_checkpoint(checkpoint_path)
            except Exception:
                self.encoder = None
        if calibrator_path.is_file():
            try:
                candidate = joblib.load(calibrator_path)
                if isinstance(candidate, DistanceCalibrator) and self.encoder and candidate.model_version == self.encoder.model_version:
                    self.calibrator = candidate
            except Exception:
                self.calibrator = None

    @property
    def model_version(self) -> str | None:
        return self.encoder.model_version if self.encoder else None

    @property
    def verification_ready(self) -> bool:
        return self.encoder is not None and self.calibrator is not None

    def status(self) -> dict[str, object]:
        rf_path = Path("models/random_forest.joblib")
        return {
            "random_forest_available": rf_path.is_file(),
            "lstm_available": self.encoder is not None,
            "verification_calibrator_available": self.calibrator is not None,
            "model_version": self.model_version,
            "verification_ready": self.verification_ready,
        }

    def extract(self, events: list[dict[str, Any]], session_id: str) -> FeatureResult:
        result = extract_sample({"session_id": session_id, "events": events})
        if result.accepted_event_count < settings.minimum_typing_events:
            raise ValueError(f"at least {settings.minimum_typing_events} valid key events are required")
        return result

    def encode(self, result: FeatureResult) -> np.ndarray | None:
        return self.encoder.encode_sequence(result.sequence) if self.encoder else None

    def encrypted_sample(self, result: FeatureResult, embedding: np.ndarray | None, context: str, include_sequence: bool = True) -> str:
        payload: dict[str, object] = {
            "version": 1,
            "features": result.features,
            "accepted_event_count": result.accepted_event_count,
            "rejected_event_count": result.rejected_event_count,
        }
        if include_sequence:
            payload["sequence"] = result.sequence.tolist()
        if embedding is not None and self.model_version:
            payload["embedding"] = embedding.tolist()
            payload["model_version"] = self.model_version
        return encrypt_json(payload, context)

    def _load_embeddings(self, db: Session, user_id: int) -> list[np.ndarray]:
        if not self.encoder:
            return []
        rows = db.scalars(
            select(TypingSession).where(TypingSession.user_id == user_id, TypingSession.session_type == "enrollment")
        ).all()
        embeddings: list[np.ndarray] = []
        for row in rows:
            payload = decrypt_json(row.sample_metadata, f"typing:{row.user_id}:{row.id}")
            vector = payload.get("embedding")
            if payload.get("model_version") == self.model_version and isinstance(vector, list):
                embeddings.append(np.asarray(vector, dtype=np.float32))
                continue
            sequence = payload.get("sequence")
            if not isinstance(sequence, list):
                continue
            encoded = self.encoder.encode_sequence(np.asarray(sequence, dtype=np.float32))
            payload["embedding"] = encoded.tolist()
            payload["model_version"] = self.model_version
            row.sample_metadata = encrypt_json(payload, f"typing:{row.user_id}:{row.id}")
            embeddings.append(encoded)
        return embeddings

    def rebuild_profile(self, db: Session, user_id: int) -> BiometricProfile | None:
        if not self.encoder:
            return db.scalar(select(BiometricProfile).where(BiometricProfile.user_id == user_id))
        embeddings = self._load_embeddings(db, user_id)
        if len(embeddings) < settings.minimum_enrollment_samples:
            return db.scalar(select(BiometricProfile).where(BiometricProfile.user_id == user_id))
        profile = create_profile(
            str(user_id), np.vstack(embeddings), self.encoder.model_version, min_samples=settings.minimum_enrollment_samples
        )
        payload = {
            "centroid": profile.centroid.tolist(),
            "scale": profile.scale.tolist(),
            "enrollment_samples": profile.enrollment_samples,
            "model_version": profile.model_version,
        }
        protected = encrypt_json(payload, f"profile:{user_id}")
        row = db.scalar(select(BiometricProfile).where(BiometricProfile.user_id == user_id))
        if row is None:
            row = BiometricProfile(user_id=user_id, model_version=profile.model_version, protected_profile_data=protected, enrollment_samples=profile.enrollment_samples)
            db.add(row)
        else:
            row.model_version = profile.model_version
            row.protected_profile_data = protected
            row.enrollment_samples = profile.enrollment_samples
        db.flush()
        return row

    def load_profile(self, row: BiometricProfile) -> EnrollmentProfile:
        payload = decrypt_json(row.protected_profile_data, f"profile:{row.user_id}")
        return EnrollmentProfile(
            user_id=str(row.user_id),
            model_version=str(payload["model_version"]),
            centroid=np.asarray(payload["centroid"], dtype=np.float32),
            scale=np.asarray(payload["scale"], dtype=np.float32),
            enrollment_samples=int(payload["enrollment_samples"]),
        )

    def authenticate(self, profile_row: BiometricProfile, result: FeatureResult) -> dict[str, object]:
        if not self.encoder:
            raise RuntimeError("LSTM model is unavailable")
        profile = self.load_profile(profile_row)
        engine = AuthenticationEngine(
            self.encoder,
            {profile.user_id: profile},
            self.calibrator,
            threshold=settings.verification_threshold,
            minimum_sequence_length=settings.minimum_typing_events,
        )
        response = engine.verify(profile.user_id, result.sequence)
        return {
            "status": response.status,
            "accepted": response.accepted,
            "similarity_score": response.similarity_score,
            "threshold": response.threshold,
            "reason": response.reason,
        }


_model_service: ModelService | None = None


def get_model_service() -> ModelService:
    global _model_service
    if _model_service is None:
        _model_service = ModelService()
    return _model_service
