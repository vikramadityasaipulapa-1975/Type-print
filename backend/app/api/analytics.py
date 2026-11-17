from datetime import timezone

from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.core.security import decrypt_json
from app.database.session import get_db
from app.models.entities import AuthenticationLog, BiometricProfile, TypingSession, User
from app.schemas.api import AnalyticsDashboardResponse, AuthenticationHistoryItem, TimingHistoryItem
from ml.features.keystroke_features import KEY_FAMILIES
from app.services.biometric import ModelService, get_model_service

router = APIRouter(prefix="/api/analytics", tags=["analytics"])


def _read_payload(row: TypingSession) -> dict[str, object] | None:
    try:
        return decrypt_json(row.sample_metadata, f"typing:{row.user_id}:{row.id}")
    except Exception:
        return None


def _read_stats(row: TypingSession) -> dict[str, float | int] | None:
    payload = _read_payload(row)
    if not payload:
        return None
    features = payload.get("features")
    if not isinstance(features, dict):
        return None
    try:
        return {
            "average_dwell_ms": float(features["dwell_ms_mean"]),
            "average_flight_ms": float(features["flight_ms_mean"]),
            "typing_consistency": float(features["typing_consistency"]),
            "characters_per_minute": float(features["characters_per_minute"]),
            "duration_ms": float(features["duration_ms"]),
            "event_count": int(payload["accepted_event_count"]),
        }
    except (KeyError, TypeError, ValueError):
        return None


@router.get("/dashboard", response_model=AnalyticsDashboardResponse)
def dashboard(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    service: ModelService = Depends(get_model_service),
) -> AnalyticsDashboardResponse:
    attempts = db.scalar(select(func.count()).select_from(AuthenticationLog).where(AuthenticationLog.user_id == user.id)) or 0
    successes = db.scalar(
        select(func.count()).select_from(AuthenticationLog).where(
            AuthenticationLog.user_id == user.id,
            AuthenticationLog.authentication_status == "authenticated",
        )
    ) or 0
    sample_rows = db.scalars(
        select(TypingSession).where(TypingSession.user_id == user.id).order_by(TypingSession.session_timestamp.desc()).limit(100)
    ).all()
    summaries = [_read_stats(row) for row in sample_rows]
    summaries = [summary for summary in summaries if summary]
    average = lambda name: sum(float(item[name]) for item in summaries) / len(summaries) if summaries else None
    profile = db.scalar(select(BiometricProfile).where(BiometricProfile.user_id == user.id))
    enrollment_samples = db.scalar(
        select(func.count()).select_from(TypingSession).where(TypingSession.user_id == user.id, TypingSession.session_type == "enrollment")
    ) or 0
    return AnalyticsDashboardResponse(
        typing_speed_cpm=average("characters_per_minute"),
        average_dwell_ms=average("average_dwell_ms"),
        average_flight_ms=average("average_flight_ms"),
        typing_consistency=average("typing_consistency"),
        authentication_attempts=int(attempts),
        successful_authentications=successes,
        failed_authentications=attempts - successes,
        enrollment_samples=enrollment_samples,
        profile_ready=bool(profile and service.verification_ready and profile.model_version == service.model_version),
    )


@router.get("/history", response_model=list[AuthenticationHistoryItem])
def history(user: User = Depends(get_current_user), db: Session = Depends(get_db), limit: int = 50) -> list[AuthenticationHistoryItem]:
    limit = max(1, min(limit, 100))
    rows = db.scalars(
        select(AuthenticationLog).where(AuthenticationLog.user_id == user.id).order_by(AuthenticationLog.timestamp.desc()).limit(limit)
    ).all()
    return [
        AuthenticationHistoryItem(
            timestamp=row.timestamp.replace(tzinfo=timezone.utc) if row.timestamp.tzinfo is None else row.timestamp,
            authentication_status=row.authentication_status,
            similarity_score=row.similarity_score,
            session_id=row.session_id,
        )
        for row in rows
    ]


@router.get("/timing", response_model=list[TimingHistoryItem])
def timing(user: User = Depends(get_current_user), db: Session = Depends(get_db), limit: int = 50) -> list[TimingHistoryItem]:
    limit = max(1, min(limit, 100))
    rows = db.scalars(
        select(TypingSession).where(TypingSession.user_id == user.id).order_by(TypingSession.session_timestamp.desc()).limit(limit)
    ).all()
    output: list[TimingHistoryItem] = []
    for row in rows:
        payload = _read_payload(row) or {}
        stats = _read_stats(row) or {}
        raw_sequence = payload.get("sequence")
        sequence = None
        if isinstance(raw_sequence, list):
            sequence = []
            for point in raw_sequence[-80:]:
                if not isinstance(point, list) or len(point) < 8:
                    continue
                family_index = max(range(6), key=lambda index: point[index + 2])
                sequence.append({"key_family": KEY_FAMILIES[family_index], "dwell_ms": float(point[0]), "flight_ms": float(point[1])})
        timestamp = row.session_timestamp.replace(tzinfo=timezone.utc) if row.session_timestamp.tzinfo is None else row.session_timestamp
        output.append(TimingHistoryItem(
            timestamp=timestamp,
            session_type=row.session_type,
            average_dwell_ms=stats.get("average_dwell_ms"),
            average_flight_ms=stats.get("average_flight_ms"),
            characters_per_minute=stats.get("characters_per_minute"),
            event_count=stats.get("event_count"),
            sequence=sequence,
        ))
    return output
