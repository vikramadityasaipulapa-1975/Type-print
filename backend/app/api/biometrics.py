from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.core.config import settings
from app.core.rate_limit import rate_limit
from app.core.security import SecurityConfigurationError
from app.database.session import get_db
from app.models.entities import AuthenticationLog, BiometricProfile, TypingSession, User
from app.schemas.api import EnrollmentResponse, ProfileResponse, TypingSampleRequest
from app.services.biometric import ModelService, get_model_service

router = APIRouter(prefix="/api/biometrics", tags=["biometrics"])


def _events(payload: TypingSampleRequest) -> list[dict[str, object]]:
    return [event.model_dump(by_alias=True, exclude_none=False) for event in payload.events]


def _check_consent(payload: TypingSampleRequest) -> None:
    if not payload.consent:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Explicit biometric consent is required")
    if not payload.passage_completed:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Complete the fixed passage before submitting a sample")


@router.post("/enroll", response_model=EnrollmentResponse, status_code=status.HTTP_201_CREATED)
def enroll(
    payload: TypingSampleRequest,
    request: Request,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    service: ModelService = Depends(get_model_service),
) -> EnrollmentResponse:
    rate_limit(request, f"enroll:{user.id}", 20)
    _check_consent(payload)
    try:
        result = service.extract(_events(payload), str(payload.session_id))
        embedding = service.encode(result)
        database_session_id = str(uuid4())
        protected = service.encrypted_sample(result, embedding, f"typing:{user.id}:{database_session_id}")
    except SecurityConfigurationError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc

    session = TypingSession(
        id=database_session_id,
        user_id=user.id,
        collection_session_id=str(payload.session_id),
        session_type="enrollment",
        sample_metadata=protected,
    )
    db.add(session)
    try:
        db.flush()
        sample_count = db.scalar(select(func.count()).select_from(TypingSession).where(TypingSession.user_id == user.id, TypingSession.session_type == "enrollment")) or 0
        profile = service.rebuild_profile(db, user.id)
        db.commit()
    except Exception:
        db.rollback()
        raise
    ready = bool(profile and service.verification_ready and profile.model_version == service.model_version)
    if ready:
        current_status = "enrolled"
    elif sample_count >= settings.minimum_enrollment_samples and not service.encoder:
        current_status = "model_unavailable"
    elif sample_count >= settings.minimum_enrollment_samples:
        current_status = "calibration_unavailable"
    else:
        current_status = "collecting"
    return EnrollmentResponse(
        status=current_status,
        enrollment_samples=sample_count,
        samples_remaining=max(0, settings.minimum_enrollment_samples - sample_count),
        profile_ready=ready,
        model_version=profile.model_version if ready and profile else None,
    )


@router.get("/profile", response_model=ProfileResponse)
def get_profile(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    service: ModelService = Depends(get_model_service),
) -> ProfileResponse:
    try:
        profile = service.rebuild_profile(db, user.id)
        db.commit()
    except SecurityConfigurationError as exc:
        db.rollback()
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    sample_count = db.scalar(select(func.count()).select_from(TypingSession).where(TypingSession.user_id == user.id, TypingSession.session_type == "enrollment")) or 0
    ready = bool(profile and service.verification_ready and profile.model_version == service.model_version)
    return ProfileResponse(
        enrolled=sample_count > 0,
        enrollment_samples=sample_count,
        minimum_samples=settings.minimum_enrollment_samples,
        profile_ready=ready,
        model_version=profile.model_version if ready and profile else None,
    )


@router.delete("/profile", status_code=status.HTTP_204_NO_CONTENT)
def delete_profile(user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> Response:
    db.query(BiometricProfile).filter(BiometricProfile.user_id == user.id).delete(synchronize_session=False)
    db.query(TypingSession).filter(TypingSession.user_id == user.id).delete(synchronize_session=False)
    db.query(AuthenticationLog).filter(AuthenticationLog.user_id == user.id).delete(synchronize_session=False)
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)
