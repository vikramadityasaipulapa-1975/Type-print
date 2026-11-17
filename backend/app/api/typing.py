from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.api.biometrics import _check_consent, _events
from app.core.rate_limit import rate_limit
from app.core.security import SecurityConfigurationError
from app.database.session import get_db
from app.models.entities import AuthenticationLog, BiometricProfile, TypingSession, User
from app.schemas.api import AuthenticateResponse, TypingSampleRequest, TypingStatsResponse
from app.services.biometric import ModelService, get_model_service

router = APIRouter(prefix="/api/typing", tags=["typing"])


def _stats(result: object) -> dict[str, float | int]:
    features = result.features
    return {
        "average_dwell_ms": features["dwell_ms_mean"],
        "average_flight_ms": features["flight_ms_mean"],
        "typing_consistency": features["typing_consistency"],
        "characters_per_minute": features["characters_per_minute"],
        "event_count": result.accepted_event_count,
        "duration_ms": features["duration_ms"],
    }


@router.post("/analyze", response_model=TypingStatsResponse)
def analyze(
    payload: TypingSampleRequest,
    request: Request,
    user: User = Depends(get_current_user),
    service: ModelService = Depends(get_model_service),
) -> dict[str, float | int]:
    rate_limit(request, f"analyze:{user.id}", 60)
    _check_consent(payload)
    try:
        result = service.extract(_events(payload), str(payload.session_id))
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    return _stats(result)


@router.post("/authenticate", response_model=AuthenticateResponse)
def authenticate(
    payload: TypingSampleRequest,
    request: Request,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    service: ModelService = Depends(get_model_service),
) -> AuthenticateResponse:
    rate_limit(request, f"authenticate:{user.id}", 20)
    _check_consent(payload)
    try:
        result = service.extract(_events(payload), str(payload.session_id))
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    profile_row = db.scalar(select(BiometricProfile).where(BiometricProfile.user_id == user.id))
    if profile_row is None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="No biometric profile is enrolled")
    if not service.encoder:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="LSTM verification model is unavailable")
    try:
        decision = service.authenticate(profile_row, result)
        database_session_id = str(uuid4())
        protected = service.encrypted_sample(result, None, f"typing:{user.id}:{database_session_id}")
    except SecurityConfigurationError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc

    db.add(TypingSession(
        id=database_session_id,
        user_id=user.id,
        collection_session_id=str(payload.session_id),
        session_type="authentication",
        sample_metadata=protected,
    ))
    db.add(AuthenticationLog(
        user_id=user.id,
        similarity_score=decision["similarity_score"],
        authentication_status=str(decision["status"]),
        session_id=str(payload.session_id),
    ))
    db.commit()
    return AuthenticateResponse(session_id=payload.session_id, **decision)
