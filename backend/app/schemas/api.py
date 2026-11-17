from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator


class RegisterRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    username: str = Field(min_length=3, max_length=48, pattern=r"^[A-Za-z0-9_.-]+$")
    email: EmailStr
    password: str = Field(min_length=12, max_length=128)

    @field_validator("email")
    @classmethod
    def normalize_email(cls, value: EmailStr) -> str:
        return str(value).lower()

    @field_validator("username")
    @classmethod
    def normalize_username(cls, value: str) -> str:
        return value.strip().lower()


class LoginRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    identifier: str = Field(min_length=3, max_length=254)
    password: str = Field(min_length=1, max_length=128)


class UserResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    username: str
    email: EmailStr
    created_at: datetime


class TokenResponse(BaseModel):
    access_token: str
    token_type: Literal["bearer"] = "bearer"
    expires_at: datetime


class TypingEvent(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")
    id: int | None = None
    code: str = Field(pattern=r"^(Key[A-Z]|Digit[0-9]|Numpad[0-9]|Space|Backspace|Minus|Equal|BracketLeft|BracketRight|Backslash|Semicolon|Quote|Comma|Period|Slash|Backquote)$")
    down_ms: float | None = Field(default=None, alias="downMs", ge=0)
    up_ms: float | None = Field(default=None, alias="upMs", ge=0)
    dwell_ms: float | None = Field(default=None, alias="dwellMs", ge=0, le=2500)
    flight_ms: float | None = Field(default=None, alias="flightMs", ge=-2500, le=10000)


class TypingSampleRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sample_id: UUID
    session_id: UUID
    consent: bool
    passage_completed: bool
    events: list[TypingEvent] = Field(min_length=2, max_length=256)


class TypingStatsResponse(BaseModel):
    average_dwell_ms: float
    average_flight_ms: float
    typing_consistency: float
    characters_per_minute: float
    event_count: int
    duration_ms: float


class EnrollmentResponse(BaseModel):
    status: str
    enrollment_samples: int
    samples_remaining: int
    profile_ready: bool
    model_version: str | None = None


class ProfileResponse(BaseModel):
    enrolled: bool
    enrollment_samples: int
    minimum_samples: int
    profile_ready: bool
    model_version: str | None = None


class AuthenticateResponse(BaseModel):
    session_id: UUID
    status: str
    accepted: bool | None
    similarity_score: float | None
    threshold: float | None
    reason: str | None = None


class AnalyticsDashboardResponse(BaseModel):
    typing_speed_cpm: float | None
    average_dwell_ms: float | None
    average_flight_ms: float | None
    typing_consistency: float | None
    authentication_attempts: int
    successful_authentications: int
    failed_authentications: int
    enrollment_samples: int
    profile_ready: bool


class AuthenticationHistoryItem(BaseModel):
    timestamp: datetime
    authentication_status: str
    similarity_score: float | None
    session_id: UUID | None


class TimingPoint(BaseModel):
    key_family: str
    dwell_ms: float
    flight_ms: float


class TimingHistoryItem(BaseModel):
    timestamp: datetime
    session_type: str
    average_dwell_ms: float | None
    average_flight_ms: float | None
    characters_per_minute: float | None
    event_count: int | None
    sequence: list[TimingPoint] | None = None


class ModelStatusResponse(BaseModel):
    random_forest_available: bool
    lstm_available: bool
    verification_calibrator_available: bool
    model_version: str | None
    verification_ready: bool


class ModelMetricsResponse(BaseModel):
    random_forest: dict[str, object] | None
    verification: dict[str, object] | None
