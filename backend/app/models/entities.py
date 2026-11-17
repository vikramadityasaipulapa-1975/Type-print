from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Index, Integer, String, Text, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.session import Base


class User(Base):
    __tablename__ = "users"
    __table_args__ = (UniqueConstraint("username", name="uq_users_username"), UniqueConstraint("email", name="uq_users_email"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    username: Mapped[str] = mapped_column(String(48), nullable=False, index=True)
    email: Mapped[str] = mapped_column(String(254), nullable=False, index=True)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    biometric_profile: Mapped["BiometricProfile | None"] = relationship(back_populates="user", cascade="all, delete-orphan", uselist=False)
    sessions: Mapped[list["TypingSession"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    authentication_logs: Mapped[list["AuthenticationLog"]] = relationship(back_populates="user", cascade="all, delete-orphan")


class SessionToken(Base):
    __tablename__ = "session_tokens"

    jti: Mapped[str] = mapped_column(String(36), primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class BiometricProfile(Base):
    __tablename__ = "biometric_profiles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False, unique=True, index=True)
    model_version: Mapped[str] = mapped_column(String(64), nullable=False)
    protected_profile_data: Mapped[str] = mapped_column(Text, nullable=False)
    enrollment_samples: Mapped[int] = mapped_column(Integer, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    user: Mapped[User] = relationship(back_populates="biometric_profile")


class TypingSession(Base):
    __tablename__ = "typing_sessions"
    __table_args__ = (Index("ix_typing_sessions_user_type_time", "user_id", "session_type", "session_timestamp"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    collection_session_id: Mapped[str | None] = mapped_column(String(36), index=True)
    session_type: Mapped[str] = mapped_column(String(24), nullable=False)
    session_timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False, index=True)
    sample_metadata: Mapped[str] = mapped_column(Text, nullable=False)
    user: Mapped[User] = relationship(back_populates="sessions")


class AuthenticationLog(Base):
    __tablename__ = "authentication_logs"
    __table_args__ = (Index("ix_auth_logs_user_timestamp", "user_id", "timestamp"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    similarity_score: Mapped[float | None] = mapped_column(Float)
    authentication_status: Mapped[str] = mapped_column(String(24), nullable=False, index=True)
    session_id: Mapped[str | None] = mapped_column(String(36))
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False, index=True)
    user: Mapped[User] = relationship(back_populates="authentication_logs")


class ModelMetric(Base):
    __tablename__ = "model_metrics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    model_name: Mapped[str] = mapped_column(String(96), nullable=False, index=True)
    far: Mapped[float | None] = mapped_column(Float)
    frr: Mapped[float | None] = mapped_column(Float)
    eer: Mapped[float | None] = mapped_column(Float)
    roc_auc: Mapped[float | None] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
