from __future__ import annotations

import base64
import json
import secrets
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from app.core.config import settings

_password_hasher = PasswordHasher()


class SecurityConfigurationError(RuntimeError):
    pass


def hash_password(password: str) -> str:
    return _password_hasher.hash(password)


def verify_password(password: str, encoded: str) -> bool:
    try:
        return _password_hasher.verify(encoded, password)
    except (VerifyMismatchError, VerificationError, InvalidHashError):
        return False


def create_access_token(user_id: int) -> tuple[str, str, datetime]:
    if len(settings.jwt_secret_key) < 32:
        raise SecurityConfigurationError("JWT_SECRET_KEY must contain at least 32 characters")
    jti = str(uuid4())
    expires_at = datetime.now(timezone.utc) + timedelta(minutes=settings.jwt_expire_minutes)
    token = jwt.encode(
        {"sub": str(user_id), "jti": jti, "iat": datetime.now(timezone.utc), "exp": expires_at},
        settings.jwt_secret_key,
        algorithm="HS256",
    )
    return token, jti, expires_at


def decode_access_token(token: str) -> dict[str, object]:
    if len(settings.jwt_secret_key) < 32:
        raise SecurityConfigurationError("JWT_SECRET_KEY must contain at least 32 characters")
    return jwt.decode(token, settings.jwt_secret_key, algorithms=["HS256"])


def _encryption_key() -> bytes:
    if not settings.biometric_encryption_key:
        raise SecurityConfigurationError("BIOMETRIC_ENCRYPTION_KEY is not configured")
    try:
        key = base64.urlsafe_b64decode(settings.biometric_encryption_key.encode("ascii"))
    except (ValueError, UnicodeEncodeError) as exc:
        raise SecurityConfigurationError("BIOMETRIC_ENCRYPTION_KEY must be a URL-safe base64 encoded 32-byte key") from exc
    if len(key) != 32:
        raise SecurityConfigurationError("BIOMETRIC_ENCRYPTION_KEY must decode to exactly 32 bytes")
    return key


def encrypt_json(payload: dict[str, object], context: str = "general") -> str:
    nonce = secrets.token_bytes(12)
    plaintext = json.dumps(payload, separators=(",", ":"), allow_nan=False).encode("utf-8")
    associated_data = b"typing-fingerprint:v1:" + context.encode("utf-8")
    encrypted = AESGCM(_encryption_key()).encrypt(nonce, plaintext, associated_data)
    return "v1:" + base64.urlsafe_b64encode(nonce + encrypted).decode("ascii")


def decrypt_json(ciphertext: str, context: str = "general") -> dict[str, object]:
    if not ciphertext.startswith("v1:"):
        raise ValueError("unsupported biometric ciphertext version")
    value = base64.urlsafe_b64decode(ciphertext[3:].encode("ascii"))
    nonce, encrypted = value[:12], value[12:]
    associated_data = b"typing-fingerprint:v1:" + context.encode("utf-8")
    plaintext = AESGCM(_encryption_key()).decrypt(nonce, encrypted, associated_data)
    data = json.loads(plaintext)
    if not isinstance(data, dict):
        raise ValueError("decrypted biometric payload must be an object")
    return data
