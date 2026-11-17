from dataclasses import dataclass
from datetime import datetime, timezone

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from app.core.security import SecurityConfigurationError, decode_access_token
from app.database.session import get_db
from app.models.entities import SessionToken, User

bearer_scheme = HTTPBearer(auto_error=False)


@dataclass(frozen=True)
class AuthContext:
    user: User
    session_token: SessionToken


def get_auth_context(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    db: Session = Depends(get_db),
) -> AuthContext:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required", headers={"WWW-Authenticate": "Bearer"})
    try:
        claims = decode_access_token(credentials.credentials)
        user_id = int(claims["sub"])
        jti = str(claims["jti"])
    except SecurityConfigurationError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    except (jwt.InvalidTokenError, KeyError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired session", headers={"WWW-Authenticate": "Bearer"}) from exc
    token_row = db.get(SessionToken, jti)
    now = datetime.now(timezone.utc)
    expiry = token_row.expires_at if token_row else now
    if expiry.tzinfo is None:
        expiry = expiry.replace(tzinfo=timezone.utc)
    if token_row is None or token_row.revoked_at is not None or expiry <= now or token_row.user_id != user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired session", headers={"WWW-Authenticate": "Bearer"})
    user = db.get(User, user_id)
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid session", headers={"WWW-Authenticate": "Bearer"})
    return AuthContext(user, token_row)


def get_current_user(context: AuthContext = Depends(get_auth_context)) -> User:
    return context.user
