"""FastAPI dependencies for database sessions and authentication."""

from datetime import UTC, datetime
from hashlib import sha256
from typing import Annotated

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.security import TokenType, decode_token
from app.db.session import get_db
from app.models.user import APIToken, User, UserRole

DBSessionDep = Annotated[AsyncSession, Depends(get_db)]
bearer_scheme = HTTPBearer(auto_error=False)
api_key_header = APIKeyHeader(name="X-API-Token", auto_error=False)
BearerCredentialsDep = Annotated[
    HTTPAuthorizationCredentials | None, Depends(bearer_scheme)
]
APIKeyDep = Annotated[str | None, Depends(api_key_header)]


def credentials_error() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )


async def get_current_user(
    db: DBSessionDep,
    credentials: BearerCredentialsDep,
) -> User:
    """Authenticate an access token and reload authoritative user state."""
    if credentials is None:
        raise credentials_error()
    try:
        payload = decode_token(credentials.credentials, TokenType.ACCESS)
        user_id = int(payload["sub"])
    except (jwt.InvalidTokenError, KeyError, TypeError, ValueError) as exc:
        raise credentials_error() from exc
    user = await db.get(User, user_id)
    if user is None or not user.is_active:
        raise credentials_error()
    return user


AuthUserDep = Annotated[User, Depends(get_current_user)]


async def get_current_staff(current_user: AuthUserDep) -> User:
    if current_user.role not in {UserRole.SUPPORT_AGENT, UserRole.ADMIN}:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Staff role required"
        )
    return current_user


StaffUserDep = Annotated[User, Depends(get_current_staff)]


async def get_current_user_token(db: DBSessionDep, api_token: APIKeyDep) -> User:
    if not api_token:
        raise credentials_error()
    token_hash = sha256(api_token.encode("utf-8")).hexdigest()
    result = await db.execute(
        select(APIToken)
        .options(selectinload(APIToken.user))
        .where(APIToken.token_hash == token_hash)
    )
    record = result.scalar_one_or_none()
    now = datetime.now(UTC)
    if (
        record is None
        or record.revoked_at is not None
        or (record.expires_at is not None and _as_utc(record.expires_at) <= now)
        or not record.user.is_active
    ):
        raise credentials_error()
    record.last_used_at = now
    await db.commit()
    return record.user


def _as_utc(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


TokenUserDep = Annotated[User, Depends(get_current_user_token)]
