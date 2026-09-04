"""Authentication business logic."""

import secrets
from datetime import UTC, datetime

import jwt
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import (
    DUMMY_PASSWORD_HASH,
    TokenType,
    create_access_token,
    create_refresh_token,
    decode_token,
    hash_token,
    verify_and_update_password,
)
from app.models.user import RefreshToken, User


def _as_utc(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


async def authenticate_user(
    db: AsyncSession, username: str, password: str
) -> User | None:
    """Authenticate without revealing whether the username exists through timing."""
    result = await db.execute(select(User).where(User.username == username))
    user = result.scalar_one_or_none()
    stored_hash = user.hashed_password if user is not None else DUMMY_PASSWORD_HASH
    valid, updated_hash = await verify_and_update_password(password, stored_hash)
    if user is None or not valid or not user.is_active:
        return None
    if updated_hash is not None:
        user.hashed_password = updated_hash
        await db.commit()
    return user


async def create_tokens_for_user(db: AsyncSession, user: User) -> tuple[str, str]:
    """Issue an access token and persist only a hash of its refresh token."""
    access = create_access_token(user.username, user.id, user.role.value)
    refresh = create_refresh_token(user.username, user.id, user.role.value)
    db.add(
        RefreshToken(
            jti=refresh.jti,
            token_hash=hash_token(refresh.value),
            user_id=user.id,
            expires_at=refresh.expires_at,
        )
    )
    await db.commit()
    return access.value, refresh.value


async def rotate_refresh_token(
    raw_token: str, db: AsyncSession
) -> tuple[str, str] | None:
    """Revoke a refresh token and replace it atomically with a new token pair."""
    try:
        payload = decode_token(raw_token, TokenType.REFRESH)
        user_id = int(payload["sub"])
        jti = str(payload["jti"])
    except (jwt.InvalidTokenError, KeyError, TypeError, ValueError):
        return None

    result = await db.execute(select(RefreshToken).where(RefreshToken.jti == jti))
    stored = result.scalar_one_or_none()
    now = datetime.now(UTC)
    if (
        stored is None
        or stored.user_id != user_id
        or stored.revoked_at is not None
        or _as_utc(stored.expires_at) <= now
        or not secrets.compare_digest(stored.token_hash, hash_token(raw_token))
    ):
        return None

    user = await db.get(User, user_id)
    if user is None or not user.is_active:
        return None

    access = create_access_token(user.username, user.id, user.role.value)
    refresh = create_refresh_token(user.username, user.id, user.role.value)

    # Claim the token with a conditional update rather than a row lock:
    # SQLAlchemy omits FOR UPDATE on SQLite, so two concurrent redemptions
    # would both pass the checks above and each be issued a valid pair.
    # Exactly one caller can move revoked_at away from NULL.
    claimed = await db.execute(
        update(RefreshToken)
        .where(RefreshToken.jti == jti, RefreshToken.revoked_at.is_(None))
        .values(revoked_at=now, replaced_by_jti=refresh.jti)
    )
    if claimed.rowcount != 1:
        await db.rollback()
        return None

    db.add(
        RefreshToken(
            jti=refresh.jti,
            token_hash=hash_token(refresh.value),
            user_id=user.id,
            expires_at=refresh.expires_at,
        )
    )
    await db.commit()
    return access.value, refresh.value


async def revoke_refresh_token(
    raw_token: str, db: AsyncSession, current_user: User
) -> bool:
    """Revoke a refresh token owned by the authenticated user."""
    try:
        payload = decode_token(raw_token, TokenType.REFRESH)
        jti = str(payload["jti"])
        user_id = int(payload["sub"])
    except (jwt.InvalidTokenError, KeyError, TypeError, ValueError):
        return False
    if user_id != current_user.id:
        return False
    claimed = await db.execute(
        update(RefreshToken)
        .where(RefreshToken.jti == jti, RefreshToken.revoked_at.is_(None))
        .values(revoked_at=datetime.now(UTC))
    )
    if claimed.rowcount != 1:
        await db.rollback()
        return False
    await db.commit()
    return True
