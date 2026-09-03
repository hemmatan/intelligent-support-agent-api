"""Password hashing and signed-token primitives."""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from hashlib import sha256
from typing import Any
from uuid import uuid4

import jwt
from fastapi.concurrency import run_in_threadpool
from pwdlib import PasswordHash
from pwdlib.hashers.argon2 import Argon2Hasher
from pwdlib.hashers.bcrypt import BcryptHasher

from app.core.config import settings

_password_hash = PasswordHash((Argon2Hasher(), BcryptHasher()))
DUMMY_PASSWORD_HASH = _password_hash.hash("not-a-real-password")


class TokenType(StrEnum):
    ACCESS = "access"
    REFRESH = "refresh"


@dataclass(frozen=True)
class IssuedToken:
    value: str
    jti: str
    expires_at: datetime


# Argon2 is deliberately expensive: roughly 150ms of CPU per call. Running it
# inline would block the event loop for the whole request handler, so both
# entry points hand the work to a worker thread.


def _verify_and_update(plain: str, hashed: str) -> tuple[bool, str | None]:
    return _password_hash.verify_and_update(plain, hashed)


async def verify_and_update_password(
    plain_password: str, hashed_password: str
) -> tuple[bool, str | None]:
    """Verify a password and return an upgraded Argon2 hash when needed."""
    return await run_in_threadpool(_verify_and_update, plain_password, hashed_password)


async def get_password_hash(password: str) -> str:
    """Hash new passwords with Argon2."""
    return await run_in_threadpool(_password_hash.hash, password)


def hash_token(token: str) -> str:
    """Create a one-way lookup value for stored bearer tokens."""
    return sha256(token.encode("utf-8")).hexdigest()


def _issue_token(
    *,
    username: str,
    user_id: int,
    role: str,
    token_type: TokenType,
    expires_delta: timedelta,
) -> IssuedToken:
    now = datetime.now(UTC)
    expires_at = now + expires_delta
    jti = str(uuid4())
    payload: dict[str, Any] = {
        "sub": str(user_id),
        "username": username,
        "role": role,
        "type": token_type.value,
        "jti": jti,
        "iat": now,
        "nbf": now,
        "exp": expires_at,
        "iss": settings.JWT_ISSUER,
        "aud": settings.JWT_AUDIENCE,
    }
    value = jwt.encode(
        payload,
        settings.SECRET_KEY.get_secret_value(),
        algorithm=settings.ALGORITHM,
    )
    return IssuedToken(value=value, jti=jti, expires_at=expires_at)


def create_access_token(
    username: str,
    user_id: int,
    role: str = "customer",
    expires_delta: timedelta | None = None,
) -> IssuedToken:
    return _issue_token(
        username=username,
        user_id=user_id,
        role=role,
        token_type=TokenType.ACCESS,
        expires_delta=expires_delta
        or timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES),
    )


def create_refresh_token(
    username: str,
    user_id: int,
    role: str = "customer",
    expires_delta: timedelta | None = None,
) -> IssuedToken:
    return _issue_token(
        username=username,
        user_id=user_id,
        role=role,
        token_type=TokenType.REFRESH,
        expires_delta=expires_delta
        or timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS),
    )


def decode_token(token: str, expected_type: TokenType) -> dict[str, Any]:
    """Decode a JWT and enforce issuer, audience, and token purpose."""
    payload: dict[str, Any] = jwt.decode(
        token,
        settings.SECRET_KEY.get_secret_value(),
        algorithms=[settings.ALGORITHM],
        audience=settings.JWT_AUDIENCE,
        issuer=settings.JWT_ISSUER,
        options={"require": ["sub", "type", "jti", "iat", "nbf", "exp"]},
    )
    if payload.get("type") != expected_type.value:
        raise jwt.InvalidTokenError("Unexpected token type")
    return payload
