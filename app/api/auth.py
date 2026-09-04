"""Authentication routes."""

import secrets
from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, HTTPException, Response, status
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from app.api.deps import AuthUserDep, DBSessionDep, TokenUserDep
from app.core.security import get_password_hash, hash_token
from app.models.user import APIToken, User
from app.schemas.token import (
    APITokenCreate,
    APITokenOut,
    LogoutRequest,
    RefreshTokenRequest,
    Token,
)
from app.schemas.user import UserCreate, UserLogin, UserOut
from app.services.auth import (
    authenticate_user,
    create_tokens_for_user,
    revoke_refresh_token,
    rotate_refresh_token,
)

router = APIRouter()


@router.post("/signup", response_model=UserOut, status_code=status.HTTP_201_CREATED)
async def create_user(user_data: UserCreate, db: DBSessionDep) -> User:
    """Register a new customer account."""
    result = await db.execute(
        select(User.id).where(User.username == user_data.username)
    )
    if result.scalar_one_or_none() is not None:
        raise HTTPException(status_code=409, detail="Username is already registered")
    user = User(
        username=user_data.username,
        hashed_password=await get_password_hash(user_data.password),
        preferred_locale=user_data.preferred_locale,
    )
    db.add(user)
    try:
        await db.commit()
    except IntegrityError as exc:
        await db.rollback()
        raise HTTPException(
            status_code=409, detail="Username is already registered"
        ) from exc
    await db.refresh(user)
    return user


@router.post("/login", response_model=Token)
async def login(data: UserLogin, db: DBSessionDep) -> Token:
    """Authenticate a user and issue an access and refresh token pair."""
    user = await authenticate_user(db, data.username, data.password)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token, refresh_token = await create_tokens_for_user(db, user)
    return Token(access_token=access_token, refresh_token=refresh_token)


@router.post("/token/refresh", response_model=Token)
async def refresh_token(data: RefreshTokenRequest, db: DBSessionDep) -> Token:
    """Exchange a refresh token for a new pair, revoking the presented token."""
    tokens = await rotate_refresh_token(data.refresh_token, db)
    if tokens is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return Token(access_token=tokens[0], refresh_token=tokens[1])


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(
    data: LogoutRequest, db: DBSessionDep, current_user: AuthUserDep
) -> Response:
    """Revoke a refresh token belonging to the authenticated user."""
    if not await revoke_refresh_token(data.refresh_token, db, current_user):
        raise HTTPException(status_code=401, detail="Invalid refresh token")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/me", response_model=UserOut)
async def read_users_me(current_user: AuthUserDep) -> User:
    """Return the authenticated user, identified by an access token."""
    return current_user


@router.get("/api-me", response_model=UserOut)
async def read_users_api_me(current_user: TokenUserDep) -> User:
    """Return the authenticated user, identified by an API token."""
    return current_user


@router.post("/api-tokens", response_model=APITokenOut, status_code=201)
async def create_api_token(
    data: APITokenCreate, db: DBSessionDep, user: AuthUserDep
) -> APITokenOut:
    """Create an API token. The raw value is shown once and stored only as a hash."""
    token_value = secrets.token_urlsafe(32)
    expires_at = (
        datetime.now(UTC) + timedelta(days=data.expires_in_days)
        if data.expires_in_days is not None
        else None
    )
    db.add(
        APIToken(
            name=data.name,
            token_hash=hash_token(token_value),
            user_id=user.id,
            expires_at=expires_at,
        )
    )
    await db.commit()
    return APITokenOut(
        api_token=token_value,
        name=data.name,
        expires_at=expires_at,
    )
