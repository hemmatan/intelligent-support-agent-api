"""Authentication API and token lifecycle tests."""

import asyncio
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio
from httpx import AsyncClient
from pwdlib.hashers.bcrypt import BcryptHasher
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.security import create_access_token, get_password_hash, hash_token
from app.db.base import Base
from app.models.user import APIToken, RefreshToken, User
from app.services.auth import create_tokens_for_user, rotate_refresh_token


@pytest.fixture
def test_user_password() -> str:
    return "secret123"


@pytest_asyncio.fixture
async def test_user(session: AsyncSession) -> User:
    hashed = await get_password_hash("secret123")
    user = User(username="testuser", hashed_password=hashed)
    session.add(user)
    await session.commit()
    await session.refresh(user)
    return user


@pytest_asyncio.fixture
async def jwt_token(test_user: User) -> str:
    token = create_access_token(
        test_user.username, test_user.id, test_user.role.value
    ).value
    return f"Bearer {token}"


@pytest_asyncio.fixture
async def api_token(test_user: User, session: AsyncSession) -> str:
    raw_token = "test-api-token"
    session.add(
        APIToken(
            name="test",
            token_hash=hash_token(raw_token),
            user_id=test_user.id,
        )
    )
    await session.commit()
    return raw_token


@pytest.mark.asyncio
async def test_signup_success(async_client: AsyncClient) -> None:
    response = await async_client.post(
        "/api/v1/auth/signup",
        json={"username": "newuser", "password": "pass1234", "preferred_locale": "fr"},
    )
    assert response.status_code == 201
    assert response.json() == {
        "id": 1,
        "username": "newuser",
        "role": "customer",
        "preferred_locale": "fr",
        "is_active": True,
    }


@pytest.mark.asyncio
async def test_signup_duplicate_username(
    async_client: AsyncClient, test_user: User
) -> None:
    response = await async_client.post(
        "/api/v1/auth/signup",
        json={"username": "testuser", "password": "secret123"},
    )
    assert response.status_code == 409
    assert response.json()["detail"] == "Username is already registered"


@pytest.mark.asyncio
async def test_login_and_refresh_rotation(
    async_client: AsyncClient, test_user: User, session: AsyncSession
) -> None:
    login = await async_client.post(
        "/api/v1/auth/login",
        json={"username": "testuser", "password": "secret123"},
    )
    assert login.status_code == 200
    original = login.json()

    refreshed = await async_client.post(
        "/api/v1/auth/token/refresh",
        json={"refresh_token": original["refresh_token"]},
    )
    assert refreshed.status_code == 200
    assert refreshed.json()["refresh_token"] != original["refresh_token"]

    replay = await async_client.post(
        "/api/v1/auth/token/refresh",
        json={"refresh_token": original["refresh_token"]},
    )
    assert replay.status_code == 401

    records = (await session.execute(select(RefreshToken))).scalars().all()
    assert len(records) == 2
    assert any(record.revoked_at is not None for record in records)
    assert all(record.token_hash != original["refresh_token"] for record in records)


@pytest.mark.asyncio
async def test_login_invalid_credentials(async_client: AsyncClient) -> None:
    response = await async_client.post(
        "/api/v1/auth/login",
        json={"username": "fakeuser", "password": "wrongpass"},
    )
    assert response.status_code == 401
    assert response.json()["detail"] == "Incorrect username or password"


@pytest.mark.asyncio
async def test_access_token_cannot_be_refreshed(
    async_client: AsyncClient, jwt_token: str
) -> None:
    response = await async_client.post(
        "/api/v1/auth/token/refresh",
        json={"refresh_token": jwt_token.removeprefix("Bearer ")},
    )
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_refresh_token_cannot_authenticate_api(
    async_client: AsyncClient, test_user: User
) -> None:
    login = await async_client.post(
        "/api/v1/auth/login",
        json={"username": "testuser", "password": "secret123"},
    )
    response = await async_client.get(
        "/api/v1/auth/me",
        headers={"Authorization": f"Bearer {login.json()['refresh_token']}"},
    )
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_read_users_me(async_client: AsyncClient, jwt_token: str) -> None:
    response = await async_client.get(
        "/api/v1/auth/me", headers={"Authorization": jwt_token}
    )
    assert response.status_code == 200
    assert response.json()["username"] == "testuser"


@pytest.mark.asyncio
async def test_database_role_overrides_stale_token_claim(
    async_client: AsyncClient, test_user: User
) -> None:
    misleading_token = create_access_token(
        test_user.username, test_user.id, role="admin"
    ).value
    response = await async_client.get(
        "/api/v1/auth/me",
        headers={"Authorization": f"Bearer {misleading_token}"},
    )
    assert response.status_code == 200
    assert response.json()["role"] == "customer"


@pytest.mark.asyncio
async def test_api_token_is_hashed_and_authenticates(
    async_client: AsyncClient, jwt_token: str, session: AsyncSession
) -> None:
    created = await async_client.post(
        "/api/v1/auth/api-tokens",
        headers={"Authorization": jwt_token},
        json={"name": "automation", "expires_in_days": 30},
    )
    assert created.status_code == 201
    raw_token = created.json()["api_token"]
    stored = (await session.execute(select(APIToken))).scalar_one()
    assert stored.token_hash == hash_token(raw_token)
    assert stored.token_hash != raw_token

    authenticated = await async_client.get(
        "/api/v1/auth/api-me", headers={"X-API-Token": raw_token}
    )
    assert authenticated.status_code == 200
    assert authenticated.json()["username"] == "testuser"


@pytest.mark.asyncio
async def test_logout_revokes_refresh_token(
    async_client: AsyncClient, test_user: User
) -> None:
    login = await async_client.post(
        "/api/v1/auth/login",
        json={"username": "testuser", "password": "secret123"},
    )
    tokens = login.json()
    response = await async_client.post(
        "/api/v1/auth/logout",
        headers={"Authorization": f"Bearer {tokens['access_token']}"},
        json={"refresh_token": tokens["refresh_token"]},
    )
    assert response.status_code == 204
    refresh = await async_client.post(
        "/api/v1/auth/token/refresh",
        json={"refresh_token": tokens["refresh_token"]},
    )
    assert refresh.status_code == 401


@pytest.mark.asyncio
async def test_legacy_bcrypt_hash_upgrades_on_login(
    async_client: AsyncClient, session: AsyncSession
) -> None:
    user = User(
        username="legacy-user",
        hashed_password=BcryptHasher().hash("legacy-password"),
    )
    session.add(user)
    await session.commit()

    response = await async_client.post(
        "/api/v1/auth/login",
        json={"username": "legacy-user", "password": "legacy-password"},
    )
    assert response.status_code == 200
    await session.refresh(user)
    assert user.hashed_password.startswith("$argon2")


@pytest.mark.asyncio
async def test_unauthenticated_requests_are_rejected(async_client: AsyncClient) -> None:
    assert (await async_client.get("/api/v1/auth/me")).status_code == 401
    assert (await async_client.get("/api/v1/auth/api-me")).status_code == 401


@pytest.mark.asyncio
async def test_concurrent_refresh_redeems_a_token_only_once(
    tmp_path: Path, test_user_password: str
) -> None:
    """Regression: FOR UPDATE is a no-op on SQLite, so rotation must not rely on it.

    Uses a file-backed database because concurrent redemption needs two real
    connections; the shared in-memory pool would serialise them and hide the race.
    """
    barrier = asyncio.Barrier(2)

    class BarrierSession(AsyncSession):
        """Hold both redemptions between reading the token and claiming it."""

        async def get(self, *args: Any, **kwargs: Any) -> Any:
            if self.info.get("synchronise"):
                await barrier.wait()
            return await super().get(*args, **kwargs)

    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'race.sqlite3'}")
    sessionmaker = async_sessionmaker(
        engine, class_=BarrierSession, expire_on_commit=False
    )
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)

    async with sessionmaker() as session:
        user = User(
            username="racer",
            hashed_password=await get_password_hash(test_user_password),
        )
        session.add(user)
        await session.commit()
        await session.refresh(user)
        _, refresh_token = await create_tokens_for_user(session, user)

    async def redeem() -> tuple[str, str] | None:
        # rotate_refresh_token loads the user between validating the stored
        # token and claiming it, so pausing there parks both callers inside
        # the critical section regardless of scheduling.
        async with sessionmaker(info={"synchronise": True}) as session:
            return await rotate_refresh_token(refresh_token, session)

    outcomes = await asyncio.gather(redeem(), redeem())
    assert sum(outcome is not None for outcome in outcomes) == 1

    async with sessionmaker() as session:
        stored = (await session.execute(select(RefreshToken))).scalars().all()
    assert len(stored) == 2, "one original plus exactly one replacement"
    await engine.dispose()
