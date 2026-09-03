"""Shared isolated database and API client fixtures."""

from collections.abc import AsyncGenerator

import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.pool import StaticPool

from app.db.base import Base
from app.db.session import DatabaseSessionManager, get_db
from app.models.user import APIToken, RefreshToken, User
from main import app


@pytest_asyncio.fixture
async def test_db() -> AsyncGenerator[DatabaseSessionManager, None]:
    """Create a fresh in-memory database in the test's event loop."""
    manager = DatabaseSessionManager(
        "sqlite+aiosqlite:///:memory:", {"poolclass": StaticPool}
    )
    assert manager._engine is not None
    async with manager._engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)

    async def override_get_db() -> AsyncGenerator[AsyncSession, None]:
        async with manager.session() as session:
            yield session

    app.dependency_overrides[get_db] = override_get_db
    yield manager
    app.dependency_overrides.pop(get_db, None)
    async with manager._engine.begin() as connection:
        await connection.run_sync(Base.metadata.drop_all)
    await manager.close()


@pytest_asyncio.fixture
async def async_client(
    test_db: DatabaseSessionManager,
) -> AsyncGenerator[AsyncClient, None]:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest_asyncio.fixture
async def session(
    test_db: DatabaseSessionManager,
) -> AsyncGenerator[AsyncSession, None]:
    async with test_db.session() as db_session:
        yield db_session
