"""
Database session management.
"""

import contextlib
from collections.abc import AsyncGenerator, AsyncIterator
from typing import Any

from sqlalchemy.ext.asyncio import (
    AsyncConnection,
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.config import settings


class DatabaseNotInitializedError(RuntimeError):
    """Raised when a closed session manager is reused."""


class DatabaseSessionManager:
    def __init__(self, host: str, engine_kwargs: dict[str, Any] | None = None):
        self._engine: AsyncEngine | None = create_async_engine(
            host, **(engine_kwargs or {})
        )
        self._sessionmaker: async_sessionmaker[AsyncSession] | None = (
            async_sessionmaker(
                autocommit=False,
                expire_on_commit=False,
                bind=self._engine,
            )
        )

    async def close(self) -> None:
        if self._engine is None:
            raise DatabaseNotInitializedError(
                "DatabaseSessionManager is not initialized"
            )
        await self._engine.dispose()

        self._engine = None
        self._sessionmaker = None

    @contextlib.asynccontextmanager
    async def connect(self) -> AsyncIterator[AsyncConnection]:
        if self._engine is None:
            raise DatabaseNotInitializedError(
                "DatabaseSessionManager is not initialized"
            )

        async with self._engine.begin() as connection:
            try:
                yield connection
            except Exception:
                await connection.rollback()
                raise

    @contextlib.asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        if self._sessionmaker is None:
            raise DatabaseNotInitializedError(
                "DatabaseSessionManager is not initialized"
            )

        session = self._sessionmaker()
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


sessionmanager = DatabaseSessionManager(
    settings.DATABASE_URL,
    # {"echo": settings.echo_sql}
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with sessionmanager.session() as session:
        yield session
