"""The application lifespan must release database resources on shutdown."""

from unittest.mock import AsyncMock

import pytest

from main import app


@pytest.mark.asyncio
async def test_shutdown_disposes_the_database_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: the lifespan was defined but never registered on the app."""
    close = AsyncMock()
    monkeypatch.setattr("main.sessionmanager.close", close)

    async with app.router.lifespan_context(app):
        close.assert_not_awaited()

    close.assert_awaited_once()
