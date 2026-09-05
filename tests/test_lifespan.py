"""What the application does on the way up, and on the way down."""

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from app.agent.knowledge import PolicyCorpusError, load_corpus
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


@pytest.mark.asyncio
async def test_startup_loads_the_policy_corpus() -> None:
    async with app.router.lifespan_context(app):
        corpus = app.state.policies
    assert {entry.id for entry in corpus} == {"returns.standard", "shipping.times"}


@pytest.mark.asyncio
async def test_a_broken_corpus_stops_the_application_starting(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A real corpus failure, not a mocked one: the file is genuinely invalid."""
    (tmp_path / "returns.standard.en.v1.toml").write_text("id = 'unterminated")
    monkeypatch.setattr("main.load_corpus", lambda: load_corpus(tmp_path))

    with pytest.raises(PolicyCorpusError):
        async with app.router.lifespan_context(app):
            pass  # pragma: no cover — startup raises before the body runs
