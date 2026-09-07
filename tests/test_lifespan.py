"""What the application does on the way up, and on the way down."""

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from app.agent.embedding import EmbeddingMisconfiguredError
from app.agent.knowledge import PolicyCorpusError, load_corpus
from app.core.config import settings
from main import app


@pytest.fixture(autouse=True)
def _no_embedder(monkeypatch: pytest.MonkeyPatch) -> None:
    """Start the application without reaching for a hosted model.

    Every test here is about startup and shutdown, and none of them is about
    embeddings. Left to the ambient configuration they warmed a real index
    against whatever token the developer had, which is a network call, a
    ten-second timeout each way, and a result that depends on somebody's .env.

    The two tests that are about the embedder override this themselves.
    """
    monkeypatch.setattr(settings, "HUGGINGFACE_API_TOKEN", None)


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


@pytest.mark.asyncio
async def test_startup_builds_a_searchable_index() -> None:
    async with app.router.lifespan_context(app):
        index = app.state.policy_index
    hits = await index.search("how long do I have to return a jacket", "en")
    assert hits[0].entry.id == "returns.standard"


@pytest.mark.asyncio
async def test_without_a_token_the_index_is_lexical_and_the_app_still_starts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No embedder configured is a supported deployment, not a broken one.

    The absence is arranged rather than assumed. Reading it off whatever
    happened to be configured meant the assertion held until somebody put a
    real token in their .env, and then failed for a reason unconnected to
    anything this covers.
    """
    assert settings.HUGGINGFACE_API_TOKEN is None
    async with app.router.lifespan_context(app):
        assert app.state.policy_index.semantic_ready is False


@pytest.mark.asyncio
async def test_a_misconfigured_embedder_stops_the_application_starting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A token that will be refused every time is not a degraded mode.

    Running for months on half the retrieval while every health check reports
    success is the outcome this prevents.
    """

    class Refused:
        async def embed(self, texts: object) -> list[tuple[float, ...]]:
            raise EmbeddingMisconfiguredError("token refused")

    monkeypatch.setattr("main._embedder", Refused)
    with pytest.raises(EmbeddingMisconfiguredError):
        async with app.router.lifespan_context(app):
            pass  # pragma: no cover — startup raises before the body runs
