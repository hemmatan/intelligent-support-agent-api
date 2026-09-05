"""The hosted embedder, and what it does when the host will not answer."""

import os

import httpx
import pytest
from pydantic import SecretStr

from app.agent.embedding import (
    EmbeddingMisconfiguredError,
    EmbeddingUnavailableError,
    cosine_similarity,
)
from app.agent.inference import HuggingFaceEmbedder
from app.core.config import Settings


def configured() -> Settings:
    return Settings(
        _env_file=None,  # type: ignore[call-arg]
        HUGGINGFACE_API_TOKEN=SecretStr("hf_not_a_real_token"),
    )


def responding(handler: object) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_vectors_come_back_in_the_order_they_were_sent() -> None:
    sent: list[list[str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        sent.append(json.loads(request.content)["inputs"])
        return httpx.Response(200, json=[[1.0, 0.0], [0.0, 1.0]])

    async with responding(handler) as client:
        vectors = await HuggingFaceEmbedder(configured(), client).embed(["a", "b"])

    assert sent == [["a", "b"]]
    assert vectors == [(1.0, 0.0), (0.0, 1.0)]


@pytest.mark.asyncio
async def test_nothing_to_embed_makes_no_request() -> None:
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError("an empty batch should not reach the network")

    async with responding(handler) as client:
        assert await HuggingFaceEmbedder(configured(), client).embed([]) == []


@pytest.mark.parametrize(
    ("label", "response"),
    [
        ("model loading", httpx.Response(503, json={"error": "currently loading"})),
        ("server error", httpx.Response(500, text="upstream failure")),
        ("not json", httpx.Response(200, text="<html>gateway</html>")),
        ("wrong count", httpx.Response(200, json=[[1.0, 0.0]])),
        ("token-level output", httpx.Response(200, json=[[[1.0], [2.0]], [[3.0]]])),
    ],
    ids=lambda value: value if isinstance(value, str) else "",
)
@pytest.mark.asyncio
async def test_every_failure_looks_the_same_to_the_caller(
    label: str, response: httpx.Response
) -> None:
    """One decision to make, so one exception to make it about."""
    async with responding(lambda request: response) as client:
        with pytest.raises(EmbeddingUnavailableError):
            await HuggingFaceEmbedder(configured(), client).embed(["a", "b"])


@pytest.mark.asyncio
async def test_a_timeout_is_unavailability_not_a_crash() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("took too long", request=request)

    async with responding(handler) as client:
        with pytest.raises(EmbeddingUnavailableError, match="ConnectTimeout"):
            await HuggingFaceEmbedder(configured(), client).embed(["a"])


def test_an_unconfigured_token_is_refused_at_construction() -> None:
    """Not at the first search, when a customer is waiting."""
    with pytest.raises(ValueError, match="HUGGINGFACE_API_TOKEN"):
        HuggingFaceEmbedder(Settings(_env_file=None))  # type: ignore[call-arg]


@pytest.mark.skipif(
    not os.environ.get("DORNASHOP_HUGGINGFACE_API_TOKEN"),
    reason="needs a real token; set DORNASHOP_HUGGINGFACE_API_TOKEN to run",
)
@pytest.mark.asyncio
async def test_a_real_model_places_a_paraphrase_near_the_policy() -> None:
    """The claim the stand-in deliberately does not make.

    Everything else here asserts that our code handles vectors correctly. This
    asserts that the model produces useful ones — that a question worded
    nothing like the returns policy still lands closer to it than to shipping.
    Skipped by default: it needs a token and the network, and CI has neither.
    """
    embedder = HuggingFaceEmbedder(Settings())
    question, returns, shipping = await embedder.embed(
        [
            "what happens if I change my mind about something I bought",
            "You can return most items within 30 days of delivery for a refund.",
            "Standard delivery takes 2 to 3 business days from dispatch.",
        ]
    )
    assert cosine_similarity(question, returns) > cosine_similarity(question, shipping)


@pytest.mark.parametrize(
    ("label", "status"),
    [("refused token", 401), ("forbidden", 403), ("unknown model", 404)],
    ids=lambda value: value if isinstance(value, str) else "",
)
@pytest.mark.asyncio
async def test_a_mistake_in_the_configuration_is_not_weather(
    label: str, status: int
) -> None:
    """Retrying a typo produces the same answer forever."""
    async with responding(lambda request: httpx.Response(status, json={})) as client:
        with pytest.raises(EmbeddingMisconfiguredError, match="DORNASHOP_"):
            await HuggingFaceEmbedder(configured(), client).embed(["a"])


@pytest.mark.parametrize(
    ("label", "body"),
    [
        ("unequal dimensions", [[1.0], [1.0, 2.0]]),
        ("booleans", [[True, False], [1.0, 0.0]]),
        ("not a number", [[float("nan"), 1.0], [1.0, 0.0]]),
        ("infinite", [[float("inf"), 1.0], [1.0, 0.0]]),
        ("empty vector", [[], []]),
    ],
    ids=lambda value: value if isinstance(value, str) else "",
)
@pytest.mark.asyncio
async def test_vectors_that_would_break_a_comparison_are_refused(
    label: str, body: list[list[float]]
) -> None:
    """Nothing downstream looks again.

    Unequal lengths raise when compared, and one NaN makes every similarity
    involving it NaN, which sorts unpredictably and never looks like an error.
    """
    async with responding(lambda request: httpx.Response(200, json=body)) as client:
        with pytest.raises(EmbeddingUnavailableError):
            await HuggingFaceEmbedder(configured(), client).embed(["a", "b"])
