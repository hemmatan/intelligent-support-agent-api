"""The endpoint a customer talks to, wired to the agent it answers from."""

from collections.abc import AsyncGenerator, Callable

import pytest
import pytest_asyncio
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.answering import Sources
from app.agent.knowledge import load_corpus
from app.agent.messages import load_messages
from app.agent.responses import load_templates
from app.agent.retrieval import PolicyIndex
from app.api.support import support_agent
from app.core.security import create_access_token
from app.models.user import User
from app.services.support import SupportAgent
from main import app

MESSAGES = "/api/v1/support/messages"


@pytest.fixture(scope="module")
def agent() -> SupportAgent:
    """The real corpus, wording and messages. Lexical ranking, no network."""
    return SupportAgent(
        sources=Sources(knowledge_base=PolicyIndex(load_corpus())),
        templates=load_templates(),
        messages=load_messages(),
    )


@pytest_asyncio.fixture
async def wired(agent: SupportAgent) -> AsyncGenerator[None, None]:
    app.dependency_overrides[support_agent] = lambda: agent
    yield
    app.dependency_overrides.pop(support_agent, None)


@pytest_asyncio.fixture
async def signed_in(
    session: AsyncSession,
) -> Callable[..., AsyncGenerator[dict[str, str], None]]:
    """A customer, and the header that speaks for them."""

    async def make(locale: str = "en", linked: int | None = 4471) -> dict[str, str]:
        customer = User(
            username=f"asker-{locale}-{linked}",
            hashed_password="x",
            preferred_locale=locale,
            external_customer_id=linked,
        )
        session.add(customer)
        await session.commit()
        await session.refresh(customer)
        issued = create_access_token(
            customer.username, customer.id, customer.role.value
        )
        return {"Authorization": f"Bearer {issued.value}"}

    return make  # type: ignore[return-value]


@pytest.mark.asyncio
async def test_a_question_the_corpus_answers_comes_back_answered(
    async_client: AsyncClient, wired: None, signed_in: Callable[..., object]
) -> None:
    headers = await signed_in()  # type: ignore[misc]
    response = await async_client.post(
        MESSAGES,
        json={"message": "How long do I have to return a jacket?"},
        headers=headers,
    )
    assert response.status_code == 200
    body = response.json()
    assert body["route"] == "direct_response"
    assert body["reply"] == "Returns are accepted within 30 days of delivery."
    assert body["citations"][0]["reference"] == "kb:returns.standard.en.v1"
    # Words alone, with no embedder configured here: relevance tops out below
    # ready, and the weakest factor is what the answer is rated at. It still
    # goes out, which is the difference between acceptable and review only.
    assert body["reliability"]["factors"]["relevance"] == "acceptable"
    assert body["reliability"]["level"] == "acceptable"
    assert body["reliability"]["ordinal"] == 2


@pytest.mark.asyncio
async def test_a_customer_is_asked_for_what_the_request_needs(
    async_client: AsyncClient, wired: None, signed_in: Callable[..., object]
) -> None:
    headers = await signed_in()  # type: ignore[misc]
    response = await async_client.post(
        MESSAGES, json={"message": "Where is my order?"}, headers=headers
    )
    assert response.status_code == 200
    body = response.json()
    assert body["route"] == "clarification"
    assert body["reason"] == "missing_order_id"
    assert body["message"].startswith("Please send us your order number")


@pytest.mark.asyncio
async def test_the_account_decides_the_language(
    async_client: AsyncClient, wired: None, signed_in: Callable[..., object]
) -> None:
    """Nothing in the request body can ask for a different one."""
    headers = await signed_in(locale="fr")  # type: ignore[misc]
    response = await async_client.post(
        MESSAGES,
        json={"message": "Combien de temps pour retourner un article ?"},
        headers=headers,
    )
    body = response.json()
    assert body["route"] == "direct_response"
    assert body["reply"].startswith("Les retours sont acceptés")


@pytest.mark.asyncio
async def test_a_stored_language_nothing_is_written_in_still_gets_an_answer(
    async_client: AsyncClient, wired: None, signed_in: Callable[..., object]
) -> None:
    """The column takes any five characters, and only our own API guards it.

    An enquiry refuses one, so a request would have ended as a five hundred
    for a fault the customer had no part in.
    """
    headers = await signed_in(locale="de")  # type: ignore[misc]
    response = await async_client.post(
        MESSAGES,
        json={"message": "How long do I have to return a jacket?"},
        headers=headers,
    )
    assert response.status_code == 200
    assert response.json()["route"] == "direct_response"


@pytest.mark.asyncio
async def test_nobody_unknown_is_answered(
    async_client: AsyncClient, wired: None
) -> None:
    response = await async_client.post(MESSAGES, json={"message": "Hello"})
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_a_malformed_request_never_reaches_the_agent(
    async_client: AsyncClient, wired: None, signed_in: Callable[..., object]
) -> None:
    headers = await signed_in()  # type: ignore[misc]
    for sent in ({"message": "   "}, {"message": "hi", "oder_id": "ORD-1"}):
        response = await async_client.post(MESSAGES, json=sent, headers=headers)
        assert response.status_code == 422, sent


@pytest.mark.asyncio
async def test_startup_builds_the_agent_the_endpoint_asks_for(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The dependency fetches it rather than assembling one per request."""
    from app.core.config import settings

    monkeypatch.setattr(settings, "HUGGINGFACE_API_TOKEN", None)
    async with app.router.lifespan_context(app):
        assert isinstance(app.state.support, SupportAgent)
        assert len(app.state.support.templates)
        assert len(app.state.support.messages)
