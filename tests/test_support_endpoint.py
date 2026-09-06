"""The endpoint a customer talks to, wired to the agent it answers from."""

from collections.abc import AsyncGenerator, Callable
from dataclasses import replace
from itertools import count

import pytest
import pytest_asyncio
from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.answering import Sources
from app.agent.enquiry import MAX_MESSAGE, Enquiry
from app.agent.intent import ClassifierUnavailableError, Intent
from app.agent.knowledge import Locale, load_corpus
from app.agent.messages import load_messages
from app.agent.responses import load_templates
from app.agent.retrieval import PolicyIndex
from app.api.support import support_agent
from app.core.security import create_access_token
from app.models.support import SupportCase
from app.models.user import User
from app.schemas.support import SupportReply
from app.services.cases import NEEDS_SOMEBODY
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
    """A customer, and the header that speaks for them.

    Each one is distinct: the commerce link is unique in the schema, so a
    test wanting two customers used to collide on the second.
    """
    made = count()

    async def make(locale: str = "en", linked: bool = True) -> dict[str, str]:
        nth = next(made)
        customer = User(
            username=f"asker-{nth}",
            hashed_password="x",
            preferred_locale=locale,
            external_customer_id=4471 + nth if linked else None,
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


# One row per decision the service can reach, asserted through HTTP rather
# than in process. Everything below the route handler is exercised by its own
# tests; this table is the claim that a customer writing in actually gets
# these, and it is the table the README quotes.
GOLDEN: list[tuple[str, str, dict[str, str], str, str]] = [
    (
        "en",
        "a question the policy answers",
        {"message": "How long do I have to return a jacket?"},
        "direct_response",
        "return_window",
    ),
    (
        "fr",
        "the same question in French",
        {"message": "Combien de temps pour retourner un article ?"},
        "direct_response",
        "return_window",
    ),
    (
        "en",
        "a question about delivery",
        {"message": "How long does delivery take?"},
        "direct_response",
        "standard_delivery_time",
    ),
    (
        "fr",
        "delivery in a language nothing covers",
        {"message": "Quel est le delai de livraison ?"},
        "human_escalation",
        "evidence_does_not_cover_the_question",
    ),
    (
        "en",
        "a carve-out the claims do not state",
        {"message": "Can I return underwear?"},
        "internal_review",
        "evidence_does_not_cover_the_question",
    ),
    (
        "en",
        "nothing approved says anything about it",
        {"message": "Do you ship to Belgium?"},
        "human_escalation",
        "no_supporting_evidence",
    ),
    (
        "en",
        "trouble reported in passing",
        {"message": "I need to return this because a stranger used my card"},
        "human_escalation",
        "suspected_fraud",
    ),
    (
        "en",
        "a source nobody connected",
        {"message": "Where is my order?", "order_id": "ORD-4471"},
        "internal_review",
        "source_unavailable",
    ),
    (
        "en",
        "something only the customer can supply",
        {"message": "Where is my order?"},
        "clarification",
        "missing_order_id",
    ),
    (
        "en",
        "two questions in one message",
        {"message": "Where is my order, and can I return it once it arrives?"},
        "clarification",
        "multiple_intents",
    ),
    (
        "en",
        "nothing anybody can place",
        {"message": "I need help with my purchase"},
        "clarification",
        "unresolved_intent",
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("locale", "about", "sent", "route", "why"),
    GOLDEN,
    ids=[f"{locale}: {about}" for locale, about, _, _, _ in GOLDEN],
)
async def test_a_customer_writing_in_gets_the_decision_they_should(
    async_client: AsyncClient,
    wired: None,
    signed_in: Callable[..., object],
    locale: str,
    about: str,
    sent: dict[str, str],
    route: str,
    why: str,
) -> None:
    headers = await signed_in(locale=locale)  # type: ignore[misc]
    response = await async_client.post(MESSAGES, json=sent, headers=headers)

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["route"] == route, body
    if route == "direct_response":
        assert why in body["wording"][0]
        assert body["reply"]
        assert body["citations"]
    else:
        assert why in body.get("reasons", [body.get("reason")]), body
        # Nothing leaves here as a bare code, whatever stopped it.
        assert body["message"], body
        assert body["wording"].startswith("say:"), body


@pytest.mark.asyncio
async def test_an_identifier_reaches_the_step_that_needed_it(
    async_client: AsyncClient, wired: None, signed_in: Callable[..., object]
) -> None:
    """Supplying it changes the decision, which is the only proof it arrived.

    Recorded as a marker and dropped, the request kept asking for an order
    number it had already been given.
    """
    headers = await signed_in()  # type: ignore[misc]
    without = await async_client.post(
        MESSAGES, json={"message": "Where is my order?"}, headers=headers
    )
    with_it = await async_client.post(
        MESSAGES,
        json={"message": "Where is my order?", "order_id": "ORD-4471"},
        headers=headers,
    )
    assert without.json()["reason"] == "missing_order_id"
    assert with_it.json()["reasons"] == ["source_unavailable"]


@pytest.mark.asyncio
async def test_an_unlinked_customer_is_told_what_is_being_done(
    async_client: AsyncClient, wired: None, signed_in: Callable[..., object]
) -> None:
    """They cannot supply the link, so asking them would waste their turn."""
    headers = await signed_in(linked=False)  # type: ignore[misc]
    response = await async_client.post(
        MESSAGES,
        json={"message": "Where is my order?", "order_id": "ORD-4471"},
        headers=headers,
    )
    body = response.json()
    assert body["route"] == "human_escalation"
    assert body["reasons"] == ["customer_not_linked"]
    assert "not yet connected" in body["message"]


class Down:
    """A classifier that cannot answer today."""

    async def classify(self, message: str, locale: Locale) -> Intent | None:
        raise ClassifierUnavailableError("the provider timed out")


class MustNotBeAsked:
    """Raises if consulted. A spy counting calls passes when never wired up."""

    async def classify(self, message: str, locale: Locale) -> Intent | None:
        raise AssertionError("a model saw a message the rules had settled")


@pytest_asyncio.fixture
async def wired_with(
    agent: SupportAgent,
) -> AsyncGenerator[Callable[[object], None], None]:
    def use(classifier: object) -> None:
        app.dependency_overrides[support_agent] = lambda: replace(
            agent,
            classifier=classifier,  # type: ignore[arg-type]
        )

    yield use
    app.dependency_overrides.pop(support_agent, None)


@pytest.mark.asyncio
async def test_a_model_that_cannot_answer_holds_the_request_here(
    async_client: AsyncClient,
    wired_with: Callable[[object], None],
    signed_in: Callable[..., object],
) -> None:
    """Not a five hundred. The fault is ours and the customer is told so."""
    wired_with(Down())
    headers = await signed_in()  # type: ignore[misc]
    response = await async_client.post(
        MESSAGES, json={"message": "I need help with my purchase"}, headers=headers
    )
    assert response.status_code == 200
    body = response.json()
    assert body["route"] == "internal_review"
    assert body["reasons"] == ["intent_check_unavailable"]
    assert body["message"]


@pytest.mark.asyncio
async def test_reported_trouble_never_waits_on_a_model(
    async_client: AsyncClient,
    wired_with: Callable[[object], None],
    signed_in: Callable[..., object],
) -> None:
    """The rules read the message first and leave, so nothing else runs.

    The model here raises when spoken to. It is not spoken to, which is why a
    provider having a bad afternoon cannot hold up a fraud report.
    """
    wired_with(MustNotBeAsked())
    headers = await signed_in()  # type: ignore[misc]
    response = await async_client.post(
        MESSAGES, json={"message": "I was charged twice"}, headers=headers
    )
    assert response.status_code == 200
    assert response.json()["reasons"] == ["payment_dispute"]


@pytest.mark.asyncio
async def test_a_message_past_the_ceiling_is_refused_at_the_edge(
    async_client: AsyncClient, wired: None, signed_in: Callable[..., object]
) -> None:
    """Before ranking tokenises it or anything is asked to read it."""
    headers = await signed_in()  # type: ignore[misc]
    response = await async_client.post(
        MESSAGES, json={"message": "a" * (MAX_MESSAGE + 1)}, headers=headers
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_a_request_held_here_does_not_claim_to_have_gone_elsewhere(
    async_client: AsyncClient, wired: None, signed_in: Callable[..., object]
) -> None:
    """The same shortfall reaches two routes, and only one sentence is true.

    Chosen by the code alone, a customer whose message never left the service
    was told a colleague had taken it on personally.
    """
    headers = await signed_in()  # type: ignore[misc]
    held = await async_client.post(
        MESSAGES, json={"message": "Can I return underwear?"}, headers=headers
    )
    passed = await async_client.post(
        MESSAGES,
        json={"message": "Quel est le delai de livraison ?"},
        headers=await signed_in(locale="fr"),  # type: ignore[misc]
    )
    assert held.json()["route"] == "internal_review"
    assert passed.json()["route"] == "human_escalation"
    assert held.json()["reasons"] == passed.json()["reasons"]
    assert "checking your request" in held.json()["message"]
    assert "transmis" in passed.json()["message"]


@pytest.mark.asyncio
async def test_saying_it_reached_a_person_means_it_reached_the_queue(
    async_client: AsyncClient,
    wired: None,
    signed_in: Callable[..., object],
    session: AsyncSession,
) -> None:
    """The sentence was true of nothing until there was a row behind it.

    A customer was told their message had gone to a colleague while the
    service returned and forgot it. Nothing was written down, so nothing was
    anywhere a colleague would look.
    """
    headers = await signed_in()  # type: ignore[misc]
    response = await async_client.post(
        MESSAGES, json={"message": "I was charged twice"}, headers=headers
    )
    body = response.json()
    assert body["route"] == "human_escalation"

    case = (
        await session.execute(
            select(SupportCase).where(SupportCase.reference == body["case"])
        )
    ).scalar_one()
    assert case.route == "human_escalation"
    assert case.reasons == ["payment_dispute"]
    assert case.message == "I was charged twice"
    assert case.reply is None
    assert case.wording == [body["wording"]]
    # Open, which is what makes this a queue and not a log.
    assert case.closed_at is None


@pytest.mark.asyncio
async def test_an_answer_that_went_out_is_written_down_too(
    async_client: AsyncClient,
    wired: None,
    signed_in: Callable[..., object],
    session: AsyncSession,
) -> None:
    """Every decision, not only the ones somebody has to pick up.

    A reply nobody kept a record of is the one that cannot be looked into,
    and it is the one that reached a customer.
    """
    headers = await signed_in()  # type: ignore[misc]
    response = await async_client.post(
        MESSAGES,
        json={"message": "How long do I have to return a jacket?"},
        headers=headers,
    )
    body = response.json()
    case = (
        await session.execute(
            select(SupportCase).where(SupportCase.reference == body["case"])
        )
    ).scalar_one()
    assert case.route == "direct_response"
    assert case.intent == "return_policy"
    assert case.reply == body["reply"]
    assert case.citations[0]["reference"] == "kb:returns.standard.en.v1"
    assert case.wording == body["wording"]
    assert case.reliability is not None
    assert case.reliability["level"] == "acceptable"


@pytest.mark.asyncio
async def test_what_a_person_has_to_work_can_be_found(
    async_client: AsyncClient,
    wired: None,
    signed_in: Callable[..., object],
    session: AsyncSession,
) -> None:
    """The queue is a query, and answered requests are not in it."""
    headers = await signed_in()  # type: ignore[misc]
    for message in (
        "I was charged twice",
        "Where is my order?",
        "How long do I have to return a jacket?",
    ):
        await async_client.post(MESSAGES, json={"message": message}, headers=headers)

    waiting = (
        (
            await session.execute(
                select(SupportCase).where(
                    SupportCase.route.in_([str(r) for r in NEEDS_SOMEBODY]),
                    SupportCase.closed_at.is_(None),
                )
            )
        )
        .scalars()
        .all()
    )
    assert [case.message for case in waiting] == ["I was charged twice"]


@pytest.mark.asyncio
async def test_a_record_that_cannot_be_written_is_not_answered_around(
    agent: SupportAgent,
) -> None:
    """Losing the record quietly is how the false sentence came back.

    A reply returned with nothing behind it is exactly the state this was
    added to prevent, so a failure to write is a failure to reply.
    """

    class Broken:
        async def record(
            self,
            reference: str,
            enquiry: Enquiry,
            reply: SupportReply,
            customer: int,
        ) -> None:
            raise RuntimeError("the queue is down")

    with pytest.raises(RuntimeError, match="the queue is down"):
        await agent.answer(
            Enquiry(message="I was charged twice"), cases=Broken(), customer=1
        )
