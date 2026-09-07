"""The other half of escalating: a person who can actually take it."""

import asyncio
from collections.abc import AsyncGenerator, Callable
from itertools import count

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
from app.models.user import User, UserRole
from app.services.support import SupportAgent
from main import app

MESSAGES = "/api/v1/support/messages"
CASES = "/api/v1/support/cases"


@pytest.fixture(scope="module")
def agent() -> SupportAgent:
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
async def person(session: AsyncSession) -> Callable[..., object]:
    """Somebody with an account, at whichever role the test needs."""
    made = count()

    async def make(role: UserRole = UserRole.CUSTOMER) -> dict[str, str]:
        nth = next(made)
        user = User(
            username=f"{role.value}-{nth}",
            hashed_password="x",
            role=role,
            preferred_locale="en",
            external_customer_id=8800 + nth,
        )
        session.add(user)
        await session.commit()
        await session.refresh(user)
        issued = create_access_token(user.username, user.id, user.role.value)
        return {"Authorization": f"Bearer {issued.value}"}

    return make  # type: ignore[return-value]


@pytest.mark.asyncio
async def test_a_customer_escalates_and_a_person_picks_it_up(
    async_client: AsyncClient, wired: None, person: Callable[..., object]
) -> None:
    """The whole handoff, end to end.

    Until this existed the service told people their message had reached a
    colleague and gave no colleague any way to reach it.
    """
    customer = await person()  # type: ignore[misc]
    escalated = await async_client.post(
        MESSAGES,
        json={"message": "I was charged twice", "order_id": "ORD-4471"},
        headers=customer,
    )
    reference = escalated.json()["case"]
    assert escalated.json()["route"] == "human_escalation"

    staff = await person(UserRole.SUPPORT_AGENT)  # type: ignore[misc]
    queue = await async_client.get(CASES, headers=staff)
    assert queue.status_code == 200
    waiting = queue.json()
    assert [case["reference"] for case in waiting] == [reference]

    # Everything the decision rested on, without going back to the customer.
    case = waiting[0]
    assert case["message"] == "I was charged twice"
    assert case["order_id"] == "ORD-4471"
    assert case["external_customer_id"] is not None
    assert case["reasons"] == ["payment_dispute"]
    assert case["sent"].startswith("We have passed this")
    assert case["assigned_to"] is None

    claimed = await async_client.post(f"{CASES}/{reference}/claim", headers=staff)
    assert claimed.status_code == 200
    assert claimed.json()["assigned_to"] is not None

    resolved = await async_client.post(
        f"{CASES}/{reference}/resolve",
        json={"note": "Refunded the duplicate charge and wrote to the customer."},
        headers=staff,
    )
    assert resolved.status_code == 200
    assert resolved.json()["closed_at"] is not None
    assert resolved.json()["resolution"].startswith("Refunded")

    # Off the queue, because it is done.
    assert (await async_client.get(CASES, headers=staff)).json() == []


@pytest.mark.asyncio
async def test_a_customer_cannot_read_the_queue(
    async_client: AsyncClient, wired: None, person: Callable[..., object]
) -> None:
    """It holds other people's messages and everything decided about them."""
    customer = await person()  # type: ignore[misc]
    assert (await async_client.get(CASES, headers=customer)).status_code == 403
    assert (await async_client.get(CASES)).status_code == 401


@pytest.mark.asyncio
async def test_an_answered_request_is_a_record_and_not_somebody_s_work(
    async_client: AsyncClient, wired: None, person: Callable[..., object]
) -> None:
    """A queue listing things nobody has to do is a queue nobody reads."""
    customer = await person()  # type: ignore[misc]
    answered = await async_client.post(
        MESSAGES,
        json={"message": "How long do I have to return a jacket?"},
        headers=customer,
    )
    reference = answered.json()["case"]
    staff = await person(UserRole.SUPPORT_AGENT)  # type: ignore[misc]

    assert (await async_client.get(CASES, headers=staff)).json() == []
    # And it cannot be claimed, rather than being claimable but invisible.
    claimed = await async_client.post(f"{CASES}/{reference}/claim", headers=staff)
    assert claimed.status_code == 404


@pytest.mark.asyncio
async def test_a_case_is_not_resolved_twice(
    async_client: AsyncClient, wired: None, person: Callable[..., object]
) -> None:
    """The second note would replace the account of whoever did the work."""
    customer = await person()  # type: ignore[misc]
    reference = (
        await async_client.post(
            MESSAGES, json={"message": "My lawyer will be in touch"}, headers=customer
        )
    ).json()["case"]
    staff = await person(UserRole.SUPPORT_AGENT)  # type: ignore[misc]

    first = await async_client.post(
        f"{CASES}/{reference}/resolve", json={"note": "Passed to legal."}, headers=staff
    )
    assert first.status_code == 200
    second = await async_client.post(
        f"{CASES}/{reference}/resolve", json={"note": "Something else."}, headers=staff
    )
    assert second.status_code == 409
    assert first.json()["resolution"] == "Passed to legal."


@pytest.mark.asyncio
async def test_resolving_without_claiming_still_names_who_did_it(
    async_client: AsyncClient, wired: None, person: Callable[..., object]
) -> None:
    """Somebody dealing with it straight away should not need a two-step."""
    customer = await person()  # type: ignore[misc]
    reference = (
        await async_client.post(
            MESSAGES, json={"message": "Someone got into my account"}, headers=customer
        )
    ).json()["case"]
    staff = await person(UserRole.SUPPORT_AGENT)  # type: ignore[misc]
    resolved = await async_client.post(
        f"{CASES}/{reference}/resolve",
        json={"note": "Reset their access."},
        headers=staff,
    )
    assert resolved.status_code == 200
    assert resolved.json()["assigned_to"] is not None


@pytest.mark.asyncio
async def test_a_reference_nobody_issued_is_not_found(
    async_client: AsyncClient, person: Callable[..., object]
) -> None:
    staff = await person(UserRole.SUPPORT_AGENT)  # type: ignore[misc]
    for path in (f"{CASES}/nope/claim", f"{CASES}/nope/resolve"):
        response = await async_client.post(path, json={"note": "x"}, headers=staff)
        assert response.status_code == 404


@pytest.mark.asyncio
async def test_a_resolution_has_to_say_something(
    async_client: AsyncClient, wired: None, person: Callable[..., object]
) -> None:
    """An empty note records that somebody closed it and nothing about why."""
    customer = await person()  # type: ignore[misc]
    reference = (
        await async_client.post(
            MESSAGES, json={"message": "I was charged twice"}, headers=customer
        )
    ).json()["case"]
    staff = await person(UserRole.SUPPORT_AGENT)  # type: ignore[misc]
    response = await async_client.post(
        f"{CASES}/{reference}/resolve", json={"note": "   "}, headers=staff
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_a_case_already_taken_is_not_handed_to_somebody_else(
    async_client: AsyncClient, wired: None, person: Callable[..., object]
) -> None:
    """Claiming replaced whoever held it and returned success to both.

    Two people would each have been told the case was theirs, which is the
    single thing claiming exists to prevent.
    """
    customer = await person()  # type: ignore[misc]
    reference = (
        await async_client.post(
            MESSAGES, json={"message": "I was charged twice"}, headers=customer
        )
    ).json()["case"]
    alice = await person(UserRole.SUPPORT_AGENT)  # type: ignore[misc]
    bob = await person(UserRole.SUPPORT_AGENT)  # type: ignore[misc]

    first = await async_client.post(f"{CASES}/{reference}/claim", headers=alice)
    assert first.status_code == 200
    hers = first.json()["assigned_to"]

    second = await async_client.post(f"{CASES}/{reference}/claim", headers=bob)
    assert second.status_code == 409
    still = await async_client.get(CASES, headers=bob)
    assert still.json()[0]["assigned_to"] == hers


@pytest.mark.asyncio
async def test_two_people_claiming_at_once_produce_one_winner(
    async_client: AsyncClient, wired: None, person: Callable[..., object]
) -> None:
    """The condition is in the statement, so the database settles it.

    Read the row, decide in Python, write it back, and there is a window
    between the decision and the write that both requests pass through.
    """
    customer = await person()  # type: ignore[misc]
    reference = (
        await async_client.post(
            MESSAGES, json={"message": "Someone got into my account"}, headers=customer
        )
    ).json()["case"]
    racers = [await person(UserRole.SUPPORT_AGENT) for _ in range(4)]  # type: ignore[misc]

    results = await asyncio.gather(
        *(
            async_client.post(f"{CASES}/{reference}/claim", headers=headers)
            for headers in racers
        )
    )
    codes = sorted(response.status_code for response in results)
    assert codes == [200, 409, 409, 409], codes


@pytest.mark.asyncio
async def test_whoever_finished_it_is_who_the_record_names(
    async_client: AsyncClient, wired: None, person: Callable[..., object]
) -> None:
    """A case taken by one person and closed by another credited the first.

    Somebody covering a colleague's shift did the work and the record said
    their colleague had.
    """
    customer = await person()  # type: ignore[misc]
    reference = (
        await async_client.post(
            MESSAGES, json={"message": "My lawyer will be in touch"}, headers=customer
        )
    ).json()["case"]
    alice = await person(UserRole.SUPPORT_AGENT)  # type: ignore[misc]
    bob = await person(UserRole.SUPPORT_AGENT)  # type: ignore[misc]

    claimed = (
        await async_client.post(f"{CASES}/{reference}/claim", headers=alice)
    ).json()
    resolved = (
        await async_client.post(
            f"{CASES}/{reference}/resolve",
            json={"note": "Bob picked this up and passed it to legal."},
            headers=bob,
        )
    ).json()

    assert resolved["assigned_to"] == claimed["assigned_to"]
    assert resolved["resolved_by"] != resolved["assigned_to"]


@pytest.mark.asyncio
async def test_two_people_resolving_at_once_produce_one_account(
    async_client: AsyncClient, wired: None, person: Callable[..., object]
) -> None:
    """The second note would replace the first, and both callers hear success."""
    customer = await person()  # type: ignore[misc]
    reference = (
        await async_client.post(
            MESSAGES, json={"message": "I was charged twice"}, headers=customer
        )
    ).json()["case"]
    racers = [await person(UserRole.SUPPORT_AGENT) for _ in range(3)]  # type: ignore[misc]

    results = await asyncio.gather(
        *(
            async_client.post(
                f"{CASES}/{reference}/resolve",
                json={"note": f"Handled by number {n}."},
                headers=headers,
            )
            for n, headers in enumerate(racers)
        )
    )
    assert sorted(r.status_code for r in results) == [200, 409, 409]
    won = next(r for r in results if r.status_code == 200).json()
    assert won["resolution"].startswith("Handled by number")


@pytest.mark.asyncio
async def test_a_case_records_what_the_request_was_allowed_to_read(
    async_client: AsyncClient, wired: None, person: Callable[..., object]
) -> None:
    """The design notes say a source plan is kept, and none was.

    Derivable from the intent today, and stored anyway for the reason the
    delivered words are stored: a profile can be edited afterwards, and the
    record answers what was permitted then.
    """
    customer = await person()  # type: ignore[misc]
    escalated = await async_client.post(
        MESSAGES,
        json={"message": "Do you ship to Belgium?"},
        headers=customer,
    )
    staff = await person(UserRole.SUPPORT_AGENT)  # type: ignore[misc]
    case = next(
        c
        for c in (await async_client.get(CASES, headers=staff)).json()
        if c["reference"] == escalated.json()["case"]
    )
    # A shipping question is placed, so the plan is the one its profile names.
    assert case["sources"] == ["history", "knowledge_base"]


@pytest.mark.asyncio
async def test_a_request_that_never_got_placed_names_no_sources(
    async_client: AsyncClient, wired: None, person: Callable[..., object]
) -> None:
    """Trouble is reported before any profile is chosen, so there is no plan.

    Recording one anyway would put a guess where an audit expects a fact.
    """
    customer = await person()  # type: ignore[misc]
    escalated = await async_client.post(
        MESSAGES, json={"message": "I was charged twice"}, headers=customer
    )
    staff = await person(UserRole.SUPPORT_AGENT)  # type: ignore[misc]
    case = next(
        c
        for c in (await async_client.get(CASES, headers=staff)).json()
        if c["reference"] == escalated.json()["case"]
    )
    assert case["sources"] == []
