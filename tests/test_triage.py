"""What happens to a message before anything is asked on its behalf."""

import pytest

from app.agent.intent import Intent
from app.agent.knowledge import Locale
from app.agent.profiles import Input, Source
from app.agent.reasons import ReasonCode
from app.agent.triage import Clarify, Escalate, Proceed, triage

LINKED = frozenset({Input.COMMERCE_ACCOUNT})
LINKED_WITH_ORDER = frozenset({Input.COMMERCE_ACCOUNT, Input.ORDER_ID})


class MustNotBeAsked:
    """Fails the test if triage consults it.

    Counting calls and asserting zero passes just as well when the spy was
    never wired up. Raising cannot.
    """

    async def classify(self, message: str, locale: Locale) -> Intent | None:
        raise AssertionError("a classifier saw a message the rules had settled")


class Insists:
    """A classifier that answers whatever it is asked."""

    def __init__(self, answer: Intent | None):
        self.answer = answer
        self.seen: list[str] = []

    async def classify(self, message: str, locale: Locale) -> Intent | None:
        self.seen.append(message)
        return self.answer


@pytest.mark.asyncio
async def test_a_report_of_trouble_reaches_a_person_before_anything_else_runs() -> None:
    """The classifier raises if consulted. It is not consulted."""
    outcome = await triage("I was charged twice", classifier=MustNotBeAsked())
    assert outcome == Escalate(reasons=frozenset({ReasonCode.PAYMENT_DISPUTE}))


@pytest.mark.asyncio
async def test_every_kind_of_trouble_in_the_message_is_carried_forward() -> None:
    outcome = await triage("Someone hacked my account and used my card fraudulently")
    assert outcome == Escalate(
        reasons=frozenset({ReasonCode.ACCOUNT_COMPROMISE, ReasonCode.SUSPECTED_FRAUD})
    )


@pytest.mark.asyncio
async def test_a_policy_question_says_where_it_may_be_answered_from() -> None:
    outcome = await triage("How long do I have to return a jacket?")
    assert isinstance(outcome, Proceed)
    assert outcome.intent is Intent.RETURN_POLICY
    assert outcome.profile.required_sources == {Source.KNOWLEDGE_BASE}
    assert Source.COMMERCE not in outcome.sources


@pytest.mark.asyncio
async def test_a_classifier_is_never_asked_about_a_message_the_rules_placed() -> None:
    outcome = await triage(
        "How long do I have to return a jacket?", classifier=MustNotBeAsked()
    )
    assert isinstance(outcome, Proceed)


@pytest.mark.asyncio
async def test_a_classifier_sees_what_the_rules_could_not_place() -> None:
    classifier = Insists(Intent.RETURN_POLICY)
    outcome = await triage("I need help with my purchase", classifier=classifier)
    assert classifier.seen == ["I need help with my purchase"]
    assert isinstance(outcome, Proceed)
    assert outcome.intent is Intent.RETURN_POLICY


@pytest.mark.asyncio
async def test_a_classifier_that_cannot_place_it_either_asks_the_customer() -> None:
    outcome = await triage("I need help with my purchase", classifier=Insists(None))
    assert outcome == Clarify(reason=ReasonCode.UNRESOLVED_INTENT)


@pytest.mark.asyncio
async def test_without_a_classifier_an_unplaced_message_asks_the_customer() -> None:
    assert await triage("I need help with my purchase") == Clarify(
        reason=ReasonCode.UNRESOLVED_INTENT
    )


@pytest.mark.asyncio
async def test_two_requests_get_a_question_rather_than_half_an_answer() -> None:
    outcome = await triage(
        "Where is my order, and can I return it once it arrives?",
        known=LINKED_WITH_ORDER,
    )
    assert outcome == Clarify(reason=ReasonCode.MULTIPLE_INTENTS)


@pytest.mark.asyncio
async def test_a_missing_order_number_is_asked_for() -> None:
    outcome = await triage("Where is my order?", known=LINKED)
    assert outcome == Clarify(reason=ReasonCode.MISSING_ORDER_ID)


@pytest.mark.asyncio
async def test_an_unlinked_customer_goes_to_a_person() -> None:
    """They cannot supply what is missing, so asking them wastes their time."""
    outcome = await triage("Where is my order?", known=frozenset({Input.ORDER_ID}))
    assert outcome == Escalate(reasons=frozenset({ReasonCode.CUSTOMER_NOT_LINKED}))


@pytest.mark.asyncio
async def test_the_more_serious_absence_decides() -> None:
    """Both missing. Asking for an order number would not have helped."""
    outcome = await triage("Where is my order?", known=frozenset())
    assert outcome == Escalate(reasons=frozenset({ReasonCode.CUSTOMER_NOT_LINKED}))


@pytest.mark.asyncio
async def test_everything_present_proceeds() -> None:
    outcome = await triage("Where is my order?", known=LINKED_WITH_ORDER)
    assert isinstance(outcome, Proceed)
    assert outcome.intent is Intent.ORDER_STATUS
    assert Source.HISTORY in outcome.sources


@pytest.mark.asyncio
async def test_asking_after_an_unnamed_product_asks_which_one() -> None:
    outcome = await triage("Is it still available?")
    assert outcome == Clarify(reason=ReasonCode.MISSING_PRODUCT_REFERENCE)


@pytest.mark.asyncio
async def test_chasing_a_refund_is_not_answered_from_the_policy() -> None:
    """No intent covers it, so it asks rather than quoting a return window."""
    assert await triage("Where is my refund?") == Clarify(
        reason=ReasonCode.UNRESOLVED_INTENT
    )
