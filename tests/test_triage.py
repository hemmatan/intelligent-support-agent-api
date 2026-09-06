"""What happens to a message before anything is asked on its behalf."""

from dataclasses import fields

import pytest

from app.agent.enquiry import Enquiry
from app.agent.intent import ClassifierUnavailableError, Intent, intents_in
from app.agent.knowledge import Locale
from app.agent.profiles import Input, Source, profile_for
from app.agent.reasons import (
    BlockedReason,
    ClarificationReason,
    ReviewReason,
    RiskReason,
)
from app.agent.triage import (
    Clarify,
    Escalate,
    Proceed,
    Review,
    UnexplainedEscalationError,
    triage,
)

LINKED = frozenset({Input.COMMERCE_ACCOUNT})
LINKED_WITH_ORDER = frozenset({Input.COMMERCE_ACCOUNT, Input.ORDER_ID})


def asked(
    message: str, known: frozenset[Input] = frozenset(), locale: Locale = "en"
) -> Enquiry:
    """An enquiry holding values for exactly the inputs named, and no others.

    Tests talk about inputs because that is the vocabulary profiles are
    written in. An enquiry carries what those inputs stand for, and derives
    the rest, so a test cannot describe a request that says it has an order
    number while holding none.
    """
    return Enquiry(
        message=message,
        locale=locale,
        customer=7 if Input.COMMERCE_ACCOUNT in known else None,
        order="ORD-4471" if Input.ORDER_ID in known else None,
        product="SKU-9" if Input.PRODUCT_REFERENCE in known else None,
    )


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
    outcome = await triage(
        asked("I was charged twice"),
        classifier=MustNotBeAsked(),
    )
    assert outcome == Escalate(reasons=frozenset({RiskReason.PAYMENT_DISPUTE}))


@pytest.mark.asyncio
async def test_every_kind_of_trouble_in_the_message_is_carried_forward() -> None:
    outcome = await triage(
        asked("Someone hacked my account and used my card fraudulently"),
    )
    assert outcome == Escalate(
        reasons=frozenset({RiskReason.ACCOUNT_COMPROMISE, RiskReason.SUSPECTED_FRAUD})
    )


@pytest.mark.asyncio
async def test_a_policy_question_says_where_it_may_be_answered_from() -> None:
    outcome = await triage(
        asked("How long do I have to return a jacket?"),
    )
    assert isinstance(outcome, Proceed)
    assert outcome.intent is Intent.RETURN_POLICY
    assert outcome.profile.required_sources == {Source.KNOWLEDGE_BASE}
    assert Source.COMMERCE not in outcome.sources


@pytest.mark.asyncio
async def test_a_model_is_not_asked_about_a_message_the_rules_placed() -> None:
    """It raises when spoken to, and is not spoken to.

    A classifier answers only where the phrases produced nothing. Measured
    against messages they could not place, the model evaluated for the job
    named the returns policy for a question about when a parcel would arrive,
    so what it may revise is nothing at all.
    """
    outcome = await triage(
        asked("How long do I have to return a jacket?"), classifier=MustNotBeAsked()
    )
    assert isinstance(outcome, Proceed)
    assert outcome.intent is Intent.RETURN_POLICY


@pytest.mark.asyncio
async def test_a_classifier_sees_what_the_rules_could_not_place() -> None:
    classifier = Insists(Intent.RETURN_POLICY)
    outcome = await triage(
        asked("I need help with my purchase"),
        classifier=classifier,
    )
    assert classifier.seen == ["I need help with my purchase"]
    assert isinstance(outcome, Proceed)
    assert outcome.intent is Intent.RETURN_POLICY


@pytest.mark.asyncio
async def test_a_classifier_that_cannot_place_it_either_asks_the_customer() -> None:
    outcome = await triage(
        asked("I need help with my purchase"),
        classifier=Insists(None),
    )
    assert outcome == Clarify(reason=ClarificationReason.UNRESOLVED_INTENT)


@pytest.mark.asyncio
async def test_without_a_classifier_an_unplaced_message_asks_the_customer() -> None:
    assert await triage(
        asked("I need help with my purchase"),
    ) == Clarify(reason=ClarificationReason.UNRESOLVED_INTENT)


@pytest.mark.asyncio
async def test_two_requests_get_a_question_rather_than_half_an_answer() -> None:
    outcome = await triage(
        asked(
            "Where is my order, and can I return it once it arrives?", LINKED_WITH_ORDER
        ),
    )
    assert outcome == Clarify(reason=ClarificationReason.MULTIPLE_INTENTS)


@pytest.mark.asyncio
async def test_a_missing_order_number_is_asked_for() -> None:
    outcome = await triage(
        asked("Where is my order?", LINKED),
    )
    assert outcome == Clarify(reason=ClarificationReason.MISSING_ORDER_ID)


@pytest.mark.asyncio
async def test_an_unlinked_customer_goes_to_a_person() -> None:
    """They cannot supply what is missing, so asking them wastes their time."""
    outcome = await triage(
        asked("Where is my order?", frozenset({Input.ORDER_ID})),
    )
    assert outcome == Escalate(reasons=frozenset({BlockedReason.CUSTOMER_NOT_LINKED}))


@pytest.mark.asyncio
async def test_the_more_serious_absence_decides() -> None:
    """Both missing. Asking for an order number would not have helped."""
    outcome = await triage(
        asked("Where is my order?", frozenset()),
    )
    assert outcome == Escalate(reasons=frozenset({BlockedReason.CUSTOMER_NOT_LINKED}))


@pytest.mark.asyncio
async def test_everything_present_proceeds() -> None:
    outcome = await triage(
        asked("Where is my order?", LINKED_WITH_ORDER),
    )
    assert isinstance(outcome, Proceed)
    assert outcome.intent is Intent.ORDER_STATUS
    assert Source.HISTORY in outcome.sources


@pytest.mark.asyncio
async def test_asking_after_an_unnamed_product_asks_which_one() -> None:
    outcome = await triage(
        asked("Is it still available?"),
    )
    assert outcome == Clarify(reason=ClarificationReason.MISSING_PRODUCT_REFERENCE)


@pytest.mark.asyncio
async def test_chasing_a_refund_is_a_question_about_money_not_about_policy() -> None:
    outcome = await triage(
        asked("Where is my refund?", LINKED_WITH_ORDER),
    )
    assert isinstance(outcome, Proceed)
    assert outcome.intent is Intent.REFUND_STATUS
    assert outcome.profile.required_sources == {Source.COMMERCE}
    assert Source.KNOWLEDGE_BASE not in outcome.sources


@pytest.mark.asyncio
async def test_a_model_cannot_hand_back_a_policy_the_rules_withheld() -> None:
    """The rules keep the returns policy from somebody chasing money.

    Asked about the leftover, a classifier used to hand it straight back.
    There is no leftover now, and its answer is not taken for a placed
    request, so neither route reopens.
    """
    outcome = await triage(
        asked("Where is my refund?", LINKED_WITH_ORDER),
        classifier=Insists(Intent.RETURN_POLICY),
    )
    assert isinstance(outcome, Proceed)
    assert outcome.intent is Intent.REFUND_STATUS


@pytest.mark.asyncio
async def test_an_order_and_a_refund_are_still_two_questions() -> None:
    """Discarding the refund match answered the order half and said nothing."""
    outcome = await triage(
        asked("Where is my order, and where is my refund?", LINKED_WITH_ORDER),
    )
    assert outcome == Clarify(reason=ClarificationReason.MULTIPLE_INTENTS)


@pytest.mark.asyncio
async def test_saying_an_order_turned_up_is_context_for_the_return() -> None:
    outcome = await triage(
        asked("My order arrived and I want to return it."),
    )
    assert isinstance(outcome, Proceed)
    assert outcome.intent is Intent.RETURN_POLICY


def test_an_escalation_has_to_say_why() -> None:
    """A queue entry nobody can act on is worse than none."""
    with pytest.raises(UnexplainedEscalationError):
        Escalate(reasons=frozenset())


def test_a_plan_cannot_be_given_a_profile_belonging_to_something_else() -> None:
    """It held both and nothing compared them.

    A return-policy request carrying the order-status profile is a request
    authorised to read commerce, which nothing about it justified. There is
    no longer a field to put one in: it is looked up from the intent, so the
    two cannot be set to disagree.
    """
    assert "profile" not in {field.name for field in fields(Proceed)}
    for intent in Intent:
        carried = Proceed(intent=intent, enquiry=asked("anything")).profile
        assert carried is profile_for(intent)


@pytest.mark.asyncio
async def test_a_model_has_no_way_to_say_a_message_is_fine() -> None:
    """The only asymmetry that makes a model safe here.

    A classifier returning nothing at all cannot rescue a message the rules
    escalated, because the rules returned before it would have been reached.
    """
    quiet = Insists(None)
    outcome = await triage(
        asked("I was charged twice"),
        classifier=quiet,
    )
    assert outcome == Escalate(reasons=frozenset({RiskReason.PAYMENT_DISPUTE}))
    assert quiet.seen == []


@pytest.mark.asyncio
async def test_trouble_inside_an_ordinary_request_still_reaches_a_person() -> None:
    """The phrases recognise the return and have to catch the rest as well.

    A model was brought in for exactly this sentence and then measured, and it
    could not tell a report of fraud from a question about avoiding it. So the
    wording is listed, and the limit of what is listed is the limit of what is
    caught — written down in the design notes rather than left to be found.
    """
    message = "I need to return this because a stranger made purchases using my account"
    assert intents_in(message) == {Intent.RETURN_POLICY}
    outcome = await triage(asked(message))
    assert outcome == Escalate(reasons=frozenset({RiskReason.SUSPECTED_FRAUD}))


@pytest.mark.asyncio
async def test_a_message_the_rules_escalated_is_shown_to_nobody() -> None:
    """Consulting it on everything else does not mean consulting it on this.

    It has no way to lower a risk because it never sees a message carrying
    one, which is what makes always asking safe rather than merely useful.
    """
    outcome = await triage(
        asked("I was charged twice"),
        classifier=MustNotBeAsked(),
    )
    assert outcome == Escalate(reasons=frozenset({RiskReason.PAYMENT_DISPUTE}))


class Down:
    """A classifier that cannot answer today."""

    async def classify(self, message: str, locale: Locale) -> Intent | None:
        raise ClassifierUnavailableError("the provider timed out")


@pytest.mark.asyncio
async def test_a_message_nobody_could_read_waits_for_somebody_here() -> None:
    """Only for a message the phrases could not place; nothing else asks it.

    The failure used to leave the endpoint, which is the wrong place: an
    ordinary question and a provider having a bad afternoon became a five
    hundred.
    """
    outcome = await triage(asked("I need help with my purchase"), classifier=Down())
    assert outcome == Review(reason=ReviewReason.INTENT_CHECK_UNAVAILABLE)


@pytest.mark.asyncio
async def test_danger_the_rules_saw_survives_the_model_being_down() -> None:
    """It left before the call, so the call failing changes nothing."""
    outcome = await triage(
        asked("I was charged twice"),
        classifier=Down(),
    )
    assert outcome == Escalate(reasons=frozenset({RiskReason.PAYMENT_DISPUTE}))
