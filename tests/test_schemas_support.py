"""The shape a client is given, and the shapes it can never be given."""

import pytest
from pydantic import TypeAdapter, ValidationError

from app.agent.answering import Citation as Cited
from app.agent.answering import Handover, Plan
from app.agent.answering import Review as PlanReview
from app.agent.enquiry import MAX_MESSAGE
from app.agent.intent import Intent
from app.agent.knowledge import Locale
from app.agent.messages import load_messages
from app.agent.profiles import Source
from app.agent.reasons import (
    BlockedReason,
    ClarificationReason,
    EvidenceReason,
    ReviewReason,
    RiskReason,
)
from app.agent.reliability import Assessment, Factor, ReliabilityLevel, Route
from app.agent.responses import ApprovedReply
from app.agent.triage import Clarify, Escalate
from app.agent.triage import Review as TriageReview
from app.schemas.support import (
    SCALE,
    Answer,
    Clarification,
    Escalation,
    InternalReview,
    Reliability,
    SupportMessage,
    SupportReply,
    replied,
)

REPLIES: TypeAdapter[SupportReply] = TypeAdapter(SupportReply)
BOOK = load_messages()


CASE = "11111111-2222-3333-4444-555555555555"


def spoken(outcome: object, locale: Locale = "en") -> SupportReply:
    return replied(
        outcome,  # type: ignore[arg-type]
        messages=BOOK,
        locale=locale,
        case=CASE,
    )


READY = Assessment(required={Factor.COVERAGE: ReliabilityLevel.READY})
SUNK = Assessment(required={Factor.COVERAGE: ReliabilityLevel.UNUSABLE})
HELD = Assessment(required={Factor.COVERAGE: ReliabilityLevel.REVIEW_ONLY})
CITED = (
    Cited(
        source=Source.KNOWLEDGE_BASE, reference="kb:x.en.v1", content_hash="sha256:a"
    ),
)


def answered() -> Plan:
    return Plan(
        intent=Intent.RETURN_POLICY,
        requested=frozenset(),
        assessment=READY,
        citations=CITED,
        reply=ApprovedReply(
            text="Returns are accepted.", said=("say:x.en.v1@sha256:b",)
        ),
    )


def test_a_question_answered_carries_what_it_rested_on() -> None:
    reply = spoken(answered())
    assert isinstance(reply, Answer)
    assert reply.route is Route.DIRECT_RESPONSE
    assert reply.reply == "Returns are accepted."
    assert [c.reference for c in reply.citations] == ["kb:x.en.v1"]
    assert reply.wording == ["say:x.en.v1@sha256:b"]
    assert reply.reliability.level == "ready"
    assert reply.reliability.scale == int(max(ReliabilityLevel))


@pytest.mark.parametrize(
    ("outcome", "expected", "route"),
    [
        (
            Escalate(frozenset({RiskReason.PAYMENT_DISPUTE})),
            Escalation,
            Route.HUMAN_ESCALATION,
        ),
        (
            Clarify(ClarificationReason.MISSING_ORDER_ID, intent=Intent.ORDER_STATUS),
            Clarification,
            Route.CLARIFICATION,
        ),
        (
            TriageReview(ReviewReason.INTENT_CHECK_UNAVAILABLE),
            InternalReview,
            Route.INTERNAL_REVIEW,
        ),
        (
            PlanReview(ReviewReason.SOURCE_UNAVAILABLE),
            InternalReview,
            Route.INTERNAL_REVIEW,
        ),
        (
            Handover(frozenset({BlockedReason.NO_SUPPORTING_EVIDENCE})),
            Escalation,
            Route.HUMAN_ESCALATION,
        ),
    ],
    ids=["risk", "missing input", "no reading", "no source", "no evidence"],
)
def test_every_way_a_request_can_stop_has_a_shape(
    outcome: object, expected: type, route: Route
) -> None:
    reply = spoken(outcome)  # type: ignore[arg-type]
    assert isinstance(reply, expected)
    assert reply.route is route


def stopped(assessment: Assessment) -> Plan:
    return Plan(
        intent=Intent.RETURN_POLICY,
        requested=frozenset(),
        assessment=assessment,
        citations=CITED,
    )


def test_a_rating_that_sank_an_answer_says_what_it_was() -> None:
    """A queue entry arrives with the rating that put it there."""
    reply = spoken(stopped(SUNK))
    assert isinstance(reply, Escalation)
    assert reply.route is Route.HUMAN_ESCALATION
    assert reply.reasons == [EvidenceReason.NOT_COVERED]
    assert reply.reliability is not None
    assert reply.reliability.level == "unusable"


def test_a_rating_that_held_an_answer_back_says_what_it_was() -> None:
    reply = spoken(stopped(HELD))
    assert isinstance(reply, InternalReview)
    assert reply.route is Route.INTERNAL_REVIEW
    assert reply.reasons == [EvidenceReason.NOT_COVERED]
    assert reply.reliability is not None
    assert reply.reliability.level == "review_only"


def test_a_reply_is_told_apart_by_its_route_alone() -> None:
    """A client reads one field and knows which of four shapes it holds."""
    for outcome in (answered(), Clarify(ClarificationReason.MULTIPLE_INTENTS)):
        as_sent = spoken(outcome).model_dump(mode="json")
        assert REPLIES.validate_python(as_sent).route == as_sent["route"]


def test_an_answer_cannot_be_described_without_what_it_said() -> None:
    """Each of these is otherwise complete, so only the named fault fails it.

    Written the lazy way — leaving several fields out at once — they passed
    on whichever pydantic noticed first, and proved nothing about the rule
    they were named for.
    """
    whole = spoken(answered()).model_dump(mode="json")

    for missing in ("reply", "citations", "wording"):
        with pytest.raises(ValidationError, match=missing):
            REPLIES.validate_python(
                {**whole, missing: [] if missing != "reply" else ""}
            )

    stopped = spoken(Escalate(frozenset({RiskReason.PAYMENT_DISPUTE}))).model_dump(
        mode="json"
    )
    with pytest.raises(ValidationError, match="reasons"):
        REPLIES.validate_python({**stopped, "reasons": []})
    with pytest.raises(ValidationError, match="reply"):
        REPLIES.validate_python({**stopped, "reply": "Returns are accepted."})
    with pytest.raises(ValidationError):
        REPLIES.validate_python({"route": "clarification", "reason": "payment_dispute"})


def test_a_published_rating_has_to_be_a_rating() -> None:
    """The scale is a promise, and an unconstrained field is not one."""
    sound = spoken(answered()).model_dump(mode="json")["reliability"]
    for broken in (
        {**sound, "level": "banana"},
        {**sound, "scale": -2},
        {**sound, "ordinal": 99},
        {**sound, "factors": {"coverage": "maybe"}},
        {**sound, "factors": {"vibes": "ready"}},
        {**sound, "extra": 1},
    ):
        with pytest.raises(ValidationError):
            Reliability.model_validate(broken)


def test_the_two_ways_a_rating_states_itself_have_to_agree() -> None:
    """A caller reading the name and one reading the number must not differ."""
    sound = spoken(answered()).model_dump(mode="json")["reliability"]
    with pytest.raises(ValidationError, match="on this scale"):
        Reliability.model_validate({**sound, "ordinal": 0})


def test_the_published_scale_still_matches_the_levels() -> None:
    """Stated as a constant to callers; the import fails if a level is added."""
    assert int(max(ReliabilityLevel)) == SCALE


@pytest.mark.parametrize(
    "sent",
    [
        {"message": ""},
        {"message": "   "},
        {"message": "a" * (MAX_MESSAGE + 1)},
        {"message": "hello", "order_id": ""},
        {"message": "hello", "order_id": "   "},
        {"message": "hello", "order_id": "ORD 4471"},
        {"message": "hello", "order_id": "a" * 65},
        {"message": "hello", "product_reference": "../../etc/passwd"},
        {"message": "hello", "oder_id": "ORD-4471"},
    ],
    ids=[
        "empty",
        "spaces",
        "too long",
        "blank id",
        "whitespace id",
        "id with a space",
        "id too long",
        "id that is a path",
        "a misspelled field",
    ],
)
def test_a_malformed_request_is_refused_before_anything_reads_it(
    sent: dict[str, str],
) -> None:
    with pytest.raises(ValidationError):
        SupportMessage.model_validate(sent)


def test_a_well_formed_request_arrives_trimmed() -> None:
    asked = SupportMessage.model_validate(
        {"message": "  Where is my order?  ", "order_id": " ORD-4471 "}
    )
    assert asked.message == "Where is my order?"
    assert asked.order_id == "ORD-4471"


def test_the_language_is_not_the_client_s_to_choose() -> None:
    """It comes from the account, so nobody can ask for one nothing is written in."""
    assert "locale" not in SupportMessage.model_fields


def test_a_generated_client_is_told_how_to_choose_between_the_shapes() -> None:
    """The discriminator is a promise to whoever consumes the schema.

    Without it a client is left matching shapes by trial, and every route this
    service can take has to be present in the mapping or one of them arrives
    as something a generated type cannot express.
    """
    schema = REPLIES.json_schema()
    assert schema["discriminator"]["propertyName"] == "route"
    assert set(schema["discriminator"]["mapping"]) == {route.value for route in Route}


@pytest.mark.parametrize("locale", ["en", "fr"], ids=["english", "french"])
def test_nothing_reaches_a_customer_as_a_bare_code(locale: Locale) -> None:
    """A code is not something to show anybody.

    Being asked, being handed on and being held back are three of the four
    outcomes, and all three used to arrive as a reason and nothing else, which
    left the customer with silence.
    """
    stopping = [
        Escalate(frozenset({RiskReason.PAYMENT_DISPUTE})),
        Clarify(ClarificationReason.MISSING_ORDER_ID, intent=Intent.ORDER_STATUS),
        TriageReview(ReviewReason.INTENT_CHECK_UNAVAILABLE),
        Handover(frozenset({BlockedReason.NO_SUPPORTING_EVIDENCE})),
        stopped(SUNK),
        stopped(HELD),
    ]
    for outcome in stopping:
        reply = spoken(outcome, locale)
        assert isinstance(reply, Clarification | Escalation | InternalReview)
        assert reply.message.strip(), outcome
        assert reply.wording.startswith("say:"), outcome
        assert "@sha256:" in reply.wording, outcome
        # The code the client branches on is still there beside it.
        assert getattr(reply, "reason", None) or getattr(reply, "reasons", None)


def test_a_customer_is_told_one_thing_however_much_went_wrong() -> None:
    """Two apologies stitched together read as a form letter."""
    both = Escalate(frozenset({RiskReason.PAYMENT_DISPUTE, RiskReason.LEGAL_THREAT}))
    reply = spoken(both)
    assert isinstance(reply, Escalation)
    assert reply.message
    assert len(reply.reasons) == 2
    alone = spoken(Escalate(frozenset({RiskReason.PAYMENT_DISPUTE})))
    assert isinstance(alone, Escalation)
    assert reply.message == alone.message


def test_the_language_reaches_the_wording() -> None:
    asked = Clarify(ClarificationReason.MISSING_ORDER_ID, intent=Intent.ORDER_STATUS)
    english, french = spoken(asked, "en"), spoken(asked, "fr")
    assert isinstance(english, Clarification)
    assert isinstance(french, Clarification)
    assert english.message != french.message
    assert french.wording.startswith("say:ask_for_order_number.fr.")


def test_an_unlinked_account_is_told_what_is_being_done_about_it() -> None:
    """Not the generic handover: the design says what this customer hears."""
    reply = spoken(Escalate(frozenset({BlockedReason.CUSTOMER_NOT_LINKED})))
    assert isinstance(reply, Escalation)
    assert reply.message
    assert reply.wording.startswith("say:account_not_linked.en.")
    alone = spoken(Escalate(frozenset({RiskReason.PAYMENT_DISPUTE})))
    assert isinstance(alone, Escalation)
    assert reply.message != alone.message


@pytest.mark.parametrize("blanked", ["message", "wording"], ids=["message", "wording"])
def test_a_stopped_request_cannot_be_described_without_telling_them(
    blanked: str,
) -> None:
    """The rule, not the current behaviour.

    Asserting only what this service produces leaves the constraint untested:
    the pipeline never builds an empty one, so removing the requirement
    changed nothing any test could see.
    """
    for outcome in (
        Clarify(ClarificationReason.MISSING_ORDER_ID, intent=Intent.ORDER_STATUS),
        Escalate(frozenset({RiskReason.PAYMENT_DISPUTE})),
        TriageReview(ReviewReason.SOURCE_UNAVAILABLE),
    ):
        whole = spoken(outcome).model_dump(mode="json")
        with pytest.raises(ValidationError, match=blanked):
            REPLIES.validate_python({**whole, blanked: ""})


@pytest.mark.parametrize("blank", ["   ", "\t", "\n"], ids=["spaces", "tab", "newline"])
def test_whitespace_is_not_something_to_send_anybody(blank: str) -> None:
    """A minimum length of one is satisfied by a space.

    Which reaches a customer as an empty bubble, and an auditor as a citation
    pointing nowhere.
    """
    asked = spoken(
        Clarify(ClarificationReason.MISSING_ORDER_ID, intent=Intent.ORDER_STATUS)
    ).model_dump(mode="json")
    for field in ("message", "wording"):
        with pytest.raises(ValidationError, match=field):
            REPLIES.validate_python({**asked, field: blank})

    answer = spoken(answered()).model_dump(mode="json")
    with pytest.raises(ValidationError, match="reply"):
        REPLIES.validate_python({**answer, "reply": blank})
    with pytest.raises(ValidationError, match="wording"):
        REPLIES.validate_python({**answer, "wording": [blank]})


def test_every_reply_names_the_record_it_was_written_into() -> None:
    """A customer can quote it, and an auditor can find the decision behind it."""
    for outcome in (
        answered(),
        Clarify(ClarificationReason.MISSING_ORDER_ID, intent=Intent.ORDER_STATUS),
        Escalate(frozenset({RiskReason.PAYMENT_DISPUTE})),
        TriageReview(ReviewReason.SOURCE_UNAVAILABLE),
        stopped(SUNK),
    ):
        assert spoken(outcome).case == CASE, outcome
