"""The shape a client is given, and the shapes it can never be given."""

import pytest
from pydantic import TypeAdapter, ValidationError

from app.agent.answering import Citation as Cited
from app.agent.answering import Handover, Plan
from app.agent.answering import Review as PlanReview
from app.agent.enquiry import MAX_MESSAGE
from app.agent.intent import Intent
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
    reply = replied(answered())
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
            Clarify(ClarificationReason.MISSING_ORDER_ID),
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
    reply = replied(outcome)  # type: ignore[arg-type]
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
    reply = replied(stopped(SUNK))
    assert isinstance(reply, Escalation)
    assert reply.route is Route.HUMAN_ESCALATION
    assert reply.reasons == [EvidenceReason.NOT_COVERED]
    assert reply.reliability is not None
    assert reply.reliability.level == "unusable"


def test_a_rating_that_held_an_answer_back_says_what_it_was() -> None:
    reply = replied(stopped(HELD))
    assert isinstance(reply, InternalReview)
    assert reply.route is Route.INTERNAL_REVIEW
    assert reply.reasons == [EvidenceReason.NOT_COVERED]
    assert reply.reliability is not None
    assert reply.reliability.level == "review_only"


def test_a_reply_is_told_apart_by_its_route_alone() -> None:
    """A client reads one field and knows which of four shapes it holds."""
    for outcome in (answered(), Clarify(ClarificationReason.MULTIPLE_INTENTS)):
        as_sent = replied(outcome).model_dump(mode="json")
        assert REPLIES.validate_python(as_sent).route == as_sent["route"]


def test_an_answer_cannot_be_described_without_what_it_said() -> None:
    """Each of these is otherwise complete, so only the named fault fails it.

    Written the lazy way — leaving several fields out at once — they passed
    on whichever pydantic noticed first, and proved nothing about the rule
    they were named for.
    """
    whole = replied(answered()).model_dump(mode="json")

    for missing in ("reply", "citations", "wording"):
        with pytest.raises(ValidationError, match=missing):
            REPLIES.validate_python(
                {**whole, missing: [] if missing != "reply" else ""}
            )

    stopped = replied(Escalate(frozenset({RiskReason.PAYMENT_DISPUTE}))).model_dump(
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
    sound = replied(answered()).model_dump(mode="json")["reliability"]
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
    sound = replied(answered()).model_dump(mode="json")["reliability"]
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
