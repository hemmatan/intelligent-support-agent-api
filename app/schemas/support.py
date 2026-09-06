"""What a client sends the support agent, and what comes back.

One shape per outcome, told apart by the route, rather than one shape with
most of its fields empty. A single model would have carried a reply beside an
escalation reason and left the pairing to whoever read it; here an answer
without wording, or a clarification carrying a fraud code, cannot be built.

Every one of these is a two hundred. Being asked a question, being sent to a
person and being held for review are all decisions the service reached, and
the route says which. Status codes are kept for the request being wrong,
the caller being unknown, and the service being broken.
"""

from typing import Annotated, Literal

from pydantic import BaseModel, Field, StringConstraints

from app.agent.answering import Handover, Outcome, Plan
from app.agent.answering import Review as PlanReview
from app.agent.enquiry import MAX_MESSAGE
from app.agent.intent import Intent
from app.agent.reasons import (
    ClarificationReason,
    EscalationReason,
    EvidenceReason,
    ReviewReason,
)
from app.agent.reliability import ReliabilityLevel, Route
from app.agent.triage import Clarify, Escalate, Proceed
from app.agent.triage import Review as TriageReview

# Trimmed here as well as in the enquiry, because the two answer different
# questions: this one refuses a malformed request, and the enquiry keeps its
# guarantee for every caller, HTTP or not.
Message = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_MESSAGE),
]
# Deliberately narrow. An identifier is copied from an email or a receipt, so
# it holds no spaces and no punctuation beyond what a reference number uses;
# anything else is a client sending the wrong field.
Identifier = Annotated[
    str,
    StringConstraints(strip_whitespace=True, pattern=r"^[\w.\-]{1,64}$"),
]


class SupportMessage(BaseModel):
    """One question from a signed-in customer.

    The language is not taken from here. It comes from the account, which is
    where the customer set it, so a client cannot ask for an answer in a
    language nothing is written in.
    """

    message: Message
    order_id: Identifier | None = None
    product_reference: Identifier | None = None


class Citation(BaseModel):
    """One piece of evidence an answer rests on."""

    source: str
    reference: str
    content_hash: str


class Reliability(BaseModel):
    """How far the evidence carried, named on a scale rather than a percentage."""

    level: str
    ordinal: int
    scale: int
    factors: dict[str, str]


class Answer(BaseModel):
    """Wording a person approved, and everything it was built from."""

    route: Literal[Route.DIRECT_RESPONSE] = Route.DIRECT_RESPONSE
    intent: Intent
    reply: str
    citations: list[Citation]
    wording: list[str] = Field(
        description="Approved sentences used, each with the digest it had"
    )
    reliability: Reliability


class Clarification(BaseModel):
    """Something is missing that the customer can supply."""

    route: Literal[Route.CLARIFICATION] = Route.CLARIFICATION
    reason: ClarificationReason


class Escalation(BaseModel):
    """A person takes this one. No wording is included, because none was made."""

    route: Literal[Route.HUMAN_ESCALATION] = Route.HUMAN_ESCALATION
    reasons: list[EscalationReason | EvidenceReason]
    reliability: Reliability | None = None


class InternalReview(BaseModel):
    """Somebody here finishes it. Nothing is wrong with the request."""

    route: Literal[Route.INTERNAL_REVIEW] = Route.INTERNAL_REVIEW
    reasons: list[ReviewReason | EvidenceReason]
    reliability: Reliability | None = None


SupportReply = Annotated[
    Answer | Clarification | Escalation | InternalReview,
    Field(discriminator="route"),
]


def _reliability(plan: Plan) -> Reliability:
    """Read off the assessment rather than its audit dump.

    The dump is untyped on purpose — it goes to a log — and rebuilding a
    response from strings would lose every guarantee the levels carry.
    """
    rated = plan.assessment
    return Reliability(
        level=rated.level.name.lower(),
        ordinal=int(rated.level),
        scale=int(max(ReliabilityLevel)),
        factors={
            factor.value: level.name.lower()
            for factor, level in {**rated.required, **rated.contextual}.items()
        },
    )


def replied(outcome: Outcome | Escalate | Clarify | TriageReview) -> SupportReply:
    """Turn a decision into the one shape that can express it.

    Every branch is reachable: a request stops at triage, at a gate, or at a
    rating, and the last of those can land on any of three routes depending on
    what the evidence turned out to be worth.
    """
    match outcome:
        case Escalate():
            return Escalation(reasons=sorted(outcome.reasons, key=str))
        case Clarify():
            return Clarification(reason=outcome.reason)
        case TriageReview() | PlanReview():
            return InternalReview(reasons=[outcome.reason])
        case Handover():
            return Escalation(reasons=sorted(outcome.reasons, key=str))
        case Plan():
            return _from_plan(outcome)
    raise TypeError(f"{outcome!r} is not an outcome")


def _from_plan(plan: Plan) -> SupportReply:
    reasons = sorted(plan.reasons, key=str)
    if plan.route is Route.HUMAN_ESCALATION:
        return Escalation(reasons=list(reasons), reliability=_reliability(plan))
    if plan.route is Route.INTERNAL_REVIEW:
        return InternalReview(reasons=list(reasons), reliability=_reliability(plan))
    assert plan.reply is not None  # the type refuses a direct answer without one
    return Answer(
        intent=plan.intent,
        reply=plan.reply.text,
        citations=[
            Citation(
                source=cited.source,
                reference=cited.reference,
                content_hash=cited.content_hash,
            )
            for cited in plan.citations
        ],
        wording=list(plan.reply.said),
        reliability=_reliability(plan),
    )


__all__ = [
    "Answer",
    "Citation",
    "Clarification",
    "Escalation",
    "InternalReview",
    "Proceed",
    "Reliability",
    "SupportMessage",
    "SupportReply",
    "replied",
]
