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

from datetime import datetime
from typing import Annotated, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from app.agent.answering import Handover, Outcome, Plan
from app.agent.answering import Review as PlanReview
from app.agent.enquiry import MAX_MESSAGE
from app.agent.intent import Intent
from app.agent.knowledge import Locale
from app.agent.messages import MessageBook
from app.agent.profiles import Source
from app.agent.reasons import (
    ClarificationReason,
    EscalationReason,
    EvidenceReason,
    ReviewReason,
)
from app.agent.reliability import Factor, ReliabilityLevel, Route
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
# Anything a person or an auditor reads. Trimmed before it is measured, since
# a length of one is satisfied by a space and reaches a customer as silence.
Said = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


# Nothing outside a model is accepted anywhere here. A misspelled field was
# being dropped without complaint, so a client sending "oder_id" got a request
# with no order number and no reason to think anything had gone wrong.
_EXACT = ConfigDict(extra="forbid")

LevelName = Literal["unusable", "review_only", "acceptable", "ready"]

# Published as a constant so a generated client gets one, and checked against
# the enum here so adding a level fails at import rather than in somebody's
# parser months later.
SCALE: Final = 3
if int(max(ReliabilityLevel)) != SCALE:
    raise RuntimeError(
        f"the scale published to callers is {SCALE}, the levels go to "
        f"{int(max(ReliabilityLevel))}"
    )


class SupportMessage(BaseModel):
    """One question from a signed-in customer.

    The language is not taken from here. It comes from the account, which is
    where the customer set it, so a client cannot ask for an answer in a
    language nothing is written in.
    """

    model_config = _EXACT

    message: Message
    order_id: Identifier | None = None
    product_reference: Identifier | None = None


class Citation(BaseModel):
    """One piece of evidence an answer rests on."""

    model_config = _EXACT

    source: Source
    reference: Said
    content_hash: Said

    # Empty for written policy, which is approved rather than observed. A row
    # out of the shop's records answers all three, and the one saying none of
    # this describes a real purchase has to survive as far as the case record
    # or it stops being a warning to anybody.
    provider: str | None = None
    observed_at: datetime | None = None
    synthetic: bool | None = None


class Reliability(BaseModel):
    """How far the evidence carried, named on a scale rather than a percentage."""

    model_config = _EXACT

    level: LevelName
    ordinal: int = Field(ge=0, le=SCALE)
    scale: Literal[3]  # SCALE; a Literal cannot be spelled with a name
    factors: dict[Factor, LevelName]

    @model_validator(mode="after")
    def check_the_ordinal_belongs_to_the_level(self) -> "Reliability":
        """Two ways of saying one thing, and they have to agree.

        Published apart, a reply could name the weakest level while carrying
        the number of the strongest, and a caller reading whichever it
        preferred would draw the opposite conclusion.
        """
        if self.ordinal != int(ReliabilityLevel[self.level.upper()]):
            raise ValueError(f"{self.level} is not {self.ordinal} on this scale")
        return self


class Answer(BaseModel):
    """Wording a person approved, and everything it was built from."""

    model_config = _EXACT

    route: Literal[Route.DIRECT_RESPONSE] = Route.DIRECT_RESPONSE
    case: Said
    intent: Intent
    reply: Said
    citations: list[Citation] = Field(min_length=1)
    wording: list[Said] = Field(
        min_length=1,
        description="Approved sentences used, each with the digest it had",
    )
    reliability: Reliability


class Clarification(BaseModel):
    """Something is missing that the customer can supply, and they are asked.

    The code and the sentence are both here and are not alternatives. One is
    stable and machine-readable, so a client can open the right field and
    analytics can count how often it happens; the other is written for a
    person and gets translated. A client reading the sentence to work out what
    happened has read the wrong field.
    """

    model_config = _EXACT

    route: Literal[Route.CLARIFICATION] = Route.CLARIFICATION
    case: Said
    reason: ClarificationReason
    message: Said
    wording: Said


class Escalation(BaseModel):
    """A person takes this one, and the customer is told so.

    Several things can be wrong at once and every code is kept, because the
    queue entry is what gets acted on. One sentence goes back, chosen by a
    declared order, so the same set always reads the same way and nobody
    receives two apologies stitched together.
    """

    model_config = _EXACT

    route: Literal[Route.HUMAN_ESCALATION] = Route.HUMAN_ESCALATION
    case: Said
    reasons: list[EscalationReason | EvidenceReason] = Field(min_length=1)
    message: Said
    wording: Said
    reliability: Reliability | None = None
    citations: list[Citation] = Field(default_factory=list)


class InternalReview(BaseModel):
    """Somebody here finishes it. Nothing is wrong with the request."""

    model_config = _EXACT

    route: Literal[Route.INTERNAL_REVIEW] = Route.INTERNAL_REVIEW
    case: Said
    reasons: list[ReviewReason | EvidenceReason] = Field(min_length=1)
    message: Said
    wording: Said
    reliability: Reliability | None = None
    citations: list[Citation] = Field(default_factory=list)


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
        level=rated.level.name.lower(),  # type: ignore[arg-type]
        ordinal=int(rated.level),
        scale=SCALE,
        factors={
            factor: level.name.lower()  # type: ignore[misc]
            for factor, level in {**rated.required, **rated.contextual}.items()
        },
    )


def replied(
    outcome: Outcome | Escalate | Clarify | TriageReview,
    *,
    messages: MessageBook,
    locale: Locale,
    case: str,
) -> SupportReply:
    """Turn a decision into the one shape that can express it.

    The language is the caller's to supply, because a decision taken at triage
    does not carry one — the enquiry that did has already been left behind by
    then, and only the customer's account knows what to answer in.
    """
    match outcome:
        case Escalate():
            return _escalated(sorted(outcome.reasons, key=str), messages, locale, case)
        case Clarify():
            said = messages.tell([outcome.reason], locale, Route.CLARIFICATION)
            return Clarification(
                case=case,
                reason=outcome.reason,
                message=said.sentence,
                wording=said.cited,
            )
        case TriageReview() | PlanReview():
            said = messages.tell([outcome.reason], locale, Route.INTERNAL_REVIEW)
            return InternalReview(
                case=case,
                reasons=[outcome.reason],
                message=said.sentence,
                wording=said.cited,
            )
        case Handover():
            return _escalated(sorted(outcome.reasons, key=str), messages, locale, case)
        case Plan():
            return _from_plan(outcome, messages, locale, case)
    raise TypeError(f"{outcome!r} is not an outcome")


def _cited(plan: Plan) -> list[Citation]:
    """What the decision rested on, kept whatever became of the request.

    An escalation and a review are the outcomes somebody has to pick up, so
    they are the ones that most need the evidence attached. They were the two
    carrying none of it.
    """
    return [
        Citation(
            source=cited.source,
            reference=cited.reference,
            content_hash=cited.content_hash,
            provider=cited.provider,
            observed_at=cited.observed_at,
            synthetic=cited.synthetic,
        )
        for cited in plan.citations
    ]


def _escalated(
    reasons: list[EscalationReason | EvidenceReason],
    messages: MessageBook,
    locale: Locale,
    case: str,
    reliability: Reliability | None = None,
    citations: list[Citation] | None = None,
) -> Escalation:
    said = messages.tell(reasons, locale, Route.HUMAN_ESCALATION)
    return Escalation(
        case=case,
        reasons=reasons,
        message=said.sentence,
        wording=said.cited,
        reliability=reliability,
        citations=citations or [],
    )


def _from_plan(
    plan: Plan, messages: MessageBook, locale: Locale, case: str
) -> SupportReply:
    if plan.held_back is not None:
        # The ratings all passed. What stopped it is ours, and it is the thing
        # to tell a colleague, so it goes in the column instead of a shortfall
        # none of the dimensions actually reported.
        stopped: list[ReviewReason | EvidenceReason] = [plan.held_back]
        said = messages.tell(stopped, locale, Route.INTERNAL_REVIEW)
        return InternalReview(
            case=case,
            reasons=stopped,
            message=said.sentence,
            wording=said.cited,
            reliability=_reliability(plan),
            citations=_cited(plan),
        )
    reasons = sorted(plan.reasons, key=str)
    if plan.route is Route.HUMAN_ESCALATION:
        return _escalated(
            list(reasons), messages, locale, case, _reliability(plan), _cited(plan)
        )
    if plan.route is Route.INTERNAL_REVIEW:
        said = messages.tell(list(reasons), locale, Route.INTERNAL_REVIEW)
        return InternalReview(
            case=case,
            reasons=list(reasons),
            message=said.sentence,
            wording=said.cited,
            reliability=_reliability(plan),
            citations=_cited(plan),
        )
    assert plan.reply is not None  # the type refuses a direct answer without one
    return Answer(
        case=case,
        intent=plan.intent,
        reply=plan.reply.text,
        citations=_cited(plan),
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
