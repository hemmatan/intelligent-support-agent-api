"""Deciding what a message is before going anywhere to answer it.

Three things can come out of this, and only one of them leads to a source
being queried. They are separate types, and each holds only what it cannot be
derived from: a destination is the type itself, a reason comes from the set
belonging to that destination, and a plan carries the intent alone and looks
its profile up. An escalation delivered directly, a question to the customer
explained by a fraud report, a plan whose profile governs a different intent —
none of them can be assembled.

Order is the safety property. Risk rules read the message first, so a message
reporting trouble is escalated before a classifier has seen it and before any
source is asked. Nothing downstream can undo that, because nothing downstream
runs.
"""

from dataclasses import dataclass

from app.agent.enquiry import Enquiry
from app.agent.intent import (
    ClassifierUnavailableError,
    Intent,
    IntentClassifier,
    intents_in,
)
from app.agent.profiles import DecisionProfile, Source, profile_for
from app.agent.reasons import (
    ClarificationReason,
    EscalationReason,
    ReviewReason,
    RiskReason,
)
from app.agent.reliability import Route
from app.agent.risk import risks_in


class UnexplainedEscalationError(ValueError):
    """An escalation carrying no reason.

    Somebody picks this up and has to be told what they are looking at, so an
    empty set is a fault here rather than a thin queue entry later.
    """


@dataclass(frozen=True)
class Escalate:
    """A person takes this one. No answer is drafted."""

    reasons: frozenset[EscalationReason]

    # What the message was taken to be asking, where that was settled before
    # the request stopped. Empty is a finding rather than a gap: danger is
    # read off the sentence and returns before anything is classified, so
    # nothing was established to record.
    intent: Intent | None = None

    # Any risk at all refuses a reading, not merely a set made up entirely of
    # them. A mixture is precisely what moving the risk check below
    # classification would produce, and that is the arrangement this exists to
    # catch — so the looser test would have been blind to the only case it
    # was written for.

    def __post_init__(self) -> None:
        if not self.reasons:
            raise UnexplainedEscalationError("an escalation must say why")
        if self.intent is not None and any(
            isinstance(reason, RiskReason) for reason in self.reasons
        ):
            raise UnexplainedEscalationError(
                f"{self.intent} on an escalation raised by danger alone, which "
                f"leaves before anything reads what was wanted"
            )


@dataclass(frozen=True)
class Review:
    """Nothing is wrong with the request; something is wrong with us.

    Somebody here picks it up. The customer is not sent to a specialist for a
    fault on our side, and is not answered on less than the service normally
    knows about a message.
    """

    reason: ReviewReason

    # Held after the message was understood, in every case but one: a reading
    # that could not be taken at all is the fault being reported.
    intent: Intent | None = None


# The two questions asked because the message itself could not be placed.
# Everything else is asked after it was.
_UNPLACED = frozenset(
    {ClarificationReason.UNRESOLVED_INTENT, ClarificationReason.MULTIPLE_INTENTS}
)


@dataclass(frozen=True)
class Clarify:
    """Something is missing that the customer can supply."""

    reason: ClarificationReason

    # Set where the request was understood and stopped for want of a value.
    # Left empty where the message itself was what could not be placed: one
    # reading nobody found, and two readings nobody chose between, are both
    # states in which naming an intent would be naming a guess.
    intent: Intent | None = None

    def __post_init__(self) -> None:
        """Both ways of getting this wrong, since both are writeable.

        Naming a reading on a question asked because none was found says the
        message was understood. Omitting one on a question asked after it was
        understood throws away the thing a colleague opening the case wants.
        """
        if self.reason in _UNPLACED and self.intent is not None:
            raise UnexplainedEscalationError(
                f"{self.reason} names {self.intent}, having been asked because "
                f"no reading was arrived at"
            )
        if self.reason not in _UNPLACED and self.intent is None:
            raise UnexplainedEscalationError(
                f"{self.reason} is asked once a message is understood, and this "
                f"one names nothing"
            )


@dataclass(frozen=True)
class Proceed:
    """Enough is known to go and look, and this is where looking is allowed.

    The question travels with the reading of it, and so does everything known
    about the person who asked. Handing those along separately let a request
    be gathered for one question and labelled with another: every rating
    correct, and about different things. It is the same hazard as carrying a
    profile beside an intent, one step further out.
    """

    intent: Intent
    enquiry: Enquiry

    @property
    def profile(self) -> DecisionProfile:
        """Looked up, not supplied.

        Taking it as a field allowed a plan naming one intent while carrying
        the profile of another, which is a request permitted to read sources
        nothing about it justified.
        """
        return profile_for(self.intent)

    @property
    def sources(self) -> frozenset[Source]:
        return self.profile.required_sources | self.profile.contextual_sources


Triage = Escalate | Clarify | Proceed | Review


async def triage(
    enquiry: Enquiry, *, classifier: IntentClassifier | None = None
) -> Triage:
    """Work out what to do with a message, before doing any of it.

    The enquiry carries what the caller already has — an order number, a
    customer linked to a commerce account — as the values themselves. A
    profile asking for something absent stops the request here rather than at
    a source that would have been asked for nothing.

    A classifier, where one is configured, is asked only about messages the
    phrase rules made nothing of. It cannot report danger and it never sees a
    message the risk rules stopped, so nothing it returns revises a decision
    already taken.

    None is a supported configuration and the one that ships. A message
    nothing places is asked about instead of guessed at, which is what the
    design calls for below a confidence threshold — and, measured, is what
    the evaluated model would have been doing on four of every six messages
    it was the only thing left to read.
    """
    risks = risks_in(enquiry.message)
    if risks:
        return Escalate(reasons=risks)

    matched = intents_in(enquiry.message)
    intent = next(iter(matched)) if len(matched) == 1 else None

    if not matched and classifier is not None:
        try:
            intent = await classifier.classify(enquiry.message, enquiry.locale)
        except ClassifierUnavailableError:
            # Less is known about this message than the service answers on,
            # and nothing about that is the customer's doing.
            return Review(reason=ReviewReason.INTENT_CHECK_UNAVAILABLE)

    if len(matched) > 1:
        return Clarify(reason=ClarificationReason.MULTIPLE_INTENTS)
    if intent is None:
        return Clarify(reason=ClarificationReason.UNRESOLVED_INTENT)

    profile = profile_for(intent)
    missing = profile.required_inputs - enquiry.known
    if missing:
        # Escalations first. Where a customer is not linked to any commerce
        # record, asking them for an order number invites them to answer a
        # question that will not help.
        worst = min(
            missing,
            key=lambda need: (
                need.when_missing is not Route.HUMAN_ESCALATION,
                need.value,
            ),
        )
        reason = worst.reason
        if isinstance(reason, EscalationReason):
            return Escalate(reasons=frozenset({reason}), intent=intent)
        return Clarify(reason=reason, intent=intent)

    return Proceed(intent=intent, enquiry=enquiry)
