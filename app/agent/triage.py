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

from app.agent.intent import (
    ClassifierUnavailableError,
    Intent,
    IntentClassifier,
    intents_in,
)
from app.agent.knowledge import Locale
from app.agent.profiles import DecisionProfile, Input, Source, profile_for
from app.agent.reasons import ClarificationReason, EscalationReason, ReviewReason
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

    def __post_init__(self) -> None:
        if not self.reasons:
            raise UnexplainedEscalationError("an escalation must say why")


@dataclass(frozen=True)
class Review:
    """Nothing is wrong with the request; something is wrong with us.

    Somebody here picks it up. The customer is not sent to a specialist for a
    fault on our side, and is not answered on less than the service normally
    knows about a message.
    """

    reason: ReviewReason


@dataclass(frozen=True)
class Clarify:
    """Something is missing that the customer can supply."""

    reason: ClarificationReason


@dataclass(frozen=True)
class Proceed:
    """Enough is known to go and look, and this is where looking is allowed.

    The message travels with the reading of it. Handing the two along
    separately let a request be gathered for one question and labelled with
    another: every rating correct, and about different things. It is the same
    hazard as carrying a profile beside an intent, one step further out.
    """

    intent: Intent
    message: str
    locale: Locale = "en"

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
    message: str,
    *,
    known: frozenset[Input] = frozenset(),
    locale: Locale = "en",
    classifier: IntentClassifier,
) -> Triage:
    """Work out what to do with a message, before doing any of it.

    `known` is what the caller already has — an order number in the message,
    a customer linked to a commerce account. A profile asking for something
    absent from it stops the request here rather than at a source that would
    have been asked for nothing.

    A classifier sees every message the risk rules let through, including the
    ones the phrase rules understood. Understanding what somebody wants is not
    the same as noticing they are in trouble: "I need to return this because a
    stranger used my account" reads as a return, and the rest of the sentence
    is the part that matters. Recognising the first half was being taken as
    reason enough not to look at the second.

    What it is trusted with still depends. Risk it reports is acted on
    whatever the rules found, because it can only ever add. The intent it
    offers is taken only where the rules recognised nothing, so a model cannot
    reinterpret a request they already placed.

    It is never consulted about a message the rules escalated: that one has
    left before it would be asked, which is why nothing it returns can lower a
    risk. The cost is a call on every message that is not obviously trouble,
    which is the price of the rules being a fixed vocabulary and people not.
    """
    risks = risks_in(message)
    if risks:
        return Escalate(reasons=risks)

    matched = intents_in(message)
    intent = next(iter(matched)) if len(matched) == 1 else None

    try:
        reading = await classifier.classify(message, locale)
    except ClassifierUnavailableError:
        # Less is known about this message than the service answers on, and
        # nothing about that is the customer's doing.
        return Review(reason=ReviewReason.SAFETY_CHECK_UNAVAILABLE)

    # Trouble it saw and the rules did not outranks anything it named, and
    # outranks anything they named too.
    if reading.risks:
        return Escalate(reasons=reading.risks)
    if not matched:
        intent = reading.intent

    if len(matched) > 1:
        return Clarify(reason=ClarificationReason.MULTIPLE_INTENTS)
    if intent is None:
        return Clarify(reason=ClarificationReason.UNRESOLVED_INTENT)

    profile = profile_for(intent)
    missing = profile.required_inputs - known
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
            return Escalate(reasons=frozenset({reason}))
        return Clarify(reason=reason)

    return Proceed(intent=intent, message=message, locale=locale)
