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

from app.agent.intent import Intent, IntentClassifier, intents_in
from app.agent.knowledge import Locale
from app.agent.profiles import DecisionProfile, Input, Source, profile_for
from app.agent.reasons import ClarificationReason, EscalationReason
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
class Clarify:
    """Something is missing that the customer can supply."""

    reason: ClarificationReason


@dataclass(frozen=True)
class Proceed:
    """Enough is known to go and look, and this is where looking is allowed."""

    intent: Intent

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


Triage = Escalate | Clarify | Proceed


async def triage(
    message: str,
    *,
    known: frozenset[Input] = frozenset(),
    locale: Locale = "en",
    classifier: IntentClassifier | None = None,
) -> Triage:
    """Work out what to do with a message, before doing any of it.

    `known` is what the caller already has — an order number in the message,
    a customer linked to a commerce account. A profile asking for something
    absent from it stops the request here rather than at a source that would
    have been asked for nothing.

    A classifier is offered only messages the rules made no sense of. It can
    name an intent they missed; it is never consulted about one they placed,
    and never about a message they escalated. It cannot report risk at all, so
    there is no way for it to disagree about that either.
    """
    risks = risks_in(message)
    if risks:
        return Escalate(reasons=risks)

    matched = intents_in(message)
    if len(matched) > 1:
        return Clarify(reason=ClarificationReason.MULTIPLE_INTENTS)

    intent = next(iter(matched), None)
    if intent is None and classifier is not None:
        intent = await classifier.classify(message, locale)
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

    return Proceed(intent=intent)
