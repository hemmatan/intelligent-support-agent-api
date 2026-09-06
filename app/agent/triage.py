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
    name an intent they missed and report trouble they did not describe, and
    it is never consulted about a message they placed or escalated. It cannot
    disagree with them about risk in the one direction that would matter: a
    message they flagged has already left by the time it would be asked, and
    what it returns is only ever added.
    """
    risks = risks_in(message)
    if risks:
        return Escalate(reasons=risks)

    matched = intents_in(message)
    if len(matched) > 1:
        return Clarify(reason=ClarificationReason.MULTIPLE_INTENTS)

    intent = next(iter(matched), None)
    if intent is None and classifier is not None:
        reading = await classifier.classify(message, locale)
        # Trouble it saw and the rules did not outranks anything it named.
        if reading.risks:
            return Escalate(reasons=reading.risks)
        intent = reading.intent
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
