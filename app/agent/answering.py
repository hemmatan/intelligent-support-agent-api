"""Turning a placed request into a plan: what may be said, and how far it goes.

Triage decides which sources a request is allowed to touch. This is the step
that touches them, rates what came back, and stops. Nothing here writes to a
customer; the rating decides whether anything may be written at all.

The order is the one the design document sets out, and it is not an
implementation detail. Conditions with a known outcome are gates and are never
scored, because scoring them invites a strong rating elsewhere to average them
away. Only once the gates pass does anything become a number on a scale.
"""

from dataclasses import dataclass

from app.agent.facts import Fact, coverage_of, facts_in
from app.agent.intent import Intent
from app.agent.knowledge import Locale
from app.agent.profiles import DecisionProfile, Source
from app.agent.reasons import (
    BlockedReason,
    EscalationReason,
    EvidenceReason,
    ReviewReason,
)
from app.agent.reliability import Assessment, Factor, ReliabilityLevel, Route
from app.agent.responses import (
    ApprovedReply,
    NothingApprovedToSayError,
    TemplateLibrary,
)
from app.agent.retrieval import Hit, PolicyIndex, relevance_of
from app.agent.triage import Proceed, Review, UnexplainedEscalationError


@dataclass(frozen=True)
class Citation:
    """One piece of evidence, identified so the answer can be re-checked.

    The hash is of the entry's content, not its name. A policy edited without
    a version bump keeps its reference and changes its hash, which is the only
    way an audit six months later can tell that the text quoted then is not
    the text sitting there now.
    """

    source: Source
    reference: str
    content_hash: str


# Which rating, having decided an outcome, gets recorded as the cause of it.
_WHY: dict[Factor, EvidenceReason] = {
    Factor.AUTHORITY: EvidenceReason.UNAUTHORITATIVE,
    Factor.COVERAGE: EvidenceReason.NOT_COVERED,
    Factor.RELEVANCE: EvidenceReason.POORLY_MATCHED,
    Factor.FRESHNESS: EvidenceReason.STALE,
}


@dataclass(frozen=True)
class Handover:
    """A gate with a known outcome, decided without scoring anything."""

    reasons: frozenset[EscalationReason]

    def __post_init__(self) -> None:
        if not self.reasons:
            raise UnexplainedEscalationError("an escalation must say why")


class MisdirectedReplyError(ValueError):
    """Customer wording where none is going to a customer, or missing where it is.

    Both directions are faults. Text attached to a queue entry is text a
    reviewer may read as already sent; a delivery with nothing to deliver is a
    silence the customer has to interpret.
    """


@dataclass(frozen=True)
class Plan:
    """Evidence that survived the gates, what it is rated at, and what it says."""

    intent: Intent
    requested: frozenset[Fact]
    assessment: Assessment
    citations: tuple[Citation, ...]
    reply: ApprovedReply | None = None

    def __post_init__(self) -> None:
        going_out = self.route is Route.DIRECT_RESPONSE
        if going_out and self.reply is None:
            raise MisdirectedReplyError("a direct answer has to say something")
        if not going_out and self.reply is not None:
            raise MisdirectedReplyError(
                f"a {self.route} carries wording written for a customer"
            )
        if going_out and not self.citations:
            raise MisdirectedReplyError("an answer sent citing nothing")

    @property
    def said(self) -> tuple[str, ...]:
        """The wording this was built from, and nothing when none was."""
        return self.reply.said if self.reply else ()

    @property
    def route(self) -> Route:
        """Where the weakest required factor sends this."""
        return self.assessment.route

    @property
    def reasons(self) -> frozenset[EvidenceReason]:
        """Which ratings held this back, and nothing when none did.

        Every factor tied at the weakest level, not one picked from among
        them. A record naming coverage where relevance was equally at fault
        sends whoever reads it to fix one of two things.

        Without this a scored outcome had a destination and no cause, while a
        gated one had both — so the queue entries that needed explaining most
        were the ones arriving bare.
        """
        if self.route is Route.DIRECT_RESPONSE:
            return frozenset()
        weakest = self.assessment.level
        return frozenset(
            _WHY[factor]
            for factor, level in self.assessment.required.items()
            if level is weakest
        )


Outcome = Plan | Review | Handover


class Sources:
    """The adapters that are wired up, as against the ones profiles name.

    Commerce and conversation history have none yet. A profile requiring one
    stops at the availability gate rather than proceeding without it, which is
    the truthful outcome: an order cannot be located from a policy document,
    and the alternative is answering from whatever else was reachable.
    """

    def __init__(self, *, knowledge_base: PolicyIndex | None = None) -> None:
        self._knowledge_base = knowledge_base

    @property
    def available(self) -> frozenset[Source]:
        """Which sources can actually be asked right now."""
        if self._knowledge_base is None:
            return frozenset()
        return frozenset({Source.KNOWLEDGE_BASE})

    async def search(self, query: str, locale: Locale) -> list[Hit]:
        if self._knowledge_base is None:
            return []
        return await self._knowledge_base.search(query, locale)


def _rate(
    profile: DecisionProfile, measured: dict[Factor, ReliabilityLevel]
) -> Assessment:
    """Every factor the profile requires, whether or not one was produced.

    A required factor nothing measured is UNUSABLE rather than absent.
    Dropping it would shrink the set the minimum is taken over, so a factor
    failing to arrive would raise the result instead of sinking it — silently,
    and worst exactly when a source is broken.
    """
    return Assessment(
        required={
            factor: measured.get(factor, ReliabilityLevel.UNUSABLE)
            for factor in profile.required_factors
        },
        contextual={
            factor: measured[factor]
            for factor in profile.contextual_factors
            if factor in measured
        },
    )


def _authority_over(
    profile: DecisionProfile, citations: tuple[Citation, ...]
) -> ReliabilityLevel:
    """How far the sources actually cited may carry a claim of this kind.

    Read off what was cited rather than off everything the profile permits.
    A minimum taken across the whole plan would drag every answer down to the
    level of its weakest permitted source: history is contextual on all five
    profiles, so a reply the knowledge base supported in full would be held
    back by a source it never leaned on.

    Citing nothing is UNUSABLE rather than vacuously ready. An empty minimum
    has no answer to give, and the generous one would let a reply resting on
    no evidence rate as fully authoritative.

    Kept where every planner reaches it, because this is an argument about
    what authority means rather than a step in reading one kind of source.
    Worked out a second time somewhere else, it is free to come out different.
    """
    return min(
        (profile.authority_of(cited.source) for cited in citations),
        default=ReliabilityLevel.UNUSABLE,
    )


async def _from_knowledge_base(
    proceed: Proceed, *, sources: Sources, templates: TemplateLibrary
) -> Outcome:
    """Answering out of written policy: rank, cite the winner, rate, say it.

    Ranking is what makes this shape its own. There is a leading entry and
    there are the ones it beat, so how well it matched is a quantity somebody
    can go and measure. Against a record that either exists or does not there
    are no runners-up and nothing to measure, which is why a source is planned
    for here instead of inside one function holding every shape at once.

    Finding nothing is a gate and not a poor score. Scored, the strong ratings
    beside it would carry an empty answer out to somebody.
    """
    profile = proceed.profile

    hits = await sources.search(proceed.enquiry.message, proceed.enquiry.locale)
    if not hits:
        return Handover(reasons=frozenset({BlockedReason.NO_SUPPORTING_EVIDENCE}))

    best = hits[0]
    citations = (
        Citation(
            source=Source.KNOWLEDGE_BASE,
            reference=best.entry.reference,
            content_hash=best.entry.content_hash,
        ),
    )
    requested = facts_in(proceed.enquiry.message)
    measured = {
        Factor.RELEVANCE: relevance_of(hits),
        Factor.COVERAGE: coverage_of(requested, best.entry.facts),
        Factor.AUTHORITY: _authority_over(profile, citations),
    }
    assessment = _rate(profile, measured)
    if assessment.route is not Route.DIRECT_RESPONSE:
        return Plan(
            intent=proceed.intent,
            requested=requested,
            assessment=assessment,
            citations=citations,
        )

    try:
        reply = templates.say(requested, best.entry, proceed.enquiry.locale)
    except NothingApprovedToSayError:
        # The evidence was good enough. Nobody has written the sentence.
        return Review(reason=ReviewReason.NOTHING_APPROVED_TO_SAY)

    return Plan(
        intent=proceed.intent,
        requested=requested,
        assessment=assessment,
        citations=citations,
        reply=reply,
    )


async def plan_for(
    proceed: Proceed, *, sources: Sources, templates: TemplateLibrary
) -> Outcome:
    """See that a request can be served at all, then go and serve it.

    The request arrives whole. What was asked and what it was taken to mean
    are one value, so there is no call in which they describe different
    things.

    Nothing is read here. Whether the sources a profile calls for are
    connected is decided the same way whichever ones they are, and deciding it
    before any of them is approached is what keeps a request away from the
    ones that happen to be running: an order is a matter for commerce, and the
    knowledge base is full of documents that mention orders.
    """
    profile = proceed.profile

    missing = profile.required_sources - sources.available
    if missing:
        return Review(reason=ReviewReason.SOURCE_UNAVAILABLE)

    return await _from_knowledge_base(proceed, sources=sources, templates=templates)
