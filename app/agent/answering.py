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
from app.agent.reasons import EscalationReason, EvidenceReason, ReviewReason
from app.agent.reliability import Assessment, Factor, ReliabilityLevel, Route
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


@dataclass(frozen=True)
class Plan:
    """Evidence that survived the gates, and what it is rated at."""

    intent: Intent
    requested: frozenset[Fact]
    assessment: Assessment
    citations: tuple[Citation, ...]

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


async def plan_for(proceed: Proceed, *, sources: Sources) -> Outcome:
    """Gather what this request is allowed to gather, and rate it.

    The request arrives whole. What was asked and what it was taken to mean
    are one value, so there is no call in which they describe different
    things.

    Authority is taken over the sources actually cited, not over everything
    reachable. Rating the whole plan would drag every answer down to the level
    of its weakest permitted source: history is contextual on all five
    profiles, so an answer the knowledge base supported in full would be held
    back because a source it never leaned on is not authoritative.
    """
    profile = proceed.profile

    missing = profile.required_sources - sources.available
    if missing:
        return Review(reason=ReviewReason.SOURCE_UNAVAILABLE)

    hits = await sources.search(proceed.message, proceed.locale)
    if not hits:
        return Handover(reasons=frozenset({EscalationReason.NO_SUPPORTING_EVIDENCE}))

    best = hits[0]
    citations = (
        Citation(
            source=Source.KNOWLEDGE_BASE,
            reference=best.entry.reference,
            content_hash=best.entry.content_hash,
        ),
    )
    requested = facts_in(proceed.message)
    measured = {
        Factor.RELEVANCE: relevance_of(hits),
        Factor.COVERAGE: coverage_of(requested, best.entry.facts),
        Factor.AUTHORITY: min(
            (profile.authority_of(cited.source) for cited in citations),
            default=ReliabilityLevel.UNUSABLE,
        ),
    }
    return Plan(
        intent=proceed.intent,
        requested=requested,
        assessment=_rate(profile, measured),
        citations=citations,
    )
