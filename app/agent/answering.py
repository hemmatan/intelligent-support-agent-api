"""Turning a placed request into a plan: what may be said, and how far it goes.

Triage decides which sources a request is allowed to touch. This is the step
that touches them, rates what came back, and stops. Nothing here writes to a
customer; the rating decides whether anything may be written at all.

The order is the one the design document sets out, and it is not an
implementation detail. Conditions with a known outcome are gates and are never
scored, because scoring them invites a strong rating elsewhere to average them
away. Only once the gates pass does anything become a number on a scale.
"""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from app.agent.commerce import (
    CommerceGateway,
    CommerceUnavailableError,
    Found,
    Record,
    freshness_of,
)
from app.agent.enquiry import Enquiry
from app.agent.facts import Fact, coverage_of, facts_in
from app.agent.intent import Intent
from app.agent.knowledge import Locale
from app.agent.profiles import DecisionProfile, Source
from app.agent.reasons import (
    BlockedReason,
    ClarificationReason,
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
from app.agent.triage import Clarify, Proceed, Review, UnexplainedEscalationError


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

    # Empty for written policy, which is approved rather than observed: it has
    # no supplier, no moment of reading, and asking whether it is invented is
    # asking the wrong question of it. A row from the shop answers all three,
    # and the answers have to survive as far as the case record, or the note
    # saying none of this describes a real purchase stops here.
    provider: str | None = None
    observed_at: datetime | None = None
    synthetic: bool | None = None


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

    # Evidence that cleared every bar and still cannot be sent, because the
    # thing standing in the way is ours rather than the evidence's. Kept as a
    # plan instead of thrown away for a bare review: what the request was
    # taken to be, what it rested on and how each dimension rated are exactly
    # what the colleague picking it up needs, and they were being dropped on
    # the floor at the last step.
    held_back: ReviewReason | None = None

    def __post_init__(self) -> None:
        held = self.held_back
        if held is not None and self.assessment.route is not Route.DIRECT_RESPONSE:
            raise MisdirectedReplyError(
                f"{held} would file a {self.assessment.route} as a review"
            )
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
        """Where the weakest required factor sends this.

        Unless something here stopped it that the ratings know nothing about.
        A reply nobody has written is not weak evidence, and it may only make
        a request wait, never send one further: an assessment already bound
        for a specialist cannot be relabelled on the way past.
        """
        if self.held_back is not None:
            return Route.INTERNAL_REVIEW
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


Outcome = Plan | Review | Handover | Clarify


class RequestNotPlacedError(RuntimeError):
    """A lookup was reached without what triage promises it will have.

    Triage refuses to place a request whose profile names an input the enquiry
    does not carry, so arriving here without one means the two have come
    apart. Raised rather than worked around, because the alternative is
    querying the shop's records for an order belonging to nobody.
    """


def _utc_now() -> datetime:
    return datetime.now(UTC)


class Sources:
    """The adapters that are wired up, as against the ones profiles name.

    Conversation history has none. A profile requiring a source nobody
    connected stops at the availability gate rather than proceeding without
    it, which is the truthful outcome: an order cannot be located from a
    policy document, and the alternative is answering out of whatever else
    happened to be reachable.

    The clock and the two windows live here because they are operational
    settings rather than facts about a request, and because a test that wants
    to know what a customer is told about week-old evidence should not have to
    wait a week.
    """

    def __init__(
        self,
        *,
        knowledge_base: PolicyIndex | None = None,
        commerce: CommerceGateway | None = None,
        now: Callable[[], datetime] = _utc_now,
        ttl: timedelta = timedelta(minutes=15),
        readable_for: timedelta = timedelta(hours=6),
    ) -> None:
        self._knowledge_base = knowledge_base
        self._commerce = commerce
        self._now = now
        self._ttl = ttl
        self._readable_for = readable_for

    @property
    def available(self) -> frozenset[Source]:
        """Which sources can actually be asked right now."""
        connected = set()
        if self._knowledge_base is not None:
            connected.add(Source.KNOWLEDGE_BASE)
        if self._commerce is not None:
            connected.add(Source.COMMERCE)
        return frozenset(connected)

    async def search(self, query: str, locale: Locale) -> list[Hit]:
        if self._knowledge_base is None:
            return []
        return await self._knowledge_base.search(query, locale)

    async def look_up(self, intent: Intent, enquiry: Enquiry) -> Record | None:
        """The one row this kind of request is answered from, or nothing.

        Nothing means no such row, or none belonging to whoever asked. The
        gateway is not in a position to say which of those it was, and
        neither, therefore, is anything above it.
        """
        if self._commerce is None:
            raise CommerceUnavailableError("no commerce gateway is connected")

        if intent is Intent.PRODUCT_AVAILABILITY:
            if enquiry.product is None:
                raise RequestNotPlacedError(f"{intent} without a product reference")
            listed = await self._commerce.product(enquiry.product)
            return listed.record if isinstance(listed, Found) else None

        if enquiry.order is None or enquiry.customer is None:
            raise RequestNotPlacedError(f"{intent} without an order and an account")

        if intent is Intent.REFUND_STATUS:
            owed = await self._commerce.refund(enquiry.order, customer=enquiry.customer)
            return owed.record if isinstance(owed, Found) else None

        placed = await self._commerce.order(enquiry.order, customer=enquiry.customer)
        return placed.record if isinstance(placed, Found) else None

    def freshness(self, record: Record) -> ReliabilityLevel:
        """How far this reading's age lets it carry an answer."""
        return freshness_of(
            record, now=self._now(), ttl=self._ttl, readable_for=self._readable_for
        )


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
        return Plan(
            intent=proceed.intent,
            requested=requested,
            assessment=assessment,
            citations=citations,
            held_back=ReviewReason.NOTHING_APPROVED_TO_SAY,
        )

    return Plan(
        intent=proceed.intent,
        requested=requested,
        assessment=assessment,
        citations=citations,
        reply=reply,
    )


_NOT_FOUND: dict[Intent, ClarificationReason] = {
    Intent.ORDER_STATUS: ClarificationReason.ORDER_NOT_FOUND,
    Intent.REFUND_STATUS: ClarificationReason.ORDER_NOT_FOUND,
    Intent.PRODUCT_AVAILABILITY: ClarificationReason.PRODUCT_NOT_FOUND,
}


async def _from_commerce(proceed: Proceed, *, sources: Sources) -> Outcome:
    """Answering out of the shop's own records: look one up, cite it, rate it.

    Nothing is ranked, so there is no relevance to measure and the profiles
    do not ask for one. What takes its place is age: a policy stands until it
    is replaced, while a delivery state can be wrong by the afternoon.

    A row nobody can show this customer sends them back to check what they
    typed. It is not an escalation — nothing has gone wrong here, and there is
    nothing for a colleague to do that the customer cannot do faster — and it
    is not a report that the reference belongs to somebody else, because that
    was never established.

    A provider having a bad day is ours to answer for. The customer asked a
    fair question and our own records were the thing that did not reply, so it
    waits for one of us rather than turning into an error page.
    """
    profile = proceed.profile

    try:
        record = await sources.look_up(proceed.intent, proceed.enquiry)
    except CommerceUnavailableError:
        return Review(reason=ReviewReason.SOURCE_UNAVAILABLE)
    if record is None:
        return Clarify(reason=_NOT_FOUND[proceed.intent])

    citations = (
        Citation(
            source=Source.COMMERCE,
            reference=record.cited_as,
            content_hash=record.content_hash,
            provider=record.provider,
            observed_at=record.observed_at,
            synthetic=record.synthetic,
        ),
    )
    requested = facts_in(proceed.enquiry.message)
    measured = {
        Factor.COVERAGE: coverage_of(requested, record.facts),
        Factor.AUTHORITY: _authority_over(profile, citations),
        Factor.FRESHNESS: sources.freshness(record),
    }
    assessment = _rate(profile, measured)
    return Plan(
        intent=proceed.intent,
        requested=requested,
        assessment=assessment,
        citations=citations,
        # Evidence good enough to send, and no approved sentence to send it
        # in: the phrase book covers written policy and stops there. The row
        # travels with the request either way, so whoever picks it up has
        # what the decision was taken on.
        held_back=(
            ReviewReason.NOTHING_APPROVED_TO_SAY
            if assessment.route is Route.DIRECT_RESPONSE
            else None
        ),
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

    if Source.COMMERCE in profile.required_sources:
        return await _from_commerce(proceed, sources=sources)
    return await _from_knowledge_base(proceed, sources=sources, templates=templates)
