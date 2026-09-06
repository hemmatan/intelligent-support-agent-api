"""Gathering what a placed request may gather, and rating what came back."""

import inspect

import pytest

from app.agent.answering import (
    Citation,
    Handover,
    Plan,
    Review,
    Sources,
    _rate,
    plan_for,
)
from app.agent.facts import Fact
from app.agent.intent import Intent
from app.agent.knowledge import load_corpus
from app.agent.profiles import Source, profile_for
from app.agent.reasons import EscalationReason, ReviewReason
from app.agent.reliability import Assessment, Factor, ReliabilityLevel, Route
from app.agent.retrieval import PolicyIndex
from app.agent.triage import Proceed


@pytest.fixture
def sources() -> Sources:
    """The real corpus, lexical ranking only. No embedder, no network."""
    return Sources(knowledge_base=PolicyIndex(load_corpus()))


@pytest.mark.asyncio
async def test_coverage_is_the_only_thing_standing_between_this_and_a_customer(
    sources: Sources,
) -> None:
    """A French delivery question, answered from the French returns policy.

    There is no French shipping entry. The returns one says refunds happen
    within thirty days of delivery, so it shares the asked-about word and BM25
    returns it. Retrieval is working. The source is the one the profile named.
    Every rating except coverage is satisfied, and the reply would have been a
    return window offered to somebody asking when their parcel arrives.
    """
    outcome = await plan_for(
        Proceed(
            intent=Intent.SHIPPING_POLICY,
            message="Quel est le delai de livraison ?",
            locale="fr",
        ),
        sources=sources,
    )
    assert isinstance(outcome, Plan)
    assert [cited.reference for cited in outcome.citations] == [
        "kb:returns.standard.fr.v1"
    ]
    assert outcome.assessment.required[Factor.RELEVANCE] > ReliabilityLevel.REVIEW_ONLY
    assert outcome.assessment.required[Factor.AUTHORITY] is ReliabilityLevel.READY
    assert outcome.assessment.required[Factor.COVERAGE] is ReliabilityLevel.UNUSABLE
    assert outcome.route is Route.HUMAN_ESCALATION

    without_coverage = Assessment(
        required={
            factor: level
            for factor, level in outcome.assessment.required.items()
            if factor is not Factor.COVERAGE
        }
    )
    assert without_coverage.route is Route.DIRECT_RESPONSE


@pytest.mark.asyncio
async def test_a_question_the_corpus_answers_is_answered(sources: Sources) -> None:
    outcome = await plan_for(
        Proceed(intent=Intent.SHIPPING_POLICY, message="How long does delivery take?"),
        sources=sources,
    )
    assert isinstance(outcome, Plan)
    assert outcome.route is Route.DIRECT_RESPONSE
    assert outcome.requested == {Fact.STANDARD_DELIVERY_TIME}
    assert [cited.reference for cited in outcome.citations] == [
        "kb:shipping.times.en.v1"
    ]


@pytest.mark.asyncio
async def test_a_source_nobody_wired_up_waits_for_somebody_here(
    sources: Sources,
) -> None:
    """Not answered from the sources that happen to be running.

    Order status requires commerce. Only the knowledge base is connected, and
    the knowledge base is full of documents that mention orders.
    """
    outcome = await plan_for(
        Proceed(intent=Intent.ORDER_STATUS, message="Where is my order?"),
        sources=sources,
    )
    assert outcome == Review(reason=ReviewReason.SOURCE_UNAVAILABLE)


@pytest.mark.asyncio
async def test_nothing_found_is_a_gate_and_not_a_low_score(sources: Sources) -> None:
    """Scoring it would let strong ratings elsewhere carry an empty answer."""
    outcome = await plan_for(
        Proceed(intent=Intent.RETURN_POLICY, message="Do you ship to Belgium?"),
        sources=sources,
    )
    assert outcome == Handover(
        reasons=frozenset({EscalationReason.NO_SUPPORTING_EVIDENCE})
    )


@pytest.mark.asyncio
async def test_a_source_the_answer_never_leaned_on_does_not_hold_it_back(
    sources: Sources,
) -> None:
    """Authority is rated over what was cited, not over what was permitted.

    History is contextual on every profile, so taking the minimum across the
    whole plan would rate every answer REVIEW_ONLY and send the lot to a
    queue — including answers the knowledge base supported by itself.
    """
    outcome = await plan_for(
        Proceed(
            intent=Intent.RETURN_POLICY,
            message="How long do I have to return a jacket?",
        ),
        sources=sources,
    )
    assert isinstance(outcome, Plan)
    assert Source.HISTORY in profile_for(Intent.RETURN_POLICY).contextual_sources
    assert outcome.assessment.required[Factor.AUTHORITY] is ReliabilityLevel.READY
    assert outcome.route is Route.DIRECT_RESPONSE


@pytest.mark.asyncio
async def test_a_citation_can_outlive_the_text_it_points_at(
    sources: Sources,
) -> None:
    """An edit without a version bump keeps the reference and moves the hash."""
    outcome = await plan_for(
        Proceed(
            intent=Intent.RETURN_POLICY,
            message="How long do I have to return a jacket?",
        ),
        sources=sources,
    )
    assert isinstance(outcome, Plan)
    entry = next(e for e in load_corpus() if e.reference == "kb:returns.standard.en.v1")
    assert outcome.citations == (
        Citation(
            source=Source.KNOWLEDGE_BASE,
            reference=entry.reference,
            content_hash=entry.content_hash,
        ),
    )


def test_a_required_factor_nobody_measured_sinks_the_answer() -> None:
    """Leaving it out would shrink the set the minimum runs over.

    A factor that failed to arrive would then raise the result rather than
    lower it, which is silent and worst precisely when something is broken.
    """
    orders = profile_for(Intent.ORDER_STATUS)
    assert Factor.FRESHNESS in orders.required_factors
    assessed = _rate(orders, {Factor.AUTHORITY: ReliabilityLevel.READY})
    assert assessed.required[Factor.FRESHNESS] is ReliabilityLevel.UNUSABLE
    assert assessed.route is Route.HUMAN_ESCALATION


def test_a_reading_cannot_exist_apart_from_what_was_read() -> None:
    """Passed separately, they could describe different requests.

    A return-policy reading of a delivery question gathered the shipping
    entry, rated every factor correctly, and came out a direct response
    labelled as being about returns. A template chosen from that label would
    have quoted a return window off shipping evidence. There is now no reading
    without its message, and nowhere to hand a second one in.
    """
    with pytest.raises(TypeError):
        Proceed(intent=Intent.RETURN_POLICY)  # type: ignore[call-arg]

    positional = [
        name
        for name, parameter in inspect.signature(plan_for).parameters.items()
        if parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    ]
    assert positional == ["proceed"]
