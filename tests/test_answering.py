"""Gathering what a placed request may gather, and rating what came back."""

import inspect
import logging
from datetime import UTC, datetime, timedelta

import pytest

from app.agent.answering import (
    Citation,
    Handover,
    MisdirectedReplyError,
    Plan,
    Review,
    Sources,
    _rate,
    plan_for,
)
from app.agent.commerce import (
    CommerceUnavailableError,
    Found,
    Observation,
    OrderRecord,
    OrderState,
    ProductRecord,
)
from app.agent.demo import DemoStorefront
from app.agent.enquiry import Enquiry
from app.agent.facts import Fact
from app.agent.intent import Intent
from app.agent.knowledge import Locale, load_corpus
from app.agent.profiles import Source, profile_for
from app.agent.reasons import (
    BlockedReason,
    ClarificationReason,
    EvidenceReason,
    ReviewReason,
)
from app.agent.reliability import Assessment, Factor, ReliabilityLevel, Route
from app.agent.responses import ApprovedReply, TemplateLibrary, load_templates
from app.agent.retrieval import Hit, PolicyIndex
from app.agent.triage import Clarify, Proceed, UnexplainedEscalationError


@pytest.fixture
def templates() -> TemplateLibrary:
    """The approved wording as it ships."""
    return load_templates()


@pytest.fixture
def sources() -> Sources:
    """The real corpus, lexical ranking only. No embedder, no network."""
    return Sources(knowledge_base=PolicyIndex(load_corpus()))


@pytest.mark.asyncio
async def test_coverage_is_the_only_thing_standing_between_this_and_a_customer(
    sources: Sources, templates: TemplateLibrary
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
            enquiry=Enquiry(message="Quel est le delai de livraison ?", locale="fr"),
        ),
        sources=sources,
        templates=templates,
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
async def test_a_question_the_corpus_answers_is_answered(
    sources: Sources, templates: TemplateLibrary
) -> None:
    outcome = await plan_for(
        Proceed(
            intent=Intent.SHIPPING_POLICY,
            enquiry=Enquiry(message="How long does delivery take?"),
        ),
        sources=sources,
        templates=templates,
    )
    assert isinstance(outcome, Plan)
    assert outcome.route is Route.DIRECT_RESPONSE
    assert outcome.requested == {Fact.STANDARD_DELIVERY_TIME}
    assert [cited.reference for cited in outcome.citations] == [
        "kb:shipping.times.en.v1"
    ]


@pytest.mark.asyncio
async def test_a_source_nobody_wired_up_waits_for_somebody_here(
    sources: Sources, templates: TemplateLibrary
) -> None:
    """Not answered from the sources that happen to be running.

    Order status requires commerce. Only the knowledge base is connected, and
    the knowledge base is full of documents that mention orders.
    """
    outcome = await plan_for(
        Proceed(
            intent=Intent.ORDER_STATUS,
            enquiry=Enquiry(message="Where is my order?"),
        ),
        sources=sources,
        templates=templates,
    )
    assert outcome == Review(
        reason=ReviewReason.SOURCE_UNAVAILABLE, intent=Intent.ORDER_STATUS
    )


class _RefusesToBeAsked:
    """An index that fails the run if anything consults it.

    A spy counting calls and asserting nought passes equally well having never
    been connected to anything. Reaching this one at all is the failure, so it
    cannot report a guarantee it was not in a position to observe.
    """

    async def search(self, query: str, locale: Locale) -> list[Hit]:
        raise AssertionError("a source was read before the gate had run")


@pytest.mark.asyncio
async def test_the_gate_decides_before_a_source_is_read(
    templates: TemplateLibrary,
) -> None:
    """Ordering is the guarantee, and nothing here was checking it.

    The test above arrives at its verdict whether or not anything was read.
    The knowledge base holds documents that mention orders, so a search
    running ahead of the gate would come back with hits, the gate would refuse
    the request regardless, and the assertion would not move. Nothing is
    fetched on a message's behalf until the request has been placed, and until
    now that held at this layer by arrangement rather than by test.
    """
    outcome = await plan_for(
        Proceed(
            intent=Intent.ORDER_STATUS,
            enquiry=Enquiry(message="Where is my order?"),
        ),
        # Not a PolicyIndex, and it does not need to be: the only thing being
        # observed is whether anything reaches it.
        sources=Sources(knowledge_base=_RefusesToBeAsked()),  # type: ignore[arg-type]
        templates=templates,
    )
    assert outcome == Review(
        reason=ReviewReason.SOURCE_UNAVAILABLE, intent=Intent.ORDER_STATUS
    )


@pytest.mark.asyncio
async def test_nothing_found_is_a_gate_and_not_a_low_score(
    sources: Sources, templates: TemplateLibrary
) -> None:
    """Scoring it would let strong ratings elsewhere carry an empty answer."""
    outcome = await plan_for(
        Proceed(
            intent=Intent.RETURN_POLICY,
            enquiry=Enquiry(message="Do you ship to Belgium?"),
        ),
        sources=sources,
        templates=templates,
    )
    assert outcome == Handover(
        reasons=frozenset({BlockedReason.NO_SUPPORTING_EVIDENCE}),
        intent=Intent.RETURN_POLICY,
    )


@pytest.mark.asyncio
async def test_a_source_the_answer_never_leaned_on_does_not_hold_it_back(
    sources: Sources, templates: TemplateLibrary
) -> None:
    """Authority is rated over what was cited, not over what was permitted.

    History is contextual on every profile, so taking the minimum across the
    whole plan would rate every answer REVIEW_ONLY and send the lot to a
    queue — including answers the knowledge base supported by itself.
    """
    outcome = await plan_for(
        Proceed(
            intent=Intent.RETURN_POLICY,
            enquiry=Enquiry(message="How long do I have to return a jacket?"),
        ),
        sources=sources,
        templates=templates,
    )
    assert isinstance(outcome, Plan)
    assert Source.HISTORY in profile_for(Intent.RETURN_POLICY).contextual_sources
    assert outcome.assessment.required[Factor.AUTHORITY] is ReliabilityLevel.READY
    assert outcome.route is Route.DIRECT_RESPONSE


@pytest.mark.asyncio
async def test_a_citation_can_outlive_the_text_it_points_at(
    sources: Sources, templates: TemplateLibrary
) -> None:
    """An edit without a version bump keeps the reference and moves the hash."""
    outcome = await plan_for(
        Proceed(
            intent=Intent.RETURN_POLICY,
            enquiry=Enquiry(message="How long do I have to return a jacket?"),
        ),
        sources=sources,
        templates=templates,
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
    with pytest.raises(TypeError):
        Enquiry()  # type: ignore[call-arg]

    positional = [
        name
        for name, parameter in inspect.signature(plan_for).parameters.items()
        if parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    ]
    assert positional == ["proceed"]


@pytest.mark.asyncio
async def test_a_scored_outcome_says_which_rating_decided_it(
    sources: Sources, templates: TemplateLibrary
) -> None:
    """A gated result carried a reason; a rated one arrived bare.

    Every request persists a route and a cause, so the queue entries hardest
    to act on were the ones with the most judgement behind them.
    """
    outcome = await plan_for(
        Proceed(
            intent=Intent.SHIPPING_POLICY,
            enquiry=Enquiry(message="Quel est le delai de livraison ?", locale="fr"),
        ),
        sources=sources,
        templates=templates,
    )
    assert isinstance(outcome, Plan)
    assert outcome.route is Route.HUMAN_ESCALATION
    assert outcome.reasons == {EvidenceReason.NOT_COVERED}


@pytest.mark.asyncio
async def test_an_answer_that_went_out_blames_nothing(
    sources: Sources, templates: TemplateLibrary
) -> None:
    outcome = await plan_for(
        Proceed(
            intent=Intent.RETURN_POLICY,
            enquiry=Enquiry(message="How long do I have to return a jacket?"),
        ),
        sources=sources,
        templates=templates,
    )
    assert isinstance(outcome, Plan)
    assert outcome.route is Route.DIRECT_RESPONSE
    assert outcome.reasons == frozenset()


def test_every_rating_tied_at_the_bottom_is_named() -> None:
    """Naming one of two equal faults sends a reader to fix half of it."""
    plan = Plan(
        intent=Intent.RETURN_POLICY,
        requested=frozenset(),
        assessment=Assessment(
            required={
                Factor.COVERAGE: ReliabilityLevel.UNUSABLE,
                Factor.RELEVANCE: ReliabilityLevel.UNUSABLE,
                Factor.AUTHORITY: ReliabilityLevel.READY,
            }
        ),
        citations=(),
    )
    assert plan.reasons == {EvidenceReason.NOT_COVERED, EvidenceReason.POORLY_MATCHED}


def test_a_handover_has_to_say_why_here_too() -> None:
    """The invariant triage already holds. Both feed the same queue."""
    with pytest.raises(UnexplainedEscalationError):
        Handover(reasons=frozenset())


@pytest.mark.asyncio
async def test_an_answer_is_assembled_from_approved_wording(
    sources: Sources, templates: TemplateLibrary
) -> None:
    outcome = await plan_for(
        Proceed(
            intent=Intent.RETURN_POLICY,
            enquiry=Enquiry(message="How long do I have to return a jacket?"),
        ),
        sources=sources,
        templates=templates,
    )
    assert isinstance(outcome, Plan)
    assert outcome.reply is not None
    assert outcome.reply.text == "Returns are accepted within 30 days of delivery."
    assert len(outcome.said) == 1
    reference, _, digest = outcome.said[0].partition("@")
    assert reference == "say:return_window.en.v1"
    assert digest.startswith("sha256:")


@pytest.mark.asyncio
async def test_nothing_bound_for_a_queue_carries_wording_for_a_customer(
    sources: Sources, templates: TemplateLibrary
) -> None:
    """A reviewer reading it could take it for something already sent."""
    outcome = await plan_for(
        Proceed(
            intent=Intent.SHIPPING_POLICY,
            enquiry=Enquiry(message="Quel est le delai de livraison ?", locale="fr"),
        ),
        sources=sources,
        templates=templates,
    )
    assert isinstance(outcome, Plan)
    assert outcome.route is Route.HUMAN_ESCALATION
    assert outcome.reply is None
    assert outcome.said == ()


def test_neither_half_of_the_delivery_rule_can_be_broken() -> None:
    """Silence sent to a customer is as wrong as wording sent to a queue."""
    ready = Assessment(required={Factor.COVERAGE: ReliabilityLevel.READY})
    with pytest.raises(MisdirectedReplyError, match="say something"):
        Plan(
            intent=Intent.RETURN_POLICY,
            requested=frozenset(),
            assessment=ready,
            citations=(),
        )

    sunk = Assessment(required={Factor.COVERAGE: ReliabilityLevel.UNUSABLE})
    with pytest.raises(MisdirectedReplyError, match="written for a customer"):
        Plan(
            intent=Intent.RETURN_POLICY,
            requested=frozenset(),
            assessment=sunk,
            citations=(),
            reply=ApprovedReply(
                text="Returns are accepted within 30 days.",
                said=("say:return_window.en.v1@sha256:abc",),
            ),
        )


@pytest.mark.asyncio
async def test_good_evidence_nobody_wrote_a_sentence_for_waits(
    sources: Sources,
) -> None:
    """The evidence was fine. The phrase book is what is missing.

    And it arrives carrying what it was going to be answered with. A
    colleague closing this wants the entry it found, the reading of the
    question and how each dimension rated; all of it used to be dropped at
    the last step for a bare note saying no wording existed.
    """
    outcome = await plan_for(
        Proceed(
            intent=Intent.RETURN_POLICY,
            enquiry=Enquiry(message="How long do I have to return a jacket?"),
        ),
        sources=sources,
        templates=TemplateLibrary([]),
    )
    assert isinstance(outcome, Plan)
    assert outcome.route is Route.INTERNAL_REVIEW
    assert outcome.held_back is ReviewReason.NOTHING_APPROVED_TO_SAY
    assert outcome.reply is None
    assert outcome.intent is Intent.RETURN_POLICY
    assert outcome.assessment.required[Factor.COVERAGE] is ReliabilityLevel.READY
    assert [cited.reference for cited in outcome.citations] == [
        "kb:returns.standard.en.v1"
    ]


def test_a_missing_sentence_may_delay_a_reply_and_never_divert_one() -> None:
    """It is a fault here, so it can only ever make a request wait.

    Set on evidence already bound for a specialist it would relabel that as
    something a colleague could close, which is a downgrade wearing the
    clothes of a formatting problem.
    """
    unusable = Assessment(required={Factor.COVERAGE: ReliabilityLevel.UNUSABLE})
    with pytest.raises(MisdirectedReplyError, match="as a review"):
        Plan(
            intent=Intent.RETURN_POLICY,
            requested=frozenset({Fact.RETURN_WINDOW}),
            assessment=unusable,
            citations=(),
            held_back=ReviewReason.NOTHING_APPROVED_TO_SAY,
        )


def test_an_answer_sent_citing_nothing_is_refused() -> None:
    """Approved wording is not on its own a reason to have said it.

    A reply resting on no evidence at all is one nobody can re-check, which
    is the state the citation record exists to make impossible.
    """
    with pytest.raises(MisdirectedReplyError, match="citing nothing"):
        Plan(
            intent=Intent.RETURN_POLICY,
            requested=frozenset(),
            assessment=Assessment(required={Factor.COVERAGE: ReliabilityLevel.READY}),
            citations=(),
            reply=ApprovedReply(
                text="Returns are accepted within 30 days.",
                said=("say:return_window.en.v1@sha256:abc",),
            ),
        )


@pytest.fixture
def shop() -> Sources:
    """The corpus and a storefront of invented rows, on a clock we control."""
    return Sources(
        knowledge_base=PolicyIndex(load_corpus()),
        commerce=DemoStorefront(now=lambda: SHOP_NOW),
        now=lambda: SHOP_NOW,
    )


SHOP_NOW = datetime(2026, 9, 7, 12, 0, tzinfo=UTC)


def asking(message: str, **known: object) -> Proceed:
    return Proceed(
        intent=Intent.ORDER_STATUS,
        enquiry=Enquiry(message=message, customer=1, order="4471", **known),  # type: ignore[arg-type]
    )


@pytest.mark.asyncio
async def test_an_order_question_now_reaches_the_shop_instead_of_a_person(
    shop: Sources, templates: TemplateLibrary
) -> None:
    """Every rating satisfied, and no approved sentence to put it in.

    Which is the truthful outcome rather than a shortfall: the phrase book
    covers written policy and nothing else so far, so a colleague finishes
    this from the record. Before the gateway existed the same question
    stopped at the availability gate without anything having been looked up.
    """
    outcome = await plan_for(
        asking("Where is my order?"), sources=shop, templates=templates
    )
    assert isinstance(outcome, Plan)
    assert outcome.route is Route.INTERNAL_REVIEW
    assert outcome.held_back is ReviewReason.NOTHING_APPROVED_TO_SAY
    # Everything a colleague needs, which a bare note carried none of.
    assert outcome.intent is Intent.ORDER_STATUS
    assert outcome.assessment.required[Factor.FRESHNESS] is ReliabilityLevel.READY
    (cited,) = outcome.citations
    assert cited.reference == "demo:order:4471"
    assert cited.synthetic is True


@pytest.mark.asyncio
async def test_a_reference_nobody_can_show_this_customer_sends_them_back_to_check(
    shop: Sources, templates: TemplateLibrary
) -> None:
    """Somebody else's order and no such order arrive identically.

    Not an escalation: nothing has gone wrong here, and a colleague can do
    nothing about it that the person who typed the reference cannot do
    faster.
    """
    stranger = Proceed(
        intent=Intent.ORDER_STATUS,
        enquiry=Enquiry(message="Where is my order?", customer=2, order="4471"),
    )
    missing = Proceed(
        intent=Intent.ORDER_STATUS,
        enquiry=Enquiry(message="Where is my order?", customer=1, order="9999"),
    )
    expected = Clarify(
        reason=ClarificationReason.ORDER_NOT_FOUND, intent=Intent.ORDER_STATUS
    )
    assert await plan_for(stranger, sources=shop, templates=templates) == expected
    assert await plan_for(missing, sources=shop, templates=templates) == expected


@pytest.mark.asyncio
async def test_a_shop_having_a_bad_afternoon_is_ours_to_answer_for(
    templates: TemplateLibrary,
) -> None:
    """The customer asked a fair question; our own records did not reply.

    A specialist is the wrong destination for that, and so is a five hundred.
    """

    class Unreachable:
        async def order(self, reference: str, *, customer: int) -> object:
            raise CommerceUnavailableError("read timed out")

    unwell = Sources(
        knowledge_base=PolicyIndex(load_corpus()),
        commerce=Unreachable(),  # type: ignore[arg-type]
    )
    outcome = await plan_for(
        asking("Where is my order?"), sources=unwell, templates=templates
    )
    assert outcome == Review(
        reason=ReviewReason.SOURCE_UNAVAILABLE, intent=Intent.ORDER_STATUS
    )


@pytest.mark.asyncio
async def test_an_order_still_in_the_building_cannot_say_when_it_lands(
    shop: Sources, templates: TemplateLibrary
) -> None:
    """Asked for a date by a row that has none, which is not a weak answer.

    Nothing it holds bears on the question, so there is no partial reply to
    hand a colleague and it goes to somebody who can find out.
    """
    placed = Proceed(
        intent=Intent.ORDER_STATUS,
        enquiry=Enquiry(message="When will my order arrive?", customer=1, order="4472"),
    )
    outcome = await plan_for(placed, sources=shop, templates=templates)
    assert isinstance(outcome, Plan)
    assert outcome.assessment.required[Factor.COVERAGE] is ReliabilityLevel.UNUSABLE
    assert outcome.route is Route.HUMAN_ESCALATION


@pytest.mark.asyncio
async def test_evidence_read_this_morning_is_rated_on_its_age(
    templates: TemplateLibrary,
) -> None:
    """The factor every commerce profile requires and nothing used to supply.

    A reading taken hours ago is still legible and no longer something to
    send unsupervised, so it waits for one of us.
    """
    hours_later = Sources(
        knowledge_base=PolicyIndex(load_corpus()),
        commerce=DemoStorefront(now=lambda: SHOP_NOW),
        now=lambda: SHOP_NOW + timedelta(hours=2),
    )
    outcome = await plan_for(
        asking("Where is my order?"), sources=hours_later, templates=templates
    )
    assert isinstance(outcome, Plan)
    assert outcome.assessment.required[Factor.FRESHNESS] is ReliabilityLevel.REVIEW_ONLY
    assert outcome.route is Route.INTERNAL_REVIEW


@pytest.mark.asyncio
async def test_a_reply_records_which_row_it_rested_on(
    shop: Sources, templates: TemplateLibrary
) -> None:
    """Named by supplier, and carrying the note that none of it is real."""
    stale = Sources(
        knowledge_base=PolicyIndex(load_corpus()),
        commerce=DemoStorefront(now=lambda: SHOP_NOW),
        now=lambda: SHOP_NOW + timedelta(hours=2),
    )
    outcome = await plan_for(
        asking("Where is my order?"), sources=stale, templates=templates
    )
    assert isinstance(outcome, Plan)
    (cited,) = outcome.citations
    assert cited.source is Source.COMMERCE
    assert cited.reference == "demo:order:4471"
    assert cited.synthetic is True
    assert cited.observed_at == SHOP_NOW


@pytest.mark.asyncio
async def test_a_broken_integration_is_not_the_customer_s_mistake(
    templates: TemplateLibrary,
) -> None:
    """The whole point of telling the two apart.

    Read as a row that is not there, a provider returning nonsense sends back
    a request to check a reference that was typed correctly — putting the
    customer to work on a fault of ours, and leaving no sign in the case that
    anything was wrong at our end.
    """

    class Nonsense:
        async def order(self, reference: str, *, customer: int) -> object:
            return {"reference": reference, "state": "dispatched"}

    confused = Sources(
        knowledge_base=PolicyIndex(load_corpus()),
        commerce=Nonsense(),  # type: ignore[arg-type]
    )
    outcome = await plan_for(
        asking("Where is my order?"), sources=confused, templates=templates
    )
    assert outcome == Review(
        reason=ReviewReason.SOURCE_UNAVAILABLE, intent=Intent.ORDER_STATUS
    )
    assert not isinstance(outcome, Clarify)


@pytest.mark.asyncio
async def test_a_lookup_answering_about_something_else_is_not_an_answer(
    templates: TemplateLibrary,
) -> None:
    """An order question answered with a catalogue entry.

    Every rating downstream would have been taken correctly, about a row
    describing a different thing — the failure that keeping a question and
    its evidence together exists to prevent, arriving from outside.
    """

    class WrongShelf:
        async def order(self, reference: str, *, customer: int) -> object:
            return Found(
                record=ProductRecord(
                    provider="demo",
                    observed=Observation.LIVE,
                    observed_at=SHOP_NOW,
                    synthetic=True,
                    reference="12",
                    in_stock=True,
                    quantity=3,
                )
            )

    muddled = Sources(
        knowledge_base=PolicyIndex(load_corpus()),
        commerce=WrongShelf(),  # type: ignore[arg-type]
    )
    outcome = await plan_for(
        asking("Where is my order?"), sources=muddled, templates=templates
    )
    assert outcome == Review(
        reason=ReviewReason.SOURCE_UNAVAILABLE, intent=Intent.ORDER_STATUS
    )


@pytest.mark.asyncio
async def test_a_shop_answering_about_a_different_order_is_not_an_answer(
    templates: TemplateLibrary,
) -> None:
    """Held for us, not sent back to the customer as a bad reference."""

    class WrongOrder:
        async def order(self, reference: str, *, customer: int) -> object:
            return Found(record=_an_order("9999"))

    muddled = Sources(
        knowledge_base=PolicyIndex(load_corpus()),
        commerce=WrongOrder(),  # type: ignore[arg-type]
    )
    outcome = await plan_for(
        asking("Where is my order?"), sources=muddled, templates=templates
    )
    assert outcome == Review(
        reason=ReviewReason.SOURCE_UNAVAILABLE, intent=Intent.ORDER_STATUS
    )


@pytest.mark.asyncio
async def test_weather_and_a_defect_are_told_apart_where_they_are_told_apart(
    templates: TemplateLibrary, caplog: pytest.LogCaptureFixture
) -> None:
    """Both wait for a colleague, so the log is the only place this shows.

    A provider having a bad afternoon recurs and mends itself. A provider
    answering in a shape nobody described is somebody's defect and will do it
    again tomorrow. Reading the same in the reply and the case, they would be
    indistinguishable to whoever has to decide whether anything needs doing.
    """

    class Quiet:
        async def order(self, reference: str, *, customer: int) -> object:
            raise CommerceUnavailableError("read timed out")

    class Nonsense:
        async def order(self, reference: str, *, customer: int) -> object:
            return {"reference": reference}

    for gateway, level in ((Quiet(), "WARNING"), (Nonsense(), "ERROR")):
        caplog.clear()
        with caplog.at_level(logging.DEBUG, logger="app.agent.answering"):
            outcome = await plan_for(
                asking("Where is my order?"),
                sources=Sources(
                    knowledge_base=PolicyIndex(load_corpus()),
                    commerce=gateway,  # type: ignore[arg-type]
                ),
                templates=templates,
            )
        assert isinstance(outcome, Review)
        assert [record.levelname for record in caplog.records] == [level]


def _an_order(reference: str) -> OrderRecord:
    """A structurally valid order, about whichever purchase is named."""
    return OrderRecord(
        provider="demo",
        observed=Observation.LIVE,
        observed_at=SHOP_NOW,
        synthetic=True,
        reference=reference,
        state=OrderState.DISPATCHED,
    )


def _asking_stock(reference: str | None = "12", locale: str = "en") -> Proceed:
    return Proceed(
        intent=Intent.PRODUCT_AVAILABILITY,
        enquiry=Enquiry(
            message="Is it still available?"
            if locale == "en"
            else "Est-ce encore disponible ?",
            locale=locale,  # type: ignore[arg-type]
            customer=1,
            product=reference,
        ),
    )


@pytest.mark.asyncio
async def test_a_stock_question_is_answered_from_the_shop_in_both_languages(
    shop: Sources, templates: TemplateLibrary
) -> None:
    """The first non-policy journey that reaches a customer.

    The sentence says what the row holds and stops. An earlier draft read
    "available to order on our website", which asserts that ordering is
    enabled and that the item is listed — neither of which a stock flag
    establishes. Connective phrasing widening a claim is the failure approved
    wording exists to prevent, and it is invisible to every structural check
    around it.
    """
    for locale, expected in (
        ("en", "That item is in stock."),
        ("fr", "Cet article est en stock."),
    ):
        outcome = await plan_for(
            _asking_stock(locale=locale), sources=shop, templates=templates
        )
        assert isinstance(outcome, Plan)
        assert outcome.route is Route.DIRECT_RESPONSE
        assert outcome.reply is not None
        assert outcome.reply.text == expected
        (cited,) = outcome.citations
        assert cited.reference == "demo:product:12"
        assert cited.synthetic is True


@pytest.mark.asyncio
async def test_an_item_nobody_has_says_so_rather_than_going_quiet(
    shop: Sources, templates: TemplateLibrary
) -> None:
    """Out of stock is an answer, not an absence of one."""
    outcome = await plan_for(_asking_stock("13"), sources=shop, templates=templates)
    assert isinstance(outcome, Plan)
    assert outcome.route is Route.DIRECT_RESPONSE
    assert outcome.reply is not None
    assert "out of stock" in outcome.reply.text


@pytest.mark.asyncio
async def test_a_reading_taken_hours_ago_is_not_sent_unsupervised(
    templates: TemplateLibrary,
) -> None:
    """Approved wording exists and the evidence is too old to use it.

    Which is the point of rating age separately: nothing about the sentence
    or the record changed, and the request still waits for a colleague.
    """
    stale = Sources(
        knowledge_base=PolicyIndex(load_corpus()),
        commerce=DemoStorefront(now=lambda: SHOP_NOW),
        now=lambda: SHOP_NOW + timedelta(hours=2),
    )
    outcome = await plan_for(_asking_stock(), sources=stale, templates=templates)
    assert isinstance(outcome, Plan)
    assert outcome.assessment.required[Factor.FRESHNESS] is ReliabilityLevel.REVIEW_ONLY
    assert outcome.route is Route.INTERNAL_REVIEW
    assert outcome.reply is None


@pytest.mark.asyncio
async def test_a_catalogue_nobody_can_reach_still_does_not_guess(
    templates: TemplateLibrary,
) -> None:
    """Wording existing changes nothing when there is no row to put in it."""

    class Quiet:
        async def product(self, reference: str) -> object:
            raise CommerceUnavailableError("read timed out")

    outcome = await plan_for(
        _asking_stock(),
        sources=Sources(
            knowledge_base=PolicyIndex(load_corpus()),
            commerce=Quiet(),  # type: ignore[arg-type]
        ),
        templates=templates,
    )
    assert outcome == Review(
        reason=ReviewReason.SOURCE_UNAVAILABLE, intent=Intent.PRODUCT_AVAILABILITY
    )


@pytest.mark.asyncio
async def test_order_and_refund_questions_still_wait_for_wording(
    shop: Sources, templates: TemplateLibrary
) -> None:
    """Only one intent was finished, and the others say so rather than guess.

    Their states are things the records settle; what is missing is a sentence
    somebody has approved for saying them. That is a gap a person closes, and
    until they do the request goes to one.
    """
    outcome = await plan_for(
        asking("Where is my order?"), sources=shop, templates=templates
    )
    assert isinstance(outcome, Plan)
    assert outcome.held_back is ReviewReason.NOTHING_APPROVED_TO_SAY
    assert outcome.route is Route.INTERNAL_REVIEW
