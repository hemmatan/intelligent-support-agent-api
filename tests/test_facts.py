"""What a question asks to be told, and whether the evidence tells it."""

import pytest

from app.agent.commerce import OrderRecord, ProductRecord, Record, RefundRecord
from app.agent.facts import Fact, coverage_of, facts_in
from app.agent.intent import _CHASING_A_REFUND, _RULES, Intent
from app.agent.knowledge import load_corpus
from app.agent.reliability import Assessment, Factor, ReliabilityLevel, Route

SHIPPING = frozenset({Fact.STANDARD_DELIVERY_TIME, Fact.EXPRESS_DELIVERY_TIME})
RETURNS = frozenset({Fact.RETURN_WINDOW, Fact.RETURN_ELIGIBILITY})


def route_of(level: ReliabilityLevel) -> Route:
    """Where a coverage rating on its own sends a request."""
    return Assessment(required={Factor.COVERAGE: level}).route


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("How long do I have to return a jacket?", {Fact.RETURN_WINDOW}),
        ("Can I return a jacket?", {Fact.RETURN_ELIGIBILITY}),
        (
            "Can I return sale items?",
            {Fact.RETURN_ELIGIBILITY, Fact.RETURN_SALE_ITEMS},
        ),
        ("Who pays return shipping?", {Fact.RETURN_SHIPPING_COST}),
        ("Do I need a receipt?", {Fact.PROOF_OF_PURCHASE}),
        ("How long does delivery take?", {Fact.STANDARD_DELIVERY_TIME}),
        ("Is express delivery available?", {Fact.EXPRESS_DELIVERY_TIME}),
        ("Do you ship to Belgium?", {Fact.SHIPPING_DESTINATIONS}),
        ("What is the shipping cost?", {Fact.SHIPPING_COST}),
        ("Combien de temps pour retourner un article ?", {Fact.RETURN_WINDOW}),
    ],
    ids=lambda value: value if isinstance(value, str) else "",
)
def test_a_question_names_what_it_wants_told(
    question: str, expected: set[Fact]
) -> None:
    assert facts_in(question) == expected


def test_the_case_topic_matching_would_have_got_wrong() -> None:
    """The whole argument for rating facts rather than subjects.

    "Do you ship to Belgium?" is a shipping question and the shipping entry is
    the right entry, so relevance and authority are both satisfied. It states
    delivery times. Anything checking only that a shipping question met a
    shipping entry answers Belgium with two to three days.
    """
    requested = facts_in("Do you ship to Belgium?")
    assert coverage_of(requested, SHIPPING) is ReliabilityLevel.UNUSABLE
    assert route_of(coverage_of(requested, SHIPPING)) is Route.HUMAN_ESCALATION


def test_a_question_the_evidence_answers_is_covered() -> None:
    requested = facts_in("How long does delivery take?")
    assert route_of(coverage_of(requested, SHIPPING)) is Route.DIRECT_RESPONSE


def test_part_of_a_question_answered_is_its_own_rung() -> None:
    """Somebody here finishes it. Nothing part-answered is sent.

    Rating this UNUSABLE queued a specialist for a question one of our own
    people could close, having the entry in front of them. The scale already
    had a level for it.
    """
    requested = frozenset({Fact.RETURN_WINDOW, Fact.RETURN_SHIPPING_COST})
    assert coverage_of(requested, RETURNS) is ReliabilityLevel.REVIEW_ONLY
    assert route_of(coverage_of(requested, RETURNS)) is Route.INTERNAL_REVIEW


def test_a_question_sharing_nothing_with_the_evidence_is_unusable() -> None:
    """No overlap at all is a different finding from an incomplete one."""
    requested = frozenset({Fact.SHIPPING_DESTINATIONS})
    assert coverage_of(requested, RETURNS) is ReliabilityLevel.UNUSABLE


def test_a_question_nothing_could_pin_down_is_read_by_somebody() -> None:
    """Not treated as asking for nothing, which would pass every time.

    "Combien de temps pour la livraison" matches no rule here. Defaulting an
    unrecognised question to covered would hand back whichever entry ranked
    well, which is the failure this factor exists to catch.
    """
    requested = facts_in("Combien de temps pour la livraison ?")
    assert requested == frozenset()
    assert route_of(coverage_of(requested, SHIPPING)) is Route.INTERNAL_REVIEW


def test_asking_what_a_policy_says_asks_for_what_it_states() -> None:
    """All of it, not the two facts it happened to state when this was written.

    Answering the question that asks for the whole policy with a window and a
    general rule presents a part as the whole, which is worse than declining.
    """
    stated = next(entry.facts for entry in load_corpus() if "returns" in entry.id)
    requested = facts_in("What is your returns policy?")
    assert requested == stated
    assert route_of(coverage_of(requested, stated)) is Route.DIRECT_RESPONSE


def test_a_general_question_is_not_read_off_whatever_turned_up() -> None:
    """Deriving the request from the entry makes every entry answer it."""
    assert facts_in("What is your returns policy?") != facts_in(
        "What is your shipping policy?"
    )


def test_the_corpus_states_less_than_the_rules_will_accept() -> None:
    """Recorded because it is the gap, not an oversight.

    Nothing approved says where we ship or what postage costs, and both can be
    asked. Coverage is what makes that reach a person instead of a paraphrase
    of the delivery times.
    """
    stated: frozenset[Fact] = frozenset().union(
        *(entry.facts for entry in load_corpus())
    )
    assert Fact.SHIPPING_DESTINATIONS not in stated
    assert Fact.SHIPPING_COST not in stated
    for fact in Fact:
        if fact in stated:
            continue
        assert coverage_of(frozenset({fact}), stated) is ReliabilityLevel.UNUSABLE


@pytest.mark.parametrize(
    ("question", "fact"),
    [
        ("Can I return underwear?", Fact.RETURN_EXCLUDED_CATEGORIES),
        ("Can I return pierced earrings?", Fact.RETURN_EXCLUDED_CATEGORIES),
        ("Can I return a final-sale item?", Fact.RETURN_SALE_ITEMS),
        ("Puis-je retourner des articles soldes ?", Fact.RETURN_SALE_ITEMS),
        ("Do I need a receipt?", Fact.PROOF_OF_PURCHASE),
    ],
    ids=["hygiene", "hygiene, plural", "final sale", "french sale", "receipt"],
)
def test_a_carved_out_kind_of_thing_is_answerable_once_the_policy_states_it(
    question: str, fact: Fact
) -> None:
    """These went to a person while the answer sat in the searchable prose.

    One approved value stood for the whole rule, so a question naming a kind
    of thing the policy singles out was rated as covered by a value that never
    mentioned it — and then, once that was split apart, was not covered by
    anything at all. The policy states them now, so the claim can answer.
    """
    carried = next(entry.facts for entry in load_corpus() if "returns" in entry.id)
    requested = facts_in(question)
    assert fact in requested
    assert route_of(coverage_of(requested, carried)) is Route.DIRECT_RESPONSE


def test_what_the_policy_still_does_not_say_stays_unanswerable() -> None:
    """Structuring what is written down is not licence to invent the rest.

    Nothing approved says who pays to send something back, so the question
    reaches a person. Filling the field in to make the schema look complete
    would be manufacturing a business fact, which is the failure every other
    check here exists to prevent.
    """
    carried = next(entry.facts for entry in load_corpus() if "returns" in entry.id)
    requested = facts_in("Who pays return shipping?")
    assert requested == {Fact.RETURN_SHIPPING_COST}
    assert route_of(coverage_of(requested, carried)) is Route.HUMAN_ESCALATION


def test_an_ordinary_purchase_is_still_answered() -> None:
    """The split has to leave the common question working, or it is a ban."""
    carried = next(entry.facts for entry in load_corpus() if "returns" in entry.id)
    for question in ("Can I return a jacket?", "Can I send this back?"):
        assert route_of(coverage_of(facts_in(question), carried)) is (
            Route.DIRECT_RESPONSE
        )


COMMERCE_SUBJECTS: dict[Intent, type[Record]] = {
    Intent.ORDER_STATUS: OrderRecord,
    Intent.REFUND_STATUS: RefundRecord,
    Intent.PRODUCT_AVAILABILITY: ProductRecord,
}


def commerce_wordings() -> list[tuple[Intent, str]]:
    """Every wording that places a request against the shop's own records."""
    listed = [
        (intent, phrase)
        for intent, phrases in _RULES.items()
        if intent in COMMERCE_SUBJECTS
        for phrase in phrases
    ]
    listed += [(Intent.REFUND_STATUS, phrase) for phrase in _CHASING_A_REFUND]
    return listed


@pytest.mark.parametrize(("intent", "phrase"), commerce_wordings())
def test_a_request_we_can_place_is_a_request_we_can_say_what_it_wants(
    intent: Intent, phrase: str
) -> None:
    """Recognising a request and knowing what it asks for are two vocabularies.

    They were written at different times and nothing held them level, so four
    ways of asking where an order or a refund had got to were placed
    confidently and then found to be asking for nothing at all. Wanting
    nothing rates REVIEW_ONLY, so those requests could not have been answered
    however well the lookup behind them went.

    Parametrised over the wordings themselves. A list of examples would go on
    passing after somebody added a fifth way of asking.

    Held against the shop's own records and not against written policy, since
    the two are recognised on different terms. A commerce request is matched
    by wording that names a whole question, so one that turns out to want
    nothing is a hole. Policy is matched on the subject — a message with
    "return" in it is a returns question — and somebody who has said only that
    much has genuinely asked for nothing in particular. Requiring a fact there
    would answer "I want to return something" with the entire policy, which is
    the guessing that wanting nothing is meant to send to a person instead.
    """
    assert facts_in(phrase), f"{phrase!r} places a request and asks for nothing"


@pytest.mark.parametrize(("intent", "phrase"), commerce_wordings())
def test_a_request_asks_only_about_its_own_subject(intent: Intent, phrase: str) -> None:
    """An order question that also asked about stock would rate as half answered.

    Read off the record that owns the subject, so widening what an order is
    the authority on is one edit and not two.
    """
    wanted = facts_in(phrase)
    subject = COMMERCE_SUBJECTS[intent].ABOUT
    assert wanted & subject, f"{phrase!r} asks nothing {intent} could answer"
    assert not wanted - subject, f"{phrase!r} also asks for {sorted(wanted - subject)}"


@pytest.mark.parametrize(
    ("message", "wanted"),
    [
        ("has my order arrived", {Fact.ORDER_STATE}),
        ("when will my order arrive", {Fact.DELIVERY_ESTIMATE}),
        ("what is the tracking number", {Fact.TRACKING_REFERENCE}),
        ("is it sold out", {Fact.STOCK_AVAILABILITY}),
        ("where is my refund", {Fact.REFUND_STATE}),
        # Two, and the second is settled by nothing, so the first cannot
        # answer for the pair and somebody who can give a date gets it.
        ("when will i get my refund", {Fact.REFUND_STATE, Fact.REFUND_TIMING}),
        ("quand vais-je etre rembourse", {Fact.REFUND_STATE, Fact.REFUND_TIMING}),
    ],
)
def test_a_question_asks_for_the_thing_that_answers_it(
    message: str, wanted: set[Fact]
) -> None:
    """Exactly, because the looser check above cannot see this going wrong.

    Handing every order fact to every order question satisfies "asks for
    something about its own subject" and answers when a parcel lands with the
    state it is in. Asked whether an order arrived, wanting a delivery date
    as well would drag a settled question down to partly covered; asked when
    it will arrive, wanting only the state would answer a different question
    and rate it as whole.
    """
    assert facts_in(message) == wanted
