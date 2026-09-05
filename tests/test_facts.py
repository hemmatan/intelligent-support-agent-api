"""What a question asks to be told, and whether the evidence tells it."""

import pytest

from app.agent.facts import Fact, coverage_of, facts_in
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


def test_half_an_answer_is_not_a_partial_score() -> None:
    """Sent without the missing half, it reads as the whole reply."""
    requested = frozenset({Fact.RETURN_WINDOW, Fact.RETURN_SHIPPING_COST})
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
    requested = facts_in("What is your returns policy?")
    assert requested == RETURNS
    assert route_of(coverage_of(requested, RETURNS)) is Route.DIRECT_RESPONSE


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
    "question",
    [
        "Can I return underwear?",
        "Can I return pierced earrings?",
        "Can I return a final-sale item?",
        "Puis-je retourner des articles soldes ?",
    ],
    ids=["hygiene exclusion", "hygiene exclusion, plural", "final sale", "french sale"],
)
def test_naming_a_carved_out_kind_of_thing_outruns_the_general_rule(
    question: str,
) -> None:
    """One approved value stood for the whole of who may return what.

    The entry states eligibility as "standard_items" and a window of thirty
    days. That settles a question about an ordinary purchase and nothing else:
    it does not say which kinds of thing are left out, or how a final-sale
    item differs, so a question naming one was being rated as answered by a
    value that never mentioned it.

    The prose does say. It is prose, and no answer is built from a sentence,
    so these reach a person until the exclusions are claims.
    """
    carried = next(entry.facts for entry in load_corpus() if "returns" in entry.id)
    requested = facts_in(question)
    assert requested & {Fact.RETURN_EXCLUDED_CATEGORIES, Fact.RETURN_SALE_ITEMS}
    assert route_of(coverage_of(requested, carried)) is Route.HUMAN_ESCALATION


def test_an_ordinary_purchase_is_still_answered() -> None:
    """The split has to leave the common question working, or it is a ban."""
    carried = next(entry.facts for entry in load_corpus() if "returns" in entry.id)
    for question in ("Can I return a jacket?", "Can I send this back?"):
        assert route_of(coverage_of(facts_in(question), carried)) is (
            Route.DIRECT_RESPONSE
        )
