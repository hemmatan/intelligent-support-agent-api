"""What each request may be answered from, and the shapes that are refused."""

import pytest

from app.agent.intent import Intent
from app.agent.profiles import (
    PROFILES,
    DecisionProfile,
    Input,
    ProfileError,
    Source,
    profile_for,
)
from app.agent.reasons import ReasonCode
from app.agent.reliability import Assessment, Factor, ReliabilityLevel, Route


def test_every_intent_has_a_profile() -> None:
    """An intent nothing governs would be answered from whatever was reachable."""
    assert set(PROFILES) == set(Intent)


def test_an_unknown_intent_raises_rather_than_defaulting() -> None:
    with pytest.raises(ProfileError, match="no decision profile"):
        profile_for("something_new")  # type: ignore[arg-type]


def test_policy_answers_need_the_right_entry_and_the_claim_inside_it() -> None:
    profile = profile_for(Intent.RETURN_POLICY)
    assert profile.required_sources == {Source.KNOWLEDGE_BASE}
    assert Factor.RELEVANCE in profile.required_factors
    assert Factor.COVERAGE in profile.required_factors


def test_an_order_is_looked_up_not_ranked() -> None:
    """There is no better or worse match for one order. It exists or it does not."""
    profile = profile_for(Intent.ORDER_STATUS)
    assert profile.required_sources == {Source.COMMERCE}
    assert Factor.RELEVANCE not in profile.required_factors
    assert Factor.FRESHNESS in profile.required_factors


def test_policy_does_not_need_freshness_the_way_a_delivery_state_does() -> None:
    """An approved policy is current until replaced; an order moved this morning."""
    assert Factor.FRESHNESS not in profile_for(Intent.RETURN_POLICY).required_factors
    assert Factor.FRESHNESS in profile_for(Intent.ORDER_STATUS).required_factors


def test_history_is_never_a_required_source() -> None:
    """It can say which order was meant. It cannot say where that order is."""
    for intent in Intent:
        assert Source.HISTORY not in profile_for(intent).required_sources


def test_a_missing_order_number_and_a_missing_account_go_different_ways() -> None:
    """One the customer can supply. The other is our own bookkeeping."""
    assert profile_for(Intent.ORDER_STATUS).required_inputs == {
        Input.ORDER_ID,
        Input.COMMERCE_ACCOUNT,
    }
    assert Input.ORDER_ID.when_missing is Route.CLARIFICATION
    assert Input.ORDER_ID.reason is ReasonCode.MISSING_ORDER_ID
    assert Input.COMMERCE_ACCOUNT.when_missing is Route.HUMAN_ESCALATION
    assert Input.COMMERCE_ACCOUNT.reason is ReasonCode.CUSTOMER_NOT_LINKED


def test_an_input_decides_its_own_consequence() -> None:
    """Not a field somebody fills in beside it.

    Taking a route and a reason as arguments made a missing order number
    recordable as an unlinked account, and answerable directly. Nothing
    checked, because a check was the only thing that could have.
    """
    for missing in Input:
        assert missing.when_missing in {Route.CLARIFICATION, Route.HUMAN_ESCALATION}
        assert missing.reason is not None


def test_asking_after_something_only_commerce_uses_needs_commerce() -> None:
    with pytest.raises(ProfileError, match="does not query it"):
        DecisionProfile(
            required_sources=frozenset({Source.KNOWLEDGE_BASE}),
            required_factors=frozenset({Factor.COVERAGE}),
            required_inputs=frozenset({Input.ORDER_ID}),
        )


def test_a_profile_consulting_nothing_is_refused() -> None:
    with pytest.raises(ProfileError, match="no required source"):
        DecisionProfile(
            required_sources=frozenset(),
            required_factors=frozenset({Factor.AUTHORITY}),
        )


def test_availability_needs_to_know_which_product() -> None:
    """ "Is it still available?" names nothing the catalogue could look up."""
    profile = profile_for(Intent.PRODUCT_AVAILABILITY)
    assert Input.PRODUCT_REFERENCE in profile.required_inputs
    assert Source.HISTORY in profile.contextual_sources


def test_a_profile_requiring_nothing_is_refused() -> None:
    with pytest.raises(ProfileError, match="no required factors"):
        DecisionProfile(required_sources=frozenset({Source.KNOWLEDGE_BASE}))


def test_a_source_cannot_be_both_required_and_optional() -> None:
    with pytest.raises(ProfileError, match="both and neither"):
        DecisionProfile(
            required_sources=frozenset({Source.COMMERCE}),
            contextual_sources=frozenset({Source.COMMERCE}),
            required_factors=frozenset({Factor.AUTHORITY}),
        )


def test_a_factor_cannot_be_both_required_and_optional() -> None:
    with pytest.raises(ProfileError, match="both and neither"):
        DecisionProfile(
            required_sources=frozenset({Source.COMMERCE}),
            required_factors=frozenset({Factor.AUTHORITY}),
            contextual_factors=frozenset({Factor.AUTHORITY}),
        )


def test_requiring_relevance_without_the_knowledge_base_is_refused() -> None:
    """Relevance rates retrieved entries. With nothing retrieved it rates nothing."""
    with pytest.raises(ProfileError, match="names nothing"):
        DecisionProfile(
            required_sources=frozenset({Source.COMMERCE}),
            required_factors=frozenset({Factor.RELEVANCE}),
        )


def test_naming_a_ranking_score_where_nothing_is_ranked_is_refused() -> None:
    """Contextual too. It rates retrieved entries either way."""
    with pytest.raises(ProfileError, match="names nothing"):
        DecisionProfile(
            required_sources=frozenset({Source.COMMERCE}),
            required_factors=frozenset({Factor.AUTHORITY}),
            contextual_factors=frozenset({Factor.RELEVANCE}),
        )


def test_the_source_that_owns_a_claim_can_carry_it_alone() -> None:
    policy = profile_for(Intent.RETURN_POLICY)
    assert policy.authority_of(Source.KNOWLEDGE_BASE) is ReliabilityLevel.READY


def test_a_well_matched_policy_cannot_say_where_an_order_is() -> None:
    """It shares every word with the question and none of the answer.

    A shipping policy saying orders arrive in two to three days ranks highly
    against "where is my order" and knows nothing about that order. Ranking
    cannot see the difference; authority is asked instead.
    """
    orders = profile_for(Intent.ORDER_STATUS)
    assert orders.authority_of(Source.KNOWLEDGE_BASE) is ReliabilityLevel.UNUSABLE
    assert (
        Assessment(
            required={Factor.AUTHORITY: orders.authority_of(Source.KNOWLEDGE_BASE)}
        ).route
        is Route.HUMAN_ESCALATION
    )


def test_history_can_inform_an_answer_and_never_carry_one() -> None:
    """It establishes what a customer said, never that what they said is so."""
    for intent in Intent:
        assert profile_for(intent).authority_of(Source.HISTORY) is not (
            ReliabilityLevel.READY
        )


def test_an_answer_resting_only_on_history_is_read_before_it_is_sent() -> None:
    availability = profile_for(Intent.PRODUCT_AVAILABILITY)
    resting_on_history = Assessment(
        required={Factor.AUTHORITY: availability.authority_of(Source.HISTORY)}
    )
    assert resting_on_history.route is Route.INTERNAL_REVIEW


def test_a_source_this_request_never_chose_carries_nothing() -> None:
    """Reading it at all means something was picked for ranking well."""
    policy = profile_for(Intent.RETURN_POLICY)
    assert policy.authority_of(Source.COMMERCE) is ReliabilityLevel.UNUSABLE
