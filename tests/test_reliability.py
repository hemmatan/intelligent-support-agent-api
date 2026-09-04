"""Reliability aggregation and routing."""

import pytest

from app.agent.reliability import (
    Assessment,
    Factor,
    NoApplicableFactorsError,
    ReliabilityLevel,
    Route,
)


@pytest.mark.parametrize(
    ("level", "expected"),
    [
        (ReliabilityLevel.READY, Route.DIRECT_RESPONSE),
        (ReliabilityLevel.ACCEPTABLE, Route.DIRECT_RESPONSE),
        (ReliabilityLevel.REVIEW_ONLY, Route.INTERNAL_REVIEW),
        (ReliabilityLevel.UNUSABLE, Route.HUMAN_ESCALATION),
    ],
)
def test_every_level_has_a_route(level: ReliabilityLevel, expected: Route) -> None:
    assert Assessment(required={Factor.AUTHORITY: level}).route is expected


def test_the_weakest_required_factor_decides() -> None:
    """Three strong dimensions must not hide one weak one."""
    assessment = Assessment(
        required={
            Factor.AUTHORITY: ReliabilityLevel.READY,
            Factor.FRESHNESS: ReliabilityLevel.READY,
            Factor.COVERAGE: ReliabilityLevel.REVIEW_ONLY,
            Factor.RELEVANCE: ReliabilityLevel.READY,
        }
    )
    assert assessment.level is ReliabilityLevel.REVIEW_ONLY
    assert assessment.route is Route.INTERNAL_REVIEW


def test_contextual_factors_are_recorded_but_never_lower_the_level() -> None:
    """A history lookup timing out cannot block an order the gateway answered."""
    assessment = Assessment(
        required={Factor.AUTHORITY: ReliabilityLevel.READY},
        contextual={Factor.RELEVANCE: ReliabilityLevel.UNUSABLE},
    )
    assert assessment.level is ReliabilityLevel.READY
    assert assessment.route is Route.DIRECT_RESPONSE
    assert assessment.as_dict()["factors"] == {
        "authority": "ready",
        "relevance": "unusable",
    }


def test_a_profile_with_no_required_factors_is_a_configuration_fault() -> None:
    with pytest.raises(NoApplicableFactorsError):
        Assessment(required={})


def test_levels_are_ordered_weakest_first() -> None:
    assert (
        ReliabilityLevel.UNUSABLE
        < ReliabilityLevel.REVIEW_ONLY
        < ReliabilityLevel.ACCEPTABLE
        < ReliabilityLevel.READY
    )


def test_serialisation_carries_its_scale() -> None:
    """An ordinal published without its scale reads as a broken percentage."""
    payload = Assessment(
        required={Factor.COVERAGE: ReliabilityLevel.ACCEPTABLE}
    ).as_dict()
    assert payload["level"] == "acceptable"
    assert payload["ordinal"] == 2
    assert payload["scale"] == 3


def test_clarification_is_never_produced_by_a_level() -> None:
    """Clarification comes from a gate, before anything is assessed."""
    routes = {
        Assessment(required={Factor.AUTHORITY: level}).route
        for level in ReliabilityLevel
    }
    assert Route.CLARIFICATION not in routes
