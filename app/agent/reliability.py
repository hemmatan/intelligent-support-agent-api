"""How ready an answer is for automation, and where that sends it.

Reliability is an ordinal level, not a probability. The evidence available
here does not support calibrated probabilities, and a figure like 0.83 would
invent precision that nothing measured.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import IntEnum, StrEnum


class Factor(StrEnum):
    """A dimension of an answer that can be rated independently."""

    AUTHORITY = "authority"
    """Whether the source is the one that owns this kind of claim."""

    FRESHNESS = "freshness"
    """READY: live. ACCEPTABLE: cached within its TTL. REVIEW_ONLY: stale but
    readable. UNUSABLE: expired or unparseable."""

    COVERAGE = "coverage"
    """READY: every material claim is supported. ACCEPTABLE: all material
    claims supported, a minor detail missing. REVIEW_ONLY: part of the request
    can be answered. UNUSABLE: no material claim can be supported."""

    RELEVANCE = "relevance"
    """How well the retrieved knowledge-base entries match what was asked."""


class ReliabilityLevel(IntEnum):
    """Ordered, so that `min` picks the weakest and comparisons are ordinal."""

    UNUSABLE = 0
    REVIEW_ONLY = 1
    ACCEPTABLE = 2
    READY = 3


class Route(StrEnum):
    """What happens to a request once it has been assessed."""

    DIRECT_RESPONSE = "direct_response"
    CLARIFICATION = "clarification"
    INTERNAL_REVIEW = "internal_review"
    HUMAN_ESCALATION = "human_escalation"


_ROUTES: Mapping[ReliabilityLevel, Route] = {
    ReliabilityLevel.READY: Route.DIRECT_RESPONSE,
    ReliabilityLevel.ACCEPTABLE: Route.DIRECT_RESPONSE,
    ReliabilityLevel.REVIEW_ONLY: Route.INTERNAL_REVIEW,
    ReliabilityLevel.UNUSABLE: Route.HUMAN_ESCALATION,
}


class NoApplicableFactorsError(RuntimeError):
    """A decision profile declared no required factors.

    Aggregating nothing would yield an unbounded level, so this is a
    configuration fault rather than a low-reliability answer.
    """


@dataclass(frozen=True)
class Assessment:
    """Factor ratings for one request, and what they add up to.

    `required` comes from the decision profile and decides the outcome.
    `contextual` is recorded for the audit trail but never lowers the level: a
    conversation-history lookup timing out must not block an order status the
    commerce gateway answered in full.
    """

    required: Mapping[Factor, ReliabilityLevel]
    contextual: Mapping[Factor, ReliabilityLevel] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.required:
            raise NoApplicableFactorsError(
                "a decision profile must declare at least one required factor"
            )

    @property
    def level(self) -> ReliabilityLevel:
        """The weakest required factor.

        Averaging would let three strong dimensions hide one dangerous one,
        and "the retrieval was excellent, but the delivery date was invented"
        must never come out as a good answer.
        """
        return min(self.required.values())

    @property
    def route(self) -> Route:
        """Where this level sends the request.

        Never CLARIFICATION: that comes from a gate finding something missing,
        before an answer is assessed at all.
        """
        return _ROUTES[self.level]

    def as_dict(self) -> dict[str, object]:
        """Serialised with its scale attached, so nothing reads as a percentage."""
        return {
            "level": self.level.name.lower(),
            "ordinal": int(self.level),
            "scale": int(max(ReliabilityLevel)),
            "factors": {
                factor.value: level.name.lower()
                for factor, level in {**self.required, **self.contextual}.items()
            },
        }
