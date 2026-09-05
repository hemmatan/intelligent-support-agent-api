"""What each kind of request is allowed to be answered from, decided in advance.

A profile is a statement about what a safe answer needs, written before the
retrieval that might satisfy it. That ordering is the point: deciding after
the fact which sources looked promising is how the wrong one gets used because
it happened to rank well.

It also says what an answer needs that has not been built yet. A profile
requiring coverage while nothing computes coverage means requests reach
UNUSABLE and go to a person — which is true, because the check that would
have made them safe does not exist. Trimming the requirement to match the
implementation would turn an incomplete slice into a confident one.
"""

from dataclasses import dataclass
from enum import StrEnum

from app.agent.intent import Intent
from app.agent.reasons import ClarificationReason, EscalationReason, ReasonCode
from app.agent.reliability import Factor, ReliabilityLevel, Route


class Source(StrEnum):
    """Somewhere an answer can come from."""

    KNOWLEDGE_BASE = "knowledge_base"
    COMMERCE = "commerce"
    HISTORY = "history"


class Input(StrEnum):
    """Something a request needs before any source is worth asking.

    Each carries its own consequence rather than accepting one, because the
    consequence never varies with who is asking. Taking a route and a reason
    as arguments let a missing order number be recorded as an unlinked account
    and answered directly — nonsense that had to be checked for, when it could
    simply be unwriteable.
    """

    ORDER_ID = "order_id"
    COMMERCE_ACCOUNT = "commerce_account"
    PRODUCT_REFERENCE = "product_reference"

    @property
    def reason(self) -> ReasonCode:
        """Why a request stops when this is absent."""
        return _REASONS[self]

    @property
    def when_missing(self) -> Route:
        """Whether the customer can supply this, or somebody here must.

        Read off the reason rather than stored beside it. Two properties
        answering independently is two properties that can answer differently.
        """
        if isinstance(self.reason, EscalationReason):
            return Route.HUMAN_ESCALATION
        return Route.CLARIFICATION


_REASONS: dict["Input", ReasonCode] = {
    Input.ORDER_ID: ClarificationReason.MISSING_ORDER_ID,
    Input.COMMERCE_ACCOUNT: EscalationReason.CUSTOMER_NOT_LINKED,
    Input.PRODUCT_REFERENCE: ClarificationReason.MISSING_PRODUCT_REFERENCE,
}


class ProfileError(RuntimeError):
    """A profile that cannot describe a safe answer, caught at import."""


@dataclass(frozen=True)
class DecisionProfile:
    """The sources, evidence and inputs one kind of request needs.

    Required and contextual are separated because their failures differ. A
    required source that will not answer stops the request; a contextual one
    that will not answer costs some helpful colour. One set could not say
    which of those had happened.
    """

    required_sources: frozenset[Source]
    contextual_sources: frozenset[Source] = frozenset()
    required_factors: frozenset[Factor] = frozenset()
    contextual_factors: frozenset[Factor] = frozenset()
    required_inputs: frozenset[Input] = frozenset()

    def __post_init__(self) -> None:
        if not self.required_sources:
            raise ProfileError(
                "a profile with no required source would rate evidence nobody "
                "went and fetched"
            )
        if not self.required_factors:
            raise ProfileError(
                "a profile with no required factors would rate every answer on "
                "nothing at all"
            )
        overlapping_sources = self.required_sources & self.contextual_sources
        if overlapping_sources:
            raise ProfileError(f"{sorted(overlapping_sources)} is both and neither")
        overlapping_factors = self.required_factors & self.contextual_factors
        if overlapping_factors:
            raise ProfileError(f"{sorted(overlapping_factors)} is both and neither")
        rates_retrieval = Factor.RELEVANCE in (
            self.required_factors | self.contextual_factors
        )
        if rates_retrieval and Source.KNOWLEDGE_BASE not in self.required_sources:
            raise ProfileError(
                "relevance measures how well retrieved entries match a question, "
                "so naming it without the knowledge base names nothing"
            )
        if self.required_inputs and Source.COMMERCE not in self.required_sources:
            raise ProfileError(
                f"{sorted(self.required_inputs)} is asked for so that commerce can "
                f"be queried, and this profile does not query it"
            )

    def authority_of(self, source: Source) -> ReliabilityLevel:
        """How far a record from `source` may carry a claim of this kind.

        A required source owns claims like this one and can carry an answer by
        itself. A contextual source can inform one and never assert it: the
        support database can establish that a customer referred to order 4471,
        never that 4471 arrived, so an answer resting on that alone goes to
        somebody here rather than out. Anywhere else was not chosen for this
        request, and a record from it is being read because it ranked well —
        which is the thing profiles exist to stop.
        """
        if source in self.required_sources:
            return ReliabilityLevel.READY
        if source in self.contextual_sources:
            return ReliabilityLevel.REVIEW_ONLY
        return ReliabilityLevel.UNUSABLE


# Answering from written policy. Relevance says the right entry was found;
# coverage says it contains what was asked for; authority says it came from
# the source that owns claims of this kind. History can resolve what "it"
# refers to and is never the reason an answer is given.
_POLICY = DecisionProfile(
    required_sources=frozenset({Source.KNOWLEDGE_BASE}),
    contextual_sources=frozenset({Source.HISTORY}),
    required_factors=frozenset({Factor.AUTHORITY, Factor.COVERAGE, Factor.RELEVANCE}),
)

# Answering from the shop's own records. Freshness matters here and does not
# for policy: a delivery state from an hour ago may already be wrong, while an
# approved policy is current until it is replaced. Relevance does not apply —
# there is no ranking, only a record that either exists or does not.
_COMMERCE_FACTS = frozenset({Factor.AUTHORITY, Factor.FRESHNESS, Factor.COVERAGE})

PROFILES: dict[Intent, DecisionProfile] = {
    Intent.RETURN_POLICY: _POLICY,
    Intent.SHIPPING_POLICY: _POLICY,
    # Where one refund stands, which is a fact about a payment and not a rule
    # about returns. The returns policy can say somebody had thirty days; only
    # commerce can say whether their money went back.
    Intent.REFUND_STATUS: DecisionProfile(
        required_sources=frozenset({Source.COMMERCE}),
        contextual_sources=frozenset({Source.HISTORY}),
        required_factors=_COMMERCE_FACTS,
        required_inputs=frozenset({Input.COMMERCE_ACCOUNT, Input.ORDER_ID}),
    ),
    Intent.ORDER_STATUS: DecisionProfile(
        required_sources=frozenset({Source.COMMERCE}),
        contextual_sources=frozenset({Source.HISTORY}),
        required_factors=_COMMERCE_FACTS,
        required_inputs=frozenset({Input.COMMERCE_ACCOUNT, Input.ORDER_ID}),
    ),
    # "Is it still available?" names no product. History can say what "it"
    # was; where it cannot, the customer is asked rather than the catalogue
    # being searched for a thing nobody identified.
    Intent.PRODUCT_AVAILABILITY: DecisionProfile(
        required_sources=frozenset({Source.COMMERCE}),
        contextual_sources=frozenset({Source.HISTORY}),
        required_factors=_COMMERCE_FACTS,
        required_inputs=frozenset({Input.PRODUCT_REFERENCE}),
    ),
}


def profile_for(intent: Intent) -> DecisionProfile:
    """The profile governing this intent.

    Raises rather than returning a default. A permissive fallback would let a
    new intent be answered from whatever happened to be reachable, which is
    the decision profiles exist to take away from chance.
    """
    try:
        return PROFILES[intent]
    except KeyError as exc:
        raise ProfileError(f"{intent} has no decision profile") from exc
