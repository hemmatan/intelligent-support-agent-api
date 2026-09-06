"""What the shop's records may be asked, and how far an answer carries."""

import dataclasses
import inspect
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

import pytest

from app.agent.commerce import (
    CommerceGateway,
    CommerceMisconfiguredError,
    CommerceUnavailableError,
    Found,
    NotAvailable,
    Observation,
    OrderLookup,
    OrderRecord,
    OrderState,
    ProductRecord,
    RefundRecord,
    RefundState,
    freshness_of,
    names_a_customer,
)
from app.agent.facts import Fact, coverage_of
from app.agent.reliability import ReliabilityLevel

NOW = datetime(2026, 9, 6, 12, 0, tzinfo=UTC)
TTL = timedelta(minutes=15)
READABLE_FOR = timedelta(hours=6)


def order(**changed: object) -> OrderRecord:
    """A dispatched order, observed live, with everything filled in."""
    fields: dict[str, object] = {
        "provider": "demo",
        "observed": Observation.LIVE,
        "observed_at": NOW,
        "synthetic": True,
        "reference": "4471",
        "state": OrderState.DISPATCHED,
        "carrier": "DPD",
        "tracking_reference": "FR123456",
        "expected_delivery": date(2026, 9, 9),
    }
    fields.update(changed)
    return OrderRecord(**fields)  # type: ignore[arg-type]


def rated(record: OrderRecord) -> ReliabilityLevel:
    return freshness_of(record, now=NOW, ttl=TTL, readable_for=READABLE_FOR)


def test_a_reading_taken_now_is_the_only_one_that_goes_out_unsupervised() -> None:
    assert rated(order()) is ReliabilityLevel.READY


def test_something_kept_from_earlier_is_worth_less_than_a_fresh_look() -> None:
    """Inside its window it can still answer, and it is not the same thing.

    The two share a destination today. They are told apart because the
    difference is worth counting, and because tightening the bar later should
    move a route and not a score.
    """
    kept = order(observed=Observation.CACHED, observed_at=NOW - timedelta(minutes=5))
    assert rated(kept) is ReliabilityLevel.ACCEPTABLE


def test_past_its_window_but_still_legible_waits_for_somebody_here() -> None:
    stale = order(observed=Observation.CACHED, observed_at=NOW - timedelta(hours=2))
    assert rated(stale) is ReliabilityLevel.REVIEW_ONLY


def test_old_enough_and_nobody_should_be_shown_it() -> None:
    expired = order(observed=Observation.CACHED, observed_at=NOW - timedelta(days=3))
    assert rated(expired) is ReliabilityLevel.UNUSABLE


def test_a_label_saying_live_cannot_launder_something_stale() -> None:
    """The flag is the half that can be wrong, so the clock settles it.

    A record marked live and dated last Tuesday is a contradiction. Believing
    the label would let an adapter with a stuck cache send week-old delivery
    states out as though somebody had just looked.
    """
    contradictory = order(
        observed=Observation.LIVE, observed_at=NOW - timedelta(days=7)
    )
    assert rated(contradictory) is not ReliabilityLevel.READY
    assert rated(contradictory) is ReliabilityLevel.UNUSABLE


@pytest.mark.parametrize(
    "observed_at",
    [
        None,
        datetime(2026, 9, 6, 12, 0),  # no offset, so no moment in particular
        NOW + timedelta(hours=1),  # somebody else's clock, ahead of ours
    ],
)
def test_a_timestamp_nothing_can_be_worked_out_from_is_refused(
    observed_at: datetime | None,
) -> None:
    """A record dated after it was read describes something yet to happen."""
    assert rated(order(observed_at=observed_at)) is ReliabilityLevel.UNUSABLE


def test_an_order_settles_what_it_holds_and_not_what_its_kind_could_hold() -> None:
    """Placed and not yet dispatched, so there is nowhere to look it up.

    Policy works the other way round: an entry of a given kind always states
    its claims, so what it can answer belongs to the kind. An order is a
    record of a thing in progress and answers only as far as it has got.
    """
    early = order(
        state=OrderState.PLACED, tracking_reference=None, expected_delivery=None
    )
    assert early.facts == {Fact.ORDER_STATE}
    assert Fact.TRACKING_REFERENCE not in early.facts
    assert order().facts == {
        Fact.ORDER_STATE,
        Fact.TRACKING_REFERENCE,
        Fact.DELIVERY_ESTIMATE,
    }


def test_a_sum_without_a_currency_is_not_an_amount_anybody_can_be_told() -> None:
    banked = {
        "provider": "demo",
        "observed": Observation.LIVE,
        "observed_at": NOW,
        "synthetic": True,
        "order": "4471",
        "state": RefundState.PAID,
    }
    assert RefundRecord(**banked, amount=Decimal("20.00"), currency="EUR").facts == {  # type: ignore[arg-type]
        Fact.REFUND_STATE,
        Fact.REFUND_AMOUNT,
    }
    assert RefundRecord(**banked, amount=Decimal("20.00")).facts == {  # type: ignore[arg-type]
        Fact.REFUND_STATE
    }


def test_a_catalogue_entry_answers_about_stock() -> None:
    listed = ProductRecord(
        provider="demo",
        observed=Observation.LIVE,
        observed_at=NOW,
        synthetic=True,
        reference="12",
        in_stock=True,
        quantity=4,
    )
    assert listed.facts == {Fact.STOCK_AVAILABILITY}


def test_asking_an_order_something_it_cannot_settle_is_half_an_answer() -> None:
    """The same rung a partly answered policy question lands on.

    Wanting to know where a parcel is and when it lands, from an order that
    has been placed and nothing more, is a question somebody here can finish
    off the record in front of them.
    """
    asked = frozenset({Fact.ORDER_STATE, Fact.DELIVERY_ESTIMATE})
    early = order(
        state=OrderState.PLACED, tracking_reference=None, expected_delivery=None
    )
    assert coverage_of(asked, early.facts) is ReliabilityLevel.REVIEW_ONLY
    assert coverage_of(asked, order().facts) is ReliabilityLevel.READY


def test_an_order_cannot_be_asked_for_without_saying_whose() -> None:
    """Ownership is in the question, so there is no way to leave it out.

    Read off the protocol. Asserted against a list written beside it, this
    would go on passing after somebody had widened the signature.
    """
    taken = inspect.signature(CommerceGateway.order).parameters
    assert taken["customer"].kind is inspect.Parameter.KEYWORD_ONLY
    assert taken["customer"].default is inspect.Parameter.empty
    assert names_a_customer("order")
    assert names_a_customer("refund")


def test_the_catalogue_belongs_to_nobody() -> None:
    """Requiring a customer would imply a permission nothing enforces."""
    assert not names_a_customer("product")


def test_there_is_nothing_in_a_negative_answer_to_tell_the_two_apart() -> None:
    """No such order and not this customer's arrive as one value.

    Two would answer, across a pair of questions, which references exist —
    other people's orders enumerated one request at a time.
    """
    assert dataclasses.fields(NotAvailable) == ()


@pytest.mark.asyncio
async def test_a_provider_having_a_bad_day_is_not_a_provider_nobody_set_up() -> None:
    """Both are arrangements a test can make, which is what the protocol buys.

    One of them can be tried again and one is a typo that will answer exactly
    the same tomorrow, and telling them apart is what decides whether a
    request waits for somebody or a deployment stops.
    """

    class Timeout:
        async def order(self, reference: str, *, customer: int) -> OrderLookup:
            raise CommerceUnavailableError("read timed out")

    class Misspelled:
        async def order(self, reference: str, *, customer: int) -> OrderLookup:
            raise CommerceMisconfiguredError("401 from the configured address")

    with pytest.raises(CommerceUnavailableError):
        await Timeout().order("4471", customer=1)
    with pytest.raises(CommerceMisconfiguredError):
        await Misspelled().order("4471", customer=1)


@pytest.mark.asyncio
async def test_a_stand_in_can_answer_the_way_the_real_thing_would() -> None:
    class OneOrder:
        async def order(self, reference: str, *, customer: int) -> OrderLookup:
            if reference == "4471" and customer == 1:
                return Found(record=order())
            return NotAvailable()

    gateway = OneOrder()
    assert isinstance(await gateway.order("4471", customer=1), Found)
    # Somebody else's, and one that never existed. Same answer to both.
    assert await gateway.order("4471", customer=2) == NotAvailable()
    assert await gateway.order("9999", customer=1) == NotAvailable()
