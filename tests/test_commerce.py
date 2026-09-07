"""What the shop's records may be asked, and how far an answer carries."""

import dataclasses
import inspect
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

import pytest
from pydantic import ValidationError

from app.agent.commerce import (
    CommerceContractError,
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
    answered_with,
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
    [None, datetime(2026, 9, 6, 12, 0)],
    ids=["no moment at all", "a moment with no offset"],
)
def test_a_reading_has_to_be_placed_in_time_to_exist(observed_at: object) -> None:
    """Turned away on the way in, where this used to be scored on the way out.

    We write the moment down ourselves, knowing when we asked, so one that is
    absent or floating free is a fault where the record was assembled. Left to
    be rated afterwards it would arrive looking like weak evidence, which is a
    thing somebody may reasonably act on.
    """
    with pytest.raises(ValidationError):
        order(observed_at=observed_at)


def test_a_reading_taken_after_the_clock_it_is_held_against() -> None:
    """Two clocks disagreeing, which neither value can be inspected for alone."""
    ahead = order(observed_at=NOW + timedelta(hours=1))
    assert rated(ahead) is ReliabilityLevel.UNUSABLE


def test_a_scale_running_backwards_is_the_caller_s_mistake() -> None:
    """Legible for less time than it is useful for describes nothing."""
    with pytest.raises(ValueError, match="inside its"):
        freshness_of(order(), now=NOW, ttl=timedelta(hours=2), readable_for=TTL)


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


def test_a_sum_and_its_units_arrive_together_or_not_at_all() -> None:
    """Half of them settles nothing anybody can be told.

    The pair is what the fact is read off, so a figure by itself would look
    like an amount everywhere except the one place that checks.
    """
    banked: dict[str, object] = {
        "provider": "demo",
        "observed": Observation.LIVE,
        "observed_at": NOW,
        "synthetic": True,
        "order": "4471",
        "state": RefundState.PAID,
    }
    paid = RefundRecord(**banked, amount=Decimal("20.00"), currency="EUR")  # type: ignore[arg-type]
    assert paid.facts == {Fact.REFUND_STATE, Fact.REFUND_AMOUNT}
    assert RefundRecord(**banked).facts == {Fact.REFUND_STATE}  # type: ignore[arg-type]

    for broken in (
        {"amount": Decimal("20.00")},
        {"currency": "EUR"},
        {"amount": Decimal("-5"), "currency": "EUR"},
        {"amount": Decimal("NaN"), "currency": "EUR"},
        {"amount": Decimal("20.00"), "currency": "euros"},
    ):
        with pytest.raises(ValidationError):
            RefundRecord(**banked, **broken)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "broken",
    [
        {"provider": ""},
        {"provider": "   "},
        {"reference": ""},
        {"reference": " "},
        {"carrier": ""},
        {"tracking_reference": "  "},
    ],
    ids=lambda case: str(sorted(case)),
)
def test_a_value_that_is_only_whitespace_is_a_value_nobody_supplied(
    broken: dict[str, str],
) -> None:
    """Trimmed before it is measured, a rule wanting one character taking one space."""
    with pytest.raises(ValidationError):
        order(**broken)


@pytest.mark.parametrize(
    ("in_stock", "quantity"),
    [(False, 4), (True, 0), (True, -1), (False, -1)],
)
def test_a_count_and_an_answer_that_disagree_cannot_both_be_written_down(
    in_stock: bool, quantity: int
) -> None:
    """Whichever a reply were built from, the other sits there contradicting it.

    Which one gets used would then come down to the order somebody happened
    to read the fields in.
    """
    with pytest.raises(ValidationError):
        ProductRecord(
            provider="demo",
            observed=Observation.LIVE,
            observed_at=NOW,
            synthetic=True,
            reference="12",
            in_stock=in_stock,
            quantity=quantity,
        )


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


def test_a_row_is_named_by_whoever_supplied_it() -> None:
    """Two shops both numbering an order 4471 is the ordinary case.

    A bare number in the record of a decision poses a question rather than
    answering one, so the supplier and the sort of thing are carried with it.
    A refund borrows the number of the order it undoes, having none of its own.
    """
    assert order().cited_as == "demo:order:4471"
    assert (
        ProductRecord(
            provider="dummyjson",
            observed=Observation.LIVE,
            observed_at=NOW,
            synthetic=True,
            reference="13",
            in_stock=False,
            quantity=0,
        ).cited_as
        == "dummyjson:product:13"
    )
    assert (
        RefundRecord(
            provider="demo",
            observed=Observation.LIVE,
            observed_at=NOW,
            synthetic=True,
            order="4468",
            state=RefundState.PAID,
            amount=Decimal("49.99"),
            currency="EUR",
        ).cited_as
        == "demo:refund:4468"
    )


def test_reading_the_same_row_twice_does_not_look_like_a_change() -> None:
    """When we looked is outside the digest, and deliberately.

    Inside it, every reading of an untouched order would digest differently
    and the value would answer nothing. What an audit wants of it is the one
    question it can then answer: has this moved since the reply that rested
    on it went out.
    """
    later = order(observed_at=NOW + timedelta(hours=5), observed=Observation.CACHED)
    assert order().content_hash == later.content_hash

    moved = order(state=OrderState.DELIVERED)
    assert order().content_hash != moved.content_hash


def test_a_row_of_the_asked_for_kind_comes_back() -> None:
    assert answered_with(
        Found(record=order()), OrderRecord, "an order lookup", about="4471"
    ) == (order())


def test_no_row_is_not_a_complaint() -> None:
    """A reference matching nothing is an ordinary answer, not a fault."""
    assert (
        answered_with(NotAvailable(), OrderRecord, "an order lookup", about="4471")
        is None
    )


def test_a_row_of_the_wrong_kind_is_refused() -> None:
    """The annotation cannot enforce itself once the program is running.

    Nothing stops an order lookup answering with a catalogue entry. It would
    satisfy every check of shape while describing a different thing, and the
    reply would be about a product, filed against a question about an order.
    """
    catalogue = ProductRecord(
        provider="demo",
        observed=Observation.LIVE,
        observed_at=NOW,
        synthetic=True,
        reference="12",
        in_stock=True,
        quantity=3,
    )
    with pytest.raises(CommerceContractError, match="ProductRecord"):
        answered_with(
            Found(record=catalogue), OrderRecord, "an order lookup", about="4471"
        )


@pytest.mark.parametrize(
    "answer",
    [None, {"reference": "4471"}, "4471", 4471, [order()]],
    ids=["nothing", "a mapping", "a string", "a number", "a list"],
)
def test_an_answer_in_no_recognised_shape_is_refused(answer: object) -> None:
    """Read as a missing row, each of these blames the customer for our fault."""
    with pytest.raises(CommerceContractError):
        answered_with(answer, OrderRecord, "an order lookup", about="4471")


def test_the_right_kind_of_row_about_the_wrong_thing_is_refused() -> None:
    """Structurally perfect and about a different purchase.

    Every check of shape passes, and the row is then cited and rated as the
    evidence for a question it does not answer — one order's state filed
    against another order's enquiry. Keeping a question beside what was read
    for it is the rule this puts back at the edge, where the answer comes
    from somewhere that does not share it.
    """
    with pytest.raises(CommerceContractError, match="'4471'.*'4472'"):
        answered_with(
            Found(record=order(reference="4472")),
            OrderRecord,
            "an order lookup",
            about="4471",
        )
