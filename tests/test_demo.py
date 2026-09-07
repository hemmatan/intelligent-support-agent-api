"""Invented rows, and the promises they still have to keep."""

from datetime import UTC, datetime, timedelta

import pytest

from app.agent.commerce import (
    CommerceGateway,
    Found,
    NotAvailable,
    Observation,
    freshness_of,
)
from app.agent.demo import _ORDERS, _PRODUCTS, _REFUNDS, DemoStorefront
from app.agent.facts import Fact
from app.agent.reliability import ReliabilityLevel

NOW = datetime(2026, 9, 7, 12, 0, tzinfo=UTC)


@pytest.fixture
def shop() -> DemoStorefront:
    return DemoStorefront(now=lambda: NOW)


def test_one_provider_answers_everything_the_gateway_asks() -> None:
    """Which is why nothing composes two of them yet.

    A catalogue taken from somebody else's mock service would be exactly as
    invented as this one, so splitting the work in two would buy a joining
    step and no more truthfulness. When something answers about a real shop,
    the joining step arrives with it.
    """
    assert isinstance(DemoStorefront(), CommerceGateway)


@pytest.mark.asyncio
async def test_a_stranger_s_order_reads_the_same_as_no_order(
    shop: DemoStorefront,
) -> None:
    """The whole of the protection, and it is structural rather than careful.

    A row is filed under whoever placed it, so somebody else's is a key that
    is not there. Nothing is read and then compared against a name, which
    means no ordering of this code leaves a stranger's record in hand while
    the decision to discard it is being taken.
    """
    theirs = await shop.order("4471", customer=1)
    assert isinstance(theirs, Found)

    assert await shop.order("4471", customer=2) == NotAvailable()
    assert await shop.order("9999", customer=1) == NotAvailable()
    assert await shop.refund("4468", customer=2) == NotAvailable()


@pytest.mark.asyncio
async def test_nothing_here_pretends_to_be_a_real_purchase(
    shop: DemoStorefront,
) -> None:
    """Every row, not the ones a test happened to reach for.

    Marked read-just-now as well, which is a separate claim and a true one:
    the reading is live, and what it is a reading of is fiction. An audit
    wants to be able to tell those apart.
    """
    looked_up = [
        await shop.order("4471", customer=1),
        await shop.order("4472", customer=1),
        await shop.refund("4468", customer=1),
        await shop.product("12"),
    ]
    for outcome in looked_up:
        assert isinstance(outcome, Found)
        assert outcome.record.synthetic is True
        assert outcome.record.provider == "demo"
        assert outcome.record.observed is Observation.LIVE


@pytest.mark.asyncio
async def test_every_invented_row_is_one_the_rules_would_accept(
    shop: DemoStorefront,
) -> None:
    """Fixtures are refused by the same checks a real provider's answers are.

    A demonstration built on rows that could not have come from anywhere else
    proves the path works for data the path would reject, which is a
    demonstration of nothing.
    """
    for customer, reference in _ORDERS:
        assert isinstance(await shop.order(reference, customer=customer), Found)
    for customer, reference in _REFUNDS:
        assert isinstance(await shop.refund(reference, customer=customer), Found)
    for reference in _PRODUCTS:
        assert isinstance(await shop.product(reference), Found)


@pytest.mark.asyncio
async def test_a_parcel_still_in_the_building_says_only_where_it_is(
    shop: DemoStorefront,
) -> None:
    """Placed and nothing more, so there is no date and nothing to track.

    Asked when it lands, this is half an answer, and half an answer is the
    rung that goes to a colleague rather than out.
    """
    placed = await shop.order("4472", customer=1)
    assert isinstance(placed, Found)
    assert placed.record.facts == {Fact.ORDER_STATE}

    gone = await shop.order("4471", customer=1)
    assert isinstance(gone, Found)
    assert gone.record.facts == {
        Fact.ORDER_STATE,
        Fact.TRACKING_REFERENCE,
        Fact.DELIVERY_ESTIMATE,
    }


@pytest.mark.asyncio
async def test_a_delivery_date_is_counted_from_the_clock_it_was_given() -> None:
    """Otherwise a demonstration quotes dates from whenever it was written."""
    later = DemoStorefront(now=lambda: NOW + timedelta(days=30))
    first = await DemoStorefront(now=lambda: NOW).order("4471", customer=1)
    second = await later.order("4471", customer=1)
    assert isinstance(first, Found) and isinstance(second, Found)
    assert first.record.expected_delivery is not None
    assert second.record.expected_delivery is not None
    assert second.record.expected_delivery > first.record.expected_delivery


@pytest.mark.asyncio
async def test_a_row_read_just_now_is_fresh_enough_to_go_out(
    shop: DemoStorefront,
) -> None:
    """The factor every commerce profile requires and nothing has produced."""
    found = await shop.order("4471", customer=1)
    assert isinstance(found, Found)
    assert (
        freshness_of(
            found.record,
            now=NOW,
            ttl=timedelta(minutes=15),
            readable_for=timedelta(hours=6),
        )
        is ReliabilityLevel.READY
    )
