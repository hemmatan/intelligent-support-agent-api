"""A shop's records, for a shop that does not exist.

Every row here is invented, and every row says so. That is the whole of the
arrangement: the service can be shown answering an order question from end to
end, and nothing it answers with can be taken for a fact about a real
purchase.

Saying a row is made up is not saying it is old. These are read at the moment
they are asked for, so the reading is a live one; what it is a reading *of* is
fiction. The two travel as separate fields because they answer separate
questions, and anybody auditing a reply wants both.

Ownership lives in the key. A row is filed under the customer it belongs to,
so asking for somebody else's is asking for a key that is not there. Nothing
is fetched and then compared against a name, and there is no arrangement of
this file in which a record reaches somebody who should not see it.
"""

from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from typing import NamedTuple

from app.agent.commerce import (
    Found,
    NotAvailable,
    Observation,
    OrderLookup,
    OrderRecord,
    OrderState,
    ProductLookup,
    ProductRecord,
    RefundLookup,
    RefundRecord,
    RefundState,
)

PROVIDER = "demo"


def _utc_now() -> datetime:
    return datetime.now(UTC)


class _Order(NamedTuple):
    state: OrderState
    carrier: str | None = None
    tracking_reference: str | None = None
    # Counted from today, so a demonstration never quotes a delivery date
    # from last year. Negative for an order that has already landed.
    arrives_in_days: int | None = None


class _Refund(NamedTuple):
    state: RefundState
    amount: Decimal | None = None
    currency: str | None = None


class _Product(NamedTuple):
    in_stock: bool
    quantity: int


# Filed by the customer who placed it. The states cover what a reply has to
# deal with rather than what looks plausible: one that can answer everything
# asked of it, one that has not left the building and can therefore say only
# where it is, and one that is finished with.
_ORDERS: dict[tuple[int, str], _Order] = {
    (1, "4471"): _Order(OrderState.DISPATCHED, "DPD", "FR7729183044", 2),
    (1, "4472"): _Order(OrderState.PLACED),
    (1, "4468"): _Order(OrderState.DELIVERED, "Colissimo", "6A18820447715", -3),
    (2, "5510"): _Order(OrderState.IN_TRANSIT, "DPD", "FR7729184112", 1),
}

# Filed against the order it undoes and the customer who is owed the money.
_REFUNDS: dict[tuple[int, str], _Refund] = {
    (1, "4468"): _Refund(RefundState.PAID, Decimal("49.99"), "EUR"),
    (1, "4472"): _Refund(RefundState.NOT_REQUESTED),
    (2, "5510"): _Refund(RefundState.REQUESTED),
}

# A catalogue belongs to nobody, so it is filed under nobody.
_PRODUCTS: dict[str, _Product] = {
    "12": _Product(in_stock=True, quantity=34),
    "13": _Product(in_stock=False, quantity=0),
    "27": _Product(in_stock=True, quantity=2),
}


class DemoStorefront:
    """Answers every question the gateway asks, out of invented rows.

    All three of them, rather than orders alone. A catalogue borrowed from
    somebody else's mock service is no more real than one written here, so
    dividing the work between two providers would buy a composition step and
    no additional honesty. The seam that matters is the protocol: putting
    something that talks to a real shop behind it is an edit where this is
    constructed and nowhere else.

    The clock is handed in. Delivery dates are worked out from it, and a test
    that wants to know what somebody is told about a parcel arriving on
    Thursday should not have to wait until Tuesday to find out.
    """

    def __init__(self, *, now: Callable[[], datetime] = _utc_now) -> None:
        self._now = now

    def _provenance(self) -> dict[str, object]:
        """What every row says about where it came from and when it was read."""
        return {
            "provider": PROVIDER,
            "observed": Observation.LIVE,
            "observed_at": self._now(),
            "synthetic": True,
        }

    def _arriving(self, days: int | None) -> date | None:
        return None if days is None else self._now().date() + timedelta(days=days)

    async def order(self, reference: str, *, customer: int) -> OrderLookup:
        named = reference.strip()
        held = _ORDERS.get((customer, named))
        if held is None:
            return NotAvailable()
        return Found(
            record=OrderRecord(
                **self._provenance(),  # type: ignore[arg-type]
                reference=named,
                state=held.state,
                carrier=held.carrier,
                tracking_reference=held.tracking_reference,
                expected_delivery=self._arriving(held.arrives_in_days),
            )
        )

    async def refund(self, order: str, *, customer: int) -> RefundLookup:
        named = order.strip()
        held = _REFUNDS.get((customer, named))
        if held is None:
            return NotAvailable()
        return Found(
            record=RefundRecord(
                **self._provenance(),  # type: ignore[arg-type]
                order=named,
                state=held.state,
                amount=held.amount,
                currency=held.currency,
            )
        )

    async def product(self, reference: str) -> ProductLookup:
        named = reference.strip()
        held = _PRODUCTS.get(named)
        if held is None:
            return NotAvailable()
        return Found(
            record=ProductRecord(
                **self._provenance(),  # type: ignore[arg-type]
                reference=named,
                in_stock=held.in_stock,
                quantity=held.quantity,
            )
        )


def references_for(customer: int) -> dict[str, tuple[str, ...]]:
    """What this customer can ask about, read off the rows themselves.

    So that a demonstration is written from the data rather than beside it.
    The readme quoted an order reference in a format nothing here has ever
    used, which is the whole failure mode of documenting an example by hand:
    it was wrong the day it was typed and nothing was in a position to say so.
    """
    return {
        "orders": tuple(sorted(held for (whose, held) in _ORDERS if whose == customer)),
        "refunds": tuple(
            sorted(held for (whose, held) in _REFUNDS if whose == customer)
        ),
        "products": tuple(sorted(_PRODUCTS)),
    }
