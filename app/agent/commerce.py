"""The shop's own records, and the terms on which this service may ask.

The interface and the vocabulary only. What answers is a deployment decision,
and none of the answers is allowed to be invented: an adapter reports what the
shop's records say and never what they ought to say.

Ownership is part of the question instead of a check on the answer. Asking for
an order belonging to somebody, rather than asking for an order and then
reading whose it is, means a record they may not see is never held here at
all — not in memory, not in a log, not in a reply assembled before the check
ran.
"""

import inspect
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from decimal import Decimal
from enum import StrEnum
from typing import Generic, Protocol, TypeAlias, TypeVar, runtime_checkable

from app.agent.facts import Fact
from app.agent.reliability import ReliabilityLevel


class CommerceError(RuntimeError):
    """The shop's records could not be consulted."""


class CommerceUnavailableError(CommerceError):
    """A failure that might not happen next time.

    A timeout, a gateway having a moment, a provider under load. The request
    waits for somebody here: the customer asked a fair question and our own
    records were the thing that did not answer.
    """


class CommerceMisconfiguredError(CommerceError):
    """A failure that will happen every time.

    A refused key, a base address pointing nowhere. Retrying a typo is how a
    deployment runs for a month answering every order question with an
    apology while each health check reports that all is well.
    """


class Observation(StrEnum):
    """How a record came to be in front of us."""

    LIVE = "live"
    """Read from the provider in the course of answering this request."""

    CACHED = "cached"
    """Kept from an earlier reading, and therefore only as true as its age."""


class OrderState(StrEnum):
    """Where an order has got to."""

    PLACED = "placed"
    DISPATCHED = "dispatched"
    IN_TRANSIT = "in_transit"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"


class RefundState(StrEnum):
    """Where somebody's money has got to."""

    NOT_REQUESTED = "not_requested"
    REQUESTED = "requested"
    APPROVED = "approved"
    PAID = "paid"
    DECLINED = "declined"


@dataclass(frozen=True, kw_only=True)
class Record:
    """What anything out of the shop's records has to say about itself.

    Provenance travels with the value. A citation naming a reference and a
    digest says which record was read; it says nothing about who was asked,
    when they answered, or whether the answer describes a real shop at all.
    All three bear on how far the record may carry a reply, so they are on the
    record and not reassembled afterwards from whatever the caller remembers.
    """

    provider: str
    observed: Observation
    observed_at: datetime | None
    synthetic: bool


@dataclass(frozen=True, kw_only=True)
class OrderRecord(Record):
    """One order, as the shop holds it."""

    reference: str
    state: OrderState
    carrier: str | None = None
    tracking_reference: str | None = None
    expected_delivery: date | None = None

    @property
    def facts(self) -> frozenset[Fact]:
        """What this particular record settles.

        Read off the fields that are filled, and not off the kind of thing it
        is. A policy of a given kind always states its claims, so what it can
        answer is a property of the kind. An order that has not reached a
        carrier holds no tracking reference, and cannot say where the parcel
        is however ordinary a question that is to ask of an order.
        """
        settled = {Fact.ORDER_STATE}
        if self.tracking_reference is not None:
            settled.add(Fact.TRACKING_REFERENCE)
        if self.expected_delivery is not None:
            settled.add(Fact.DELIVERY_ESTIMATE)
        return frozenset(settled)


@dataclass(frozen=True, kw_only=True)
class RefundRecord(Record):
    """What became of a refund against one order."""

    order: str
    state: RefundState
    amount: Decimal | None = None
    currency: str | None = None

    @property
    def facts(self) -> frozenset[Fact]:
        """A sum with no currency beside it is not an amount anybody can be told."""
        settled = {Fact.REFUND_STATE}
        if self.amount is not None and self.currency is not None:
            settled.add(Fact.REFUND_AMOUNT)
        return frozenset(settled)


@dataclass(frozen=True, kw_only=True)
class ProductRecord(Record):
    """One catalogue entry, so far as stock goes."""

    reference: str
    in_stock: bool
    quantity: int | None = None

    @property
    def facts(self) -> frozenset[Fact]:
        return frozenset({Fact.STOCK_AVAILABILITY})


_R = TypeVar("_R", bound=Record)


@dataclass(frozen=True)
class Found(Generic[_R]):
    """The record that was asked for."""

    record: _R


@dataclass(frozen=True)
class NotAvailable:
    """No record with that reference, or none this customer may be shown.

    One outcome covering both, because the difference is not obtainable. The
    question named the customer, so nothing is all there is to come back with.

    Two outcomes would be worse than untidy. Asked twice — once as somebody
    who owns nothing, once as somebody who owns one thing — a pair of them
    reports which references exist, which is a catalogue of other people's
    orders assembled a question at a time. There is nothing in here to tell
    them apart with.
    """


Lookup: TypeAlias = Found[_R] | NotAvailable
OrderLookup: TypeAlias = Lookup[OrderRecord]
RefundLookup: TypeAlias = Lookup[RefundRecord]
ProductLookup: TypeAlias = Lookup[ProductRecord]


def freshness_of(
    record: Record, *, now: datetime, ttl: timedelta, readable_for: timedelta
) -> ReliabilityLevel:
    """How far the age of a record lets it carry an answer.

    The clock arrives as an argument. Read from the machine, every case below
    would be a test that stops meaning what it meant as the afternoon wears
    on, and the interesting rungs are the ones some hours away.

    A reading taken now is the only thing that earns the top rung, and it only
    earns it while it is recent: a record labelled live and dated last Tuesday
    is a contradiction, and the label is the half that can be wrong, so age
    settles it. That keeps a flag from laundering something stale.

    No timestamp, one without an offset, or one dated after the moment it was
    read: all refused. The last is somebody else's clock disagreeing with
    ours, and a record describing a state of affairs that has not arrived yet
    cannot be reasoned about at all.
    """
    if record.observed_at is None or record.observed_at.tzinfo is None:
        return ReliabilityLevel.UNUSABLE

    age = now - record.observed_at
    if age < timedelta(0):
        return ReliabilityLevel.UNUSABLE
    if record.observed is Observation.LIVE and age <= ttl:
        return ReliabilityLevel.READY
    if age <= ttl:
        return ReliabilityLevel.ACCEPTABLE
    if age <= readable_for:
        return ReliabilityLevel.REVIEW_ONLY
    return ReliabilityLevel.UNUSABLE


@runtime_checkable
class CommerceGateway(Protocol):
    """Anywhere the shop's own records can be asked about.

    A protocol, so that a timeout, an outage and an answer in the wrong shape
    are arrangements a test can make. Waiting for the real provider to have
    one of those days is not a plan for finding out what happens.
    """

    async def order(self, reference: str, *, customer: int) -> OrderLookup:
        """The order with this reference belonging to this customer.

        Both halves are the question. Asked for an order and afterwards told
        whose it was, this would have to hold a stranger's record for however
        long it took to decide against using it.

        Raises CommerceUnavailableError where a later attempt might do better,
        and CommerceMisconfiguredError where none ever will.
        """
        ...

    async def refund(self, order: str, *, customer: int) -> RefundLookup:
        """What became of the refund on that customer's order."""
        ...

    async def product(self, reference: str) -> ProductLookup:
        """One catalogue entry.

        Nobody is named. A catalogue belongs to no customer, and requiring one
        here would suggest a permission that is not being enforced.
        """
        ...


def names_a_customer(question: str) -> bool:
    """Whether asking this requires saying on whose behalf.

    Read off the protocol rather than listed beside it. A list is a second
    place the answer is written down, free to go on saying a lookup is
    owner-scoped after somebody has quietly widened it.
    """
    method = getattr(CommerceGateway, question)
    return "customer" in inspect.signature(method).parameters
