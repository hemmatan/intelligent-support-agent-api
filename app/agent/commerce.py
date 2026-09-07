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
from typing import (
    Annotated,
    ClassVar,
    Generic,
    Protocol,
    TypeAlias,
    TypeVar,
    runtime_checkable,
)

from pydantic import (
    BaseModel,
    ConfigDict,
    NonNegativeInt,
    StringConstraints,
    field_validator,
    model_validator,
)

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


_STRICT = ConfigDict(extra="forbid", frozen=True, strict=True)

# Copied off a receipt or a parcel, so surrounding space is a keystroke and
# not part of the value. Trimmed before it is measured, because a rule
# demanding one character is satisfied by one space.
Named = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class Record(BaseModel):
    """What anything out of the shop's records has to say about itself.

    Provenance travels with the value. A citation naming a reference and a
    digest says which record was read; it says nothing about who was asked,
    when they answered, or whether the answer describes a real shop at all.
    All three bear on how far the record may carry a reply, so they are on the
    record and not reassembled afterwards from whatever the caller remembers.

    Validated rather than merely annotated. These are built from what an
    outside service said, and a type hint stops nobody: a quantity of minus
    four and a blank provider went in happily and came out ready to be cited.
    """

    model_config = _STRICT

    provider: Named
    observed: Observation
    observed_at: datetime
    synthetic: bool

    ABOUT: ClassVar[frozenset[Fact]] = frozenset()
    """Every question this kind of record is the authority on.

    Wider than any one of them settles. An order is where somebody looks for
    a tracking reference whether or not this particular order has reached a
    courier, and the question is still an order question when the answer is
    that we cannot give one. Declared so that the wording recognising a
    request and the wording recognising what it asks for can be held against
    each other, instead of drifting until a request nobody could answer looks
    exactly like one nobody asked.
    """

    @field_validator("observed_at")
    @classmethod
    def check_the_reading_is_placed_in_time(cls, when: datetime) -> datetime:
        """An hour with no offset is an hour in nobody's day.

        The moment is ours to record — we know when we asked — so a reading
        arriving without one is a fault at the boundary and not a property of
        the record. Refusing here keeps the arithmetic downstream comparing
        two things of the same kind.
        """
        if when.tzinfo is None or when.tzinfo.utcoffset(when) is None:
            raise ValueError("observed_at carries no offset, so names no moment")
        return when


class OrderRecord(Record):
    """One order, as the shop holds it."""

    ABOUT: ClassVar[frozenset[Fact]] = frozenset(
        {Fact.ORDER_STATE, Fact.TRACKING_REFERENCE, Fact.DELIVERY_ESTIMATE}
    )

    reference: Named
    state: OrderState
    carrier: Named | None = None
    tracking_reference: Named | None = None
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


class RefundRecord(Record):
    """What became of a refund against one order."""

    ABOUT: ClassVar[frozenset[Fact]] = frozenset(
        {Fact.REFUND_STATE, Fact.REFUND_AMOUNT, Fact.REFUND_TIMING}
    )

    order: Named
    state: RefundState
    amount: Decimal | None = None
    currency: Annotated[str, StringConstraints(pattern=r"^[A-Z]{3}$")] | None = None

    @model_validator(mode="after")
    def check_the_sum_is_one_somebody_could_be_paid(self) -> "RefundRecord":
        """A figure and its units arrive together or neither arrives.

        Half of them settles nothing a customer can be told, and the pair is
        what the fact below is read off, so a sum on its own would look like
        an amount to everything except the one place that checks.

        Money that is not a number, or is a number below zero, is a refund
        running the wrong way. Neither is a smaller version of the right
        answer, and rating either would put it in front of somebody.
        """
        if (self.amount is None) != (self.currency is None):
            raise ValueError("an amount needs its currency, and a currency its amount")
        if self.amount is not None and not self.amount.is_finite():
            raise ValueError(f"{self.amount} is not a sum of money")
        if self.amount is not None and self.amount < 0:
            raise ValueError(f"{self.amount} is a refund owed the other way")
        return self

    @property
    def facts(self) -> frozenset[Fact]:
        """A sum with no currency beside it is not an amount anybody can be told."""
        settled = {Fact.REFUND_STATE}
        if self.amount is not None and self.currency is not None:
            settled.add(Fact.REFUND_AMOUNT)
        return frozenset(settled)


class ProductRecord(Record):
    """One catalogue entry, so far as stock goes."""

    ABOUT: ClassVar[frozenset[Fact]] = frozenset({Fact.STOCK_AVAILABILITY})

    reference: Named
    in_stock: bool
    quantity: NonNegativeInt | None = None

    @model_validator(mode="after")
    def check_the_count_and_the_answer_agree(self) -> "ProductRecord":
        """Four of them in the warehouse and none of them for sale is two answers.

        Whichever a reply were built from, the other is sitting in the same
        record contradicting it, and which one gets used is then a question
        about the order the fields happen to be read in.
        """
        if self.quantity is not None and self.in_stock != (self.quantity > 0):
            raise ValueError(
                f"in_stock is {self.in_stock} beside a quantity of {self.quantity}"
            )
        return self

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

    What a reading is on its own the record settles: it cannot be built
    without a moment, and the moment cannot be one with no offset. What is
    left here is what only shows up between two values. A reading dated after
    the clock it is measured against is two clocks disagreeing, and a record
    of something that has not happened yet supports no reasoning at all. A
    window of legibility shorter than the window of usefulness is a scale
    running backwards, and it is the caller's mistake rather than a poor
    rating, so it is raised instead of returned.
    """
    if readable_for < ttl:
        raise ValueError(f"readable for {readable_for}, which is inside its {ttl} life")
    if now.tzinfo is None or now.tzinfo.utcoffset(now) is None:
        raise ValueError("the moment to measure against carries no offset")

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
