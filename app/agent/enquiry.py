"""What was asked, and everything known about whoever asked it.

The values travel, not merely the fact that they exist. Recording that an
order number was supplied is enough to decide whether to ask for one, and
useless to the lookup that has to go and find that order; a later step sent
back for the value is a step where the value can differ from the one this
decision was taken on.

It is also the boundary. What arrives here came from a request body and a
database column, and neither is bound by the types this package is written
in — a locale column five characters wide will hold anything five characters
long. Whatever is built past this point can rely on what it says.
"""

from dataclasses import dataclass
from typing import get_args

from app.agent.knowledge import Locale
from app.agent.profiles import Input

MAX_MESSAGE = 4000


class NotSomethingToActOnError(ValueError):
    """The request cannot be read as a question from a customer."""


@dataclass(frozen=True)
class Enquiry:
    """One customer's question, with the context it is answered against."""

    message: str
    locale: Locale = "en"
    customer: int | None = None
    order: str | None = None
    product: str | None = None

    def __post_init__(self) -> None:
        if not self.message.strip():
            raise NotSomethingToActOnError("an enquiry with nothing in it")
        if len(self.message) > MAX_MESSAGE:
            raise NotSomethingToActOnError(
                f"an enquiry of {len(self.message)} characters, over {MAX_MESSAGE}"
            )
        if self.locale not in get_args(Locale):
            raise NotSomethingToActOnError(
                f"nothing is written in {self.locale!r}, so nothing can be said in it"
            )
        # A form that posts every field sends empty ones too, and an empty
        # order number is not an order number. Left alone it counted as
        # supplied, so a request stopped asking for what it needed and went
        # looking for a record identified by nothing.
        for named in ("order", "product"):
            given = getattr(self, named)
            if given is not None:
                object.__setattr__(self, named, given.strip() or None)

    @property
    def known(self) -> frozenset[Input]:
        """Which of the things a profile can require are actually to hand.

        Derived rather than passed alongside. Supplied as its own argument, it
        could say an order number was present while the field holding it was
        empty, and the request would stop being asked for something it had or
        proceed without something it needed.
        """
        supplied = {
            Input.COMMERCE_ACCOUNT: self.customer,
            Input.ORDER_ID: self.order,
            Input.PRODUCT_REFERENCE: self.product,
        }
        return frozenset(need for need, value in supplied.items() if value is not None)
