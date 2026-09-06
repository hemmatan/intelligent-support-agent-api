"""What a member of staff sees when they pick up a case.

Everything the decision was taken on, in one shape. Somebody working the queue
should not have to ask the customer for a detail the customer already gave, or
resolve a hash to find out what they were told.

Deliberately not the customer's shape. This one carries the identifiers, the
ratings and the evidence; a customer gets the sentence and the reason.
"""

from datetime import datetime
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

_EXACT = ConfigDict(extra="forbid", from_attributes=True)

Note = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)
]


class Case(BaseModel):
    """One request, everything it was decided on, and where it stands."""

    model_config = _EXACT

    reference: str
    created_at: datetime

    # What was asked, and what the customer had already told us.
    message: str
    locale: str
    order_id: str | None
    product_reference: str | None
    external_customer_id: int | None
    user_id: int

    # What was decided, and on what.
    route: str
    reasons: list[str]
    intent: str | None
    sent: str
    citations: list[dict[str, Any]]
    wording: list[str]
    reliability: dict[str, Any] | None

    # Where it stands.
    assigned_to: int | None
    closed_at: datetime | None
    resolution: str | None


class Resolution(BaseModel):
    """What the person handling it did about it."""

    model_config = ConfigDict(extra="forbid")

    note: Note = Field(
        description=(
            "What was done. Recorded against the case, not sent to the customer."
        )
    )
