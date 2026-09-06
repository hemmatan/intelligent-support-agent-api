"""Writing down what was decided, before the customer is told it.

A reply that reached somebody with no record of why is the one thing that
cannot be looked into later, and the interesting questions about a support
system are all asked later.

It is also the difference between escalating and saying you have. Telling a
customer their message has gone to a colleague, while putting it nowhere a
colleague will look, is a false statement dressed as a routing decision. The
record is the queue: rows carrying a route that needs a person, with nothing
closing them yet.
"""

import uuid
from typing import Protocol

from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.enquiry import Enquiry
from app.agent.reliability import Route
from app.models.support import SupportCase
from app.schemas.support import Answer, SupportReply

# The routes that leave work behind for somebody here.
NEEDS_SOMEBODY = frozenset({Route.HUMAN_ESCALATION, Route.INTERNAL_REVIEW})


def new_reference() -> str:
    """A name for a case, minted before the reply that quotes it is built."""
    return str(uuid.uuid4())


class CaseRecorder(Protocol):
    """Anywhere a decision can be written down.

    A protocol because the endpoint tests should be able to watch what gets
    recorded without a database, and because whether this is a table or a
    ticketing system is a deployment question rather than an agent one.
    """

    async def record(
        self, reference: str, enquiry: Enquiry, reply: SupportReply, customer: int
    ) -> None:
        """Persist one decision. Raises rather than losing it quietly."""
        ...


class DatabaseCases:
    """Rows in the support table, which is both the audit trail and the queue."""

    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    async def record(
        self, reference: str, enquiry: Enquiry, reply: SupportReply, customer: int
    ) -> None:
        answered = isinstance(reply, Answer)
        stated = reply.model_dump(mode="json")
        self._db.add(
            SupportCase(
                reference=reference,
                user_id=customer,
                message=enquiry.message,
                locale=enquiry.locale,
                route=str(reply.route),
                # A clarification carries one; everything else carries a list.
                reasons=stated.get("reasons") or [stated["reason"]]
                if not answered
                else [],
                intent=stated.get("intent"),
                reply=stated.get("reply"),
                citations=stated.get("citations") or [],
                # Kept as sent. Approved sentences get edited, and this has to
                # say what went out rather than what would go out today.
                wording=(
                    stated["wording"]
                    if answered
                    else [stated["wording"]]
                    if stated.get("wording")
                    else []
                ),
                reliability=stated.get("reliability"),
            )
        )
        await self._db.commit()
