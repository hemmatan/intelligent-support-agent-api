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
from collections.abc import Sequence
from typing import Protocol

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.enquiry import Enquiry
from app.agent.reliability import Route
from app.models.support import SupportCase
from app.models.user import utc_now
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
                order_id=enquiry.order,
                product_reference=enquiry.product,
                external_customer_id=enquiry.customer,
                route=str(reply.route),
                # A clarification carries one; everything else carries a list.
                reasons=stated.get("reasons") or [stated["reason"]]
                if not answered
                else [],
                intent=stated.get("intent"),
                # Whichever field carried the words: an answer has a reply,
                # everything else has a message, and every outcome sends one.
                sent=stated["reply"] if answered else stated["message"],
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


class CaseNotFoundError(LookupError):
    """No case with that reference, or none this person may act on."""


class CaseAlreadyClosedError(RuntimeError):
    """Somebody has already dealt with it.

    Closing twice would overwrite the first person's account of what they did,
    and the second entry is the one that survives — which is the wrong way
    round for a record of who handled what.
    """


class CaseDesk:
    """Reading and working the queue, as a member of staff.

    Only cases that stopped short of an answer appear. A request that was
    answered is a record; a request that was not is somebody's work.
    """

    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    async def waiting(self, limit: int = 50) -> Sequence[SupportCase]:
        """Open cases, oldest first, because that is the order they aged in."""
        found = await self._db.execute(
            select(SupportCase)
            .where(
                SupportCase.route.in_([str(route) for route in NEEDS_SOMEBODY]),
                SupportCase.closed_at.is_(None),
            )
            .order_by(SupportCase.created_at)
            .limit(limit)
        )
        return found.scalars().all()

    async def _open(self, reference: str) -> SupportCase:
        found = await self._db.execute(
            select(SupportCase).where(SupportCase.reference == reference)
        )
        case = found.scalar_one_or_none()
        if case is None or case.route not in {str(route) for route in NEEDS_SOMEBODY}:
            # An answered request is not somebody's to claim, and saying so
            # separately would tell a caller which references exist.
            raise CaseNotFoundError(reference)
        if case.closed_at is not None:
            raise CaseAlreadyClosedError(reference)
        return case

    async def claim(self, reference: str, agent: int) -> SupportCase:
        """Put a name against it, so two people do not both start."""
        case = await self._open(reference)
        case.assigned_to = agent
        await self._db.commit()
        await self._db.refresh(case)
        return case

    async def resolve(self, reference: str, agent: int, note: str) -> SupportCase:
        """Close it, recording who and what.

        Claiming first is not required. Somebody who deals with a case
        immediately should not have to perform a two-step to say so, and the
        record ends up naming them either way.
        """
        case = await self._open(reference)
        case.assigned_to = case.assigned_to or agent
        case.resolution = note
        case.closed_at = utc_now()
        await self._db.commit()
        await self._db.refresh(case)
        return case
