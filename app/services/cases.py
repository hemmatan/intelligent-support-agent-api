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

from sqlalchemy import func, select, update
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


class CaseAlreadyTakenError(RuntimeError):
    """Somebody else has it.

    Two people working the same case is the thing claiming exists to stop, and
    a claim that overwrites the previous one stops nothing.
    """


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

    Claiming and closing are single statements with their conditions in the
    WHERE clause, so the database settles who got there first. Reading a row,
    deciding in Python and writing it back leaves a window between the decision
    and the write, and two people pressing at once both come through it.
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

    async def _workable(self, reference: str) -> SupportCase:
        """The case, or the reason it could not be acted on.

        Consulted only after a conditional write has already failed, to say
        which of the reasons it was. It never decides anything.
        """
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

    async def _reload(self, reference: str) -> SupportCase:
        found = await self._db.execute(
            select(SupportCase).where(SupportCase.reference == reference)
        )
        return found.scalar_one()

    async def claim(self, reference: str, agent: int) -> SupportCase:
        """Take a case, if nobody else already has.

        One statement: it takes effect only while the case is unclaimed, open,
        and somebody's to work. A second person arriving changes no rows and is
        told so rather than quietly replacing the first.
        """
        taken = await self._db.execute(
            update(SupportCase)
            .where(
                SupportCase.reference == reference,
                SupportCase.route.in_([str(route) for route in NEEDS_SOMEBODY]),
                SupportCase.closed_at.is_(None),
                SupportCase.assigned_to.is_(None),
            )
            .values(assigned_to=agent)
        )
        if taken.rowcount == 0:
            await self._db.rollback()
            case = await self._workable(reference)
            raise CaseAlreadyTakenError(f"{reference} is with {case.assigned_to}")
        await self._db.commit()
        return await self._reload(reference)

    async def resolve(self, reference: str, agent: int, note: str) -> SupportCase:
        """Close it, recording who finished it and what they did.

        Claiming first is not required: somebody who deals with a case on
        sight should not perform a two-step to say so. Whoever writes the note
        is recorded as having done the work, which is not necessarily whoever
        took it on — a case picked up by one person and finished by another
        would otherwise credit the wrong one.
        """
        closed = await self._db.execute(
            update(SupportCase)
            .where(
                SupportCase.reference == reference,
                SupportCase.route.in_([str(route) for route in NEEDS_SOMEBODY]),
                SupportCase.closed_at.is_(None),
            )
            .values(
                assigned_to=func.coalesce(SupportCase.assigned_to, agent),
                resolved_by=agent,
                resolution=note,
                closed_at=utc_now(),
            )
        )
        if closed.rowcount == 0:
            await self._db.rollback()
            await self._workable(reference)
            raise CaseAlreadyClosedError(reference)
        await self._db.commit()
        return await self._reload(reference)
