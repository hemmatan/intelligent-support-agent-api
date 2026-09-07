"""Put a customer in the database who can reach the invented rows.

Signing up leaves the link to the shop's records empty, and nothing else sets
it: the design says a missing link is a data fault for staff to repair, not
something a customer can assert about themselves. Correct, and it left the
commerce path unreachable by anybody following the readme, because there was
no repair to perform and no staff account to perform it with.

This is that repair, done once, for a shop that does not exist. It refuses to
run anywhere the invented rows are refused, so the account it creates cannot
appear in a deployment serving real people.
"""

import asyncio
import sys

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.demo import references_for
from app.core.config import Environment, settings
from app.core.security import get_password_hash
from app.db.session import sessionmanager
from app.models.user import User

USERNAME = "demo-customer"
PASSWORD = "demo-password"  # noqa: S105 — development only; see the guard below
COMMERCE_ID = 1


class NotSomewhereToSeedError(RuntimeError):
    """This deployment does not admit invented data."""


def check_this_is_somewhere_to_do_it() -> None:
    """Both conditions, because they are two different mistakes.

    A production environment is the obvious one. A development environment
    with the invented rows switched off is the subtler one: the account would
    be created, the customer would be linked to a shop that answers nothing,
    and every order question would come back as a source that is not
    connected, which reads like a defect rather than a setting.
    """
    if settings.ENVIRONMENT is Environment.PRODUCTION:
        raise NotSomewhereToSeedError("refusing to create a demo account in production")
    if not settings.COMMERCE_DEMO_RECORDS:
        raise NotSomewhereToSeedError(
            "DORNASHOP_COMMERCE_DEMO_RECORDS is off, so the rows this account "
            "exists to reach would not be served"
        )


async def link_a_demo_customer(db: AsyncSession) -> User:
    """Create the account, or point an existing one at the same rows.

    Runnable twice. A seed that fails the second time is a seed somebody runs
    once and then works around.
    """
    check_this_is_somewhere_to_do_it()
    found = await db.execute(select(User).where(User.username == USERNAME))
    customer = found.scalar_one_or_none()
    if customer is None:
        customer = User(username=USERNAME, hashed_password="")
        db.add(customer)
    customer.hashed_password = await get_password_hash(PASSWORD)
    customer.preferred_locale = "en"
    customer.external_customer_id = COMMERCE_ID
    await db.commit()
    await db.refresh(customer)
    return customer


async def _run() -> None:
    async with sessionmanager.session() as db:
        customer = await link_a_demo_customer(db)
    reachable = references_for(COMMERCE_ID)
    print(f"username: {USERNAME}")
    print(f"password: {PASSWORD}")
    print(f"linked to commerce customer {customer.external_customer_id}")
    for kind, held in reachable.items():
        print(f"{kind}: {', '.join(held) or 'none'}")


def main() -> None:
    try:
        asyncio.run(_run())
    except NotSomewhereToSeedError as refused:
        print(refused, file=sys.stderr)
        raise SystemExit(1) from refused


if __name__ == "__main__":
    main()
