"""The one account that can reach the invented rows."""

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.commerce import Found
from app.agent.demo import DemoStorefront, references_for
from app.core.config import Environment, settings
from app.models.user import User
from app.seed import (
    COMMERCE_ID,
    USERNAME,
    NotSomewhereToSeedError,
    link_a_demo_customer,
)


@pytest.mark.asyncio
async def test_the_account_is_linked_to_the_rows_it_is_for(
    session: AsyncSession,
) -> None:
    """Signing up leaves this empty and nothing else fills it in.

    Which is right — a customer must not be able to claim a link to somebody
    else's records — and it left the whole commerce path unreachable by
    anybody without a database client.
    """
    customer = await link_a_demo_customer(session)
    assert customer.external_customer_id == COMMERCE_ID
    assert customer.hashed_password


@pytest.mark.asyncio
async def test_running_it_twice_is_not_an_error(session: AsyncSession) -> None:
    """A seed that fails on the second run is one people work around."""
    first = await link_a_demo_customer(session)
    second = await link_a_demo_customer(session)
    assert first.id == second.id

    everyone = await session.execute(select(User).where(User.username == USERNAME))
    assert len(everyone.scalars().all()) == 1


@pytest.mark.asyncio
async def test_it_refuses_where_invented_data_is_refused(
    session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two different mistakes, and the second is the easy one to miss.

    A development machine with the rows switched off would get the account,
    the link, and an unconnected source behind it — which reads as a defect
    rather than as the setting it is.
    """
    monkeypatch.setattr(settings, "ENVIRONMENT", Environment.PRODUCTION)
    with pytest.raises(NotSomewhereToSeedError, match="in production"):
        await link_a_demo_customer(session)

    monkeypatch.setattr(settings, "ENVIRONMENT", Environment.DEVELOPMENT)
    monkeypatch.setattr(settings, "COMMERCE_DEMO_RECORDS", False)
    with pytest.raises(NotSomewhereToSeedError, match="would not be served"):
        await link_a_demo_customer(session)


@pytest.mark.asyncio
async def test_every_reference_it_advertises_is_one_the_shop_answers() -> None:
    """The check the readme needed and did not have.

    It quoted an order in a shape nothing here has ever used, and no amount
    of reading either file would have caught it: the example was written
    beside the data instead of out of it. Anything documented as reachable is
    now fetched and has to come back.
    """
    shop = DemoStorefront()
    reachable = references_for(COMMERCE_ID)
    assert reachable["orders"] and reachable["products"]

    for held in reachable["orders"]:
        assert isinstance(await shop.order(held, customer=COMMERCE_ID), Found)
    for held in reachable["refunds"]:
        assert isinstance(await shop.refund(held, customer=COMMERCE_ID), Found)
    for held in reachable["products"]:
        assert isinstance(await shop.product(held), Found)
