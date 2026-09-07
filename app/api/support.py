"""The one endpoint a customer talks to."""

import logging
from typing import Annotated, get_args

from fastapi import APIRouter, Depends, Request

from app.agent.enquiry import Enquiry
from app.agent.knowledge import Locale
from app.api.deps import AuthUserDep, DBSessionDep
from app.models.user import User
from app.schemas.support import SupportMessage, SupportReply
from app.services.cases import DatabaseCases
from app.services.support import SupportAgent

_log = logging.getLogger(__name__)

router = APIRouter()

DEFAULT_LOCALE: Locale = "en"


def support_agent(request: Request) -> SupportAgent:
    """The agent built at startup.

    Fetched rather than assembled, so a request never pays for loading a
    corpus and cannot be served by a half-built one.
    """
    agent = request.app.state.support
    assert isinstance(agent, SupportAgent)
    return agent


SupportAgentDep = Annotated[SupportAgent, Depends(support_agent)]


def _language_of(customer: User) -> Locale:
    """The customer's language, or ours where theirs is not one we write in.

    The column holding this takes any five characters, and only our own API
    has ever constrained what goes in. A value from outside that set is our
    data being wrong, and refusing to answer would charge the customer for it.
    An enquiry will not accept one, so it is settled before one is built.
    """
    if customer.preferred_locale in get_args(Locale):
        stored: Locale = customer.preferred_locale  # type: ignore[assignment]
        return stored
    _log.warning(
        "user %s has locale %r, which nothing is written in; answering in %s",
        customer.id,
        customer.preferred_locale,
        DEFAULT_LOCALE,
    )
    return DEFAULT_LOCALE


@router.post("/messages")
async def answer_message(
    sent: SupportMessage,
    customer: AuthUserDep,
    agent: SupportAgentDep,
    db: DBSessionDep,
) -> SupportReply:
    """Answer a customer's question, or say why it is not being answered.

    Every outcome is a two hundred. Being asked for an order number, being
    passed to somebody, and being held for checking are decisions about a
    request that was understood, and the route in the body says which.
    """
    return await agent.answer(
        Enquiry(
            message=sent.message,
            locale=_language_of(customer),
            customer=customer.external_customer_id,
            order=sent.order_id,
            product=sent.product_reference,
        ),
        cases=DatabaseCases(db),
        customer=customer.id,
    )
