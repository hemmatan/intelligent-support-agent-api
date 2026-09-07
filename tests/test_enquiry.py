"""What was asked, what is known about the asker, and what is refused outright."""

import pytest

from app.agent.enquiry import MAX_MESSAGE, Enquiry, NotSomethingToActOnError
from app.agent.intent import Intent
from app.agent.profiles import Input
from app.agent.reasons import ClarificationReason
from app.agent.triage import Clarify, Proceed, triage


def test_which_inputs_are_to_hand_is_read_off_the_values() -> None:
    """Not stated separately, where it could say one thing and hold another."""
    enquiry = Enquiry(message="Where is my order?", customer=7, order="ORD-4471")
    assert enquiry.known == {Input.COMMERCE_ACCOUNT, Input.ORDER_ID}
    assert Input.PRODUCT_REFERENCE not in enquiry.known


def test_nothing_supplied_is_nothing_known() -> None:
    assert Enquiry(message="Where is my order?").known == frozenset()


@pytest.mark.asyncio
async def test_the_value_survives_as_far_as_the_lookup_that_needs_it() -> None:
    """The whole reason these travel rather than being counted and dropped.

    Triage decided on an order number and passed on the fact that one existed.
    Whatever goes to find that order needs the number, and going back for it
    is a second reading that can differ from the one this was decided on.
    """
    placed = await triage(
        Enquiry(message="Where is my order?", customer=7, order="ORD-4471")
    )
    assert isinstance(placed, Proceed)
    assert placed.intent is Intent.ORDER_STATUS
    assert placed.enquiry.order == "ORD-4471"
    assert placed.enquiry.customer == 7


@pytest.mark.parametrize("message", ["", "   ", "\n\t "], ids=["empty", "spaces", "ws"])
def test_an_enquiry_with_nothing_in_it_is_not_one(message: str) -> None:
    with pytest.raises(NotSomethingToActOnError, match="nothing in it"):
        Enquiry(message=message)


def test_a_message_past_the_limit_is_refused_here() -> None:
    """Before retrieval tokenises it and a model is asked to read it."""
    with pytest.raises(NotSomethingToActOnError, match="over"):
        Enquiry(message="a" * (MAX_MESSAGE + 1))
    assert Enquiry(message="a" * MAX_MESSAGE).message


def test_a_language_nothing_is_written_in_stops_at_the_boundary() -> None:
    """The column holding this is five characters wide and believes anything.

    A locale the corpus has never seen would reach retrieval, find no index,
    and be answered or escalated on the strength of that. It is caught where
    the value enters instead.
    """
    with pytest.raises(NotSomethingToActOnError, match="nothing is written in"):
        Enquiry(message="Wo ist meine Bestellung?", locale="de")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "blank", ["", "   ", "\t", "\n "], ids=["empty", "spaces", "tab", "newline"]
)
def test_an_identifier_of_whitespace_is_not_an_identifier(blank: str) -> None:
    """A form posting every field sends the empty ones too.

    Counted as supplied, the request stopped asking for what it needed and
    went looking for a record identified by nothing at all.
    """
    enquiry = Enquiry(message="Where is my order?", customer=7, order=blank)
    assert enquiry.order is None
    assert Input.ORDER_ID not in enquiry.known


def test_an_identifier_arrives_without_the_spaces_around_it() -> None:
    """Two references to one order should not depend on a stray keystroke."""
    enquiry = Enquiry(
        message="Where is my order?", order=" ORD-4471 ", product="\tSKU-9"
    )
    assert (enquiry.order, enquiry.product) == ("ORD-4471", "SKU-9")


@pytest.mark.asyncio
async def test_a_blank_order_number_is_asked_for_rather_than_looked_up() -> None:
    """End to end, because the point is what the absence does downstream."""
    outcome = await triage(
        Enquiry(message="Where is my order?", customer=7, order="   ")
    )
    assert outcome == Clarify(
        reason=ClarificationReason.MISSING_ORDER_ID, intent=Intent.ORDER_STATUS
    )
