"""Reading what a customer is asking for, or admitting it is not clear."""

import pytest

from app.agent.intent import Intent, intents_in


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("How long do I have to return a jacket?", Intent.RETURN_POLICY),
        ("Can I send this back?", Intent.RETURN_POLICY),
        ("Combien de temps pour retourner un article ?", Intent.RETURN_POLICY),
        ("What countries do you ship to?", Intent.SHIPPING_POLICY),
        ("How long does delivery take?", Intent.SHIPPING_POLICY),
        ("Quel est le délai de livraison ?", Intent.SHIPPING_POLICY),
        ("Where is my order?", Intent.ORDER_STATUS),
        ("Où est ma commande ?", Intent.ORDER_STATUS),
        ("Is the blue frock in stock?", Intent.PRODUCT_AVAILABILITY),
        ("Is that dress still available?", Intent.PRODUCT_AVAILABILITY),
        ("Avez-vous cette robe en stock ?", Intent.PRODUCT_AVAILABILITY),
        ("Cette robe est-elle encore disponible ?", Intent.PRODUCT_AVAILABILITY),
    ],
    ids=lambda value: value if isinstance(value, str) else "",
)
def test_a_clear_request_reads_as_one_thing(message: str, expected: Intent) -> None:
    assert intents_in(message) == {expected}


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("How long does shipping take for my order?", Intent.SHIPPING_POLICY),
        ("I want to return my order", Intent.RETURN_POLICY),
        ("Do you have a returns policy?", Intent.RETURN_POLICY),
        ("What is the price of returning an item?", Intent.RETURN_POLICY),
        ("Quel est le prix de retour ?", Intent.RETURN_POLICY),
    ],
    ids=[
        "shipping and order",
        "return and order",
        "question opener",
        "price of returning",
        "price of returning, french",
    ],
)
def test_shared_vocabulary_does_not_make_two_questions(
    message: str, expected: Intent
) -> None:
    """Every one of these mentions a second thing in passing. The last three
    escaped the first version of these rules, where product availability
    included "do you have", "how much is" and "price of" — question openers
    rather than references to a product, matching anything anyone asks.
    """
    assert intents_in(message) == {expected}


def test_two_requests_in_one_message_are_two_requests() -> None:
    """Answering whichever rule ran first would answer half of this silently."""
    assert intents_in("Where is my order, and can I return it once it arrives?") == {
        Intent.ORDER_STATUS,
        Intent.RETURN_POLICY,
    }


@pytest.mark.parametrize(
    "message",
    [
        "I need help with my purchase",
        "Do you offer price matching?",
        "Hello",
        "",
    ],
    ids=["vague", "no policy covers it", "greeting", "empty"],
)
def test_recognising_nothing_is_an_answer(message: str) -> None:
    """Not a failure. A classifier sees these next, or the customer does."""
    assert intents_in(message) == frozenset()


def test_accents_do_not_decide_what_was_asked() -> None:
    assert intents_in("Ou est ma commande") == intents_in("Où est ma commande")


@pytest.mark.parametrize(
    "message",
    [
        "Where is my refund?",
        "I returned my order last week. Why haven't I got my money back?",
        "Où est mon remboursement ?",
        "I still haven't received my refund",
    ],
    ids=["direct", "narrated", "french", "waiting"],
)
def test_chasing_a_refund_is_not_asking_what_the_policy_says(message: str) -> None:
    """The policy can say how long somebody has. It cannot say where the money is.

    These resolved to the returns policy, which would have replied with a
    thirty-day window to somebody asking after money they are owed. There is
    no refund-status intent to route them to, so they resolve to nothing and
    get asked about — not an answer, but not the wrong answer either.
    """
    assert Intent.RETURN_POLICY not in intents_in(message)
