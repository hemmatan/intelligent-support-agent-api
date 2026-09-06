"""What a customer is asking for, and therefore where the answer may come from.

Getting this wrong is not a ranking mistake, it is a sourcing mistake: the
intent chooses which sources are consulted, so a misread question is answered
from the wrong place before anything is retrieved.

Rules match phrases rather than words, because the words overlap and the
phrases do not. "Where is my order" and "how long does delivery take" share
almost nothing structurally while sharing most of their vocabulary with each
other and with half the corpus.

Returning nothing is a real answer. A message the rules do not recognise goes
to a classifier where one is configured, and to the customer as a question
where one is not. Neither is a guess.
"""

from collections.abc import Sequence
from enum import StrEnum
from typing import Protocol, runtime_checkable

from app.agent.knowledge import Locale
from app.agent.text import fold


class Intent(StrEnum):
    """The kinds of request this service knows how to route."""

    RETURN_POLICY = "return_policy"
    SHIPPING_POLICY = "shipping_policy"
    ORDER_STATUS = "order_status"
    PRODUCT_AVAILABILITY = "product_availability"
    REFUND_STATUS = "refund_status"


# Phrases naming a general policy, as against a particular purchase. The
# distinction is the whole reason these are phrases: "delivery" and "order"
# appear in both kinds of question and settle nothing on their own.
# Chasing one particular refund, rather than asking what the rule is. A
# returns policy can say how long somebody has; it cannot say where their
# money went, and answering from it replies to a question nobody asked.
#
# These read as the returns policy because they share its whole vocabulary,
# so the match is replaced rather than removed. Deleting it made "where is my
# order, and where is my refund" a single request about an order, and left
# nothing for a classifier to be kept away from.
#
# Written as whole phrases after a general one failed. "Any chasing wording
# anywhere in the message" also swallowed the returns half of "where is my
# order, and can I return it once it arrives", which is two questions and
# deserves to be treated as two.
_CHASING_A_REFUND = (
    "where is my refund",
    "where's my refund",
    "where is my money",
    "why haven't i got my money back",
    "why haven't i had my money back",
    "why haven't i received my refund",
    "haven't received my refund",
    "have not received my refund",
    "still waiting for my refund",
    "when will i get my refund",
    "ou est mon remboursement",
    "toujours pas recu mon remboursement",
    "quand vais-je etre rembourse",
)

_RULES: dict[Intent, Sequence[str]] = {
    Intent.RETURN_POLICY: (
        "return",
        "returns",
        "send it back",
        "send this back",
        "sending it back",
        "money back",
        "refund",
        "change my mind",
        "changed my mind",
        "retourner",
        "retour",
        "rembours",
        "renvoyer",
    ),
    Intent.SHIPPING_POLICY: (
        "how long does delivery",
        "how long does shipping",
        "how long is delivery",
        "delivery take",
        "shipping take",
        "delivery time",
        "shipping time",
        "do you ship to",
        "do you deliver to",
        "shipping cost",
        "delivery cost",
        "delai de livraison",
        "delais de livraison",
        "combien de temps la livraison",
        "livrez-vous",
    ),
    # Specific enough to name one purchase. Bare "my order" would fire on
    # "I want to return my order", which is a returns question.
    Intent.ORDER_STATUS: (
        "where is my order",
        "where's my order",
        "track my order",
        "tracking number",
        "status of my order",
        "when will my order",
        "has my order shipped",
        "has my order been",
        "has my order arrived",
        "my order arrived yet",
        "ou est ma commande",
        "suivre ma commande",
        "statut de ma commande",
        "numero de suivi",
    ),
    # About stock, not about products in general. "Do you have", "how much
    # is" and "price of" opened questions rather than naming anything, so
    # asking whether a returns policy exists read as two requests.
    Intent.PRODUCT_AVAILABILITY: (
        "in stock",
        "out of stock",
        "back in stock",
        "still available",
        "do you still have",
        "have any left",
        "sold out",
        "en stock",
        "en rupture",
        "encore disponible",
        "toujours disponible",
        "epuise",
    ),
}


class ClassifierUnavailableError(RuntimeError):
    """A model was asked to place a message and could not.

    Transient by definition: a timeout, a refused connection, a provider
    having a bad afternoon. Implementations raise this rather than letting a
    transport error out, so a request can tell "nothing was recognised" apart
    from "nobody looked".
    """


@runtime_checkable
class IntentClassifier(Protocol):
    """A model asked about a message the phrase rules made nothing of.

    It cannot report danger, and that is a finding rather than a simplification.
    The model evaluated for the job scored an ordinary returns question as a
    legal threat more strongly than a real threat, and a question about staying
    safe from fraud above a genuine fraud report. Risk is the deterministic
    rules' alone; the measurements are in docs/architecture.md.

    It is never shown a message those rules escalated, or one the phrase rules
    placed, so nothing it returns can revise a decision already taken. Where it
    cannot answer it raises, and the request waits for somebody here instead of
    being guessed at.

    Nothing implements this today. It is the shape a model would have to take
    to be admitted, kept because leaving it empty was decided rather than
    overlooked.
    """

    async def classify(self, message: str, locale: Locale) -> Intent | None:
        """Name an intent, return None, or raise ClassifierUnavailableError."""
        ...


def intents_in(message: str) -> frozenset[Intent]:
    """Every intent the rules recognise in this message.

    A set rather than a first match. One message can carry two requests —
    where an order is, and whether it can be sent back once it arrives — and
    answering whichever rule happened to be checked first would answer half
    the question without saying so.

    Somebody chasing a refund is not asking what the returns policy says, so
    the policy match is replaced by the intent that owns where money is. It
    resolves to a commerce claim, which is where a refund actually stands.

    Replacing rather than discarding matters twice. A message asking two
    things stays two things, instead of quietly becoming the one that survived
    deletion. And the request is now placed, so no classifier is consulted
    about it — which is what stopped one from handing back the returns policy
    the rules had just withheld.
    """
    folded = fold(message)
    matched = {
        intent
        for intent, phrases in _RULES.items()
        if any(phrase in folded for phrase in phrases)
    }
    if any(phrase in folded for phrase in _CHASING_A_REFUND):
        matched.discard(Intent.RETURN_POLICY)
        matched.add(Intent.REFUND_STATUS)
    return frozenset(matched)
