"""Which facts a question asks for, and which ones an entry actually states.

Relevance says an entry ranked well and authority says its source owns claims
of this kind. Neither looks inside. An entry can win both and still not state
the thing that was asked, which is the moment a confident wrong answer gets
made, so coverage is asked separately and asked about facts rather than about
topics.

Topic-level coverage does not work here, and the corpus shows why. The
shipping rules accept questions about destinations and cost; the approved
shipping entry states delivery times and nothing else. Anything checking only
that a shipping question met a shipping entry marks "do you ship to Belgium?"
answerable and replies with how long delivery takes.

A fact nothing in the corpus carries is still listed. Being able to ask for
something we cannot state is what turns that gap into an escalation instead of
a plausible answer about something else.
"""

from collections.abc import Sequence
from enum import StrEnum

from app.agent.reliability import ReliabilityLevel
from app.agent.text import fold


class Fact(StrEnum):
    """One thing a policy can state, or a customer can ask for."""

    RETURN_WINDOW = "return_window"
    RETURN_ELIGIBILITY = "return_eligibility"
    RETURN_EXCLUDED_CATEGORIES = "return_excluded_categories"
    RETURN_SALE_ITEMS = "return_sale_items"
    RETURN_FINAL_SALE = "return_final_sale"
    RETURN_SHIPPING_COST = "return_shipping_cost"
    PROOF_OF_PURCHASE = "proof_of_purchase"

    STANDARD_DELIVERY_TIME = "standard_delivery_time"
    EXPRESS_DELIVERY_TIME = "express_delivery_time"
    SHIPPING_DESTINATIONS = "shipping_destinations"
    SHIPPING_COST = "shipping_cost"

    # Settled by the shop's own records rather than by anything written down.
    # A policy says how long delivery usually takes; only the order says where
    # this parcel is, and the two questions are asked in almost the same words.
    ORDER_STATE = "order_state"
    DELIVERY_ESTIMATE = "delivery_estimate"
    TRACKING_REFERENCE = "tracking_reference"
    REFUND_STATE = "refund_state"
    REFUND_AMOUNT = "refund_amount"
    STOCK_AVAILABILITY = "stock_availability"

    # Asked constantly and settled by nothing. A refund record says where a
    # payment has got to and never when it will land, so somebody wanting a
    # date is asking for something no source here holds. Listed anyway, on the
    # same reasoning as the policy gaps above: asked alongside the state, it
    # stops the state answering for the pair and sends the question to a
    # person who can say something true about timing.
    REFUND_TIMING = "refund_timing"


# Phrases again rather than words, for the reason they are phrases in intent
# detection: "cost" and "return" belong to several of these at once and settle
# none of them alone.
_RULES: dict[Fact, Sequence[str]] = {
    Fact.RETURN_WINDOW: (
        "how long do i have",
        "how many days do i have",
        "how many days to return",
        "return window",
        "deadline to return",
        "still time to return",
        "combien de temps pour retourner",
        "combien de temps ai-je",
        "delai de retour",
        "jours pour retourner",
    ),
    # Whether there is a returns policy covering this at all. It settles a
    # question that names no particular kind of thing, and only that: the
    # approved claim is one value for the general rule, and a rule cannot say
    # what it excludes.
    Fact.RETURN_ELIGIBILITY: (
        "can i return",
        "send it back",
        "send this back",
        "eligible for a return",
        "eligible for return",
        "which items can",
        "puis-je retourner",
        "eligible au retour",
    ),
    # Naming a kind of thing policies usually carve out. Asked alongside the
    # general rule, so both are requested and the general one cannot answer
    # for the pair.
    Fact.RETURN_EXCLUDED_CATEGORIES: (
        "underwear",
        "swimwear",
        "swimsuit",
        "jewellery",
        "jewelry",
        "pierced",
        "earrings",
        "hygiene reasons",
        "sous-vetement",
        "maillot de bain",
        "bijou",
        "boucles d'oreilles",
        "raisons d'hygiene",
    ),
    # Reduced stock, which a policy usually treats like anything else. Kept
    # apart from the clause below because one value cannot answer both: a
    # claim about goods withdrawn from sale says nothing about ordinary
    # markdowns, and answering the second from the first is the mistake this
    # whole vocabulary exists to stop.
    Fact.RETURN_SALE_ITEMS: (
        "sale item",
        "sale items",
        "discounted item",
        "discounted items",
        "reduced item",
        "reduced items",
        "clearance",
        "marked down",
        "in the sale",
        "on sale",
        "article solde",
        "articles soldes",
        "en solde",
    ),
    Fact.RETURN_FINAL_SALE: (
        "final sale",
        "final-sale",
        "vente definitive",
        "ventes definitives",
    ),
    Fact.RETURN_SHIPPING_COST: (
        "who pays return",
        "who pays for return",
        "return shipping cost",
        "cost of returning",
        "price of returning",
        "pay for the return",
        "frais de retour",
        "prix de retour",
        "qui paie le retour",
    ),
    Fact.PROOF_OF_PURCHASE: (
        "proof of purchase",
        "do i need a receipt",
        "need the receipt",
        "without a receipt",
        "justificatif",
        "preuve d'achat",
    ),
    Fact.STANDARD_DELIVERY_TIME: (
        "how long does delivery",
        "how long does shipping",
        "how long is delivery",
        "delivery take",
        "shipping take",
        "delivery time",
        "shipping time",
        "when will it arrive",
        "delai de livraison",
        "delais de livraison",
        "combien de temps la livraison",
    ),
    Fact.EXPRESS_DELIVERY_TIME: (
        "express delivery",
        "express shipping",
        "next day",
        "faster delivery",
        "quicker delivery",
        "livraison express",
        "livraison rapide",
    ),
    Fact.SHIPPING_DESTINATIONS: (
        "do you ship to",
        "do you deliver to",
        "ship to belgium",
        "which countries",
        "what countries",
        "livrez-vous",
        "livrez vous en",
    ),
    Fact.SHIPPING_COST: (
        "shipping cost",
        "delivery cost",
        "how much is shipping",
        "how much is delivery",
        "how much does delivery cost",
        "free shipping",
        "free delivery",
        "frais de livraison",
        "frais de port",
    ),
    # Below here the answer is in the shop's records. Kept to whole requests
    # for the reason everything else here is: "available" and "refund" belong
    # to several of these at once and settle none of them by themselves.
    Fact.ORDER_STATE: (
        "where is my order",
        "where's my order",
        "where has my order got to",
        "status of my order",
        "my order status",
        "has my order shipped",
        # As incomplete as the wording that recognises the request, and for
        # the same reason: what follows is dispatched, sent or posted, and
        # enumerating those twice is two lists to keep level with each other.
        "has my order been",
        "has my order arrived",
        "my order arrived yet",
        "ou est ma commande",
        "statut de ma commande",
    ),
    Fact.DELIVERY_ESTIMATE: (
        "when will my order",
        "when will my parcel arrive",
        "when will it arrive",
        "when will i receive my order",
        "when will i receive my parcel",
        "quand vais-je recevoir ma commande",
        "quand ma commande arrivera",
        "quand arrivera ma commande",
    ),
    Fact.TRACKING_REFERENCE: (
        "tracking number",
        "tracking reference",
        "track my order",
        "track my parcel",
        "numero de suivi",
        "suivre ma commande",
    ),
    Fact.REFUND_STATE: (
        "where is my refund",
        "where's my refund",
        "where is my money",
        "status of my refund",
        "have i been refunded",
        "has my refund been processed",
        # Not a bare "money back", which is how somebody asks what the returns
        # policy is. These name a refund that was expected and has not come.
        "haven't i got my money back",
        "haven't i had my money back",
        "received my refund",
        "waiting for my refund",
        "when will i get my refund",
        "ou est mon remboursement",
        "statut de mon remboursement",
        "recu mon remboursement",
        "quand vais-je etre rembourse",
    ),
    Fact.REFUND_AMOUNT: (
        "how much will i get back",
        "how much was refunded",
        "how much am i getting back",
        "combien vais-je recevoir",
        "montant du remboursement",
    ),
    # Asked beside the state, never instead of it, so the state cannot answer
    # for the pair and the question reaches somebody who can date it.
    Fact.REFUND_TIMING: (
        "when will i get my refund",
        "when will i be refunded",
        "how long until i get my refund",
        "quand vais-je etre rembourse",
        "quand serai-je rembourse",
    ),
    Fact.STOCK_AVAILABILITY: (
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

_RETURNS = frozenset(
    {
        Fact.RETURN_WINDOW,
        Fact.RETURN_ELIGIBILITY,
        Fact.RETURN_EXCLUDED_CATEGORIES,
        Fact.RETURN_SALE_ITEMS,
        Fact.RETURN_FINAL_SALE,
        Fact.PROOF_OF_PURCHASE,
    }
)

# Asking what a policy says, rather than for one thing it states. Written out
# rather than inferred from whichever entry turned up: reading the request off
# the corpus makes every question answerable by whatever happens to be there.
_WHOLE_POLICY: dict[str, frozenset[Fact]] = {
    # Everything the returns policy states, not the two facts it happened to
    # state when this was written. Answering "what is your returns policy"
    # with the window and nothing else presents a partial answer as a whole
    # one, and the question is the one that asks for all of it.
    "returns policy": _RETURNS,
    "return policy": _RETURNS,
    "politique de retour": _RETURNS,
    "shipping policy": frozenset({Fact.STANDARD_DELIVERY_TIME}),
    "delivery policy": frozenset({Fact.STANDARD_DELIVERY_TIME}),
    "politique de livraison": frozenset({Fact.STANDARD_DELIVERY_TIME}),
}


def facts_in(message: str) -> frozenset[Fact]:
    """Everything this message asks to be told.

    More than one is normal and is not the ambiguity intent detection worries
    about: "how long do I have to return sale items" asks for a window and for
    what qualifies, and both have to be present for the answer to be whole.
    """
    folded = fold(message)
    requested = {
        fact
        for fact, phrases in _RULES.items()
        if any(phrase in folded for phrase in phrases)
    }
    for phrase, facts in _WHOLE_POLICY.items():
        if phrase in folded:
            requested |= facts
    return frozenset(requested)


def coverage_of(
    requested: frozenset[Fact], carried: frozenset[Fact]
) -> ReliabilityLevel:
    """Whether the evidence states everything the question asked for.

    Nothing recognised is not treated as nothing needed. A question this file
    cannot pin down is one whose answer cannot be checked, and defaulting that
    to covered would rebuild the failure this exists to stop: a phrase rule
    that misses "do you ship to Belgium" would send delivery times back as
    though they had been asked for. It goes to somebody instead.

    Answering part of it is its own rung, which is what the scale says and
    what the destination makes sensible: a partly covered question goes to
    somebody here, who has the whole entry in front of them and can finish it.
    Sending that to a specialist wastes a queue neither of them needed. What
    must never happen is the half-answer going out as though it were whole,
    and nothing at this level is delivered to anybody.
    """
    if not requested:
        return ReliabilityLevel.REVIEW_ONLY
    if requested <= carried:
        return ReliabilityLevel.READY
    if requested & carried:
        return ReliabilityLevel.REVIEW_ONLY
    return ReliabilityLevel.UNUSABLE
