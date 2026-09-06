"""Situations that must reach a person, decided before anything else runs.

These rules see the message first. No classifier has spoken, no source has
been queried, and nothing downstream may clear what they find: a model is
allowed to notice a risk the rules missed, never to overrule one they caught.

Judged sentence by sentence, and within a sentence the report wins. A message
can ask a general question and describe an incident in the same breath, with
or without a full stop between them:

    "How do you protect accounts? Mine was hacked yesterday."
    "How do you protect accounts when mine was hacked yesterday?"

Two failures are possible and they are not equally bad. Escalating something
harmless costs a person a few seconds and the customer a slower reply.
Missing a real report costs considerably more. So a topic escalates unless
the sentence is plainly asking about policy, and `risks_in` records where
that leaning is wrong.
"""

import re
from dataclasses import dataclass

from app.agent.reasons import RiskReason
from app.agent.text import fold

_SENTENCE = re.compile(r"[.!?\n]+")


@dataclass(frozen=True)
class _Category:
    """A kind of trouble, and the phrases that name it in either language."""

    reason: RiskReason
    topics: tuple[str, ...]


_CATEGORIES = (
    _Category(
        RiskReason.PAYMENT_DISPUTE,
        (
            "charged twice",
            "charged me twice",
            "double charge",
            "duplicate charge",
            "billed twice",
            "charged for it twice",
            "debite deux fois",
            "preleve deux fois",
            "facture deux fois",
        ),
    ),
    _Category(
        RiskReason.SUSPECTED_FRAUD,
        (
            "fraud",
            "fraudulent",
            "unauthorized",
            "unauthorised",
            "did not order",
            "didn't order",
            "didnt order",
            "never ordered",
            "stolen card",
            "card was stolen",
            "card is stolen",
            "card stolen",
            # Somebody else spending on this account, described without any of
            # the words above. A model was tried for this and rated an
            # ordinary returns question a legal threat above a real one, so
            # the wording is listed instead of inferred.
            "stranger made purchases",
            "stranger used my",
            "someone else used my",
            "someone used my card",
            "someone used my account",
            "somebody else used my",
            "purchases i did not make",
            "purchases i didn't make",
            "charges i did not make",
            "charges i didn't make",
            "bought things i never",
            "ordered things i never",
            "fraude",
            "frauduleux",
            "carte volee",
            "carte a ete volee",
            "carte a ete piratee",
            "on a vole ma carte",
            "pas commande",
            "n'ai pas commande",
            "quelqu'un a utilise ma carte",
            "quelqu'un a utilise mon compte",
            "achats que je n'ai pas",
        ),
    ),
    _Category(
        RiskReason.ACCOUNT_COMPROMISE,
        (
            "hacked",
            "compromised",
            "broke into my account",
            "got into my account",
            "logged into my account",
            "logged in to my account",
            "accessed my account",
            "access to my account",
            "someone else is in my account",
            "quelqu'un a accede a mon compte",
            "acces a mon compte",
            "someone is using my account",
            "someone changed my password",
            "somebody changed my password",
            "my password was changed",
            "password has been changed",
            "compte pirate",
            "compte a ete pirate",
            "compte est pirate",
            "piratage",
            "compte compromis",
            "compte est compromis",
            "change mon mot de passe sans",
        ),
    ),
    _Category(
        RiskReason.LEGAL_THREAT,
        (
            "my lawyer",
            "my solicitor",
            "legal action",
            "take you to court",
            "sue you",
            "small claims",
            "mon avocat",
            "poursuite judiciaire",
            "porter plainte",
        ),
    ),
)

# Somebody speaking about their own account, rather than about the subject in
# general. Only ever used to overrule the advisory list below, never to
# require anything, because demanding it is what an earlier version did and it
# lost "this transaction is fraudulent" — the plainest report there is.
_FIRST_PERSON = (
    "i was",
    "i've been",
    "i have been",
    "i am",
    "i'm",
    "my ",
    "mine",
    "j'ai",
    "je suis",
    "mon ",
    "ma ",
    "mes ",
    "on m'a",
)

# Asking about policy or prevention. Suppresses a topic only where nobody is
# describing their own account, since one sentence can do both:
#
#     "How do you protect accounts when mine was hacked yesterday?"
#
# Splitting on punctuation is not enough for that one. There is no full stop
# in it.
_ADVISORY = (
    "how can i prevent",
    "how do i prevent",
    "how do you prevent",
    "how do you protect",
    "how do you keep",
    # Asking how to stay out of trouble, which is not being in it. The list
    # grew by one wording at a time and each addition is a customer who would
    # otherwise have been sent to a specialist for asking a sensible question.
    "how can i keep",
    "how do i keep",
    "how can i protect",
    "how do i protect",
    "how can i avoid",
    "how do i avoid",
    "how can i secure",
    "how to prevent",
    "how to protect",
    "how to avoid",
    "how to keep",
    "is it safe to",
    "how safe is",
    "how do you handle",
    "what is your policy",
    "what's your policy",
    "what should someone do",
    "what should i do if",
    "do you offer",
    "comment prevenir",
    "comment proteger",
    "comment protegez",
    "comment eviter",
    "comment securiser",
    "comment garder",
    "comment faire pour proteger",
    "comment eviter que",
    "est-il sur de",
    "quelle est votre politique",
    "que faire si",
)


# Something already happened, as against something that might. Asking how to
# keep one's account safe is not reporting that it was broken into, and both
# sentences say "my".
_ALREADY_HAPPENED = (
    "was",
    "were",
    "has been",
    "have been",
    "hasn't",
    "haven't",
    "did not",
    "didn't",
    "never",
    "a ete",
    "ai ete",
    "ont ete",
    "n'ai pas",
)


def _is_advisory(sentence: str) -> bool:
    """Whether this sentence asks about policy instead of reporting something.

    A customer describing their own account overrules the question — but only
    where they describe it in the past. "How do you protect accounts when mine
    was hacked" is a report wearing a question mark; "how can I prevent fraud
    on my account" is the question it is dressed as, and both say "my".
    """
    speaking_about_themselves = any(
        marker in sentence for marker in _FIRST_PERSON
    ) and any(marker in sentence for marker in _ALREADY_HAPPENED)
    if speaking_about_themselves:
        return False
    return any(phrase in sentence for phrase in _ADVISORY)


def risks_in(message: str) -> frozenset[RiskReason]:
    """Every kind of trouble this message reports.

    A set, not a single answer: one message can describe both a compromised
    account and charges nobody recognises, and an audit record that kept only
    the first would be missing half of what was said.

    Known limitation. Negation and resolution are not understood, so "I was
    not charged twice" and "I thought my card was stolen, but I found it"
    both escalate. Recognising them reliably needs more than phrase matching,
    and the failure is in the direction that costs a person a moment rather
    than the direction that loses a fraud report. The tests say so plainly
    rather than leaving it to be discovered.
    """
    folded = fold(message)
    found = set()
    for sentence in _SENTENCE.split(folded):
        if _is_advisory(sentence):
            continue
        for category in _CATEGORIES:
            if any(topic in sentence for topic in category.topics):
                found.add(category.reason)
    return frozenset(found)
