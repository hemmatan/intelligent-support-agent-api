"""What must reach a person, and what must not be sent to one needlessly."""

import pytest

from app.agent.reasons import RiskReason
from app.agent.risk import risks_in


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("I was charged twice for the same order", RiskReason.PAYMENT_DISPUTE),
        ("This transaction is fraudulent", RiskReason.SUSPECTED_FRAUD),
        ("Someone broke into my account", RiskReason.ACCOUNT_COMPROMISE),
        ("My lawyer will be in touch", RiskReason.LEGAL_THREAT),
        ("J'ai ete debite deux fois", RiskReason.PAYMENT_DISPUTE),
        ("J'ai été débité deux fois", RiskReason.PAYMENT_DISPUTE),
        ("Mon compte a été piraté", RiskReason.ACCOUNT_COMPROMISE),
        ("Je n'ai pas commandé cet article", RiskReason.SUSPECTED_FRAUD),
    ],
    ids=lambda value: value if isinstance(value, str) else "",
)
def test_reports_reach_a_person(message: str, expected: RiskReason) -> None:
    assert expected in risks_in(message)


@pytest.mark.parametrize(
    "message",
    [
        "How can I prevent payment fraud?",
        "How do you protect accounts?",
        "What is your policy on unauthorized charges?",
        "What should someone do if their card is stolen?",
        "Is my card safe on your site?",
        "How long do I have to return a jacket?",
        "Comment prévenir la fraude ?",
    ],
    ids=lambda value: value,
)
def test_asking_about_policy_is_not_reporting_an_incident(message: str) -> None:
    """Escalating these would train customers not to ask, and staff to skim."""
    assert risks_in(message) == frozenset()


@pytest.mark.parametrize(
    "message",
    [
        "How do you protect accounts? Mine was hacked yesterday.",
        "How do you protect accounts when mine was hacked yesterday?",
    ],
    ids=["separate sentences", "one sentence"],
)
def test_a_general_question_cannot_excuse_a_report_beside_it(message: str) -> None:
    """Splitting on punctuation is not enough: the second has no full stop."""
    assert risks_in(message) == {RiskReason.ACCOUNT_COMPROMISE}


@pytest.mark.parametrize(
    "message",
    [
        "I didn't order this",
        "I didn\u2019t order this",
        "Je n'ai pas commandé cet article",
    ],
    ids=["typed apostrophe", "typographic apostrophe", "french"],
)
def test_a_contraction_reports_as_clearly_as_the_long_form(message: str) -> None:
    """Which apostrophe a keyboard produced is not a fact about the message."""
    assert RiskReason.SUSPECTED_FRAUD in risks_in(message)


@pytest.mark.parametrize(
    "message",
    [
        "I changed my password yesterday",
        "Nous avons trouvé un compromis",
        "The pirate ship costume arrived damaged",
    ],
    ids=["own action", "compromis meaning agreement", "pirate as a noun"],
)
def test_words_that_only_sound_like_trouble(message: str) -> None:
    """Bare substrings for compromis and pirate escalated all three of these."""
    assert risks_in(message) == frozenset()


def test_somebody_else_changing_the_password_is_different() -> None:
    assert RiskReason.ACCOUNT_COMPROMISE in risks_in("Someone changed my password")


def test_one_message_can_report_more_than_one_kind_of_trouble() -> None:
    """Keeping only the first would lose half of what was said."""
    assert risks_in("Someone hacked my account and used my card fraudulently") == {
        RiskReason.ACCOUNT_COMPROMISE,
        RiskReason.SUSPECTED_FRAUD,
    }


def test_reporting_something_you_want_to_report_still_counts() -> None:
    """Phrased as a question, but the person asking has fraud to report."""
    assert RiskReason.SUSPECTED_FRAUD in risks_in(
        "How do I report a fraudulent charge?"
    )


def test_hedging_is_not_absence() -> None:
    assert RiskReason.PAYMENT_DISPUTE in risks_in(
        "I think I might have been charged twice"
    )


def test_a_report_about_someone_else_still_reaches_a_person() -> None:
    """Cheaper than deciding, from a sentence, whose money it was.

    The reply telling them their friend has to write in themselves is a
    response concern; the routing is the same either way.
    """
    assert RiskReason.PAYMENT_DISPUTE in risks_in("My friend was charged twice")


@pytest.mark.parametrize(
    "message",
    [
        "I was not charged twice",
        "I thought my card was stolen, but I found it",
    ],
    ids=["negation", "resolved"],
)
def test_negation_is_not_understood_and_escalates_anyway(message: str) -> None:
    """A known false positive, recorded rather than left to be discovered.

    Phrase matching cannot tell these from reports. Handling them needs more
    than this file has, and the cost of being wrong here is a person reading
    a message that turns out to be fine.
    """
    assert risks_in(message) != frozenset()


def test_an_ordinary_question_reaches_nobody() -> None:
    assert risks_in("How long does delivery usually take?") == frozenset()
    assert risks_in("") == frozenset()


@pytest.mark.parametrize(
    "message",
    [
        "How can I prevent fraud on my account?",
        "What should I do if my card is stolen?",
        "Comment protéger mon compte contre le piratage ?",
    ],
    ids=["prevention", "hypothetical", "prevention, french"],
)
def test_asking_about_your_own_account_is_not_reporting_it(message: str) -> None:
    """Saying "my" is not saying something happened.

    An earlier version let any first-person mention overrule an advisory
    question, which sent every customer asking how to stay safe to a person.
    The override wants the past tense as well.
    """
    assert risks_in(message) == frozenset()


# Every category, in both languages, as the same subject asked about and
# reported. Either column alone passes for the wrong reason — a rule that never
# fires clears all the questions, one that always fires clears all the reports —
# and the expected category is named, so a report answered with the wrong kind
# of trouble does not count as caught.
BOUNDARIES = [
    (
        "en",
        RiskReason.PAYMENT_DISPUTE,
        "What is your policy on double billing?",
        "I was charged twice",
    ),
    (
        "fr",
        RiskReason.PAYMENT_DISPUTE,
        "Quelle est votre politique en cas de double facturation ?",
        "J'ai ete debite deux fois",
    ),
    (
        "en",
        RiskReason.SUSPECTED_FRAUD,
        "How can I avoid unauthorised charges?",
        "Someone used my card",
    ),
    (
        "fr",
        RiskReason.SUSPECTED_FRAUD,
        "Comment prevenir la fraude ?",
        "Je n'ai pas commande cet article",
    ),
    (
        "en",
        RiskReason.ACCOUNT_COMPROMISE,
        "How do you protect accounts?",
        "Someone logged into my account last night",
    ),
    (
        "fr",
        RiskReason.ACCOUNT_COMPROMISE,
        "Comment garder mon compte en securite ?",
        "Quelqu'un a accede a mon compte",
    ),
    (
        "en",
        RiskReason.LEGAL_THREAT,
        "What is your policy on disputes?",
        "My lawyer will be in touch",
    ),
    (
        "fr",
        RiskReason.LEGAL_THREAT,
        "Quelle est votre politique en cas de litige ?",
        "Mon avocat va vous contacter",
    ),
]


@pytest.mark.parametrize(
    ("locale", "reason", "question", "report"),
    BOUNDARIES,
    ids=[f"{locale} {reason.value}" for locale, reason, _, _ in BOUNDARIES],
)
def test_asking_about_a_subject_and_reporting_it_go_different_ways(
    locale: str, reason: RiskReason, question: str, report: str
) -> None:
    """The distinction the whole detector exists to draw."""
    assert risks_in(question) == frozenset(), question
    assert reason in risks_in(report), report


@pytest.mark.parametrize(
    ("reason", "message"),
    [
        (
            RiskReason.ACCOUNT_COMPROMISE,
            "How do you protect accounts when someone got into my account?",
        ),
        (
            RiskReason.SUSPECTED_FRAUD,
            "How can I prevent fraud if someone used my card?",
        ),
        (
            RiskReason.ACCOUNT_COMPROMISE,
            "Comment proteger les comptes quand quelqu'un a accede a mon compte ?",
        ),
        (
            RiskReason.LEGAL_THREAT,
            "What is your policy if my lawyer contacts you?",
        ),
    ],
    ids=["account, one sentence", "fraud, one sentence", "account, french", "legal"],
)
def test_a_question_wrapped_round_a_report_is_still_a_report(
    reason: RiskReason, message: str
) -> None:
    """Every one of these was being discarded whole.

    An advisory opening set the sentence aside unless the customer also used
    one of a short list of past-tense verbs, and "got", "used" and "a accede"
    were not on it. Lengthening that list is not a fix — it is the same bet
    made again. Wordings that can only be a report are now recognised whatever
    is wrapped around them, and the subject words alone are what an advisory
    sentence sets aside.
    """
    assert reason in risks_in(message)


def test_a_report_is_still_read_when_it_arrives_inside_a_request() -> None:
    """The sentence a model was brought in for, and could not tell from a
    question about the same subject. It is listed instead.
    """
    assert RiskReason.SUSPECTED_FRAUD in risks_in(
        "I need to return this because a stranger made purchases using my account"
    )


def test_what_the_vocabulary_does_not_cover_is_recorded_not_assumed() -> None:
    """A wording nobody listed is a wording nobody catches.

    This is the bound the design accepts, and it is written down in
    docs/architecture.md rather than discovered. The example is deliberate: it
    reports a real incident in words the lists do not carry, and it proceeds.
    """
    unlisted = "an individual has been helping themselves to my funds"
    assert risks_in(unlisted) == frozenset()
