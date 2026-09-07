"""What a customer is told when no answer is being sent."""

from pathlib import Path

import pytest

from app.agent.knowledge import Locale
from app.agent.messages import (
    _FIRST,
    DEFAULT_MESSAGE_DIR,
    MessageBook,
    MessageKey,
    NothingToTellThemError,
    OutcomeMessage,
    OutcomeMessageError,
    load_messages,
)
from app.agent.reasons import (
    BlockedReason,
    ClarificationReason,
    EvidenceReason,
    ReasonCode,
    ReviewReason,
    RiskReason,
)
from app.agent.reliability import Route

WORDING = {
    "key": "being_checked_here",
    "locale": "en",
    "version": 1,
    "approved": True,
    "sentence": "We are checking your request before replying.",
}


# Which codes each route can actually carry, so the table below covers what
# the service produces rather than every combination the types allow.
CARRIED: dict[Route, list[ReasonCode]] = {
    Route.CLARIFICATION: list(ClarificationReason),
    Route.HUMAN_ESCALATION: [*RiskReason, *BlockedReason, *EvidenceReason],
    Route.INTERNAL_REVIEW: [*ReviewReason, *EvidenceReason],
}


@pytest.mark.parametrize("locale", ["en", "fr"], ids=["english", "french"])
def test_everything_a_customer_can_meet_has_something_to_say(locale: Locale) -> None:
    """A code with no wording is somebody receiving silence.

    Adding a reason is when this gets forgotten, and where it shows up is a
    reply with nothing in it.
    """
    book = load_messages()
    for route, reasons in CARRIED.items():
        for reason in reasons:
            assert book.tell([reason], locale, route).sentence, (route, reason)


def test_what_a_customer_is_told_follows_where_the_request_went() -> None:
    """One code, two routes, and only one of the sentences is true.

    Coverage falling short holds a request here or sends it onward depending
    on how far short. Chosen by the code alone, somebody whose message never
    left was told a colleague had taken it on.
    """
    book = load_messages()
    held = book.tell([EvidenceReason.NOT_COVERED], "en", Route.INTERNAL_REVIEW)
    passed = book.tell([EvidenceReason.NOT_COVERED], "en", Route.HUMAN_ESCALATION)
    assert held.key is MessageKey.BEING_CHECKED_HERE
    assert passed.key is MessageKey.HANDED_TO_A_SPECIALIST
    assert held.sentence != passed.sentence


def test_several_things_wrong_come_to_one_sentence() -> None:
    """Chosen by a declared order, checked over every pair there is.

    Taking whichever the set happened to yield first passed a single example
    and would have varied between runs, since these hash by their text.
    """
    book = load_messages()
    speaks_for: dict[MessageKey, ReasonCode] = {
        MessageKey.ASK_FOR_ORDER_NUMBER: ClarificationReason.MISSING_ORDER_ID,
        MessageKey.ASK_WHICH_PRODUCT: ClarificationReason.MISSING_PRODUCT_REFERENCE,
        MessageKey.ASK_WHICH_FIRST: ClarificationReason.MULTIPLE_INTENTS,
        MessageKey.ASK_WHAT_IS_MEANT: ClarificationReason.UNRESOLVED_INTENT,
        MessageKey.ACCOUNT_NOT_LINKED: BlockedReason.CUSTOMER_NOT_LINKED,
        MessageKey.HANDED_TO_A_SPECIALIST: RiskReason.PAYMENT_DISPUTE,
    }
    for first in speaks_for:
        for second in speaks_for:
            if first is second:
                continue
            pair: list[ReasonCode] = [speaks_for[first], speaks_for[second]]
            expected = min({first, second}, key=_FIRST.index)
            route = Route.HUMAN_ESCALATION
            assert book.tell(pair, "en", route).key is expected, pair
            assert book.tell(list(reversed(pair)), "en", route).key is expected, pair


def test_what_went_wrong_inside_is_not_what_the_customer_hears() -> None:
    """Which rating fell short is a fact about us, and useless to them."""
    book = load_messages()
    for reason in EvidenceReason:
        told = book.tell([reason], "en", Route.HUMAN_ESCALATION)
        assert told.key is MessageKey.HANDED_TO_A_SPECIALIST


def test_a_sentence_here_states_no_figure() -> None:
    with pytest.raises(ValueError, match="states a figure"):
        OutcomeMessage.model_validate(
            {**WORDING, "sentence": "We reply within 2 days."}
        )


@pytest.mark.parametrize(
    "sentence",
    [
        "We will reply soon.",
        "Somebody will look at this shortly.",
        "We are on it and will respond today.",
        "Nous repondrons rapidement.",
    ],
    ids=["soon", "shortly", "today", "french"],
)
def test_a_sentence_here_promises_no_timescale(sentence: str) -> None:
    """A wording with no figure in it can still commit us to one.

    This is a guard rather than a judgement: what makes a sentence safe is
    somebody approving it, and this catches the commonest way of getting it
    wrong.
    """
    with pytest.raises(ValueError, match="promises"):
        OutcomeMessage.model_validate({**WORDING, "sentence": sentence})


def test_a_sentence_here_has_nothing_to_fill_in() -> None:
    """These carry no claims, so a slot would render as itself."""
    with pytest.raises(ValueError, match="slot"):
        OutcomeMessage.model_validate(
            {**WORDING, "sentence": "We are checking order {order_id}."}
        )


def test_one_approved_way_to_say_a_thing_in_a_language() -> None:
    with pytest.raises(OutcomeMessageError, match="two approved ways"):
        MessageBook(
            [
                OutcomeMessage.model_validate(WORDING),
                OutcomeMessage.model_validate({**WORDING, "sentence": "We are on it."}),
            ]
        )


def test_a_draft_is_not_wording() -> None:
    assert (
        len(
            MessageBook([OutcomeMessage.model_validate({**WORDING, "approved": False})])
        )
        == 0
    )


def test_a_reason_nobody_wrote_for_is_refused_rather_than_answered_blankly() -> None:
    thin = MessageBook([OutcomeMessage.model_validate(WORDING)])
    with pytest.raises(NothingToTellThemError):
        thin.tell([ClarificationReason.MISSING_ORDER_ID], "en", Route.CLARIFICATION)
    with pytest.raises(NothingToTellThemError):
        thin.tell([], "en", Route.CLARIFICATION)


def test_a_file_has_to_say_what_it_is(tmp_path: Path) -> None:
    (tmp_path / "wrong_name.en.v1.toml").write_text(
        'key = "being_checked_here"\nlocale = "en"\nversion = 1\n'
        'approved = true\nsentence = "We are checking your request."\n'
    )
    with pytest.raises(OutcomeMessageError, match="declares itself"):
        load_messages(tmp_path)


def test_rewriting_a_sentence_moves_its_digest() -> None:
    before = OutcomeMessage.model_validate(WORDING)
    after = OutcomeMessage.model_validate({**WORDING, "sentence": "We are checking."})
    assert before.reference == after.reference
    assert before.content_hash != after.content_hash


def test_no_shipped_sentence_states_a_figure() -> None:
    """Read off the files, so wording added later is held to it too."""
    for message in load_messages():
        assert not any(character.isdigit() for character in message.sentence)


def test_a_sentence_of_whitespace_is_not_a_sentence() -> None:
    """A length of one is met by a space, which arrives as silence.

    Which is the outcome this whole file was added to stop, reintroduced by
    the check meant to prevent it.
    """
    with pytest.raises(ValueError, match="at least 1"):
        OutcomeMessage.model_validate({**WORDING, "sentence": "   "})


def test_a_promise_written_the_way_people_write_it_is_still_a_promise() -> None:
    """Lowercasing is not folding.

    The list is spelled without accents, and "bientot" does not occur inside
    "bientôt", so ordinary French went through untouched.
    """
    with pytest.raises(ValueError, match="promises"):
        OutcomeMessage.model_validate(
            {**WORDING, "locale": "fr", "sentence": "Nous répondrons bientôt."}
        )


def test_a_language_missing_from_the_image_stops_the_process(tmp_path: Path) -> None:
    """Not the first customer who writes in that language.

    A file left out of a build is not a mistake a test suite is present to
    catch, and the promise that every key exists in every language lived only
    in one.
    """
    for shipped in DEFAULT_MESSAGE_DIR.glob("*.toml"):
        if not shipped.name.endswith(".fr.v1.toml"):
            (tmp_path / shipped.name).write_text(shipped.read_text())
    with pytest.raises(OutcomeMessageError, match="nothing approved says"):
        load_messages(tmp_path)


def test_the_shipped_set_is_complete() -> None:
    book = load_messages()
    assert len(book) == len(MessageKey) * 2
