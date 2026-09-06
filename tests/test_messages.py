"""What a customer is told when no answer is being sent."""

from pathlib import Path

import pytest

from app.agent.messages import (
    _FIRST,
    _SAYS,
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

WORDING = {
    "key": "being_checked_here",
    "locale": "en",
    "version": 1,
    "approved": True,
    "sentence": "We are checking your request before replying.",
}


def test_every_code_a_customer_can_meet_has_something_to_say() -> None:
    """A code with no wording is a customer receiving silence.

    Adding a reason is when this gets forgotten, and the place it shows up is
    a reply with nothing in it.
    """
    reachable = {
        *ClarificationReason,
        *RiskReason,
        *BlockedReason,
        *ReviewReason,
        *EvidenceReason,
    }
    assert reachable - set(_SAYS) == set()


def test_everything_there_is_to_say_is_written_in_both_languages() -> None:
    book = load_messages()
    for key in MessageKey:
        for locale in ("en", "fr"):
            assert book.tell([next(r for r, k in _SAYS.items() if k is key)], locale)  # type: ignore[arg-type]


def test_several_things_wrong_come_to_one_sentence() -> None:
    """Chosen by a declared order, checked over every pair there is.

    Taking whichever the set happened to yield first passed a single example
    and would have varied between runs, since these hash by their text.
    """
    book = load_messages()
    speaks_for = {
        key: next(r for r, k in _SAYS.items() if k is key) for key in MessageKey
    }
    for first in MessageKey:
        for second in MessageKey:
            if first is second:
                continue
            pair: list[ReasonCode] = [speaks_for[first], speaks_for[second]]
            expected = min({first, second}, key=_FIRST.index)
            assert book.tell(pair, "en").key is expected, pair
            assert book.tell(list(reversed(pair)), "en").key is expected, pair


def test_what_went_wrong_inside_is_not_what_the_customer_hears() -> None:
    """Which rating fell short is a fact about us, and useless to them."""
    book = load_messages()
    for reason in EvidenceReason:
        assert book.tell([reason], "en").key is MessageKey.HANDED_TO_A_SPECIALIST


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
        thin.tell([ClarificationReason.MISSING_ORDER_ID], "en")
    with pytest.raises(NothingToTellThemError):
        thin.tell([], "en")


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
