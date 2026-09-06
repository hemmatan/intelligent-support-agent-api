"""The only sentences a customer is sent, and what stops anything else."""

import re
from pathlib import Path

import pytest

from app.agent.facts import Fact
from app.agent.knowledge import PolicyEntry, load_corpus
from app.agent.responses import (
    NothingApprovedToSayError,
    ResponseTemplate,
    ResponseTemplateError,
    TemplateLibrary,
    load_templates,
)

WORDING = {
    "fact": "return_window",
    "locale": "en",
    "version": 1,
    "approved": True,
    "sentence": "Returns are accepted within {return_window_days} days.",
}


def entry(reference: str) -> PolicyEntry:
    return next(e for e in load_corpus() if e.reference.endswith(reference))


def test_a_figure_reaches_a_customer_only_from_a_claim() -> None:
    """The same rule the searchable prose lives under, for the same reason.

    A digit typed into wording is a second place the number is written down,
    and the two can drift apart without either looking wrong on its own.
    """
    with pytest.raises(ValueError, match="literal digit"):
        ResponseTemplate.model_validate(
            {**WORDING, "sentence": "Returns are accepted within 30 days."}
        )


def test_wording_that_rests_on_nothing_is_refused() -> None:
    """A sentence naming no claim asserts something with no evidence under it."""
    with pytest.raises(ValueError, match="names no claim"):
        ResponseTemplate.model_validate(
            {**WORDING, "sentence": "Returns are usually straightforward."}
        )


def test_wording_cannot_name_a_claim_no_entry_states() -> None:
    """Caught on load, not in front of somebody waiting for a reply."""
    with pytest.raises(ValueError, match="declares"):
        ResponseTemplate.model_validate(
            {**WORDING, "sentence": "Returns cost {postage_refund} to send."}
        )


def test_wording_cannot_answer_with_a_figure_belonging_to_another_fact() -> None:
    """Correct sentence, wrong question, filed as the answer to the other one.

    Every field of a model stating the fact used to be fillable, rather than
    the fields that state it. So the approved reply about standard delivery
    could quote the express figure and be rated as covering the question.
    """
    with pytest.raises(ValueError, match="nothing stating"):
        ResponseTemplate.model_validate(
            {
                **WORDING,
                "fact": "standard_delivery_time",
                "sentence": "Delivery takes {express_delivery_days} working day.",
            }
        )


def test_a_fact_carried_by_two_figures_is_not_settled_by_one() -> None:
    """A range quoted by its lower end is a shorter promise than the policy."""
    with pytest.raises(ValueError, match="names only"):
        ResponseTemplate.model_validate(
            {
                **WORDING,
                "fact": "standard_delivery_time",
                "sentence": "Delivery takes {standard_delivery_days_min} days.",
            }
        )


def test_one_approved_way_to_say_a_thing_in_a_language() -> None:
    """Two would leave the choice to whichever loaded first."""
    with pytest.raises(ResponseTemplateError, match="two approved ways"):
        TemplateLibrary(
            [
                ResponseTemplate.model_validate(WORDING),
                ResponseTemplate.model_validate(
                    {**WORDING, "sentence": "You have {return_window_days} days."}
                ),
            ]
        )


def test_a_draft_is_not_wording(tmp_path: Path) -> None:
    library = TemplateLibrary(
        [ResponseTemplate.model_validate({**WORDING, "approved": False})]
    )
    assert len(library) == 0


def test_the_shipped_wording_all_loads() -> None:
    library = load_templates()
    assert len(library) == 6
    for template in library:
        assert template.approved


def test_the_reply_reads_the_same_way_round_every_time() -> None:
    """Ordered by the vocabulary, not by which the question mentioned first."""
    library = load_templates()
    both = frozenset({Fact.RETURN_WINDOW, Fact.RETURN_ELIGIBILITY})
    text, used = library.say(both, entry("returns.standard.en.v1"), "en")
    assert text == (
        "Returns are accepted within 30 days of delivery. Most items can be "
        "returned if unworn and in the original packaging."
    )
    assert [reference.split("@")[0] for reference in used] == [
        "say:return_window.en.v1",
        "say:return_eligibility.en.v1",
    ]


def test_a_named_choice_reaches_a_customer_as_words() -> None:
    """ "standard_items" is how we store it, not something to send anybody."""
    library = load_templates()
    for locale, expected in (("en", "Most items"), ("fr", "La plupart des articles")):
        text, _ = library.say(
            frozenset({Fact.RETURN_ELIGIBILITY}),
            entry(f"returns.standard.{locale}.v1"),
            locale,  # type: ignore[arg-type]
        )
        assert text.startswith(expected)
        assert "_" not in text


def test_a_language_nobody_wrote_for_says_nothing(tmp_path: Path) -> None:
    """Rather than answering in the language we happen to have."""
    library = load_templates()
    with pytest.raises(NothingApprovedToSayError, match="standard_delivery_time"):
        library.say(
            frozenset({Fact.STANDARD_DELIVERY_TIME}),
            entry("shipping.times.en.v1"),
            "fr",
        )


def test_no_shipped_sentence_states_a_figure_of_its_own() -> None:
    """Read off the files, so wording added later is held to it too."""
    for path in sorted(Path("app/agent/responses").glob("*.toml")):
        assert not re.search(r"\d", path.read_text().split("sentence = ")[1])


def test_no_claim_leaves_the_corpus_already_turned_into_prose() -> None:
    """Values come out as authored. Turning them into words is the phrase book."""
    for policy in load_corpus():
        assert policy.claims.values() == policy.claims.model_dump()
