"""The only sentences a customer is sent, and what stops anything else."""

import re
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Literal

import pytest

from app.agent.commerce import Observation, ProductRecord
from app.agent.facts import Fact
from app.agent.knowledge import (
    ItemCategory,
    PolicyEntry,
    ReturnPolicyClaims,
    ShippingPolicyClaims,
    load_corpus,
)
from app.agent.responses import (
    ApprovedReply,
    NothingApprovedToSayError,
    ResponseTemplate,
    ResponseTemplateError,
    TemplateLibrary,
    UnattributedReplyError,
    _how_to_say,
    _Rendering,
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
    assert len(library) == 14
    for template in library:
        assert template.approved


def test_the_reply_reads_the_same_way_round_every_time() -> None:
    """Ordered by the vocabulary, not by which the question mentioned first."""
    library = load_templates()
    both = frozenset({Fact.RETURN_WINDOW, Fact.RETURN_ELIGIBILITY})
    reply = library.say(both, entry("returns.standard.en.v1").claims.values(), "en")
    assert reply.text == (
        "Returns are accepted within 30 days of delivery. Most items can be "
        "returned if unworn and in the original packaging."
    )
    assert [reference.split("@")[0] for reference in reply.said] == [
        "say:return_window.en.v1",
        "say:return_eligibility.en.v1",
    ]


def test_a_named_choice_reaches_a_customer_as_words() -> None:
    """ "standard_items" is how we store it, not something to send anybody."""
    library = load_templates()
    for locale, expected in (("en", "Most items"), ("fr", "La plupart des articles")):
        reply = library.say(
            frozenset({Fact.RETURN_ELIGIBILITY}),
            entry(f"returns.standard.{locale}.v1").claims.values(),
            locale,  # type: ignore[arg-type]
        )
        assert reply.text.startswith(expected)
        assert "_" not in reply.text


def test_a_language_nobody_wrote_for_says_nothing(tmp_path: Path) -> None:
    """Rather than answering in the language we happen to have."""
    library = load_templates()
    with pytest.raises(NothingApprovedToSayError, match="standard_delivery_time"):
        library.say(
            frozenset({Fact.STANDARD_DELIVERY_TIME}),
            entry("shipping.times.en.v1").claims.values(),
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


def test_wording_that_cannot_name_its_source_is_not_a_reply() -> None:
    """A bare string carries the same words with nothing behind them.

    Anywhere one is accepted, a string composed on the spot is accepted too,
    so the reply is a thing that cannot exist without its provenance.
    """
    with pytest.raises(UnattributedReplyError, match="names no approved wording"):
        ApprovedReply(text="Returns are accepted within 30 days.", said=())
    with pytest.raises(UnattributedReplyError, match="says nothing"):
        ApprovedReply(text="  ", said=("say:return_window.en.v1@sha256:abc",))


ELIGIBILITY_WORDS = {
    "standard_items": "Most items",
    "all_items": "All items",
    "selected_items": "Selected items",
}
CHOICES = {
    **WORDING,
    "fact": "return_eligibility",
    "sentence": "{eligibility} can be returned.",
    "words": ELIGIBILITY_WORDS,
}


def test_every_value_a_slot_can_hold_has_words_in_the_same_file() -> None:
    """Adding a choice to a claim is exactly when this gets forgotten.

    Where it used to surface was the reply, because the wording lived beside
    the data rather than beside the sentence spending it.
    """
    with pytest.raises(ValueError, match="no approved words"):
        ResponseTemplate.model_validate(
            {**CHOICES, "words": {"standard_items": "Most items"}}
        )


def test_words_for_a_value_no_claim_can_take_are_refused() -> None:
    """Wording nothing can select is wording nobody reviews."""
    with pytest.raises(ValueError, match="not a value any claim can take"):
        ResponseTemplate.model_validate(
            {**CHOICES, "words": {**ELIGIBILITY_WORDS, "clearance": "Clearance items"}}
        )


def test_rewriting_approved_words_moves_the_hash_under_one_version() -> None:
    """A version says which wording was meant, not what was under it.

    Editing approved copy in place is the change nobody announces and the one
    an audit most needs to see.
    """
    before = ResponseTemplate.model_validate(CHOICES)
    after = ResponseTemplate.model_validate(
        {**CHOICES, "words": {**ELIGIBILITY_WORDS, "standard_items": "Nearly all"}}
    )
    assert before.reference == after.reference
    assert before.content_hash != after.content_hash


def test_a_reply_records_the_exact_wording_it_used() -> None:
    library = load_templates()
    reply = library.say(
        frozenset({Fact.RETURN_WINDOW}),
        entry("returns.standard.en.v1").claims.values(),
        "en",
    )
    template = next(t for t in library if t.reference == "say:return_window.en.v1")
    assert reply.said == (f"say:return_window.en.v1@{template.content_hash}",)


def test_a_yes_or_no_reaches_a_customer_as_words() -> None:
    """A boolean printed raw says "False" to somebody who asked a question.

    It is an int in this language, so it has to be recognised before numbers
    are, or it prints as one.
    """
    library = load_templates()
    reply = library.say(
        frozenset({Fact.RETURN_FINAL_SALE}),
        entry("returns.standard.en.v1").claims.values(),
        "en",
    )
    assert "cannot be sent back" in reply.text
    assert "False" not in reply.text


def test_a_list_reaches_a_customer_as_a_sentence() -> None:
    library = load_templates()
    for locale, expected in (
        ("en", "underwear, swimwear and pierced jewellery"),
        ("fr", "sous-vêtements, maillots de bain et bijoux percés"),
    ):
        reply = library.say(
            frozenset({Fact.RETURN_EXCLUDED_CATEGORIES}),
            entry(f"returns.standard.{locale}.v1").claims.values(),
            locale,  # type: ignore[arg-type]
        )
        assert expected in reply.text
        assert "[" not in reply.text and "_" not in reply.text


def test_wording_must_cover_every_answer_a_yes_or_no_can_give() -> None:
    """Approved once, and then asked to say whatever the claim turns out to
    hold. Words for one of the two answers is words for half the policies.
    """
    with pytest.raises(ValueError, match="no approved words"):
        ResponseTemplate.model_validate(
            {
                **WORDING,
                "fact": "return_final_sale",
                "sentence": "Final sale items {final_sale_returnable}.",
                "words": {"final_sale_returnable_false": "cannot be sent back"},
            }
        )


def test_wording_must_cover_every_member_of_a_named_set() -> None:
    """A category added to the policy would otherwise render as its token."""
    with pytest.raises(ValueError, match="no approved words"):
        ResponseTemplate.model_validate(
            {
                **WORDING,
                "fact": "return_excluded_categories",
                "sentence": "Not {excluded_categories}.",
                "words": {"underwear": "underwear", "swimwear": "swimwear"},
            }
        )


def test_a_figure_is_printed_and_not_looked_up() -> None:
    """Thirty reads as thirty wherever it lands, so nobody approves it."""
    assert _how_to_say("return_window_days", int) == (
        _Rendering.AS_ITSELF,
        frozenset(),
    )


def test_a_choice_asks_for_words_for_every_option_it_has() -> None:
    """Not merely the one an entry happens to be holding.

    Wording is signed off once and used against whatever the claim later
    turns out to say, so an option nobody wrote for is a hole that opens on
    the day the policy is edited rather than the day it was approved.
    """
    how, keys = _how_to_say("eligibility", Literal["standard_items", "all_items"])
    assert how is _Rendering.FROM_WORDS
    assert keys == {"standard_items", "all_items"}


def test_a_yes_or_no_keys_its_words_by_the_field() -> None:
    """Two in one sentence would otherwise be reaching for the same key."""
    how, keys = _how_to_say("proof_of_purchase_required", bool)
    assert how is _Rendering.FROM_WORDS
    assert keys == {
        "proof_of_purchase_required_true",
        "proof_of_purchase_required_false",
    }


def test_a_set_asks_for_words_for_every_member_it_draws_from() -> None:
    how, keys = _how_to_say("excluded_categories", tuple[ItemCategory, ...])
    assert how is _Rendering.FROM_WORDS
    assert keys == {member.value for member in ItemCategory}


@pytest.mark.parametrize("declared", [str, date, Decimal, float, Literal[1, 2]])
def test_a_value_nothing_knows_how_to_say_is_refused(declared: object) -> None:
    """These two answers used to be the same answer.

    Asked what a figure needed spelling out, it said nothing; asked the same
    about free text it said nothing again, and the checks reading that reply
    are set comparisons which are satisfied by an empty one. A reference
    copied off a parcel, or a date, would have loaded without complaint and
    come apart later with somebody waiting on the reply.
    """
    with pytest.raises(ValueError, match="no rule here"):
        _how_to_say("whatever", declared)


def test_every_claim_a_policy_can_state_can_be_put_into_words() -> None:
    """What catches a field arriving that none of this can speak.

    A template is only measured against the claims it names, so a field no
    sentence has been written for yet would sit unexamined until one was.
    """
    for claims in (ReturnPolicyClaims, ShippingPolicyClaims):
        for field, declared in claims.model_fields.items():
            how, _ = _how_to_say(field, declared.annotation)
            assert isinstance(how, _Rendering)


STOCK = {
    "fact": "stock_availability",
    "locale": "en",
    "version": 1,
    "approved": True,
    "sentence": "That item {in_stock} at the moment.",
    "words": {"in_stock_true": "is in stock", "in_stock_false": "is out of stock"},
}


def a_product(*, in_stock: bool) -> ProductRecord:
    return ProductRecord(
        provider="demo",
        observed=Observation.LIVE,
        observed_at=datetime(2026, 9, 7, 12, 0, tzinfo=UTC),
        synthetic=True,
        reference="12",
        in_stock=in_stock,
        quantity=4 if in_stock else 0,
    )


def test_a_row_from_the_shop_can_be_held_against_approved_wording() -> None:
    """The check that used to know only about written policy.

    A sentence naming a slot nothing declares is refused when the file loads,
    and until now nothing declared what a row out of the shop states. So no
    wording for one could exist: not because none was written, but because
    none could have been accepted.

    No such wording ships yet. What this shows is that the refusal has moved
    from the shape of the evidence to whether anybody has approved a sentence.
    """
    approved = ResponseTemplate.model_validate(STOCK)
    assert approved.fact is Fact.STOCK_AVAILABILITY


def test_a_row_from_the_shop_fills_the_slots_a_policy_would() -> None:
    """One phrase book, two kinds of evidence, and it knows about neither.

    The library is handed values rather than the thing holding them, so a
    catalogue row and a policy entry arrive in the same shape and the wording
    cannot tell which it is serving.
    """
    library = TemplateLibrary([ResponseTemplate.model_validate(STOCK)])
    asked = frozenset({Fact.STOCK_AVAILABILITY})

    held = library.say(asked, a_product(in_stock=True).values(), "en")
    assert held.text == "That item is in stock at the moment."
    assert held.said == (f"{approved_reference()}@{approved_hash()}",)

    gone = library.say(asked, a_product(in_stock=False).values(), "en")
    assert gone.text == "That item is out of stock at the moment."


def approved_reference() -> str:
    return ResponseTemplate.model_validate(STOCK).reference


def approved_hash() -> str:
    return ResponseTemplate.model_validate(STOCK).content_hash


def test_written_policy_still_answers_exactly_as_before() -> None:
    """The boundary widened; nothing about the existing path moved.

    Every shipped policy answer goes through the same call, so a regression
    here would be a regression in the only thing that currently reaches a
    customer.
    """
    library = load_templates()
    both = frozenset({Fact.RETURN_WINDOW, Fact.RETURN_ELIGIBILITY})
    reply = library.say(both, entry("returns.standard.en.v1").claims.values(), "en")
    assert reply.text == (
        "Returns are accepted within 30 days of delivery. Most items can be "
        "returned if unworn and in the original packaging."
    )
