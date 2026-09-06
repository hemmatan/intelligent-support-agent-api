"""Searchable prose is rendered from claims, so a figure has one source."""

import pytest
from pydantic import ValidationError

from app.agent.knowledge import (
    PolicyCorpus,
    PolicyCorpusError,
    ReturnPolicyEntry,
    ShippingPolicyEntry,
    load_corpus,
)

TEMPLATE = "You can return {eligibility} within {return_window_days} days."


def entry(
    *,
    locale: str = "en",
    version: int = 1,
    days: int = 30,
    approved: bool | None = None,
    prose_template: str = TEMPLATE,
    words: dict[str, str] | None = None,
    categories: list[str] | None = None,
    proof_required: bool = True,
) -> ReturnPolicyEntry:
    return ReturnPolicyEntry.model_validate(
        {
            "id": "returns.standard",
            "locale": locale,
            "version": version,
            "approved": locale == "en" if approved is None else approved,
            "kind": "return_policy",
            "prose_template": prose_template,
            "claims": {
                "return_window_days": days,
                "eligibility": "standard_items",
                "excluded_categories": (
                    ["underwear"] if categories is None else categories
                ),
                "sale_items_follow_standard_window": True,
                "final_sale_returnable": False,
                "proof_of_purchase_required": proof_required,
            },
            "words": words
            or {
                "standard_items": "most items",
                "underwear": "underwear",
                "proof_of_purchase_required_true": "returned with a receipt",
                "proof_of_purchase_required_false": "returned without a receipt",
                "sale_items_follow_standard_window_true": "are treated the same way",
                "sale_items_follow_standard_window_false": "may not be sent back",
                "final_sale_returnable_true": "may be sent back too",
                "final_sale_returnable_false": "may not",
            },
        }
    )


def flat(text: str) -> str:
    """Compare on words, not on where the author happened to wrap a line."""
    return " ".join(text.split())


def test_the_repository_corpus_renders() -> None:
    prose = {e.reference: flat(e.prose) for e in load_corpus()}
    assert "within 30 days" in prose["kb:returns.standard.en.v1"]
    assert "les 30 jours" in prose["kb:returns.standard.fr.v1"]
    assert "takes 2 to 3 business days" in prose["kb:shipping.times.en.v1"]


def test_the_shipping_entry_still_opens_on_the_question_it_cannot_answer() -> None:
    """Step 3 needs a document that matches "where is my order" and cannot answer it."""
    shipping = next(e for e in load_corpus() if e.id == "shipping.times")
    assert flat(shipping.prose).startswith("Where is my order?")


def test_a_repeated_placeholder_renders_every_time() -> None:
    rendered = entry(
        prose_template="{return_window_days} days, and still {return_window_days}."
    ).prose
    assert rendered == "30 days, and still 30."


def test_a_literal_digit_in_a_template_is_rejected() -> None:
    """The drift this design removes: a figure written down a second time."""
    with pytest.raises(ValidationError, match="literal digit"):
        entry(prose_template="You can return {eligibility} within 30 days.")


def test_a_placeholder_naming_no_claim_is_rejected() -> None:
    with pytest.raises(ValidationError, match=r"undeclared \['refund_days'\]"):
        entry(prose_template="Refunds take {refund_days} and {return_window_days}.")


def test_a_numeric_claim_the_template_never_states_is_rejected() -> None:
    """Otherwise an author could stop using placeholders and drift again."""
    with pytest.raises(ValidationError, match=r"never states \['return_window_days'\]"):
        entry(prose_template="You can return {eligibility}. Conditions apply.")


def test_a_non_numeric_claim_need_not_appear() -> None:
    """eligibility renders as an identifier, not as English."""
    assert "standard_items" not in entry().prose


def test_the_wrong_sentence_case_can_no_longer_be_written() -> None:
    """Prose said "2 to 4 business days" while the claim said 3, satisfied by "3pm"."""
    with pytest.raises(ValidationError, match="literal digit"):
        ShippingPolicyEntry.model_validate(
            {
                "id": "shipping.times",
                "locale": "en",
                "version": 1,
                "approved": True,
                "kind": "shipping_policy",
                "prose_template": (
                    "Standard delivery takes 2 to 4 business days. "
                    "Orders after 3pm ship the next day."
                ),
                "claims": {
                    "standard_delivery_days_min": 2,
                    "standard_delivery_days_max": 3,
                    "express_delivery_days": 1,
                },
            }
        )


def test_changing_a_claim_changes_both_the_prose_and_the_hash() -> None:
    original, revised = entry(days=30), entry(days=14)
    assert "30 days" in original.prose and "14 days" in revised.prose
    assert original.content_hash != revised.content_hash


def test_rewording_a_template_changes_the_hash() -> None:
    reworded = entry(prose_template="Send items back within {return_window_days} days.")
    assert reworded.prose != entry().prose
    assert reworded.content_hash != entry().content_hash


def test_locales_of_one_version_must_still_agree_on_the_rule() -> None:
    with pytest.raises(PolicyCorpusError, match=r"\['return_window_days'\]"):
        PolicyCorpus(
            [
                entry(locale="en", days=30),
                entry(
                    locale="fr",
                    days=14,
                    prose_template="Retour sous {return_window_days} jours.",
                ),
            ]
        )


def test_a_value_the_claims_hold_may_not_be_typed_into_the_prose() -> None:
    """The digit ban, applied to everything that is not a digit.

    A word spelled out by hand is a second copy of a claim, free to drift from
    the first. The drift is the interesting part: an entry becomes findable by
    a word it no longer states, or unfindable by one it does, and neither shows
    up as a wrong answer until somebody asks.
    """
    with pytest.raises(ValidationError, match="spells out"):
        entry(prose_template="Return most items within {return_window_days} days.")


def test_a_value_with_no_words_here_is_refused() -> None:
    """Otherwise the searchable text carries a slug nobody would type."""
    with pytest.raises(ValidationError, match="no words for"):
        ReturnPolicyEntry.model_validate(
            {
                "id": "returns.standard",
                "locale": "en",
                "version": 1,
                "approved": True,
                "kind": "return_policy",
                "prose_template": (
                    "Return {eligibility} within {return_window_days} days. "
                    "Not {excluded_categories}."
                ),
                "claims": {
                    "return_window_days": 30,
                    "eligibility": "standard_items",
                    "excluded_categories": ["underwear", "swimwear"],
                    "sale_items_follow_standard_window": True,
                    "final_sale_returnable": False,
                    "proof_of_purchase_required": True,
                },
                "words": {
                    "standard_items": "most items",
                    "underwear": "underwear",
                    "proof_of_purchase_required_true": "returned with a receipt",
                    "proof_of_purchase_required_false": "returned without a receipt",
                    "sale_items_follow_standard_window_true": (
                        "are treated the same way"
                    ),
                    "sale_items_follow_standard_window_false": "may not be sent back",
                    "final_sale_returnable_true": "may be sent back too",
                    "final_sale_returnable_false": "may not",
                },
            }
        )


def test_words_for_a_value_the_claims_never_hold_are_refused() -> None:
    """Wording nothing can select is wording nobody reviews."""
    complete = entry().words
    with pytest.raises(ValidationError, match="is not a value these claims hold"):
        entry(words=dict(complete) | {"swimwear": "swimwear"})


def test_a_list_reads_as_a_sentence_in_either_language() -> None:
    """Rendered, not printed. A raw sequence would put brackets and internal
    tokens into the text retrieval searches, matching nothing anybody types.
    """
    english, french = (
        next(e for e in load_corpus() if e.id == "returns.standard" and e.locale == loc)
        for loc in ("en", "fr")
    )
    assert "underwear, swimwear and pierced jewellery" in english.prose
    assert "sous-vêtements, maillots de bain et bijoux percés" in french.prose
    for rendered in (english.prose, french.prose):
        assert "[" not in rendered and "_" not in rendered


@pytest.mark.parametrize(
    ("kinds", "complaint"),
    [([], "at least 1"), (["underwear", "underwear"], "more than once")],
    ids=["empty", "duplicated"],
)
def test_a_list_that_cannot_be_said_is_refused(
    kinds: list[str], complaint: str
) -> None:
    """An empty one rendered as a hole in the middle of a sentence.

    "We cannot accept returns of , for hygiene reasons" went to a customer as
    a direct answer. A repeated one reads back as a stutter. Both are caught
    where the claim is written rather than where it is spoken.
    """
    with pytest.raises(ValidationError, match=complaint):
        entry(categories=kinds)


def test_a_yes_or_no_named_in_the_prose_needs_words_for_both_answers() -> None:
    """It validated, and then raised on the way to being read.

    Wording was collected for text values only, so a policy naming one of
    these passed every check and failed at rendering — the one place a fault
    reaches somebody waiting.
    """
    with pytest.raises(ValidationError, match="no words for"):
        entry(words={"standard_items": "most items", "underwear": "underwear"})


def test_a_yes_or_no_cannot_drift_from_the_sentence_describing_it() -> None:
    """Flipping the claim used to leave the prose saying the opposite.

    Both are now rendered, so the sentence follows the claim or the entry does
    not load at all.
    """
    naming = (
        "Return {eligibility} in {return_window_days} days, "
        "{proof_of_purchase_required}."
    )
    assert "returned with a receipt" in entry(prose_template=naming).prose
    assert (
        "returned without a receipt"
        in entry(prose_template=naming, proof_required=False).prose
    )
