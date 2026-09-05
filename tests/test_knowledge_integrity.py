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

TEMPLATE = "You can return most items within {return_window_days} days."


def entry(
    *,
    locale: str = "en",
    version: int = 1,
    days: int = 30,
    approved: bool | None = None,
    prose_template: str = TEMPLATE,
) -> ReturnPolicyEntry:
    return ReturnPolicyEntry.model_validate(
        {
            "id": "returns.standard",
            "locale": locale,
            "version": version,
            "approved": locale == "en" if approved is None else approved,
            "kind": "return_policy",
            "prose_template": prose_template,
            "claims": {"return_window_days": days, "eligibility": "standard_items"},
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
        entry(prose_template="You can return most items within 30 days.")


def test_a_placeholder_naming_no_claim_is_rejected() -> None:
    with pytest.raises(ValidationError, match=r"undeclared \['refund_days'\]"):
        entry(prose_template="Refunds take {refund_days} and {return_window_days}.")


def test_a_numeric_claim_the_template_never_states_is_rejected() -> None:
    """Otherwise an author could stop using placeholders and drift again."""
    with pytest.raises(ValidationError, match=r"never states \['return_window_days'\]"):
        entry(prose_template="You can return most items. Conditions apply.")


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
