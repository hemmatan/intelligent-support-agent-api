"""A corpus whose numbers disagree with themselves does not load."""

import pytest

from app.agent.knowledge import (
    PolicyCorpus,
    PolicyCorpusError,
    ReturnPolicyEntry,
    load_corpus,
)


def entry(
    *,
    locale: str = "en",
    version: int = 1,
    days: int = 30,
    approved: bool | None = None,
    prose: str = "You can return most items within 30 days of delivery.",
) -> ReturnPolicyEntry:
    return ReturnPolicyEntry.model_validate(
        {
            "id": "returns.standard",
            "locale": locale,
            "version": version,
            "approved": locale == "en" if approved is None else approved,
            "kind": "return_policy",
            "prose": prose,
            "claims": {"return_window_days": days, "eligibility": "standard_items"},
        }
    )


def test_the_repository_corpus_is_internally_consistent() -> None:
    assert len(load_corpus()) == 3


def test_prose_edited_without_the_claim_is_rejected() -> None:
    """The failure this exists for: 14 in the text, 30 in the slot."""
    with pytest.raises(PolicyCorpusError, match="return_window_days = 30"):
        PolicyCorpus([entry(prose="You can return most items within 14 days.")])


def test_a_number_inside_a_longer_number_does_not_count() -> None:
    """A substring check would find 30 in 130 and pass."""
    with pytest.raises(PolicyCorpusError, match="does not appear in its prose"):
        PolicyCorpus([entry(prose="We process up to 130 returns each day.")])


def test_a_number_beside_a_non_digit_does_count() -> None:
    PolicyCorpus([entry(days=3, prose="Orders placed after 3pm ship the next day.")])


def test_non_numeric_claims_are_not_checked_against_prose() -> None:
    """eligibility = standard_items will never appear in a sentence."""
    PolicyCorpus([entry(prose="Most items may be sent back within 30 days.")])


def test_locales_of_one_version_must_agree_on_the_rule() -> None:
    with pytest.raises(PolicyCorpusError, match=r"\['return_window_days'\]"):
        PolicyCorpus(
            [
                entry(locale="en", days=30),
                entry(
                    locale="fr",
                    days=14,
                    prose="Vous pouvez retourner vos articles sous 14 jours.",
                ),
            ]
        )


def test_translations_may_differ_while_the_rule_does_not() -> None:
    PolicyCorpus(
        [
            entry(locale="en", days=30),
            entry(
                locale="fr",
                days=30,
                prose="Vous pouvez retourner vos articles sous 30 jours.",
            ),
        ]
    )


def test_different_versions_are_allowed_to_state_different_rules() -> None:
    """A policy change is the point of a new version."""
    PolicyCorpus(
        [
            entry(version=1, days=30, approved=False),
            entry(
                version=2,
                days=14,
                prose="You can return most items within 14 days of delivery.",
            ),
        ]
    )
