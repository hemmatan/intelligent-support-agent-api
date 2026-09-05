"""Loading the policy corpus, and the identities it refuses to accept."""

from pathlib import Path

import pytest

from app.agent.knowledge import (
    PolicyCorpus,
    PolicyCorpusError,
    ReturnPolicyEntry,
    load_corpus,
)

ENTRY = """
id = "returns.standard"
locale = "en"
version = {version}
approved = {approved}
kind = "return_policy"
prose = "You can return most items within 30 days of delivery."

[claims]
return_window_days = 30
eligibility = "standard_items"
"""


def write(directory: Path, name: str, **fields: object) -> None:
    body = ENTRY.format(
        version=fields.get("version", 1),
        approved=str(fields.get("approved", True)).lower(),
    )
    (directory / name).write_text(body)


def entry(
    version: int = 1, approved: bool = True, locale: str = "en"
) -> ReturnPolicyEntry:
    return ReturnPolicyEntry.model_validate(
        {
            "id": "returns.standard",
            "locale": locale,
            "version": version,
            "approved": approved,
            "kind": "return_policy",
            "prose": "You can return most items within 30 days of delivery.",
            "claims": {"return_window_days": 30, "eligibility": "standard_items"},
        }
    )


def test_the_repository_corpus_loads() -> None:
    corpus = load_corpus()
    assert {e.reference for e in corpus} == {
        "kb:returns.standard.en.v1",
        "kb:returns.standard.fr.v1",
        "kb:shipping.times.en.v1",
    }


def test_iterating_yields_approved_entries_only() -> None:
    """The safe set is the one you get without asking for anything."""
    corpus = PolicyCorpus(
        [entry(version=1, approved=True), entry(version=2, approved=False)]
    )
    assert [e.version for e in corpus] == [1]
    assert [e.version for e in corpus.including_drafts()] == [1, 2]


def test_the_same_identity_twice_is_rejected() -> None:
    with pytest.raises(PolicyCorpusError, match="defined more than once"):
        PolicyCorpus([entry(version=1), entry(version=1)])


def test_two_approved_versions_of_one_policy_are_rejected() -> None:
    """Otherwise which policy is in force depends on iteration order."""
    with pytest.raises(PolicyCorpusError, match="two approved versions"):
        PolicyCorpus([entry(version=1), entry(version=2)])


def test_an_older_version_may_stay_alongside_the_approved_one() -> None:
    corpus = PolicyCorpus([entry(version=1, approved=False), entry(version=2)])
    assert [e.version for e in corpus.including_drafts()] == [1, 2]
    assert [e.version for e in corpus] == [2]


def test_malformed_toml_is_rejected_at_load(tmp_path: Path) -> None:
    (tmp_path / "returns.standard.en.v1.toml").write_text("id = 'unterminated")
    with pytest.raises(PolicyCorpusError, match="not a valid policy"):
        load_corpus(tmp_path)


def test_content_that_fails_the_model_is_rejected_at_load(tmp_path: Path) -> None:
    (tmp_path / "returns.standard.en.v1.toml").write_text(
        ENTRY.format(version=1, approved="true").replace("30", '"about a month"')
    )
    with pytest.raises(PolicyCorpusError, match="not a valid policy"):
        load_corpus(tmp_path)


def test_a_filename_that_disagrees_with_its_contents_is_rejected(
    tmp_path: Path,
) -> None:
    """The commonest mistake here is copying a file and half-editing it."""
    write(tmp_path, "returns.standard.fr.v1.toml")
    with pytest.raises(PolicyCorpusError, match="declares itself to be"):
        load_corpus(tmp_path)


def test_an_empty_directory_is_a_fault_not_an_empty_corpus(tmp_path: Path) -> None:
    with pytest.raises(PolicyCorpusError, match="no policy files"):
        load_corpus(tmp_path)


def test_one_policy_cannot_be_approved_at_two_versions_in_two_languages() -> None:
    """Otherwise English and French customers are told different rules."""
    with pytest.raises(PolicyCorpusError, match="approved at en v2, fr v1"):
        PolicyCorpus(
            [
                entry(version=1, approved=False),
                entry(version=2, approved=True),
                entry(version=1, approved=True, locale="fr"),
            ]
        )


def test_translations_must_be_the_same_kind_of_policy() -> None:
    from app.agent.knowledge import ShippingPolicyEntry

    shipping = ShippingPolicyEntry.model_validate(
        {
            "id": "returns.standard",
            "locale": "fr",
            "version": 1,
            "approved": True,
            "kind": "shipping_policy",
            "prose": "Livraison en 2 a 3 jours, express en 1 jour.",
            "claims": {
                "standard_delivery_days_min": 2,
                "standard_delivery_days_max": 3,
                "express_delivery_days": 1,
            },
        }
    )
    with pytest.raises(PolicyCorpusError, match="return_policy and a shipping_policy"):
        PolicyCorpus([entry(), shipping])
