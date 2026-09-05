"""Searching the policy corpus."""

import pytest

from app.agent.knowledge import PolicyCorpus, ReturnPolicyEntry, load_corpus
from app.agent.retrieval import PolicyIndex, tokenise


@pytest.fixture(scope="module")
def index() -> PolicyIndex:
    return PolicyIndex(load_corpus())


def entry(*, approved: bool, version: int) -> ReturnPolicyEntry:
    return ReturnPolicyEntry.model_validate(
        {
            "id": "returns.standard",
            "locale": "en",
            "version": version,
            "approved": approved,
            "kind": "return_policy",
            "prose_template": "Return most items within {return_window_days} days.",
            "claims": {"return_window_days": 30, "eligibility": "standard_items"},
        }
    )


def test_a_returns_question_finds_the_returns_policy(index: PolicyIndex) -> None:
    hits = index.search("how long do I have to return a jacket", "en")
    assert hits[0].entry.id == "returns.standard"


def test_matching_every_word_is_not_the_same_as_answering(index: PolicyIndex) -> None:
    """The reason source selection cannot be left to a retrieval score.

    "Where is my order" matches the shipping article so strongly that it is the
    only hit, and the shipping article cannot say where any particular order
    is. Ranking is being confident and useless at the same time.
    """
    hits = index.search("where is my order", "en")
    assert [hit.entry.id for hit in hits] == ["shipping.times"]
    assert hits[0].score > index.search("how long to return", "en")[0].score


def test_a_french_question_finds_the_french_entry(index: PolicyIndex) -> None:
    hits = index.search("combien de temps pour retourner un article", "fr")
    assert hits[0].entry.reference == "kb:returns.standard.fr.v1"


def test_a_language_never_sees_another_languages_entries(index: PolicyIndex) -> None:
    """An answer the customer cannot read is absent, not merely ranked lower.

    "hygiene" is one of only two tokens the English and French return policies
    share, so it is one of the few queries that would reach across languages if
    nothing stopped it. A word only one of them contains proves nothing.
    """
    assert [hit.entry.locale for hit in index.search("hygiene", "en")] == ["en"]
    assert [hit.entry.locale for hit in index.search("hygiene", "fr")] == ["fr"]


def test_accents_do_not_decide_whether_a_customer_finds_the_paragraph(
    index: PolicyIndex,
) -> None:
    assert index.search("hygiene", "fr")[0].entry.locale == "fr"
    assert index.search("hygiène", "fr")[0].entry.locale == "fr"


def test_nothing_matching_returns_nothing(index: PolicyIndex) -> None:
    """Not the least bad entry. There is no policy about this."""
    assert index.search("warranty on electrical goods", "en") == []
    assert index.search("", "en") == []


def test_drafts_are_not_searchable() -> None:
    corpus = PolicyCorpus(
        [entry(approved=True, version=1), entry(approved=False, version=2)]
    )
    hits = PolicyIndex(corpus).search("return", "en")
    assert [hit.entry.version for hit in hits] == [1]


def test_a_language_with_no_entries_returns_nothing(index: PolicyIndex) -> None:
    corpus = PolicyCorpus([entry(approved=True, version=1)])
    assert PolicyIndex(corpus).search("retourner", "fr") == []


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Return Within 30 Days", ["return", "within", "30", "days"]),
        ("hygiène", ["hygiene"]),
        ("non-portés, d'origine", ["non", "portes", "d", "origine"]),
        ("", []),
    ],
    ids=["lowercased", "accents-folded", "split-on-punctuation", "empty"],
)
def test_tokenisation(text: str, expected: list[str]) -> None:
    assert tokenise(text) == expected
