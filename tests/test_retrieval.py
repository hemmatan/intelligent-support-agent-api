"""Searching the policy corpus."""

from collections.abc import Sequence

import pytest

from app.agent.embedding import EmbeddingUnavailableError, Vector
from app.agent.knowledge import PolicyCorpus, ReturnPolicyEntry, load_corpus
from app.agent.retrieval import PolicyIndex, tokenise


class ConceptEmbedder:
    """A stand-in that recognises meaning we decide in advance.

    Not a model. It exists so a test can say "given an embedder that considers
    these two texts related, does fusion find the entry BM25 missed" without
    also asserting that some real model agrees they are related. That is a
    separate claim about a separate thing, and it belongs in a test that
    actually calls one.
    """

    CONCEPTS = {
        "returns": ("return", "retourner", "refund", "rembours", "change my mind"),
        "shipping": ("delivery", "livraison", "order", "dispatch", "shipped"),
    }

    def __init__(self, *, unavailable: bool = False):
        self.unavailable = unavailable
        self.calls = 0

    async def embed(self, texts: Sequence[str]) -> list[Vector]:
        self.calls += 1
        if self.unavailable:
            raise EmbeddingUnavailableError("no embedder configured for this test")
        return [self._vector(text.casefold()) for text in texts]

    def _vector(self, text: str) -> Vector:
        return tuple(
            float(any(word in text for word in words))
            for words in self.CONCEPTS.values()
        )


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


@pytest.mark.asyncio
async def test_a_returns_question_finds_the_returns_policy(index: PolicyIndex) -> None:
    hits = await index.search("how long do I have to return a jacket", "en")
    assert hits[0].entry.id == "returns.standard"


@pytest.mark.asyncio
async def test_matching_every_word_is_not_the_same_as_answering(
    index: PolicyIndex,
) -> None:
    """The reason source selection cannot be left to a retrieval score.

    "Where is my order" matches the shipping article so strongly that it is the
    only hit, and the shipping article cannot say where any particular order
    is. Ranking is being confident and useless at the same time.
    """
    hits = await index.search("where is my order", "en")
    assert [hit.entry.id for hit in hits] == ["shipping.times"]
    assert hits[0].lexical_rank == 1


@pytest.mark.asyncio
async def test_a_french_question_finds_the_french_entry(index: PolicyIndex) -> None:
    hits = await index.search("combien de temps pour retourner un article", "fr")
    assert hits[0].entry.reference == "kb:returns.standard.fr.v1"


@pytest.mark.asyncio
async def test_a_language_never_sees_another_languages_entries(
    index: PolicyIndex,
) -> None:
    """An answer the customer cannot read is absent, not merely ranked lower.

    "hygiene" is one of only two tokens the English and French return policies
    share, so it is one of the few queries that would reach across languages if
    nothing stopped it. A word only one of them contains proves nothing.
    """
    assert [hit.entry.locale for hit in await index.search("hygiene", "en")] == ["en"]
    assert [hit.entry.locale for hit in await index.search("hygiene", "fr")] == ["fr"]


@pytest.mark.asyncio
async def test_accents_do_not_decide_whether_a_customer_finds_the_paragraph(
    index: PolicyIndex,
) -> None:
    assert (await index.search("hygiene", "fr"))[0].entry.locale == "fr"
    assert (await index.search("hygiène", "fr"))[0].entry.locale == "fr"


@pytest.mark.asyncio
async def test_nothing_matching_returns_nothing(index: PolicyIndex) -> None:
    """Not the least bad entry. There is no policy about this."""
    assert await index.search("warranty on electrical goods", "en") == []
    assert await index.search("", "en") == []


@pytest.mark.asyncio
async def test_drafts_are_not_searchable() -> None:
    corpus = PolicyCorpus(
        [entry(approved=True, version=1), entry(approved=False, version=2)]
    )
    hits = await PolicyIndex(corpus).search("return", "en")
    assert [hit.entry.version for hit in hits] == [1]


@pytest.mark.asyncio
async def test_a_language_with_no_entries_returns_nothing(index: PolicyIndex) -> None:
    corpus = PolicyCorpus([entry(approved=True, version=1)])
    assert await PolicyIndex(corpus).search("retourner", "fr") == []


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


PARAPHRASE = "what happens if I change my mind about something I bought"


@pytest.mark.asyncio
async def test_lexical_search_alone_misses_a_paraphrase(index: PolicyIndex) -> None:
    """No content word shared with the returns policy: not return, not refund.

    An answerable question looks unanswerable, and would escalate to a human
    for no better reason than vocabulary. Before stopwords were removed this
    was worse than nothing: the question shared "my" with "where is my order",
    so words alone confidently returned the shipping article.
    """
    assert await index.search(PARAPHRASE, "en") == []


@pytest.mark.asyncio
async def test_fusion_finds_what_words_alone_could_not() -> None:
    hybrid = PolicyIndex(load_corpus(), ConceptEmbedder())
    await hybrid.warm()
    hits = await hybrid.search(PARAPHRASE, "en")
    assert [hit.entry.id for hit in hits] == ["returns.standard"]
    assert hits[0].lexical_rank is None
    assert hits[0].semantic_rank == 1


@pytest.mark.asyncio
async def test_an_entry_both_rankers_find_outranks_one_only_one_finds() -> None:
    """Agreement is what fusion adds; neither ranking states it alone.

    "days" appears in both English entries, so words alone put the shipping
    article second rather than nowhere. Only the returns policy is also found
    by meaning, and that is what separates them.
    """
    hybrid = PolicyIndex(load_corpus(), ConceptEmbedder())
    await hybrid.warm()
    hits = await hybrid.search("how many days do I have to return something", "en")
    assert [hit.entry.id for hit in hits] == ["returns.standard", "shipping.times"]
    assert (hits[0].lexical_rank, hits[0].semantic_rank) == (1, 1)
    assert (hits[1].lexical_rank, hits[1].semantic_rank) == (2, None)
    assert hits[0].score > hits[1].score


@pytest.mark.asyncio
async def test_an_unavailable_embedder_leaves_search_lexical() -> None:
    """One optional ranker being unreachable must not take the corpus offline."""
    degraded = PolicyIndex(load_corpus(), ConceptEmbedder(unavailable=True))
    await degraded.warm()
    assert degraded.semantic_ready is False
    hits = await degraded.search("how long do I have to return a jacket", "en")
    assert hits[0].entry.id == "returns.standard"
    assert hits[0].semantic_rank is None


@pytest.mark.asyncio
async def test_the_corpus_is_embedded_once_not_once_per_search() -> None:
    embedder = ConceptEmbedder()
    hybrid = PolicyIndex(load_corpus(), embedder)
    await hybrid.warm()
    for _ in range(3):
        await hybrid.search("return", "en")
    assert embedder.calls == 1 + 3  # the corpus once, then one query each
