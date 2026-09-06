"""Searching the policy corpus."""

from collections.abc import Sequence

import pytest

from app.agent.embedding import EmbeddingUnavailableError, Vector
from app.agent.knowledge import PolicyCorpus, ReturnPolicyEntry, load_corpus
from app.agent.reliability import ReliabilityLevel
from app.agent.retrieval import Hit, PolicyIndex, relevance_of, tokenise


class ConceptEmbedder:
    """A stand-in shaped like a real embedder, not like a convenient one.

    Vectors are dense and every pair scores well above zero, because that is
    what a sentence model does — its embeddings sit in a cone and unrelated
    texts land around 0.5 to 0.9, never at 0. A one-hot fake would make an
    unrelated query score exactly zero and quietly hide the fact that cosine
    similarity has no threshold below which a document stops being returned.

    It recognises meaning decided in advance. It exists to ask whether fusion
    surfaces what BM25 missed, not whether a real model finds two texts alike.
    That is a separate claim and needs a real model.
    """

    CONCEPTS = {
        "returns": ("return", "retourner", "refund", "rembours", "change my mind"),
        "shipping": ("delivery", "livraison", "order", "dispatch", "shipped"),
    }
    # Every text starts here, so nothing is ever orthogonal to anything.
    BASELINE = 0.45

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
            self.BASELINE + float(any(word in text for word in words))
            for words in self.CONCEPTS.values()
        )


@pytest.fixture(scope="module")
def index() -> PolicyIndex:
    return PolicyIndex(load_corpus())


def entry(
    *, approved: bool, version: int, policy_id: str = "returns.standard"
) -> ReturnPolicyEntry:
    return ReturnPolicyEntry.model_validate(
        {
            "id": policy_id,
            "locale": "en",
            "version": version,
            "approved": approved,
            "kind": "return_policy",
            "prose_template": "Return {eligibility} within {return_window_days} days.",
            "claims": {
                "return_window_days": 30,
                "eligibility": "standard_items",
                "excluded_categories": ["underwear"],
                "final_sale_returnable": False,
                "proof_of_purchase_required": True,
            },
            "words": {"standard_items": "most items", "underwear": "underwear"},
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
async def test_fusion_surfaces_what_words_alone_could_not() -> None:
    """Meaning finds the right entry. It does not get to send it unreviewed.

    Words alone return nothing here, so without an embedder this question
    reaches a human as a ticket. With one it reaches a staff member with the
    right policy already drafted, which is the improvement being claimed —
    not a direct answer.
    """
    hybrid = PolicyIndex(load_corpus(), ConceptEmbedder())
    await hybrid.warm()
    hits = await hybrid.search(PARAPHRASE, "en")
    assert hits[0].entry.id == "returns.standard"
    assert hits[0].lexical_rank is None
    assert relevance_of(hits) is ReliabilityLevel.REVIEW_ONLY


@pytest.mark.asyncio
async def test_meaning_cannot_tell_a_paraphrase_from_an_unanswerable_question() -> None:
    """The limit of retrieval, and the reason coverage exists.

    Nothing in the corpus mentions price matching. A real embedder returns the
    nearest entries anyway, because cosine similarity has no floor, so this
    produces the same shape as a question the corpus really can answer.
    Separating them needs a check on whether the entry carries the claim being
    asked for, which retrieval is not the place for.
    """
    hybrid = PolicyIndex(load_corpus(), ConceptEmbedder())
    await hybrid.warm()

    unanswerable = await hybrid.search("do you offer price matching", "en")
    paraphrase = await hybrid.search(PARAPHRASE, "en")

    assert unanswerable, "a real embedder returns something for anything"
    assert [(h.lexical_rank, h.semantic_rank) for h in unanswerable] == [
        (h.lexical_rank, h.semantic_rank) for h in paraphrase
    ]
    assert relevance_of(unanswerable) is relevance_of(paraphrase)


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
    assert hits[0].score > hits[1].score
    assert relevance_of(hits) is ReliabilityLevel.READY


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


def hit(policy_id: str, score: float, lexical: int | None, semantic: int | None) -> Hit:
    return Hit(
        entry=entry(approved=True, version=1, policy_id=policy_id),
        score=score,
        lexical_rank=lexical,
        semantic_rank=semantic,
    )


def test_nothing_retrieved_is_unusable() -> None:
    assert relevance_of([]) is ReliabilityLevel.UNUSABLE


def test_agreement_between_rankers_is_the_strongest_signal() -> None:
    hits = [
        hit("returns.standard", 0.0328, 1, 1),
        hit("shipping.times", 0.0161, 2, None),
    ]
    assert relevance_of(hits) is ReliabilityLevel.READY


def test_appearing_in_both_rankings_is_not_agreement() -> None:
    """The fused winner need not be either ranker's choice.

    Second place with one and first with the other outscores first place with
    one and third with the other, so an entry neither ranker preferred can
    lead while the two disagree about the subject of the question.
    """
    hits = [
        hit("shipping.times", 1 / 62 + 1 / 61, 2, 1),
        hit("returns.standard", 1 / 61 + 1 / 63, 1, 3),
    ]
    assert hits[0].score > hits[1].score
    assert relevance_of(hits) is ReliabilityLevel.REVIEW_ONLY


def test_one_ranker_alone_is_usable_but_not_the_strongest() -> None:
    """A lexical-only deployment answers; it never claims two opinions."""
    hits = [
        hit("returns.standard", 0.0164, 1, None),
        hit("shipping.times", 0.0161, 2, None),
    ]
    assert relevance_of(hits) is ReliabilityLevel.ACCEPTABLE


def test_rankers_disagreeing_about_the_subject_goes_to_review() -> None:
    """Two methods disagreeing about the subject, not two close answers."""
    hits = [
        hit("shipping.times", 0.0164, 1, None),
        hit("returns.standard", 0.0164, None, 1),
    ]
    assert relevance_of(hits) is ReliabilityLevel.REVIEW_ONLY


def test_a_single_hit_found_by_one_ranker_is_acceptable() -> None:
    assert (
        relevance_of([hit("returns.standard", 0.0164, 1, None)])
        is ReliabilityLevel.ACCEPTABLE
    )


@pytest.mark.asyncio
async def test_the_levels_a_real_search_produces(index: PolicyIndex) -> None:
    hybrid = PolicyIndex(load_corpus(), ConceptEmbedder())
    await hybrid.warm()

    agreed = await hybrid.search("how many days do I have to return something", "en")
    assert relevance_of(agreed) is ReliabilityLevel.READY

    lexical_only = await index.search("how long do I have to return a jacket", "en")
    assert relevance_of(lexical_only) is ReliabilityLevel.ACCEPTABLE

    assert relevance_of(await index.search("warranty on electrical goods", "en")) is (
        ReliabilityLevel.UNUSABLE
    )
