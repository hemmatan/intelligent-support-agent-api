"""Search over the approved policy corpus.

Ranking decides which document inside the knowledge base answers a question.
It never decides whether the knowledge base is the right place to look at all:
"where is my order" matches a shipping article on every word and cannot be
answered by one. That decision belongs to the intent, not to a score.

Two rankers run where an embedder is configured. Words find what the customer
literally wrote; vectors find what they meant. Neither alone covers both.
"""

import logging
import math
import re
from collections import Counter
from dataclasses import dataclass

from app.agent.embedding import (
    Embedder,
    EmbeddingUnavailableError,
    Vector,
    cosine_similarity,
)
from app.agent.knowledge import (
    Locale,
    PolicyCorpus,
    ReturnPolicyEntry,
    ShippingPolicyEntry,
)
from app.agent.reliability import ReliabilityLevel
from app.agent.text import fold

_log = logging.getLogger(__name__)

PolicyEntry = ReturnPolicyEntry | ShippingPolicyEntry

# Saturation and length normalisation, at the values BM25 is usually published
# with. Neither has been tuned against this corpus, which is far too small to
# tune anything against.
_K1 = 1.5
_B = 0.75

# Reciprocal rank fusion, at the constant it is usually published with. It
# damps the advantage of a first place, so one ranker being certain does not
# overrule the other being certain about something else.
_RRF_K = 60

_WORD = re.compile(r"\w+")

# Words carrying no topic. At three documents the rarity term cannot damp them
# on its own: "my" appears in one English entry of two, so a question sharing
# nothing but "my" with a shipping article still ranked it first. Held per
# language because they have nothing to do with each other, and applied before
# scoring rather than inside tokenise, so that tokenise stays a statement about
# words rather than about relevance.
_ENGLISH_STOPWORDS = frozenset(
    (
        "a",
        "about",
        "all",
        "am",
        "an",
        "and",
        "any",
        "are",
        "as",
        "at",
        "be",
        "been",
        "but",
        "by",
        "can",
        "do",
        "does",
        "for",
        "from",
        "had",
        "has",
        "have",
        "how",
        "i",
        "if",
        "in",
        "is",
        "it",
        "its",
        "me",
        "my",
        "no",
        "not",
        "of",
        "on",
        "or",
        "our",
        "so",
        "than",
        "that",
        "the",
        "their",
        "them",
        "then",
        "there",
        "these",
        "they",
        "this",
        "to",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "will",
        "with",
        "would",
        "you",
        "your",
    )
)

_FRENCH_STOPWORDS = frozenset(
    (
        "au",
        "aux",
        "avec",
        "ce",
        "ces",
        "dans",
        "de",
        "des",
        "du",
        "elle",
        "en",
        "est",
        "et",
        "eux",
        "il",
        "je",
        "la",
        "le",
        "les",
        "leur",
        "lui",
        "ma",
        "mais",
        "me",
        "meme",
        "mes",
        "moi",
        "mon",
        "ne",
        "nos",
        "notre",
        "nous",
        "on",
        "ou",
        "par",
        "pas",
        "pour",
        "qu",
        "que",
        "qui",
        "sa",
        "se",
        "ses",
        "son",
        "sur",
        "ta",
        "te",
        "tes",
        "toi",
        "ton",
        "tu",
        "un",
        "une",
        "vos",
        "votre",
        "vous",
        "y",
    )
)

_STOPWORDS = _ENGLISH_STOPWORDS | _FRENCH_STOPWORDS


def _content_terms(text: str) -> list[str]:
    return [term for term in tokenise(text) if term not in _STOPWORDS]


def tokenise(text: str) -> list[str]:
    """Split into lowercase words with accents folded away.

    Customers type "hygiene" for "hygiène" and the reverse, and neither should
    decide whether they find the paragraph they were looking for.
    """
    return _WORD.findall(fold(text))


@dataclass(frozen=True)
class Hit:
    """One entry the query matched, and which rankers found it.

    The ranks are kept because agreement between two rankers says more about
    a result than either ranking says alone, and a fused score has thrown that
    away by the time it is a single number.
    """

    entry: PolicyEntry
    score: float
    lexical_rank: int | None = None
    semantic_rank: int | None = None


class _LocaleIndex:
    """Term statistics for one language.

    Held separately because they are not comparable across languages: how rare
    a word is among the English entries says nothing about the French ones, and
    pooling them would let the size of one corpus move the ranking of another.
    """

    def __init__(self, entries: list[PolicyEntry]):
        self._entries = entries
        self._documents = [_content_terms(entry.prose) for entry in entries]
        self._frequencies = [Counter(document) for document in self._documents]
        self._lengths = [len(document) for document in self._documents]
        self._average_length = sum(self._lengths) / len(self._lengths)
        self._document_frequency = Counter(
            term for document in self._documents for term in set(document)
        )

    def _inverse_document_frequency(self, term: str) -> float:
        total = len(self._entries)
        containing = self._document_frequency[term]
        return math.log(1 + (total - containing + 0.5) / (containing + 0.5))

    def search(self, query: str) -> list[Hit]:
        terms = _content_terms(query)
        hits = []
        for position, entry in enumerate(self._entries):
            frequencies = self._frequencies[position]
            length = self._lengths[position]
            score = 0.0
            for term in terms:
                occurrences = frequencies[term]
                if not occurrences:
                    continue
                saturated = occurrences * (_K1 + 1)
                normaliser = occurrences + _K1 * (
                    1 - _B + _B * length / self._average_length
                )
                score += self._inverse_document_frequency(term) * saturated / normaliser
            if score > 0:
                hits.append(Hit(entry=entry, score=score))
        return sorted(hits, key=lambda hit: hit.score, reverse=True)


def _fuse(rankings: list[list[PolicyEntry]]) -> list[Hit]:
    """Combine rankings by reciprocal rank.

    Adding the rankers' own scores would mean converting between a BM25 score
    and a cosine similarity, which have no common unit. Positions do: being
    first means the same thing to both.
    """
    scores: dict[str, float] = {}
    entries: dict[str, PolicyEntry] = {}
    positions: list[dict[str, int]] = []
    for ranking in rankings:
        placed = {}
        for position, entry in enumerate(ranking, start=1):
            scores[entry.reference] = scores.get(entry.reference, 0.0) + 1 / (
                _RRF_K + position
            )
            entries[entry.reference] = entry
            placed[entry.reference] = position
        positions.append(placed)

    lexical, semantic = (positions + [{}, {}])[:2]
    hits = [
        Hit(
            entry=entry,
            score=scores[reference],
            lexical_rank=lexical.get(reference),
            semantic_rank=semantic.get(reference),
        )
        for reference, entry in entries.items()
    ]
    return sorted(hits, key=lambda hit: hit.score, reverse=True)


class PolicyIndex:
    """Searchable view of the approved entries, one index per language."""

    def __init__(self, corpus: PolicyCorpus, embedder: Embedder | None = None):
        grouped: dict[str, list[PolicyEntry]] = {}
        for entry in corpus:
            grouped.setdefault(entry.locale, []).append(entry)
        self._entries_by_locale = grouped
        self._by_locale = {
            locale: _LocaleIndex(entries) for locale, entries in grouped.items()
        }
        self._embedder = embedder
        self._vectors: dict[str, Vector] = {}

    @property
    def semantic_ready(self) -> bool:
        """Whether vectors are available. False means searches are lexical only."""
        return bool(self._vectors)

    async def warm(self) -> None:
        """Embed the corpus once, if an embedder was supplied.

        A temporary failure leaves the index lexical, loudly. Refusing to start
        because a second ranker is briefly unreachable would take the whole
        knowledge base offline over a degraded feature.

        A misconfigured embedder is not caught here. It will fail identically
        on every future call, and a deployment that quietly runs for months on
        half its retrieval while reporting itself healthy is worse than one
        that will not start.
        """
        if self._embedder is None:
            return
        entries = [
            entry for group in self._entries_by_locale.values() for entry in group
        ]
        try:
            vectors = await self._embedder.embed([entry.prose for entry in entries])
        except EmbeddingUnavailableError as exc:
            _log.warning("searching lexically: the embedder is unavailable (%s)", exc)
            return
        self._vectors = {
            entry.reference: vector
            for entry, vector in zip(entries, vectors, strict=True)
        }

    async def _semantic_ranking(self, query: str, locale: Locale) -> list[PolicyEntry]:
        if self._embedder is None or not self._vectors:
            return []
        try:
            embedded = await self._embedder.embed([query])
        except EmbeddingUnavailableError as exc:
            _log.warning("ranking this query lexically: %s", exc)
            return []
        scored = [
            (cosine_similarity(embedded[0], self._vectors[entry.reference]), entry)
            for entry in self._entries_by_locale.get(locale, [])
            if entry.reference in self._vectors
        ]
        ranked = sorted(
            (pair for pair in scored if pair[0] > 0),
            key=lambda pair: pair[0],
            reverse=True,
        )
        return [entry for _, entry in ranked]

    async def search(self, query: str, locale: Locale, limit: int = 5) -> list[Hit]:
        """Rank the entries written in `locale` against `query`.

        Entries in another language are not ranked lower, they are absent. An
        answer a customer cannot read is not a worse answer, it is not one.
        Nothing matching returns nothing, rather than the least bad entry.
        """
        index = self._by_locale.get(locale)
        lexical = [hit.entry for hit in index.search(query)] if index else []
        semantic = await self._semantic_ranking(query, locale)
        if not lexical and not semantic:
            return []
        return _fuse([lexical, semantic])[:limit]


def relevance_of(hits: list[Hit]) -> ReliabilityLevel:
    """How far retrieval can be trusted about which entry answers the question.

    Read from which entry each ranker chose, not from the size of the winning
    number. A cutoff would need a figure nobody measured; whether two
    independent methods picked the same document needs no unit at all.

    Agreeing means leading with the same entry. Merely appearing in both
    rankings is weaker than it looks, because fusion can turn it into a win:
    an entry placed second by one ranker and first by the other outscores one
    placed first and third, so a document neither ranker preferred outright
    can lead while the two disagree about what the question is about.

    Meaning alone never reaches a customer unreviewed. Cosine similarity has
    no zero: unrelated texts sit well above it, so a high one says a document
    is the closest of those available, not that it answers anything. Words
    corroborating it is evidence; the number on its own is not. Whether the
    entry actually carries the claim an answer needs is a question for
    coverage, which can answer it definitely rather than by degree.
    """
    if not hits:
        return ReliabilityLevel.UNUSABLE

    lexical_first = next((hit for hit in hits if hit.lexical_rank == 1), None)
    semantic_first = next((hit for hit in hits if hit.semantic_rank == 1), None)

    if lexical_first is None:
        return ReliabilityLevel.REVIEW_ONLY
    if semantic_first is None:
        return ReliabilityLevel.ACCEPTABLE
    if lexical_first.entry.reference == semantic_first.entry.reference:
        return ReliabilityLevel.READY
    return ReliabilityLevel.REVIEW_ONLY
