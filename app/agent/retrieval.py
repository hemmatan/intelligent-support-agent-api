"""Lexical search over the approved policy corpus.

Ranking decides which document inside the knowledge base answers a question.
It never decides whether the knowledge base is the right place to look at all:
"where is my order" matches a shipping article on every word and cannot be
answered by one. That decision belongs to the intent, not to a score.
"""

import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass

from app.agent.knowledge import (
    Locale,
    PolicyCorpus,
    ReturnPolicyEntry,
    ShippingPolicyEntry,
)

PolicyEntry = ReturnPolicyEntry | ShippingPolicyEntry

# Saturation and length normalisation, at the values BM25 is usually published
# with. Neither has been tuned against this corpus, which is far too small to
# tune anything against.
_K1 = 1.5
_B = 0.75

_WORD = re.compile(r"\w+")


def tokenise(text: str) -> list[str]:
    """Split into lowercase words with accents folded away.

    Customers type "hygiene" for "hygiène" and the reverse, and neither should
    decide whether they find the paragraph they were looking for.
    """
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    folded = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return _WORD.findall(folded)


@dataclass(frozen=True)
class Hit:
    """One entry the query matched, and how strongly."""

    entry: PolicyEntry
    score: float


class _LocaleIndex:
    """Term statistics for one language.

    Held separately because they are not comparable across languages: how rare
    a word is among the English entries says nothing about the French ones, and
    pooling them would let the size of one corpus move the ranking of another.
    """

    def __init__(self, entries: list[PolicyEntry]):
        self._entries = entries
        self._documents = [tokenise(entry.prose) for entry in entries]
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
        terms = tokenise(query)
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


class PolicyIndex:
    """Searchable view of the approved entries, one index per language."""

    def __init__(self, corpus: PolicyCorpus):
        grouped: dict[str, list[PolicyEntry]] = {}
        for entry in corpus:
            grouped.setdefault(entry.locale, []).append(entry)
        self._by_locale = {
            locale: _LocaleIndex(entries) for locale, entries in grouped.items()
        }

    def search(self, query: str, locale: Locale, limit: int = 5) -> list[Hit]:
        """Rank the entries written in `locale` against `query`.

        Entries in another language are not ranked lower, they are absent. An
        answer a customer cannot read is not a worse answer, it is not one.
        Nothing matching returns nothing, rather than the least bad entry.
        """
        index = self._by_locale.get(locale)
        if index is None:
            return []
        return index.search(query)[:limit]
