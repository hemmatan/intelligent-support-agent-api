"""Turning text into vectors, and comparing them.

The interface only. What produces the vectors is a deployment decision: a
hosted inference API in production, a hand-written stand-in in tests. Neither
is allowed to be a source of facts — an embedding decides which paragraph a
question is about, never what the paragraph says.
"""

import math
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

Vector = tuple[float, ...]


class EmbeddingUnavailableError(RuntimeError):
    """The embedder could not answer.

    Raised rather than returned so a caller has to decide what to do about it.
    Retrieval degrades to lexical search; it does not quietly return nothing
    and let an answerable question look unanswerable.
    """


@runtime_checkable
class Embedder(Protocol):
    """Anything that can turn text into comparable vectors."""

    async def embed(self, texts: Sequence[str]) -> list[Vector]:
        """Return one vector per input, in the order given.

        Raises EmbeddingUnavailableError if it cannot.
        """
        ...


def cosine_similarity(left: Vector, right: Vector) -> float:
    """Similarity of direction, ignoring magnitude.

    Zero for a zero vector rather than a division error: a text nothing could
    be said about is unlike everything, which is the answer callers want.
    """
    if len(left) != len(right):
        raise ValueError(f"vectors of length {len(left)} and {len(right)}")
    magnitude = math.sqrt(sum(value * value for value in left)) * math.sqrt(
        sum(value * value for value in right)
    )
    if magnitude == 0:
        return 0.0
    return sum(a * b for a, b in zip(left, right, strict=True)) / magnitude
