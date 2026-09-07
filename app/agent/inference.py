"""Embeddings from the Hugging Face inference API.

Hosted rather than local. A sentence-transformers install would put torch in
the application image for one optional ranker, and the fallback that already
exists — searching lexically when meaning is unavailable — covers the outage
this trades for.
"""

import math
from collections.abc import Sequence

import httpx

from app.agent.embedding import (
    EmbeddingMisconfiguredError,
    EmbeddingUnavailableError,
    Vector,
)
from app.core.config import Settings

# Answers that will be identical however many times they are asked for.
_CONFIGURATION_FAILURES = frozenset((401, 403, 404))

_ENDPOINT = "https://router.huggingface.co/hf-inference/models/{model}/pipeline/feature-extraction"


class HuggingFaceEmbedder:
    """Calls a hosted feature-extraction model.

    Failures divide by whether trying again could help. A timeout or a model
    still warming up is worth carrying on without. A refused token or a model
    that does not exist is a mistake in the configuration, and pretending it
    is weather means nobody finds out.
    """

    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None):
        if settings.HUGGINGFACE_API_TOKEN is None:
            raise ValueError("HUGGINGFACE_API_TOKEN is not configured")
        self._token = settings.HUGGINGFACE_API_TOKEN.get_secret_value()
        self._url = _ENDPOINT.format(model=settings.EMBEDDING_MODEL)
        self._timeout = settings.EMBEDDING_TIMEOUT_SECONDS
        self._client = client

    async def embed(self, texts: Sequence[str]) -> list[Vector]:
        if not texts:
            return []
        payload = {"inputs": list(texts), "options": {"wait_for_model": True}}
        headers = {"Authorization": f"Bearer {self._token}"}
        try:
            if self._client is not None:
                response = await self._client.post(
                    self._url, json=payload, headers=headers, timeout=self._timeout
                )
            else:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    response = await client.post(
                        self._url, json=payload, headers=headers
                    )
            if response.status_code in _CONFIGURATION_FAILURES:
                raise EmbeddingMisconfiguredError(
                    f"{self._url} answered {response.status_code}; "
                    f"check DORNASHOP_HUGGINGFACE_API_TOKEN and "
                    f"DORNASHOP_EMBEDDING_MODEL"
                )
            response.raise_for_status()
            body = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise EmbeddingUnavailableError(f"{type(exc).__name__}: {exc}") from exc

        return _vectors_from(body, expected=len(texts))


def _vectors_from(body: object, expected: int) -> list[Vector]:
    """Read one vector per input out of the response.

    Feature extraction returns a vector per input for a sentence-transformers
    model, and a vector per token for a plain encoder. Only the first shape is
    usable here, so the second is refused rather than quietly averaged into
    something that would rank and mean nothing.

    Everything about the shape is checked here because nothing downstream
    checks it again. Vectors of unequal length raise when compared, and a
    single NaN turns every similarity involving it into NaN, which sorts
    unpredictably and never looks like an error.
    """
    if not isinstance(body, list) or len(body) != expected:
        raise EmbeddingUnavailableError(
            f"expected {expected} vectors, got {body!r:.80}"
        )

    vectors: list[Vector] = []
    dimension: int | None = None
    for item in body:
        if not isinstance(item, list) or not item:
            raise EmbeddingUnavailableError(
                "expected one non-empty vector per input; the model returned "
                "token-level output or nothing at all"
            )
        if dimension is None:
            dimension = len(item)
        elif len(item) != dimension:
            raise EmbeddingUnavailableError(
                f"vectors of {dimension} and {len(item)} dimensions in one response"
            )
        values = []
        for value in item:
            # bool is an int in Python, and True in a vector is a wrong answer
            # dressed as a number.
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise EmbeddingUnavailableError(f"{value!r} is not a coordinate")
            if not math.isfinite(value):
                raise EmbeddingUnavailableError(f"{value!r} is not a finite coordinate")
            values.append(float(value))
        vectors.append(tuple(values))
    return vectors
