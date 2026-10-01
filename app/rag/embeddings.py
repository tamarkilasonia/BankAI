"""Embeddings: turning text into vectors whose distance reflects meaning.

Two implementations behind one interface:

* ``OpenAICompatibleEmbedder`` - a real embedding model (Ollama's
  nomic-embed-text locally, or OpenAI's text-embedding-3-small with
  ``dimensions=768``). Captures meaning: "wire money abroad" is close to
  "international transfer".
* ``HashingEmbedder`` - deterministic, offline, free. Words (and word pairs) are
  hashed into a 768-slot vector, so similarity = shared vocabulary. Used by
  tests and CI; it is a *lexical* baseline, not semantic search, and it is
  labelled as such everywhere.

All vectors have 768 dimensions because that is the size of the database column.
"""

import hashlib
import math
import re
from collections.abc import Sequence
from itertools import pairwise
from typing import Protocol

import httpx2
from openai import APIConnectionError, APIStatusError, APITimeoutError, AsyncOpenAI

EMBEDDING_DIMENSIONS = 768

_STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does", "for", "from", "how", "i",
    "in", "is", "it", "my", "of", "on", "or", "the", "to", "what", "when", "where", "which", "who",
    "why", "will", "with", "you", "your", "me", "we", "our", "this", "that", "there", "their", "about",
})  # fmt: skip


class EmbeddingError(Exception):
    """The embedding model could not produce vectors."""


class Embedder(Protocol):
    @property
    def name(self) -> str:
        """Identifies the vector space; vectors from different embedders are never compared."""
        ...

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    async def embed_query(self, text: str) -> list[float]: ...

    async def aclose(self) -> None: ...


def _normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vector))
    return [x / norm for x in vector] if norm else vector


def tokens(text: str) -> list[str]:
    """Lower-cased words without stop words, with naive plural stripping ('fees' -> 'fee')."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [
        w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w
        for w in words
        if w not in _STOPWORDS
    ]


class HashingEmbedder:
    """Offline lexical embeddings via the 'hashing trick' (MOCK-grade, deterministic)."""

    def __init__(self, dimensions: int = EMBEDDING_DIMENSIONS) -> None:
        self._dimensions = dimensions

    @property
    def name(self) -> str:
        return f"hashing-v1-{self._dimensions}"

    def _vector(self, text: str) -> list[float]:
        words = tokens(text)
        features = [(w, 1.0) for w in words] + [(f"{a} {b}", 0.5) for a, b in pairwise(words)]
        vector = [0.0] * self._dimensions
        for feature, weight in features:
            digest = int.from_bytes(hashlib.blake2b(feature.encode(), digest_size=8).digest(), "big")
            sign = 1.0 if digest >> 63 else -1.0
            vector[digest % self._dimensions] += sign * weight
        return _normalize(vector)

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    async def embed_query(self, text: str) -> list[float]:
        return self._vector(text)

    async def aclose(self) -> None:
        return None


class OpenAICompatibleEmbedder:
    """Embeddings from any server speaking the OpenAI embeddings API (Ollama, OpenAI)."""

    def __init__(
        self,
        *,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        query_prefix: str = "",
        document_prefix: str = "",
        request_dimensions: bool = False,
        timeout_seconds: float = 30.0,
        batch_size: int = 32,
        http_client: httpx2.AsyncClient | None = None,
    ) -> None:
        self._model = model
        self._query_prefix = query_prefix
        self._document_prefix = document_prefix
        self._request_dimensions = request_dimensions  # OpenAI text-embedding-3 can shrink to 768
        self._batch_size = batch_size
        self._client = AsyncOpenAI(
            api_key=api_key or "not-needed-for-local-servers",
            base_url=base_url,
            timeout=timeout_seconds,
            http_client=http_client,
        )

    @property
    def name(self) -> str:
        return self._model

    async def _embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = list(texts[start : start + self._batch_size])
            try:
                if self._request_dimensions:
                    response = await self._client.embeddings.create(
                        model=self._model, input=batch, dimensions=EMBEDDING_DIMENSIONS
                    )
                else:
                    response = await self._client.embeddings.create(model=self._model, input=batch)
            except (APITimeoutError, APIConnectionError) as exc:
                raise EmbeddingError("the embedding model is unreachable") from exc
            except APIStatusError as exc:
                raise EmbeddingError(f"the embedding server returned HTTP {exc.status_code}") from exc
            for item in sorted(response.data, key=lambda d: d.index):
                if len(item.embedding) != EMBEDDING_DIMENSIONS:
                    got = len(item.embedding)
                    raise EmbeddingError(
                        f"{self._model} returned {got} dimensions, expected {EMBEDDING_DIMENSIONS}"
                    )
                vectors.append(_normalize(list(item.embedding)))
        return vectors

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return await self._embed([self._document_prefix + t for t in texts])

    async def embed_query(self, text: str) -> list[float]:
        return (await self._embed([self._query_prefix + text]))[0]

    async def aclose(self) -> None:
        await self._client.close()
