"""Vector stores: where chunk embeddings live and how nearest neighbours are found.

* ``PgVectorStore`` - PostgreSQL with the pgvector extension. Similarity search is
  one SQL query (``ORDER BY embedding <=> :query LIMIT k``), accelerated by an
  HNSW index. Relational data and vectors share one database, one backup and
  one transaction model.
* ``InMemoryVectorStore`` - a Python list; for unit tests and for running
  without pgvector.

Scores are cosine *similarities* (1 = same direction, 0 = unrelated).
Vectors from different embedders live in different spaces, so every row is
tagged with its embedder name and searches never mix them.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.models import KnowledgeChunk
from app.rag.documents import Chunk


@dataclass(frozen=True)
class ScoredChunk:
    chunk: Chunk
    score: float


class VectorStore(Protocol):
    async def replace_all(
        self, embedder: str, chunks: Sequence[Chunk], vectors: Sequence[list[float]]
    ) -> None:
        """Make the stored chunks for ``embedder`` exactly ``chunks`` (idempotent re-indexing)."""
        ...

    async def search(self, embedder: str, vector: list[float], k: int) -> list[ScoredChunk]: ...

    async def fingerprint(self, embedder: str) -> set[tuple[str, str]]:
        """(chunk_id, content_hash) pairs currently stored - to skip re-indexing unchanged content."""
        ...


class InMemoryVectorStore:
    def __init__(self) -> None:
        self._rows: dict[str, list[tuple[Chunk, list[float]]]] = {}

    async def replace_all(
        self, embedder: str, chunks: Sequence[Chunk], vectors: Sequence[list[float]]
    ) -> None:
        self._rows[embedder] = list(zip(chunks, vectors, strict=True))

    async def search(self, embedder: str, vector: list[float], k: int) -> list[ScoredChunk]:
        scored = [
            ScoredChunk(chunk, _cosine(vector, stored)) for chunk, stored in self._rows.get(embedder, [])
        ]
        return sorted(scored, key=lambda s: s.score, reverse=True)[:k]

    async def fingerprint(self, embedder: str) -> set[tuple[str, str]]:
        return {(chunk.chunk_id, chunk.content_hash) for chunk, _ in self._rows.get(embedder, [])}


class PgVectorStore:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = session_factory

    async def replace_all(
        self, embedder: str, chunks: Sequence[Chunk], vectors: Sequence[list[float]]
    ) -> None:
        async with self._sessions() as session, session.begin():
            await session.execute(delete(KnowledgeChunk).where(KnowledgeChunk.embedding_model == embedder))
            session.add_all(
                KnowledgeChunk(
                    chunk_id=chunk.chunk_id,
                    source=chunk.source,
                    title=chunk.title,
                    section=chunk.section,
                    content=chunk.text,
                    content_hash=chunk.content_hash,
                    embedding_model=embedder,
                    embedding=vector,
                )
                for chunk, vector in zip(chunks, vectors, strict=True)
            )

    async def search(self, embedder: str, vector: list[float], k: int) -> list[ScoredChunk]:
        distance = KnowledgeChunk.embedding.cosine_distance(vector).label("distance")
        stmt = (
            select(KnowledgeChunk, distance)
            .where(KnowledgeChunk.embedding_model == embedder)
            .order_by(distance)
            .limit(k)
        )
        async with self._sessions() as session:
            rows = (await session.execute(stmt)).all()
        return [
            ScoredChunk(
                Chunk(row.chunk_id, row.source, row.title, row.section, row.content), 1.0 - float(dist)
            )
            for row, dist in rows
        ]

    async def fingerprint(self, embedder: str) -> set[tuple[str, str]]:
        stmt = select(KnowledgeChunk.chunk_id, KnowledgeChunk.content_hash).where(
            KnowledgeChunk.embedding_model == embedder
        )
        async with self._sessions() as session:
            return {(chunk_id, digest) for chunk_id, digest in (await session.execute(stmt)).all()}

    async def count(self, embedder: str) -> int:
        stmt = (
            select(func.count()).select_from(KnowledgeChunk).where(KnowledgeChunk.embedding_model == embedder)
        )
        async with self._sessions() as session:
            return await session.scalar(stmt) or 0


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0
