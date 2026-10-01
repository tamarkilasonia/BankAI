"""Retrieval with the real pgvector store in PostgreSQL."""

from collections.abc import AsyncIterator

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker

from app.rag.documents import Chunk, vocabulary
from app.rag.embeddings import HashingEmbedder
from app.rag.factory import DEFAULT_MIN_SCORES
from app.rag.indexing import index_chunks
from app.rag.retriever import Retriever
from app.rag.store import PgVectorStore

pytestmark = pytest.mark.postgres


@pytest.fixture
async def pg_store(pg_engine: AsyncEngine) -> AsyncIterator[PgVectorStore]:
    async with pg_engine.begin() as conn:
        await conn.execute(text("DELETE FROM knowledge_chunks"))
    yield PgVectorStore(async_sessionmaker(pg_engine, expire_on_commit=False))


async def test_index_and_search_in_pgvector(pg_store: PgVectorStore, knowledge_chunks: list[Chunk]) -> None:
    embedder = HashingEmbedder()
    report = await index_chunks(knowledge_chunks, embedder, pg_store)

    hits = await pg_store.search(embedder.name, await embedder.embed_query("How do I block a lost card?"), 3)

    assert report.changed and await pg_store.count(embedder.name) == len(knowledge_chunks)
    assert hits[0].chunk.source == "cards.md"
    assert 0 < hits[-1].score <= hits[0].score <= 1


async def test_reindexing_is_skipped_when_unchanged(
    pg_store: PgVectorStore, knowledge_chunks: list[Chunk]
) -> None:
    embedder = HashingEmbedder()
    await index_chunks(knowledge_chunks, embedder, pg_store)

    again = await index_chunks(knowledge_chunks, embedder, pg_store)

    assert again.changed is False


async def test_embedders_never_mix(pg_store: PgVectorStore, knowledge_chunks: list[Chunk]) -> None:
    embedder = HashingEmbedder()
    await index_chunks(knowledge_chunks, embedder, pg_store)

    assert await pg_store.search("some-other-model", await embedder.embed_query("card"), 3) == []


async def test_retriever_over_pgvector_matches_the_in_memory_behaviour(
    pg_store: PgVectorStore, knowledge_chunks: list[Chunk]
) -> None:
    embedder = HashingEmbedder()
    await index_chunks(knowledge_chunks, embedder, pg_store)
    retriever = Retriever(
        embedder=embedder,
        store=pg_store,
        known_words=vocabulary(knowledge_chunks),
        min_score=DEFAULT_MIN_SCORES["hashing"],
    )

    answerable = await retriever.retrieve("What are the fees for international transfers?")
    fictional = await retriever.retrieve("What is the bank's policy on the fictional service XYZ?")

    assert answerable.sufficient and answerable.passages[0].chunk.source == "international_transfers.md"
    assert not fictional.sufficient
