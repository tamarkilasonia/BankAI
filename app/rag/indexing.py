"""Indexing: knowledge-base files -> chunks -> embeddings -> vector store.

Before anything is embedded, every chunk is screened for instructions aimed at
the assistant (indirect prompt injection, see ``app/safety``). A poisoned chunk is
never indexed; the agent screens retrieved passages again at answer time, in case
something reaches the vector store another way.
"""

from dataclasses import dataclass, field
from pathlib import Path

from app.core.logging import get_logger
from app.rag.documents import Chunk, load_knowledge_base
from app.rag.embeddings import Embedder
from app.rag.store import VectorStore
from app.safety.input_guard import GuardAction, scan_untrusted

logger = get_logger(__name__)


@dataclass(frozen=True)
class IndexReport:
    embedder: str
    chunks: int
    documents: int
    changed: bool
    rejected: list[str] = field(default_factory=list)  # chunk IDs that failed the safety screen


def screen_chunks(chunks: list[Chunk]) -> tuple[list[Chunk], list[Chunk]]:
    """Split chunks into (clean, rejected)."""
    clean, rejected = [], []
    for chunk in chunks:
        verdict = scan_untrusted(f"{chunk.section}\n{chunk.text}")
        if verdict.action is GuardAction.ALLOW:
            clean.append(chunk)
        else:
            logger.warning("rag.chunk_rejected", extra={"chunk_id": chunk.chunk_id, "rule": verdict.rule})
            rejected.append(chunk)
    return clean, rejected


async def index_chunks(
    chunks: list[Chunk], embedder: Embedder, store: VectorStore, *, force: bool = False
) -> IndexReport:
    """(Re-)index the chunks that pass the safety screen. Skips embedding when nothing changed."""
    chunks, rejected = screen_chunks(chunks)
    rejected_ids = [c.chunk_id for c in rejected]
    documents = len({c.source for c in chunks})
    wanted = {(c.chunk_id, c.content_hash) for c in chunks}
    if not force and await store.fingerprint(embedder.name) == wanted:
        logger.info("rag.index_up_to_date", extra={"embedder": embedder.name, "chunks": len(chunks)})
        return IndexReport(embedder.name, len(chunks), documents, changed=False, rejected=rejected_ids)

    vectors = await embedder.embed_documents([c.embedding_text for c in chunks])
    await store.replace_all(embedder.name, chunks, vectors)
    logger.info(
        "rag.indexed", extra={"embedder": embedder.name, "chunks": len(chunks), "documents": documents}
    )
    return IndexReport(embedder.name, len(chunks), documents, changed=True, rejected=rejected_ids)


async def index_knowledge_base(
    directory: Path, embedder: Embedder, store: VectorStore, *, force: bool = False
) -> IndexReport:
    return await index_chunks(load_knowledge_base(directory), embedder, store, force=force)
