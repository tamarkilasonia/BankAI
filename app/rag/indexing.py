"""Indexing: knowledge-base files -> chunks -> embeddings -> vector store."""

from dataclasses import dataclass
from pathlib import Path

from app.core.logging import get_logger
from app.rag.documents import Chunk, load_knowledge_base
from app.rag.embeddings import Embedder
from app.rag.store import VectorStore

logger = get_logger(__name__)


@dataclass(frozen=True)
class IndexReport:
    embedder: str
    chunks: int
    documents: int
    changed: bool


async def index_chunks(
    chunks: list[Chunk], embedder: Embedder, store: VectorStore, *, force: bool = False
) -> IndexReport:
    """(Re-)index ``chunks``. Skips the embedding calls when nothing changed."""
    documents = len({c.source for c in chunks})
    wanted = {(c.chunk_id, c.content_hash) for c in chunks}
    if not force and await store.fingerprint(embedder.name) == wanted:
        logger.info("rag.index_up_to_date", extra={"embedder": embedder.name, "chunks": len(chunks)})
        return IndexReport(embedder.name, len(chunks), documents, changed=False)

    vectors = await embedder.embed_documents([c.embedding_text for c in chunks])
    await store.replace_all(embedder.name, chunks, vectors)
    logger.info(
        "rag.indexed", extra={"embedder": embedder.name, "chunks": len(chunks), "documents": documents}
    )
    return IndexReport(embedder.name, len(chunks), documents, changed=True)


async def index_knowledge_base(
    directory: Path, embedder: Embedder, store: VectorStore, *, force: bool = False
) -> IndexReport:
    return await index_chunks(load_knowledge_base(directory), embedder, store, force=force)
