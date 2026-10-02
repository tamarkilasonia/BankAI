"""Index the knowledge base into pgvector.

    python -m scripts.ingest_knowledge_base          # re-embeds only if documents changed
    python -m scripts.ingest_knowledge_base --force  # re-embed everything

Uses EMBEDDING_PROVIDER / EMBEDDING_MODEL from the settings. Each embedder has its
own set of rows, so switching models never mixes vector spaces.
"""

import argparse
import asyncio
import sys

from sqlalchemy.ext.asyncio import async_sessionmaker

from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger
from app.database.session import create_engine
from app.rag.embeddings import EmbeddingError
from app.rag.factory import create_embedder
from app.rag.indexing import index_knowledge_base
from app.rag.store import PgVectorStore

logger = get_logger("scripts.ingest_knowledge_base")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Index the knowledge base into pgvector.")
    parser.add_argument("--force", action="store_true", help="re-embed even if nothing changed")
    return parser.parse_args(argv)


async def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_format)
    if settings.vector_store != "pgvector":
        print("VECTOR_STORE=memory: the knowledge base is indexed in memory at startup - nothing to ingest.")
        return 0

    engine = create_engine(settings.database_url)
    embedder = create_embedder(settings)
    try:
        store = PgVectorStore(async_sessionmaker(engine, expire_on_commit=False))
        report = await index_knowledge_base(settings.knowledge_base_dir, embedder, store, force=args.force)
    except EmbeddingError as exc:
        print(f"Embedding failed: {exc}", file=sys.stderr)
        return 1
    finally:
        await embedder.aclose()
        await engine.dispose()

    state = "indexed" if report.changed else "already up to date"
    print(f"{report.chunks} chunks from {report.documents} documents {state} (embedder: {report.embedder}).")
    if report.rejected:
        print(
            f"REJECTED {len(report.rejected)} chunk(s) containing instructions aimed at the assistant: "
            + ", ".join(report.rejected),
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
