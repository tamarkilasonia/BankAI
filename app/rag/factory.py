"""Build the RAG components from settings."""

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import Settings
from app.rag.documents import load_knowledge_base, vocabulary
from app.rag.embeddings import Embedder, HashingEmbedder, OpenAICompatibleEmbedder
from app.rag.indexing import index_chunks
from app.rag.retriever import Retriever
from app.rag.store import InMemoryVectorStore, PgVectorStore, VectorStore

# Similarity thresholds measured with `python -m scripts.calibrate_rag` on the questions
# in app/rag/calibration.py (midpoint between the lowest in-scope and the highest
# out-of-scope top score):
#   hashing-v1-768:                       in >= 0.267, out <= 0.244 -> 0.255 (thin margin: lexical baseline)
#   nomic-embed-text (search_ prefixes):  in >= 0.733, out <= 0.627 -> 0.68
# A different embedding model needs its own calibration (set RAG_MIN_SCORE).
DEFAULT_MIN_SCORES = {"hashing": 0.255, "openai_compatible": 0.68}

# Relative cutoff below the best passage. With nomic-embed-text the answering section always
# ranks first and merely related ones score 0.06-0.15 lower. The lexical hashing embedder ranks
# too noisily for a cutoff (an answering section once trailed the best by 0.157).
DEFAULT_RELATIVE_WINDOWS: dict[str, float | None] = {"hashing": None, "openai_compatible": 0.12}


def create_embedder(settings: Settings) -> Embedder:
    if settings.embedding_provider == "hashing":
        return HashingEmbedder()
    return OpenAICompatibleEmbedder(
        model=settings.embedding_model,
        base_url=settings.embedding_base_url,
        api_key=settings.embedding_api_key.get_secret_value() if settings.embedding_api_key else None,
        query_prefix=settings.embedding_query_prefix,
        document_prefix=settings.embedding_document_prefix,
        request_dimensions=settings.embedding_base_url is None,  # OpenAI: shrink to 768
    )


def create_retriever(settings: Settings, session_factory: async_sessionmaker[AsyncSession]) -> Retriever:
    """A retriever over the configured store.

    * pgvector: the index is built ahead of time (``python -m scripts.ingest_knowledge_base``).
    * memory:   the knowledge base is indexed on the first question (nothing persists).
    """
    chunks = load_knowledge_base(settings.knowledge_base_dir)
    embedder = create_embedder(settings)
    store: VectorStore
    if settings.vector_store == "pgvector":
        store = PgVectorStore(session_factory)
        warmup = None
    else:
        store = InMemoryVectorStore()

        async def warmup() -> None:
            await index_chunks(chunks, embedder, store)

    min_score = settings.rag_min_score
    if min_score is None:
        min_score = DEFAULT_MIN_SCORES[settings.embedding_provider]
    return Retriever(
        embedder=embedder,
        store=store,
        known_words=vocabulary(chunks),
        min_score=min_score,
        top_k=settings.rag_top_k,
        relative_window=DEFAULT_RELATIVE_WINDOWS[settings.embedding_provider],
        warmup=warmup,
    )
