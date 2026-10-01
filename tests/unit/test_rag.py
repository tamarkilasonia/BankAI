"""RAG building blocks: chunking, embeddings, stores, indexing, retrieval, calibration."""

import asyncio
import json
import math
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx2
import pytest

from app.core.config import Settings
from app.database.seed import PRODUCTS
from app.models.enums import ProductType
from app.rag.calibration import IN_SCOPE, OUT_OF_SCOPE
from app.rag.documents import MAX_CHUNK_CHARS, Chunk, chunk_markdown, load_knowledge_base, vocabulary
from app.rag.embeddings import (
    EMBEDDING_DIMENSIONS,
    EmbeddingError,
    HashingEmbedder,
    OpenAICompatibleEmbedder,
    tokens,
)
from app.rag.factory import DEFAULT_MIN_SCORES, create_embedder, create_retriever
from app.rag.indexing import index_chunks
from app.rag.retriever import Retriever, unknown_terms
from app.rag.store import InMemoryVectorStore, PgVectorStore
from tests.conftest import PROJECT_ROOT

KB = PROJECT_ROOT / "knowledge_base"

DOC = """# Fee schedule

> MOCK fee schedule of a fictional bank.

## ATM withdrawal fees

Withdrawals at our ATMs are free.
Other banks' ATMs cost 2.00 GEL.

## Transfer fees

Transfers between own accounts are free.
"""


# --- chunking ---------------------------------------------------------------------------


def test_one_chunk_per_section_without_the_disclaimer() -> None:
    chunks = chunk_markdown("fees.md", DOC)

    assert [(c.chunk_id, c.title, c.section) for c in chunks] == [
        ("fees.md#atm-withdrawal-fees", "Fee schedule", "ATM withdrawal fees"),
        ("fees.md#transfer-fees", "Fee schedule", "Transfer fees"),
    ]
    assert chunks[0].text == "Withdrawals at our ATMs are free.\nOther banks' ATMs cost 2.00 GEL."
    assert all("MOCK" not in c.text for c in chunks)


def test_embedding_text_carries_title_and_section() -> None:
    chunk = chunk_markdown("fees.md", DOC)[1]
    assert chunk.embedding_text == "Fee schedule - Transfer fees: Transfers between own accounts are free."


def test_long_sections_are_split_at_paragraphs() -> None:
    paragraph = "word " * 150  # ~750 characters
    chunks = chunk_markdown("x.md", f"# T\n## Long\n{paragraph}\n\n{paragraph}\n\n{paragraph}")

    assert [c.chunk_id for c in chunks] == ["x.md#long-1", "x.md#long-2", "x.md#long-3"]
    assert all(len(c.text) <= MAX_CHUNK_CHARS for c in chunks)


def test_content_hash_changes_only_with_content() -> None:
    a, b = chunk_markdown("fees.md", DOC)[0], chunk_markdown("fees.md", DOC)[0]
    changed = chunk_markdown("fees.md", DOC.replace("2.00", "3.00"))[0]
    assert a.content_hash == b.content_hash != changed.content_hash


def test_real_knowledge_base_loads() -> None:
    chunks = load_knowledge_base(KB)

    assert len({c.source for c in chunks}) == 11  # README.md is skipped
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    assert all(c.text and c.title != c.source for c in chunks)


def test_missing_directory_is_reported(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_knowledge_base(tmp_path / "nope")


# --- the documents agree with the product catalogue ---------------------------------------


def _kb_text() -> str:
    return " ".join(c.text for c in load_knowledge_base(KB))


@pytest.mark.parametrize(
    "product", [p for p in PRODUCTS if p.get("interest_rate_pct") is not None], ids=lambda p: str(p["code"])
)
def test_documented_interest_rates_match_the_product_table(product: dict[str, Any]) -> None:
    """The calculator uses the table, RAG uses the documents - they must never disagree.

    The rate must also sit in the same chunk as the product's name, so a retrieved
    passage always says which product a rate belongs to.
    """
    rate = f"{product['interest_rate_pct']}%"
    chunks = [c for c in load_knowledge_base(KB) if rate in c.text]
    assert chunks, f"{rate} for {product['name']} is not documented"
    assert any(str(product["name"]) in c.section or str(product["name"]) in c.text for c in chunks), [
        c.chunk_id for c in chunks
    ]


@pytest.mark.parametrize(
    "product",
    [p for p in PRODUCTS if p["product_type"] in (ProductType.LOAN, ProductType.MORTGAGE)],
    ids=lambda p: str(p["code"]),
)
def test_documented_loan_limits_match_the_product_table(product: dict[str, Any]) -> None:
    text = _kb_text()
    assert f"{product['min_amount']:,.2f} GEL" in text
    assert f"{product['max_amount']:,.2f} GEL" in text
    assert f"from {product['min_term_months']} to {product['max_term_months']} months" in text


def test_documented_card_fees_match_the_product_table() -> None:
    gold = next(p for p in PRODUCTS if p["code"] == "CARD_DEBIT_GOLD")
    credit = next(p for p in PRODUCTS if p["code"] == "CARD_CREDIT_STANDARD")
    text = _kb_text()
    assert f"{gold['monthly_fee']} GEL per month" in text
    assert f"monthly fee of {credit['monthly_fee']} GEL" in text
    assert isinstance(credit["max_amount"], Decimal) and f"{credit['max_amount']:,.2f} GEL" in text


# --- embeddings -----------------------------------------------------------------------------


def test_tokens_drop_stopwords_and_plurals() -> None:
    assert tokens("What are the fees for international transfers?") == ["fee", "international", "transfer"]


async def test_hashing_embedder_is_deterministic_and_normalized() -> None:
    embedder = HashingEmbedder()
    first, second = await embedder.embed_query("card fees"), await embedder.embed_query("card fees")

    assert first == second
    assert len(first) == EMBEDDING_DIMENSIONS
    assert math.isclose(math.sqrt(sum(x * x for x in first)), 1.0)
    assert embedder.name == "hashing-v1-768"


async def test_hashing_similarity_follows_shared_words() -> None:
    embedder = HashingEmbedder()
    query, related, unrelated = await embedder.embed_documents(
        ["international transfer fee", "fees for international transfers", "mortgage down payment"]
    )

    def cosine(a: list[float], b: list[float]) -> float:
        return sum(x * y for x, y in zip(a, b, strict=True))

    assert cosine(query, related) > 0.5 > cosine(query, unrelated)


def _embedder_with(handler: Any, **kwargs: Any) -> OpenAICompatibleEmbedder:
    return OpenAICompatibleEmbedder(
        model="nomic-embed-text",
        base_url="http://ollama.test/v1",
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
        **kwargs,
    )


def _embedding_response(inputs: list[str], dims: int = EMBEDDING_DIMENSIONS) -> dict[str, Any]:
    return {
        "object": "list",
        "model": "nomic-embed-text",
        "data": [
            {"object": "embedding", "index": i, "embedding": [1.0] + [0.0] * (dims - 1)}
            for i in range(len(inputs))
        ],
        "usage": {"prompt_tokens": 1, "total_tokens": 1},
    }


async def test_openai_compatible_embedder_applies_prefixes_and_batches() -> None:
    requests: list[dict[str, Any]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        requests.append(body)
        return httpx2.Response(200, json=_embedding_response(body["input"]))

    embedder = _embedder_with(
        handler, query_prefix="search_query: ", document_prefix="search_document: ", batch_size=2
    )
    vectors = await embedder.embed_documents(["a", "b", "c"])
    await embedder.embed_query("q")

    assert len(vectors) == 3 and len(vectors[0]) == EMBEDDING_DIMENSIONS
    assert [r["input"] for r in requests] == [
        ["search_document: a", "search_document: b"],
        ["search_document: c"],
        ["search_query: q"],
    ]
    assert "dimensions" not in requests[0]
    await embedder.aclose()


async def test_openai_embedder_can_request_768_dimensions() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.update(json.loads(request.content))
        return httpx2.Response(200, json=_embedding_response(seen["input"]))

    await _embedder_with(handler, request_dimensions=True).embed_query("q")
    assert seen["dimensions"] == EMBEDDING_DIMENSIONS


@pytest.mark.parametrize(
    ("handler", "message"),
    [
        (lambda r: httpx2.Response(200, json=_embedding_response(["x"], dims=1024)), "1024 dimensions"),
        (lambda r: httpx2.Response(500, json={"error": {"message": "boom"}}), "HTTP 500"),
    ],
)
async def test_embedding_failures(handler: Any, message: str) -> None:
    with pytest.raises(EmbeddingError, match=message):
        await _embedder_with(handler).embed_query("q")


async def test_unreachable_embedding_server() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("refused", request=request)

    with pytest.raises(EmbeddingError, match="unreachable"):
        await _embedder_with(handler).embed_query("q")


# --- store & indexing -------------------------------------------------------------------------


class CountingEmbedder(HashingEmbedder):
    def __init__(self) -> None:
        super().__init__()
        self.document_calls = 0

    async def embed_documents(self, texts: Any) -> list[list[float]]:
        self.document_calls += 1
        return await super().embed_documents(texts)


async def test_indexing_is_idempotent_and_detects_changes() -> None:
    embedder, store = CountingEmbedder(), InMemoryVectorStore()
    chunks = chunk_markdown("fees.md", DOC)

    first = await index_chunks(chunks, embedder, store)
    second = await index_chunks(chunks, embedder, store)
    changed = await index_chunks(chunk_markdown("fees.md", DOC.replace("2.00", "3.00")), embedder, store)
    forced = await index_chunks(chunks, embedder, store, force=True)

    assert (first.changed, second.changed, changed.changed, forced.changed) == (True, False, True, True)
    assert embedder.document_calls == 3  # the unchanged run made no embedding calls


async def test_memory_store_keeps_embedders_apart() -> None:
    store = InMemoryVectorStore()
    chunk = Chunk("a#b", "a.md", "A", "B", "text")
    await store.replace_all("model-1", [chunk], [[1.0, 0.0]])

    assert await store.search("model-2", [1.0, 0.0], 3) == []
    assert (await store.search("model-1", [1.0, 0.0], 3))[0].score == pytest.approx(1.0)


# --- retrieval -------------------------------------------------------------------------------


@pytest.mark.parametrize(("question", "expected_source"), IN_SCOPE)
async def test_in_scope_questions_are_answerable(
    retriever: Retriever, question: str, expected_source: str
) -> None:
    result = await retriever.retrieve(question)

    assert result.sufficient, (result.reason, result.best_score)
    assert expected_source in {p.chunk.source for p in result.passages[:2]}


@pytest.mark.parametrize("question", OUT_OF_SCOPE)
async def test_out_of_scope_questions_are_not(retriever: Retriever, question: str) -> None:
    result = await retriever.retrieve(question)

    assert not result.sufficient
    assert result.passages == []


async def test_fictional_service_is_caught_as_unknown_term(retriever: Retriever) -> None:
    result = await retriever.retrieve("What are the fees for the XYZ transfer service?")

    assert (result.sufficient, result.reason, result.unknown_terms) == (False, "unknown_terms", ["XYZ"])


@pytest.mark.parametrize(
    ("question", "unknown"),
    [
        ("Tell me about service XYZ", ["XYZ"]),
        ('What is the "Platinum Ultra" card?', ["Platinum", "Ultra"]),
        ("fees for card ABC123?", ["ABC123"]),
        ("What is the SWIFT fee for USD in GEL?", []),
        ("How do I block my card", []),
    ],
)
def test_unknown_terms(question: str, unknown: list[str]) -> None:
    known = vocabulary(load_knowledge_base(KB))
    assert unknown_terms(question, known) == unknown


async def test_empty_index_is_insufficient() -> None:
    retriever = Retriever(
        embedder=HashingEmbedder(), store=InMemoryVectorStore(), known_words=set(), min_score=0.1
    )

    result = await retriever.retrieve("card fees")

    assert (result.sufficient, result.reason) == (False, "empty_index")


async def test_warmup_runs_exactly_once_even_concurrently() -> None:
    calls = 0

    async def warmup() -> None:
        nonlocal calls
        calls += 1
        await asyncio.sleep(0.01)

    retriever = Retriever(
        embedder=HashingEmbedder(),
        store=InMemoryVectorStore(),
        known_words=set(),
        min_score=0.1,
        warmup=warmup,
    )
    await asyncio.gather(*(retriever.retrieve("q") for _ in range(5)))

    assert calls == 1


# --- factory -----------------------------------------------------------------------------------


def test_factory_builds_the_configured_components() -> None:
    offline = Settings(_env_file=None, embedding_provider="hashing", vector_store="memory")
    ollama = Settings(
        _env_file=None,
        embedding_provider="openai_compatible",
        embedding_base_url="http://localhost:11434/v1",
        vector_store="pgvector",
        rag_min_score=0.7,
    )

    memory_retriever = create_retriever(offline, session_factory=None)  # type: ignore[arg-type]
    pg_retriever = create_retriever(ollama, session_factory=None)  # type: ignore[arg-type]

    assert isinstance(memory_retriever.store, InMemoryVectorStore)
    assert memory_retriever.min_score == DEFAULT_MIN_SCORES["hashing"]
    assert isinstance(pg_retriever.store, PgVectorStore)
    assert pg_retriever.min_score == 0.7
    assert isinstance(create_embedder(ollama), OpenAICompatibleEmbedder)


async def test_memory_retriever_indexes_on_first_question() -> None:
    retriever = create_retriever(
        Settings(_env_file=None, embedding_provider="hashing", vector_store="memory", knowledge_base_dir=KB),
        session_factory=None,  # type: ignore[arg-type]
    )

    result = await retriever.retrieve("How do I block a lost card?")

    assert result.sufficient
    assert result.passages[0].chunk.source == "cards.md"


async def test_relative_window_drops_passages_far_below_the_best() -> None:
    store = InMemoryVectorStore()
    chunks = [Chunk(f"d#{i}", "d.md", "D", f"S{i}", f"text {i}") for i in range(3)]
    await store.replace_all("e", chunks, [[1.0, 0.0], [0.9, 0.436], [0.6, 0.8]])  # cosine 1.0, 0.9, 0.6

    class FixedEmbedder(HashingEmbedder):
        @property
        def name(self) -> str:
            return "e"

        async def embed_query(self, text: str) -> list[float]:
            return [1.0, 0.0]

    windowed = Retriever(
        embedder=FixedEmbedder(), store=store, known_words=set(), min_score=0.5, relative_window=0.15
    )
    unwindowed = Retriever(embedder=FixedEmbedder(), store=store, known_words=set(), min_score=0.5)

    assert [h.chunk.chunk_id for h in (await windowed.retrieve("q")).passages] == ["d#0", "d#1"]
    assert len((await unwindowed.retrieve("q")).passages) == 3
