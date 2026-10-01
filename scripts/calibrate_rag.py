"""Measure how well the configured embedder separates answerable from unanswerable questions.

    python -m scripts.calibrate_rag

For every calibration question it embeds the question, searches an in-memory
index of the knowledge base, and reports the top similarity score. A usable
threshold lies between the lowest in-scope score and the highest out-of-scope
score; the script suggests their midpoint and shows the margin.
"""

import asyncio
from dataclasses import dataclass

from app.core.config import get_settings
from app.rag.calibration import IN_SCOPE, OUT_OF_SCOPE
from app.rag.documents import load_knowledge_base
from app.rag.embeddings import Embedder
from app.rag.factory import create_embedder
from app.rag.indexing import index_chunks
from app.rag.store import InMemoryVectorStore


@dataclass(frozen=True)
class Calibration:
    embedder: str
    lowest_in_scope: float
    highest_out_of_scope: float
    right_document_top2: int
    total_in_scope: int

    @property
    def margin(self) -> float:
        return self.lowest_in_scope - self.highest_out_of_scope

    @property
    def suggested_threshold(self) -> float:
        return round((self.lowest_in_scope + self.highest_out_of_scope) / 2, 3)


async def calibrate(embedder: Embedder) -> Calibration:
    store = InMemoryVectorStore()
    await index_chunks(load_knowledge_base(get_settings().knowledge_base_dir), embedder, store)

    async def top(question: str) -> list[tuple[str, float]]:
        hits = await store.search(embedder.name, await embedder.embed_query(question), 2)
        return [(hit.chunk.source, hit.score) for hit in hits]

    in_scores, correct = [], 0
    for question, expected_source in IN_SCOPE:
        hits = await top(question)
        in_scores.append(hits[0][1])
        correct += any(source == expected_source for source, _ in hits)
    out_scores = [(await top(question))[0][1] for question in OUT_OF_SCOPE]
    return Calibration(embedder.name, min(in_scores), max(out_scores), correct, len(IN_SCOPE))


async def main() -> int:
    embedder = create_embedder(get_settings())
    try:
        result = await calibrate(embedder)
    finally:
        await embedder.aclose()
    print(f"embedder:               {result.embedder}")
    print(f"right document in top 2: {result.right_document_top2}/{result.total_in_scope}")
    print(f"lowest in-scope score:   {result.lowest_in_scope:.3f}")
    print(f"highest out-of-scope:    {result.highest_out_of_scope:.3f}")
    print(f"margin:                  {result.margin:+.3f}")
    print(f"suggested RAG_MIN_SCORE: {result.suggested_threshold}")
    return 0 if result.margin > 0 else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
