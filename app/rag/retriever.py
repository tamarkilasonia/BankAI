"""Retrieval: question -> embedding -> nearest chunks -> "is this enough to answer?".

The "enough?" decision is the core anti-hallucination step of RAG. Two signals:

1. **Similarity threshold.** If no chunk is similar enough to the question, the
   knowledge base does not cover it.
2. **Unknown specific terms.** Embeddings judge *topic*, not facts: "the policy
   on service XYZ" is topically close to many policy passages. So named things
   in the question (acronyms, codes, quoted names) must occur in the knowledge
   base, otherwise the question is about something the documents don't describe.

If either fails, the agent says it does not have enough information instead of
letting the LLM improvise.
"""

import asyncio
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from app.core.logging import get_logger
from app.rag.embeddings import Embedder
from app.rag.store import ScoredChunk, VectorStore

logger = get_logger(__name__)

# Well-known acronyms that may legitimately be absent from the documents.
_COMMON_ACRONYMS = frozenset(
    {"gel", "usd", "eur", "gbp", "atm", "pin", "iban", "swift", "bic", "otp", "sms", "id", "pdf"}
)
_SPECIFIC_TERM = re.compile(r"\b(?:[A-Z]{2,}[A-Z0-9]*|[A-Za-z]+\d+[A-Za-z0-9]*)\b")
_QUOTED = re.compile(r"[\"“]([^\"”]{2,60})[\"”]")


@dataclass(frozen=True)
class RetrievalResult:
    query: str
    passages: list[ScoredChunk]
    sufficient: bool
    reason: str  # ok | empty_index | low_similarity | unknown_terms
    best_score: float
    unknown_terms: list[str] = field(default_factory=list)


def unknown_terms(question: str, known_words: set[str]) -> list[str]:
    """Named things in the question (XYZ, PLATINUM, "Ultra Card") that the documents never mention."""
    candidates = {m.group(0) for m in _SPECIFIC_TERM.finditer(question)}
    for quoted in _QUOTED.findall(question):
        candidates.update(quoted.split())
    unknown = []
    for term in sorted(candidates):
        words = re.findall(r"[a-z0-9]+", term.lower())
        if words and not any(w in known_words or w in _COMMON_ACRONYMS for w in words):
            unknown.append(term)
    return unknown


class Retriever:
    def __init__(
        self,
        *,
        embedder: Embedder,
        store: VectorStore,
        known_words: set[str],
        min_score: float,
        top_k: int = 4,
        relative_window: float | None = None,
        warmup: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        self.embedder = embedder
        self.store = store
        self.min_score = min_score
        self.top_k = top_k
        # Keep only passages scoring within this distance of the best one: drops "related but
        # not relevant" sections, so the LLM reads less noise and the citations stay accurate.
        self.relative_window = relative_window
        self._known_words = known_words
        self._warmup = warmup
        self._warm = warmup is None
        self._lock = asyncio.Lock()

    async def _ensure_warm(self) -> None:
        """Run one-time preparation (e.g. indexing into an in-memory store) on first use."""
        if self._warm:
            return
        async with self._lock:
            if not self._warm and self._warmup is not None:
                await self._warmup()
                self._warm = True

    async def retrieve(self, question: str, top_k: int | None = None) -> RetrievalResult:
        await self._ensure_warm()
        vector = await self.embedder.embed_query(question)
        hits = await self.store.search(self.embedder.name, vector, top_k or self.top_k)
        best = hits[0].score if hits else 0.0

        if not hits:
            logger.warning("rag.empty_index", extra={"embedder": self.embedder.name})
            return RetrievalResult(question, [], False, "empty_index", best)
        floor = (
            self.min_score
            if self.relative_window is None
            else max(self.min_score, best - self.relative_window)
        )
        relevant = [hit for hit in hits if hit.score >= floor]
        if not relevant:
            return RetrievalResult(question, [], False, "low_similarity", best)
        if missing := unknown_terms(question, self._known_words):
            return RetrievalResult(question, [], False, "unknown_terms", best, missing)
        return RetrievalResult(question, relevant, True, "ok", best)

    async def aclose(self) -> None:
        await self.embedder.aclose()
