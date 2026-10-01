"""Loading the knowledge base and cutting it into retrievable chunks.

Chunking strategy: one chunk per ``##`` section. The documents are written so
that every section answers one question on its own ("Fees for international
transfers"), which makes a section the natural unit of retrieval - small enough
to be specific, big enough to be self-contained. Unusually long sections are
split further at paragraph boundaries.

Each chunk is embedded together with its document title and section heading
("Fee schedule - ATM withdrawal fees: ..."), so a chunk about "the fee" still
knows which fee it talks about.
"""

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

MAX_CHUNK_CHARS = 1200
_SKIP_FILES = {"README.md"}


@dataclass(frozen=True)
class Chunk:
    chunk_id: str  # stable: "<file>#<section-slug>[-<n>]"
    source: str  # file name, e.g. "fees.md"
    title: str  # document title (the "# " heading)
    section: str  # section heading (the "## " heading)
    text: str

    @property
    def embedding_text(self) -> str:
        return f"{self.title} - {self.section}: {self.text}"

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.embedding_text.encode()).hexdigest()


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def _split_long(text: str, limit: int = MAX_CHUNK_CHARS) -> list[str]:
    if len(text) <= limit:
        return [text]
    pieces: list[str] = []
    current = ""
    for paragraph in text.split("\n\n"):
        candidate = f"{current}\n\n{paragraph}".strip()
        if current and len(candidate) > limit:
            pieces.append(current)
            current = paragraph
        else:
            current = candidate
    if current:
        pieces.append(current)
    return pieces


def chunk_markdown(source: str, markdown: str) -> list[Chunk]:
    """Split one Markdown document into section chunks.

    Lines starting with ``>`` (the MOCK disclaimer blockquote) are not part of any
    chunk: they carry no answerable content.
    """
    title = source
    chunks: list[Chunk] = []
    section: str | None = None
    body: list[str] = []

    def flush() -> None:
        text = "\n".join(body).strip()
        if section is None or not text:
            return
        pieces = _split_long(text)
        for index, piece in enumerate(pieces, start=1):
            suffix = f"-{index}" if len(pieces) > 1 else ""
            chunks.append(Chunk(f"{source}#{slugify(section)}{suffix}", source, title, section, piece))

    for line in markdown.splitlines():
        if line.startswith("# "):
            title = line[2:].strip()
        elif line.startswith("## "):
            flush()
            section, body = line[3:].strip(), []
        elif not line.startswith(">"):
            body.append(line)
    flush()
    return chunks


def load_knowledge_base(directory: Path) -> list[Chunk]:
    """All chunks of all ``*.md`` documents in ``directory``, in a stable order."""
    if not directory.is_dir():
        raise FileNotFoundError(f"knowledge base directory not found: {directory}")
    chunks: list[Chunk] = []
    for path in sorted(directory.glob("*.md")):
        if path.name not in _SKIP_FILES:
            chunks.extend(chunk_markdown(path.name, path.read_text(encoding="utf-8")))
    return chunks


def vocabulary(chunks: list[Chunk]) -> set[str]:
    """Every lower-cased word that occurs in the knowledge base."""
    words: set[str] = set()
    for chunk in chunks:
        words.update(re.findall(r"[a-z0-9]+", chunk.embedding_text.lower()))
    return words
