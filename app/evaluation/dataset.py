"""Loading the evaluation dataset.

The dataset is a JSONL file (one case per line) under version control. Each run
records the file's content hash, so results are only ever compared between runs
of the same dataset version.
"""

import hashlib
from collections.abc import Iterable
from pathlib import Path

from app.evaluation.schemas import EvalCase, EvalCategory

DATASET = Path(__file__).with_name("datasets") / "banking_eval.jsonl"


def dataset_version(path: Path = DATASET) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def load_cases(path: Path = DATASET) -> list[EvalCase]:
    lines = path.read_text(encoding="utf-8").splitlines()
    cases = [EvalCase.model_validate_json(line) for line in lines if line.strip()]
    duplicates = {c.id for c in cases if sum(other.id == c.id for other in cases) > 1}
    if duplicates:
        raise ValueError(f"duplicate case ids: {sorted(duplicates)}")
    return cases


def select_cases(
    cases: Iterable[EvalCase],
    *,
    categories: Iterable[str] = (),
    tags: Iterable[str] = (),
    limit: int | None = None,
) -> list[EvalCase]:
    """Filter by category and/or tag (e.g. run only PROMPT_INJECTION, or only ``holdout``)."""
    wanted = {EvalCategory(c.upper()) for c in categories}
    wanted_tags = set(tags)
    chosen = [
        c
        for c in cases
        if (not wanted or c.category in wanted) and (not wanted_tags or wanted_tags & set(c.tags))
    ]
    return chosen[:limit] if limit is not None else chosen
