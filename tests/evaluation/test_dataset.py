"""The evaluation dataset itself: complete, consistent with the tools, true to the documents."""

import json
from collections import Counter
from pathlib import Path

import pytest

from app.agents.grounding import numbers_in
from app.agents.intents import INTENT_TOOL
from app.evaluation.dataset import DATASET, dataset_version, load_cases, select_cases
from app.evaluation.schemas import Behavior, EvalCase, EvalCategory
from app.mcp.schemas import TOOLS
from tests.conftest import PROJECT_ROOT
from tests.oracle import SEED

CASES = load_cases()
CUSTOMERS = {u.customer_number for u in SEED.users}
PREFIXES = {
    "ACCOUNT": "ACC", "TRANSACTION": "TRX", "SPENDING": "SPD", "CALCULATION": "CAL", "RAG": "RAG",
    "GENERAL": "GEN", "SAFETY": "SAF", "PROMPT_INJECTION": "INJ", "AUTHORIZATION": "AUT", "VOICE": "VOI",
}  # fmt: skip


def test_size_and_coverage() -> None:
    per_category = Counter(c.category for c in CASES)

    assert len(CASES) >= 200
    assert set(per_category) == set(EvalCategory)
    assert min(per_category.values()) >= 15
    assert all(c.id.startswith(PREFIXES[c.category.value] + "-") for c in CASES)


def test_every_case_names_a_real_customer() -> None:
    assert {c.customer for c in CASES} <= CUSTOMERS


@pytest.mark.parametrize("case", [c for c in CASES if c.expected_tool], ids=lambda c: c.id)
def test_expected_tool_matches_the_intent(case: EvalCase) -> None:
    assert case.expected_tool in TOOLS
    if case.expected_intent is not None:
        assert INTENT_TOOL.get(case.expected_intent) == case.expected_tool


@pytest.mark.parametrize("case", [c for c in CASES if c.expected_arguments], ids=lambda c: c.id)
def test_expected_arguments_are_valid_tool_input(case: EvalCase) -> None:
    """An expected argument the tool would reject could never be matched."""
    assert case.expected_tool is not None
    TOOLS[case.expected_tool].input_model.model_validate(case.expected_arguments)


def test_refusals_and_confirmations_expect_no_tool() -> None:
    for case in CASES:
        if case.expected_behavior in {Behavior.REFUSE, Behavior.CONFIRM, Behavior.GENERAL}:
            assert case.expected_tool is None, case.id


def _document(name: str) -> str:
    return (PROJECT_ROOT / "knowledge_base" / name).read_text(encoding="utf-8")


@pytest.mark.parametrize("case", [c for c in CASES if c.expected_answer.sources], ids=lambda c: c.id)
def test_document_facts_really_are_in_the_cited_document(case: EvalCase) -> None:
    """The expected facts come from the documents, so they are checked against them."""
    text = " ".join(_document(source) for source in case.expected_answer.sources)
    assert {v.normalize() for v in case.expected_answer.values} <= numbers_in(text)
    for fragment in case.expected_answer.contains:
        assert fragment.lower() in text.lower()


def test_document_values_without_a_named_source_exist_somewhere() -> None:
    corpus = " ".join(p.read_text(encoding="utf-8") for p in (PROJECT_ROOT / "knowledge_base").glob("*.md"))
    for case in CASES:
        if case.category is EvalCategory.RAG and not case.expected_answer.sources:
            assert {v.normalize() for v in case.expected_answer.values} <= numbers_in(corpus), case.id


def test_version_is_a_content_hash(tmp_path: Path) -> None:
    copy = tmp_path / "cases.jsonl"
    copy.write_bytes(DATASET.read_bytes())
    assert dataset_version(copy) == dataset_version()
    copy.write_text(DATASET.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    assert dataset_version(copy) != dataset_version()
    assert len(dataset_version()) == 12


def test_duplicate_ids_are_rejected(tmp_path: Path) -> None:
    line = json.dumps({"id": "X-1", "category": "GENERAL", "input": "hi", "expected_behavior": "general"})
    path = tmp_path / "dup.jsonl"
    path.write_text(f"{line}\n{line}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate case ids"):
        load_cases(path)


def test_selection_by_category_tag_and_limit() -> None:
    assert {c.category for c in select_cases(CASES, categories=["rag"])} == {EvalCategory.RAG}
    holdout = select_cases(CASES, tags=["holdout"])
    assert holdout and all("holdout" in c.tags for c in holdout)
    assert len(select_cases(CASES, limit=5)) == 5
    assert select_cases(CASES) == CASES
