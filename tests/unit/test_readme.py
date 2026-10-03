"""The README and INTERVIEW.md, checked like code.

The README is the first thing a reader sees, so it must not drift from the project: links that
lead nowhere, a script that was renamed, or evaluation numbers that no longer match the stored
reports would quietly make it untrue.
"""

import re
from pathlib import Path

import pytest

from tests.conftest import PROJECT_ROOT

README = PROJECT_ROOT / "README.md"
INTERVIEW = PROJECT_ROOT / "INTERVIEW.md"
DOCUMENTS = [README, INTERVIEW, *sorted((PROJECT_ROOT / "docs").rglob("*.md"))]
LINK = re.compile(r"!?\[[^\]]*\]\(([^)\s]+)\)")

# The 23 README sections the project specification requires (§23), in its order.
REQUIRED_SECTIONS = [
    "Project overview", "Problem statement", "Why agentic AI", "Architecture", "Technology stack",
    "LangGraph", "MCP", "RAG", "Evaluation methodology", "Safety architecture", "Testing strategy",
    "Voice architecture", "Database architecture", "API", "Installation", "Environment variables",
    "Running locally", "Running tests", "Running the evaluation", "Example conversations",
    "Example evaluation results", "Limitations", "Future improvements",
]  # fmt: skip


def text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def outside_code(markdown: str) -> str:
    """The Markdown without fenced code blocks (their contents are not links)."""
    return re.sub(r"```.*?```", "", markdown, flags=re.DOTALL)


def github_anchor(heading: str) -> str:
    """The anchor GitHub generates for a heading: "3. Why agentic AI?" -> "3-why-agentic-ai"."""
    return re.sub(r"[^\w\- ]", "", heading.strip().lower()).replace(" ", "-")


def anchors(markdown: str) -> set[str]:
    return {github_anchor(h) for h in re.findall(r"^#{1,6} (.+)$", outside_code(markdown), re.MULTILINE)}


def test_the_readme_has_the_sections_the_specification_requires() -> None:
    headings = re.findall(r"^## (\d+)\. (.+)$", text(README), re.MULTILINE)
    assert [int(number) for number, _ in headings] == list(range(1, 24))
    for (_, heading), required in zip(headings, REQUIRED_SECTIONS, strict=True):
        assert heading.startswith(required), f"section '{heading}' should be '{required}'"


def test_the_architecture_diagram_shows_every_path_the_specification_names() -> None:
    diagrams = re.findall(r"```mermaid\n(.*?)```", text(README), re.DOTALL)
    assert len(diagrams) >= 2
    architecture = diagrams[0]
    for part in ("Web UI", "FastAPI", "LangGraph", "MCP", "PostgreSQL", "Speech-to-Text",
                 "Text-to-Speech", "Evaluation", "Dashboard"):  # fmt: skip
        assert part in architecture, part


@pytest.mark.parametrize("document", DOCUMENTS, ids=lambda p: str(p.relative_to(PROJECT_ROOT)))
def test_every_relative_link_and_anchor_resolves(document: Path) -> None:
    markdown = text(document)
    broken = []
    for target in LINK.findall(outside_code(markdown)):
        if target.startswith(("http://", "https://", "mailto:")):
            continue
        path, _, anchor = target.partition("#")
        file = (document.parent / path).resolve() if path else document
        missing_anchor = (
            file.exists() and bool(anchor) and file.suffix == ".md" and anchor not in anchors(text(file))
        )
        if not file.exists() or missing_anchor:
            broken.append(target)
    assert not broken, broken


@pytest.mark.parametrize("document", [README, INTERVIEW], ids=["README", "INTERVIEW"])
def test_every_script_the_documents_tell_you_to_run_exists(document: Path) -> None:
    scripts = set(re.findall(r"python -m scripts\.(\w+)", text(document)))
    assert scripts
    assert not [s for s in scripts if not (PROJECT_ROOT / f"scripts/{s}.py").is_file()]


def test_the_readme_evaluation_numbers_are_the_stored_reports() -> None:
    """Each "85.4% (210/246)" in the results table is a row of the report it claims to come from."""
    reports = {
        "offline": text(PROJECT_ROOT / "docs/evaluation/offline.md"),
        "llm": text(PROJECT_ROOT / "docs/evaluation/qwen2.5-3b.md"),
    }
    section = text(README).split("## 21. Example evaluation results")[1].split("\n## ")[0]
    rows = [line for line in section.splitlines() if re.match(r"\| [^-]", line)][1:]  # skip the header
    checked = 0
    for row in rows:
        cells = [cell.strip(" *") for cell in row.strip("|").split("|")]
        for report, cell in zip(("offline", "llm"), cells[1:], strict=True):
            if match := re.fullmatch(r"([\d.]+)% \((\d+/\d+)\)", cell):
                percent, fraction = match.groups()
                stored = (
                    f"**{fraction} cases passed ({percent}%).**"
                    if cells[0] == "Cases passed"
                    else f"| {percent}% | {fraction} |"
                )
                assert stored in reports[report], f"{cells[0]} ({report}): {cell} is not in the report"
                checked += 1
    assert checked >= 24


def test_the_interview_guide_has_at_least_forty_numbered_questions() -> None:
    numbers = [int(n) for n in re.findall(r"^### (\d+)\. ", text(INTERVIEW), re.MULTILINE)]
    assert len(numbers) >= 40  # the specification asks for at least 40
    assert numbers == list(range(1, len(numbers) + 1))
