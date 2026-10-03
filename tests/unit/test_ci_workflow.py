"""The CI configuration (.github/), checked like code.

A pipeline can quietly stop protecting anything: a lane that skips instead of failing, an
action that silently changes because it is pinned to a movable tag, a new test marker that no
job ever runs. These tests make such changes fail on the developer's machine, before a
broken pipeline reaches GitHub.
"""

import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

from tests.conftest import PROJECT_ROOT

WORKFLOWS = PROJECT_ROOT / ".github/workflows"
CI = "ci.yml"
LIVE = "live-models.yml"
BIN = Path(sys.executable).parent  # actionlint and shellcheck come from requirements-dev.txt


def load(name: str) -> dict[str, Any]:
    data: dict[str, Any] = yaml.safe_load((WORKFLOWS / name).read_text(encoding="utf-8"))
    return data


def triggers(workflow: dict[str, Any]) -> Any:
    return workflow.get("on", workflow.get(True))  # YAML 1.1 reads the key "on" as True


def steps(job: dict[str, Any]) -> list[dict[str, Any]]:
    return list(job["steps"])


def pytest_marker_expressions(workflow: dict[str, Any]) -> list[str]:
    """The -m expressions of every pytest command in the workflow ("postgres or ui", "docker")."""
    commands = " ".join(s.get("run", "") for job in workflow["jobs"].values() for s in steps(job))
    return [quoted or bare for quoted, bare in re.findall(r'pytest -m "([^"]+)"|pytest -m (\w+)', commands)]


def selected_markers(expression: str) -> set[str]:
    """Markers an expression runs: "not postgres and not ui" selects none, "postgres or ui" two."""
    return set(re.findall(r"(?<!not )\b(\w+)\b", expression)) - {"or", "and", "not"}


def test_the_workflows_pass_actionlint_and_shellcheck() -> None:
    actionlint = BIN / "actionlint"
    if not actionlint.exists():
        actionlint = Path(shutil.which("actionlint") or "missing")
    assert actionlint.exists(), "actionlint is in requirements-dev.txt"
    shellcheck = BIN / "shellcheck"
    result = subprocess.run(  # noqa: S603 - our own tool on our own files
        [
            str(actionlint),
            "-no-color",
            "-shellcheck",
            str(shellcheck) if shellcheck.exists() else "shellcheck",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("name", [CI, LIVE])
def test_every_action_is_pinned_to_a_commit(name: str) -> None:
    """A tag like @v7 can be moved to other code; a commit SHA cannot (supply-chain attacks)."""
    uses = [s["uses"] for job in load(name)["jobs"].values() for s in steps(job) if "uses" in s]
    assert uses
    for action in uses:
        assert re.fullmatch(r"[\w.-]+/[\w.-]+@[0-9a-f]{40}", action), (
            f"{name}: {action} is not pinned to a SHA"
        )
    text = (WORKFLOWS / name).read_text(encoding="utf-8")
    assert all(re.search(rf"{re.escape(a)} # v\d", text) for a in uses)  # with the readable version


@pytest.mark.parametrize("name", [CI, LIVE])
def test_workflows_only_read_the_repository_and_use_no_secrets(name: str) -> None:
    assert load(name)["permissions"] == {"contents": "read"}
    assert "secrets." not in (WORKFLOWS / name).read_text(encoding="utf-8")  # no API key anywhere


def test_the_pipeline_follows_the_specification() -> None:
    """install -> lint -> unit tests -> integration tests -> coverage -> Docker image."""
    jobs = load(CI)["jobs"]
    assert jobs["test"]["needs"] == "lint" and jobs["docker"]["needs"] == "test"

    def names(job: str) -> list[str]:
        return [s.get("name", "") for s in steps(jobs[job])]

    lint = names("lint")
    assert (
        lint.index("Install dependencies")
        < lint.index("Lint (ruff)")
        < lint.index("Type-check (mypy, strict)")
    )
    order = ["Unit tests", "Integration tests", "Coverage", "Mutation testing"]
    found = [next(i for i, n in enumerate(names("test")) if n.startswith(prefix)) for prefix in order]
    assert found == sorted(found)
    assert any(n.startswith("Build the image") for n in names("docker"))


def test_no_lane_can_skip_silently() -> None:
    ci, live = load(CI)["jobs"], load(LIVE)["jobs"]
    assert ci["test"]["env"]["REQUIRE_INTEGRATION_DB"] == "1"
    assert ci["test"]["env"]["REQUIRE_BROWSER"] == "1"
    assert ci["docker"]["env"]["REQUIRE_DOCKER"] == "1"
    assert live["llm"]["env"]["REQUIRE_LLM"] == "1"
    assert live["voice"]["env"]["REQUIRE_VOICE"] == "1"


def test_every_test_marker_runs_in_some_workflow() -> None:
    """A new marker that no job selects would be a set of tests that never runs in CI."""
    config = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    markers = set(re.findall(r'^\s+"(\w+): ', config, re.MULTILINE))
    expressions = pytest_marker_expressions(load(CI)) + pytest_marker_expressions(load(LIVE))
    assert markers == {"postgres", "llm", "voice", "ui", "docker"}
    assert set().union(*map(selected_markers, expressions)) == markers
    # ... and the ordinary pipeline never needs a real LLM (specification: keep those separate)
    assert all("llm" not in selected_markers(e) for e in pytest_marker_expressions(load(CI)))


def test_the_ordinary_pipeline_runs_on_every_push_and_the_live_models_only_by_hand() -> None:
    assert set(triggers(load(CI))) == {"push", "pull_request", "workflow_dispatch"}
    assert set(triggers(load(LIVE))) == {"workflow_dispatch"}


def test_dependabot_watches_every_pinned_dependency() -> None:
    config = yaml.safe_load((PROJECT_ROOT / ".github/dependabot.yml").read_text(encoding="utf-8"))
    assert {u["package-ecosystem"] for u in config["updates"]} == {"pip", "github-actions", "docker"}
