"""The mutation-testing script (scripts/mutation_test.py).

The full run takes about a minute and lives in CI; here we check that the mutant list
still matches the code (a refactored defence must update its mutant, or the mutant
would silently test nothing) and that the script reports each outcome correctly.
"""

import subprocess
from pathlib import Path

import pytest

from scripts import mutation_test
from scripts.mutation_test import MUTANTS, Mutant, check, main, run_mutant

GROUNDING_TESTS = ("tests/unit/test_grounding_and_rendering.py",)


def test_every_mutant_still_matches_the_code() -> None:
    assert check(MUTANTS) == []
    assert len(MUTANTS) >= 25


def test_an_out_of_date_mutant_is_reported(capsys: pytest.CaptureFixture[str]) -> None:
    stale = Mutant("stale", "app/nowhere.py", "x", "y", ("tests/nowhere.py",), "nothing")
    assert check([stale, stale]) == [
        "stale: the original code occurs 0 times in app/nowhere.py (expected 1)",
        "stale: no test file tests/nowhere.py",
        "stale: the original code occurs 0 times in app/nowhere.py (expected 1)",
        "stale: no test file tests/nowhere.py",
        "duplicate mutant ids",
    ]
    assert main([], mutants=[stale]) == 1
    assert "out of date" in capsys.readouterr().err


def test_list_and_unknown_ids(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--list"]) == 0
    assert "authz-account-owner" in capsys.readouterr().out
    assert main(["no-such-mutant"]) == 2
    assert "Unknown mutant" in capsys.readouterr().err


def test_killed_survived_and_broken_mutants_are_told_apart(capsys: pytest.CaptureFixture[str]) -> None:
    mutants = [
        Mutant(
            "real-bug",
            "app/agents/grounding.py",
            "if value not in allowed and (amount or value not in _SMALL_INTEGERS)",
            "if value not in allowed and value not in _SMALL_INTEGERS",
            GROUNDING_TESTS,
            "small amounts pass as counts",
        ),
        Mutant(
            "harmless",
            "app/agents/grounding.py",
            "# transfers = transfer",
            "# (comment)",
            GROUNDING_TESTS,
            "-",
        ),
        Mutant("syntax-error", "app/agents/grounding.py", "def _stem(", "def _stem((", GROUNDING_TESTS, "-"),
    ]

    assert main([], mutants=mutants) == 1

    out = capsys.readouterr().out
    assert "KILLED    real-bug" in out
    assert "SURVIVED  harmless" in out
    assert "ERROR (PYTEST EXIT" in out and "syntax-error" in out
    assert "Mutation score: 1/3 killed" in out


def test_a_hanging_test_run_is_an_error_and_the_file_is_restored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "code.py").write_text("x = 1\n", encoding="utf-8")
    mutant = Mutant("hang", "code.py", "x = 1", "x = 2", ("tests",), "-")

    def hang(*args: object, **kwargs: object) -> None:
        raise subprocess.TimeoutExpired("pytest", 1)

    monkeypatch.setattr(mutation_test.subprocess, "run", hang)
    outcome, _ = run_mutant(mutant, tmp_path)

    assert outcome == "error (timeout)"
    assert (tmp_path / "code.py").read_text(encoding="utf-8") == "x = 1\n"
