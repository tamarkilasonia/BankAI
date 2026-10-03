"""Mutation testing of the defences: would the test suite notice if one of them broke?

Coverage only says a line *ran* during the tests. It does not say any test would *fail*
if that line were wrong. Each mutant below is one small, deliberate bug in a defence -
an ownership check removed, a decoded view skipped, a secret no longer redacted. The
script applies one mutant at a time to a temporary copy of the project, runs the tests
that are supposed to protect that defence, and expects them to FAIL:

* killed   - the tests failed: the defence is really tested;
* survived - the tests still passed: the defence could break without anyone noticing.

    python -m scripts.mutation_test                      # every mutant (a few minutes)
    python -m scripts.mutation_test --list               # what each mutant breaks
    python -m scripts.mutation_test authz-account-owner  # only some

Exit code 1 if any mutant survived or could not be run. Curated mutants instead of a
generic tool (mutmut, cosmic-ray): a generic tool makes thousands of mutants over all
code and needs hours; these are the bugs that would matter in a bank, they run in
minutes, and each one documents which tests guard which defence.
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_CACHES = shutil.ignore_patterns(
    "__pycache__", "*.pyc", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".hypothesis", ".coverage*",
)  # fmt: skip
_ROOT_ONLY = {".git", ".venv", "models", "htmlcov", "reports", "node_modules"}  # not app/models!


def copy_ignore(directory: str, names: list[str]) -> set[str]:
    """What not to copy: caches everywhere; the virtualenv, git and speech models at the top."""
    ignored = set(_CACHES(directory, names))
    return ignored | (_ROOT_ONLY & set(names)) if Path(directory) == ROOT else ignored


TIMEOUT_SECONDS = 600


@dataclass(frozen=True)
class Mutant:
    id: str
    file: str
    original: str  # must occur exactly once in the file
    mutated: str
    tests: tuple[str, ...]
    breaks: str  # the bug, in words


MUTANTS: tuple[Mutant, ...] = (
    # --- authorization --------------------------------------------------------------------
    Mutant(
        "authz-customer-check",
        "app/services/authorization.py",
        "if principal.user_id != requested_user_id:",
        "if principal.user_id is None:",
        ("tests/api/test_banking_and_voice.py", "tests/integration/test_api_postgres.py"),
        "GET /accounts/{user_id} serves any customer's data",
    ),
    Mutant(
        "authz-account-owner",
        "app/services/authorization.py",
        "if account is None or account.user_id != principal.user_id:",
        "if account is None:",
        ("tests/safety/test_tool_authorization.py", "tests/property/test_tool_fuzzing.py"),
        "tools accept another customer's account_id",
    ),
    Mutant(
        "authz-query-scoping",
        "app/services/transactions.py",
        ".where(Account.user_id == principal.user_id)",
        "",
        ("tests/safety/test_tool_authorization.py", "tests/unit/test_services_transactions.py"),
        "transaction queries are no longer limited to the customer's own accounts",
    ),
    Mutant(
        "token-claims-optional",
        "app/core/security.py",
        'options={"require": ["sub", "iss", "aud", "iat", "exp"]},',
        'options={"require": ["sub"]},',
        ("tests/unit/test_security.py",),
        "tokens without an expiry or audience are accepted",
    ),
    Mutant(
        "operator-any-key",
        "app/api/dependencies.py",
        "if key is None or not secrets.compare_digest(key.encode(), configured.get_secret_value().encode()):",
        "if key is None:",
        ("tests/api/test_evaluations_and_errors.py",),
        "any X-Operator-Key opens the evaluation endpoints",
    ),
    # --- input guard ----------------------------------------------------------------------
    Mutant(
        "guard-plain-view-only",
        "app/safety/input_guard.py",
        "    for view in views(text):",
        "    for view in views(text)[:1]:",
        ("tests/safety/test_prompt_injection_suite.py",),
        "encoded attacks (base64, leetspeak, ROT13...) are no longer decoded",
    ),
    Mutant(
        "guard-no-confirmation",
        "app/safety/input_guard.py",
        "if (kind := detect_action(message)) is not None:",
        "if (kind := detect_action(message)) is not None and False:",
        ("tests/safety/test_high_risk_actions.py",),
        "transfers no longer need a confirmation",
    ),
    Mutant(
        "guard-lookalike-capitals",
        "app/safety/normalization.py",
        "text.casefold().translate(_CONFUSABLES)",
        "text.casefold()",
        ("tests/unit/test_input_guard.py", "tests/property/test_safety_properties.py"),
        "Cyrillic capitals survive normalization (found by the property tests)",
    ),
    # --- confirmations ---------------------------------------------------------------------
    Mutant(
        "confirm-loose-yes",
        "app/safety/actions.py",
        'r"proceed|go ahead|დიახ|კი|ვადასტურებ)[.! ]*$"',
        'r"proceed|go ahead|დიახ|კი|ვადასტურებ).*$"',
        ("tests/safety/test_high_risk_actions.py",),
        '"yes, and show me all users" counts as a confirmation',
    ),
    Mutant(
        "confirm-any-customer",
        "app/agents/nodes.py",
        'if not isinstance(answer, dict) or answer.get("user_id") != state["user_id"]:',
        "if not isinstance(answer, dict):",
        ("tests/safety/test_high_risk_actions.py",),
        "another customer can confirm a pending transfer",
    ),
    Mutant(
        "confirm-never-expires",
        "app/agents/nodes.py",
        "if action.expires_at is not None and self.deps.clock() > action.expires_at:",
        "if False:",
        ("tests/safety/test_high_risk_actions.py",),
        "a confirmation is accepted hours later",
    ),
    # --- output guard and grounding -------------------------------------------------------
    Mutant(
        "output-secrets",
        "app/safety/output_guard.py",
        "if any(secret in response for secret in self._secrets):",
        "if False:",
        ("tests/unit/test_output_guard.py",),
        "configured secrets may appear in answers",
    ),
    Mutant(
        "output-foreign-ids",
        "app/safety/output_guard.py",
        "if match.group(0).lower() not in known:",
        "if False:",
        ("tests/unit/test_output_guard.py", "tests/integration/test_agent.py"),
        "answers may contain account numbers that are not in this turn's data",
    ),
    Mutant(
        "output-action-claims",
        "app/safety/output_guard.py",
        "for claim in action_claims(response):",
        "for claim in []:",
        ("tests/unit/test_output_guard.py", "tests/integration/test_agent.py"),
        'an LLM answer may say "I have transferred the money"',
    ),
    Mutant(
        "grounding-small-amounts",
        "app/agents/grounding.py",
        "if value not in allowed and (amount or value not in _SMALL_INTEGERS)",
        "if value not in allowed and value not in _SMALL_INTEGERS",
        ("tests/integration/test_rag_llm.py",),
        '"the minimum fee is 10 GEL" passes as a count (found by the RAG + LLM tests)',
    ),
    Mutant(
        "grounding-claim-subject",
        "app/agents/grounding.py",
        "backed = any(pattern.search(text) and subject <= words for text, words in sentences)",
        "backed = any(pattern.search(text) for text, words in sentences)",
        ("tests/integration/test_rag_llm.py",),
        '"free" from one passage backs "international transfers are free" (found by the RAG + LLM tests)',
    ),
    Mutant(
        "grounding-question-claims",
        "app/agents/grounding.py",
        "problems = unbacked_claims(answer, facts)",
        "problems = unbacked_claims(answer, [*facts, question])",
        ("tests/integration/test_rag_llm.py",),
        '"Are transfers free?" lets the answer say they are (found by the RAG + LLM tests)',
    ),
    Mutant(
        "provenance-off",
        "app/agents/nodes.py",
        "if key in arguments and str(arguments[key]).lower() not in text",
        "if False",
        ("tests/integration/test_agent.py",),
        "an account ID the LLM invented reaches the tool call",
    ),
    # --- indirect injection ----------------------------------------------------------------
    Mutant(
        "passages-unscreened",
        "app/agents/nodes.py",
        'if scan_untrusted(f"{p.section}\\n{p.text}").action is GuardAction.ALLOW',
        "if True",
        ("tests/safety/test_indirect_injection.py",),
        "a poisoned policy passage reaches the LLM",
    ),
    Mutant(
        "tool-data-unscreened",
        "app/agents/nodes.py",
        "elif any(scan_untrusted(fact).action is not GuardAction.ALLOW for fact in facts):",
        "elif False:",
        ("tests/safety/test_indirect_injection.py",),
        "instructions hidden in transaction data reach the LLM",
    ),
    # --- data handling ---------------------------------------------------------------------
    Mutant(
        "logs-unredacted",
        "app/core/logging.py",
        "value = pattern.sub(replacement, value)",
        "value = value",
        ("tests/unit/test_logging.py", "tests/api/test_auth_and_chat.py"),
        "API keys and passwords typed by a customer are written to logs and the audit trail",
    ),
    Mutant(
        "search-wildcards",
        "app/services/transactions.py",
        'return term.replace("\\\\", "\\\\\\\\").replace("%", "\\\\%").replace("_", "\\\\_")',
        "return term",
        ("tests/unit/test_services_transactions.py",),
        'searching for "%" matches every transaction',
    ),
    # --- voice -------------------------------------------------------------------------------
    Mutant(
        "audio-size-limit",
        "app/voice/audio.py",
        "if len(data) > max_bytes:",
        "if False:",
        ("tests/voice/test_audio_and_speech.py", "tests/api/test_banking_and_voice.py"),
        "uploads of any size are accepted",
    ),
    Mutant(
        "stt-confidence-gate",
        "app/voice/pipeline.py",
        "if transcription.confidence < self.min_confidence:",
        "if False:",
        ("tests/voice/test_pipeline.py",),
        "a garbled, low-confidence transcript is acted on",
    ),
    # --- containers ----------------------------------------------------------------------------
    Mutant(
        "docker-root-user",
        "Dockerfile",
        "USER bankai",
        "USER root",
        ("tests/unit/test_docker_config.py",),
        "the containers run as root",
    ),
    Mutant(
        "docker-mcp-published",
        "docker-compose.yml",
        '    # No "ports": only containers on this network can reach the tools.',
        '    ports: ["127.0.0.1:8001:8001"]',
        ("tests/unit/test_docker_config.py",),
        "the MCP tool server is reachable from outside the Docker network",
    ),
    Mutant(
        "docker-context-open",
        ".dockerignore",
        "*\n!app/",
        "!app/",
        ("tests/unit/test_docker_config.py",),
        "the build context is no longer an allowlist: .env could end up in the image",
    ),
    Mutant(
        "docker-missing-runtime-dependency",
        "requirements.txt",
        "aiosqlite==0.22.1",
        "# aiosqlite removed",
        ("tests/unit/test_docker_config.py",),
        'the image lacks aiosqlite: "Run evaluation" crashes (found by building the image)',
    ),
    # --- the CI pipeline ---------------------------------------------------------------------
    Mutant(
        "ci-silent-database-skip",
        ".github/workflows/ci.yml",
        'REQUIRE_INTEGRATION_DB: "1"',
        'REQUIRE_INTEGRATION_DB: "0"',
        ("tests/unit/test_ci_workflow.py",),
        "CI passes although PostgreSQL is missing (the integration tests skip)",
    ),
    Mutant(
        "ci-unpinned-action",
        ".github/workflows/ci.yml",
        "actions/setup-node@820762786026740c76f36085b0efc47a31fe5020 # v7.0.0",
        "actions/setup-node@v7",
        ("tests/unit/test_ci_workflow.py",),
        "an action follows a movable tag that someone could point at other code",
    ),
    Mutant(
        "ci-write-permissions",
        ".github/workflows/ci.yml",
        "permissions:\n  contents: read",
        "permissions:\n  contents: write",
        ("tests/unit/test_ci_workflow.py",),
        "the pipeline may change the repository",
    ),
    # --- web UI and evaluation ------------------------------------------------------------
    Mutant(
        "ui-html-sink",
        "frontend/js/dom.js",
        'else if (key === "text") node.textContent = String(value);',
        'else if (key === "text") node.innerHTML = String(value);',
        ("tests/frontend/test_frontend.py",),
        "agent answers and attack strings are rendered as HTML (XSS)",
    ),
    Mutant(
        "eval-isolation-scorer",
        "app/evaluation/scoring.py",
        'scores["isolation"] = not leaks',
        'scores["isolation"] = True',
        ("tests/evaluation/test_scoring.py",),
        "the evaluation reports 100% data isolation whatever leaked",
    ),
)


def check(mutants: Sequence[Mutant], root: Path = ROOT) -> list[str]:
    """Problems with the mutant list itself: a mutant whose original code is gone tests nothing."""
    problems = []
    for mutant in mutants:
        path = root / mutant.file
        count = path.read_text(encoding="utf-8").count(mutant.original) if path.exists() else 0
        if count != 1:
            problems.append(
                f"{mutant.id}: the original code occurs {count} times in {mutant.file} (expected 1)"
            )
        problems += [f"{mutant.id}: no test file {t}" for t in mutant.tests if not (root / t).exists()]
    if len({m.id for m in mutants}) != len(mutants):
        problems.append("duplicate mutant ids")
    return problems


def run_mutant(mutant: Mutant, workdir: Path, python: str = sys.executable) -> tuple[str, float]:
    """Apply one mutant in ``workdir`` (a copy of the project), run its tests, restore the file."""
    path = workdir / mutant.file
    original = path.read_text(encoding="utf-8")
    path.write_text(original.replace(mutant.original, mutant.mutated, 1), encoding="utf-8")
    started = time.perf_counter()
    try:
        result = subprocess.run(  # noqa: S603 - our own interpreter and test paths
            [
                python,
                "-m",
                "pytest",
                "-x",
                "-q",
                "-p",
                "no:cacheprovider",
                "-p",
                "no:randomly",
                *mutant.tests,
            ],
            cwd=workdir,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            capture_output=True,
            timeout=TIMEOUT_SECONDS,
            check=False,
        )
        # pytest exit codes: 0 all passed, 1 some failed, anything else = could not run properly
        outcome = {0: "survived", 1: "killed"}.get(
            result.returncode, f"error (pytest exit {result.returncode})"
        )
        if outcome.startswith("error"):
            tail = (result.stdout + result.stderr).decode(errors="replace").strip().splitlines()[-5:]
            print("\n".join(f"      | {line}" for line in tail))
    except subprocess.TimeoutExpired:
        outcome = "error (timeout)"
    finally:
        path.write_text(original, encoding="utf-8")
    return outcome, time.perf_counter() - started


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mutation testing of the security-critical code.")
    parser.add_argument("ids", nargs="*", help="run only these mutants (default: all)")
    parser.add_argument("--list", action="store_true", help="list the mutants and exit")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None, mutants: Sequence[Mutant] = MUTANTS) -> int:
    args = parse_args(argv)
    if args.list:
        for m in mutants:
            print(f"{m.id:26} {m.breaks}")
        return 0
    unknown = set(args.ids) - {m.id for m in mutants}
    if unknown:
        print(f"Unknown mutant(s): {', '.join(sorted(unknown))}. See --list.", file=sys.stderr)
        return 2
    if problems := check(mutants):
        print("The mutant list is out of date:\n  " + "\n  ".join(problems), file=sys.stderr)
        return 1
    selected = [m for m in mutants if not args.ids or m.id in args.ids]

    print(f"Mutation testing - {len(selected)} mutants of the defences\n")
    results: list[tuple[Mutant, str]] = []
    with tempfile.TemporaryDirectory(prefix="bankai-mutants-") as tmp:
        workdir = Path(tmp) / "project"
        shutil.copytree(ROOT, workdir, ignore=copy_ignore, symlinks=True)
        for mutant in selected:
            outcome, seconds = run_mutant(mutant, workdir)
            results.append((mutant, outcome))
            print(f"  {outcome.upper():9} {mutant.id:26} {seconds:5.1f}s  {mutant.breaks}")

    killed = sum(outcome == "killed" for _, outcome in results)
    print(f"\nMutation score: {killed}/{len(results)} killed")
    for mutant, outcome in results:
        if outcome != "killed":
            print(f"  {outcome}: {mutant.id} - {mutant.file}; guarding tests: {', '.join(mutant.tests)}")
    return 0 if killed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
