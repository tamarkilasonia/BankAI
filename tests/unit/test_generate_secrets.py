"""The Docker secrets generator (scripts/generate_secrets.py)."""

import stat
from pathlib import Path

import pytest

from scripts.generate_secrets import MIN_LENGTH, SECRETS, SecretError, ensure_secrets, main

GIVEN = "operator-key-from-the-env-file-0123456789"


def mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def test_first_start_generates_private_random_secrets(tmp_path: Path) -> None:
    report = ensure_secrets(tmp_path / "secrets", {})

    assert report == dict.fromkeys(SECRETS, "generated")
    values = [(tmp_path / "secrets" / name).read_text() for name in SECRETS]
    assert len(set(values)) == len(SECRETS) and all(len(v) >= MIN_LENGTH for v in values)
    assert all(mode(tmp_path / "secrets" / name) == 0o600 for name in SECRETS)


def test_restarts_keep_the_secrets(tmp_path: Path) -> None:
    ensure_secrets(tmp_path, {})
    before = (tmp_path / "mcp_auth_secret").read_text()

    assert ensure_secrets(tmp_path, {}) == dict.fromkeys(SECRETS, "kept")
    assert (tmp_path / "mcp_auth_secret").read_text() == before


def test_a_value_from_the_environment_wins_and_empty_means_unset(tmp_path: Path) -> None:
    ensure_secrets(tmp_path, {})
    (tmp_path / "operator_api_key").chmod(0o644)

    report = ensure_secrets(tmp_path, {"OPERATOR_API_KEY": GIVEN, "MCP_AUTH_SECRET": "  "})

    assert report["operator_api_key"] == "from the environment" and report["mcp_auth_secret"] == "kept"
    assert (tmp_path / "operator_api_key").read_text() == GIVEN
    assert mode(tmp_path / "operator_api_key") == 0o600  # tightened again


def test_a_damaged_secret_file_is_replaced(tmp_path: Path) -> None:
    (tmp_path / "api_session_secret").write_text("short")
    assert ensure_secrets(tmp_path, {})["api_session_secret"] == "generated"


def test_a_weak_secret_from_the_environment_is_refused(tmp_path: Path) -> None:
    with pytest.raises(SecretError, match="at least 32 characters"):
        ensure_secrets(tmp_path, {"MCP_AUTH_SECRET": "too-short"})


def test_command_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("SECRETS_DIR", raising=False)
    assert main([]) == 2

    monkeypatch.setenv("OPERATOR_API_KEY", GIVEN)
    assert main([str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "secret operator_api_key: from the environment" in out and GIVEN not in out  # never printed

    monkeypatch.setenv("OPERATOR_API_KEY", "short")
    assert main([str(tmp_path)]) == 1
