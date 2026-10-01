import json
import logging
import sys

import pytest

from app.core.logging import (
    REDACTED,
    ConsoleFormatter,
    JsonFormatter,
    configure_logging,
    get_logger,
    redact,
    request_id_var,
    safe_request_id,
)


def make_record(message: str, *, exc: bool = False, **extra: object) -> logging.LogRecord:
    exc_info = None
    if exc:
        try:
            raise ValueError("boom")
        except ValueError:
            exc_info = sys.exc_info()
    record = logging.LogRecord("test.logger", logging.INFO, __file__, 1, message, None, exc_info)
    record.__dict__.update(extra)
    return record


# --- redact() -----------------------------------------------------------------


@pytest.mark.parametrize(
    "key", ["password", "db_password", "api_key", "OPENAI_API_KEY", "Authorization", "client_secret", "token"]
)
def test_redacts_sensitive_dictionary_keys(key: str) -> None:
    assert redact({key: "value", "safe": "visible"}) == {key: REDACTED, "safe": "visible"}


def test_redacts_nested_structures() -> None:
    data = {"request": {"headers": {"authorization": "Bearer abc"}, "items": [{"password": "x"}, "ok"]}}

    assert redact(data) == {
        "request": {"headers": {"authorization": REDACTED}, "items": [{"password": REDACTED}, "ok"]}
    }


@pytest.mark.parametrize(
    ("text", "leaked"),
    [
        ("my key is sk-proj-ABCDEFGHIJKLMNOPQRSTUV please", "sk-proj-ABCDEFGHIJKLMNOPQRSTUV"),
        ("header: Bearer eyJhbGciOiJIUzI1NiJ9.payload.sig", "eyJhbGciOiJIUzI1NiJ9"),
        ("postgresql+asyncpg://bankai:topsecret@db:5432/bankai", "topsecret"),
    ],
)
def test_redacts_secrets_inside_free_text(text: str, leaked: str) -> None:
    result = redact(text)
    assert leaked not in result
    assert REDACTED in result


def test_leaves_normal_values_untouched() -> None:
    assert redact("What is my balance?") == "What is my balance?"
    assert redact(42) == 42
    assert redact(None) is None
    assert redact(("a", "b")) == ["a", "b"]


# --- formatters ---------------------------------------------------------------


def test_json_formatter_emits_one_json_object_with_context() -> None:
    token = request_id_var.set("req-123")
    try:
        line = JsonFormatter().format(make_record("tool.executed", tool="get_balance", duration_ms=12.5))
    finally:
        request_id_var.reset(token)

    payload = json.loads(line)
    assert payload["event"] == "tool.executed"
    assert payload["level"] == "INFO"
    assert payload["logger"] == "test.logger"
    assert payload["request_id"] == "req-123"
    assert payload["tool"] == "get_balance"
    assert payload["duration_ms"] == 12.5
    assert payload["timestamp"].endswith("+00:00")


def test_json_formatter_redacts_extra_fields_and_message() -> None:
    line = JsonFormatter().format(make_record("using key sk-abcdefghijklmnopqrstuvwx", api_key="sk-zzz"))

    assert "sk-abcdefghijklmnopqrstuvwx" not in line
    assert "sk-zzz" not in line
    assert json.loads(line)["api_key"] == REDACTED


def test_json_formatter_includes_exception_details() -> None:
    payload = json.loads(JsonFormatter().format(make_record("failed", exc=True)))

    assert "ValueError: boom" in payload["exception"]


def test_console_formatter_is_readable_and_redacted() -> None:
    token = request_id_var.set("req-9")
    try:
        line = ConsoleFormatter().format(
            make_record("seed.completed", users=10, password="hunter2", exc=True)
        )
    finally:
        request_id_var.reset(token)

    assert line.startswith("INFO    test.logger: seed.completed [request_id=req-9]")
    assert '"users": 10' in line
    assert "hunter2" not in line
    assert "ValueError: boom" in line


# --- configure_logging() ------------------------------------------------------


@pytest.mark.parametrize(("fmt", "formatter"), [("json", JsonFormatter), ("console", ConsoleFormatter)])
def test_configure_logging_installs_single_handler(fmt: str, formatter: type[logging.Formatter]) -> None:
    configure_logging("DEBUG", fmt)
    configure_logging("DEBUG", fmt)  # idempotent: no duplicate handlers

    root = logging.getLogger()
    assert len(root.handlers) == 1
    assert isinstance(root.handlers[0].formatter, formatter)
    assert root.level == logging.DEBUG


def test_logged_output_is_json_on_stdout(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("INFO", "json")

    get_logger("bankai.test").info("hello", extra={"user_id": "u-1"})

    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["event"] == "hello"
    assert payload["user_id"] == "u-1"


# --- request IDs from other processes ------------------------------------------


@pytest.mark.parametrize("value", ["req-123", "a1b2c3d4-e5f6", "trace:01.2_x"])
def test_safe_request_id_accepts_normal_ids(value: str) -> None:
    assert safe_request_id(value) == value


@pytest.mark.parametrize(
    "value",
    [None, "", 42, "x" * 65, "evil\nINFO fake log line", '{"injected": true}', "id with spaces"],
)
def test_safe_request_id_rejects_log_injection(value: object) -> None:
    assert safe_request_id(value) is None
