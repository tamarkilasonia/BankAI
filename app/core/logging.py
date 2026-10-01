"""Structured (JSON) logging with request tracing and secret redaction.

Why structured logs?
    Every log line is a JSON object, so log platforms (Loki, CloudWatch,
    Datadog, ...) can filter by fields like ``request_id`` or ``intent``
    instead of grepping free text.

Why redaction lives here?
    Redaction is applied centrally in the formatter, so a developer who
    accidentally logs a password or API key still does not leak it.

Usage::

    from app.core.logging import get_logger
    logger = get_logger(__name__)
    logger.info("tool.executed", extra={"tool": "get_account_balance", "duration_ms": 12.5})
"""

import json
import logging
import re
import sys
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any

# The request ID of the request currently being handled. ContextVar is safe for
# asyncio: each request/task sees its own value.
request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)

# The request ID also travels to the MCP server (in the MCP request ``_meta``),
# so one ID traces a request across both processes.
REQUEST_ID_META_KEY = "bankai/request_id"
_REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9._:-]{1,64}$")

REDACTED = "[REDACTED]"

# Dictionary keys whose values are always secret, whatever they contain.
_SENSITIVE_KEY_PATTERN = re.compile(
    r"pass(word|wd)?|secret|token|api[_-]?key|authorization|cookie|credential|private[_-]?key",
    re.IGNORECASE,
)

# Secret-looking substrings inside free text (messages, user input, errors).
_SENSITIVE_VALUE_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # OpenAI-style API keys: sk-..., sk-proj-...
    (re.compile(r"\bsk-[A-Za-z0-9_\-]{16,}"), REDACTED),
    # HTTP bearer tokens
    (re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-]+"), f"Bearer {REDACTED}"),
    # Passwords embedded in connection URLs: scheme://user:password@host
    (re.compile(r"(?P<prefix>[a-z][a-z0-9+.\-]*://[^:/@\s]+:)[^@\s]+@"), rf"\g<prefix>{REDACTED}@"),
]

# Attributes every LogRecord has; anything else came from `extra=` and is a field.
_STANDARD_RECORD_ATTRS = frozenset(
    logging.LogRecord("", 0, "", 0, "", None, None).__dict__.keys() | {"message", "asctime", "taskName"}
)


def safe_request_id(value: object) -> str | None:
    """Accept a request ID from another process only if it looks like one.

    The value is untrusted input that ends up in every log line; restricting
    length and characters prevents log injection (fake newlines, huge values).
    """
    if isinstance(value, str) and _REQUEST_ID_PATTERN.fullmatch(value):
        return value
    return None


def redact(value: Any) -> Any:
    """Recursively mask secrets in strings, dicts and lists."""
    if isinstance(value, str):
        for pattern, replacement in _SENSITIVE_VALUE_PATTERNS:
            value = pattern.sub(replacement, value)
        return value
    if isinstance(value, dict):
        return {
            key: REDACTED if _SENSITIVE_KEY_PATTERN.search(str(key)) else redact(item)
            for key, item in value.items()
        }
    if isinstance(value, list | tuple):
        return [redact(item) for item in value]
    return value


class JsonFormatter(logging.Formatter):
    """Render each log record as a single redacted JSON line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
            "request_id": request_id_var.get(),
        }
        for key, val in record.__dict__.items():
            if key not in _STANDARD_RECORD_ATTRS and not key.startswith("_"):
                payload[key] = val
        if record.exc_info:
            # Stack traces belong in logs (for engineers), never in API responses.
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(redact(payload), default=str, ensure_ascii=False)


class ConsoleFormatter(logging.Formatter):
    """Human-friendly single-line format for local development."""

    def format(self, record: logging.LogRecord) -> str:
        extras = {
            key: val
            for key, val in record.__dict__.items()
            if key not in _STANDARD_RECORD_ATTRS and not key.startswith("_")
        }
        line = f"{record.levelname:<7} {record.name}: {record.getMessage()}"
        if request_id := request_id_var.get():
            line += f" [request_id={request_id}]"
        if extras:
            # Redact the dict first: key-based rules ("password": ...) only
            # work while the data is still structured, not on the final string.
            line += " " + json.dumps(redact(extras), default=str, ensure_ascii=False)
        if record.exc_info:
            line += "\n" + self.formatException(record.exc_info)
        return str(redact(line))


def configure_logging(level: str = "INFO", fmt: str = "json") -> None:
    """Install a single stdout handler on the root logger.

    Safe to call more than once (e.g. by the API and by scripts): the previous
    handlers are replaced rather than duplicated.
    """
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter() if fmt == "json" else ConsoleFormatter())

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)

    # SQLAlchemy's own engine logging is controlled by DATABASE_ECHO instead.
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
