"""Domain errors raised by the service layer.

Each error carries a stable machine-readable ``code``. The MCP server returns
the code to the agent, and the agent turns it into a user-facing reply - so
the codes are a contract, while messages are for humans.
"""

from typing import ClassVar


class DomainError(Exception):
    code: ClassVar[str] = "domain_error"

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class AccessDeniedError(DomainError):
    """The customer asked for data they do not own (or that does not exist).

    Both cases deliberately produce the same error, so the response cannot be
    used to probe which account IDs exist ("enumeration").
    """

    code = "access_denied"


class NotFoundError(DomainError):
    """A public, non-personal resource (e.g. a product code) does not exist."""

    code = "not_found"


class ServiceUnavailableError(DomainError):
    """A dependency (e.g. the knowledge base) is not available right now."""

    code = "unavailable"


class BusinessRuleError(DomainError):
    """The request is well-formed but violates a banking rule (e.g. loan limits)."""

    code = "rule_violation"
