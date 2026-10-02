"""Conversation state between turns, and the audit trail of every turn.

**State.** The agent needs the previous turn (for "and last month?") and a pending
high-risk confirmation (for "yes"). The client only holds an opaque
``conversation_id``; the state itself stays on the server, bound to the customer.
A conversation ID of another customer, an unknown or an expired one simply starts a
new conversation (with a new ID) - it is never an error message that reveals anything.
In-memory per process; with several API instances this would move to Redis with a TTL.

**Audit.** Every turn is written to ``conversation_logs`` (spec: timestamp, user,
request ID, input, intent, tool, tool and model latency, answer, safety status,
error) - input and answer pass through the same secret redaction as the logs.
"""

import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.agents.assistant import AssistantReply, PendingConfirmation, PreviousTurn
from app.core.logging import get_logger, redact
from app.core.time import utcnow
from app.models import ConversationLog
from app.models.enums import Channel

logger = get_logger(__name__)


@dataclass
class Conversation:
    user_id: uuid.UUID
    previous: PreviousTurn | None = None
    pending: PendingConfirmation | None = None
    updated_at: datetime | None = None


class ConversationStore:
    def __init__(self, ttl: timedelta) -> None:
        self._ttl = ttl
        self._items: dict[str, Conversation] = {}

    def open(self, conversation_id: str | None, user_id: uuid.UUID) -> tuple[str, Conversation]:
        """The customer's conversation with this ID, or a new one."""
        self._purge()
        found = self._items.get(conversation_id or "")
        if conversation_id and found is not None and found.user_id == user_id:
            return conversation_id, found
        new_id = uuid.uuid4().hex
        self._items[new_id] = Conversation(user_id=user_id, updated_at=utcnow())
        return new_id, self._items[new_id]

    def remember(self, conversation_id: str, reply: AssistantReply) -> None:
        conversation = self._items[conversation_id]
        conversation.previous = reply.as_previous_turn() or conversation.previous
        conversation.pending = reply.pending_confirmation
        conversation.updated_at = utcnow()

    def __len__(self) -> int:
        return len(self._items)

    def _purge(self) -> None:
        cutoff = utcnow() - self._ttl
        for key in [k for k, c in self._items.items() if c.updated_at and c.updated_at < cutoff]:
            del self._items[key]


def _error_code(reply: AssistantReply) -> str | None:
    if reply.tool_error:
        return reply.tool_error
    return next((e for e in reply.errors if e in {"internal_error", "output_blocked"}), None)


async def log_turn(
    sessions: async_sessionmaker[AsyncSession],
    *,
    conversation_id: str,
    user_id: uuid.UUID,
    channel: Channel,
    user_input: str,
    reply: AssistantReply | None,
    response_text: str,
    request_id: str,
    error_code: str | None = None,
) -> None:
    """Write the audit row. A failure is logged loudly but never breaks the customer's answer."""
    row = ConversationLog(
        request_id=request_id,
        conversation_id=conversation_id,
        user_id=user_id,
        channel=channel,
        user_input=redact(user_input),
        detected_intent=reply.intent.value if reply else None,
        selected_tool=reply.tool if reply else None,
        tool_arguments=redact(reply.tool_arguments) if reply else None,
        tool_latency_ms=reply.tool_latency_ms if reply else None,
        llm_latency_ms=reply.llm_latency_ms if reply else None,
        total_latency_ms=reply.latency_ms if reply else None,
        response_text=redact(response_text),
        safety_status=reply.safety_status if reply else None,
        safety_flags=reply.safety_flags if reply else [],
        error_code=_error_code(reply) if reply else error_code,
    )
    try:
        async with sessions() as session:
            session.add(row)
            await session.commit()
    except SQLAlchemyError as exc:
        logger.error("audit.write_failed", extra={"error": type(exc).__name__})
