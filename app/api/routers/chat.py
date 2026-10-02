"""POST /chat - one turn of the conversation with the banking agent."""

from fastapi import APIRouter

from app.agents.assistant import AssistantReply
from app.api.conversations import log_turn
from app.api.dependencies import PrincipalDep, ServicesDep
from app.api.schemas import ChatRequest, ChatResponse, ChatTrace, ErrorResponse, PendingActionOut
from app.core.logging import request_id_var
from app.models.enums import Channel

router = APIRouter(tags=["chat"])


def chat_response(conversation_id: str, reply: AssistantReply) -> ChatResponse:
    pending = reply.pending_confirmation
    return ChatResponse(
        conversation_id=conversation_id,
        request_id=reply.request_id,
        response=reply.response,
        language=reply.language,
        safety_status=reply.safety_status,
        pending_confirmation=PendingActionOut(
            kind=pending.action.kind, summary=pending.action.summary, expires_at=pending.expires_at
        )
        if pending
        else None,
        action_request=reply.action_request,
        latency_ms=reply.latency_ms,
        trace=ChatTrace(
            intent=reply.intent.value,
            intent_source=reply.intent_source,
            intent_confidence=reply.intent_confidence,
            tool=reply.tool,
            tool_arguments=reply.tool_arguments,
            tool_error=reply.tool_error,
            response_source=reply.response_source,
            safety_flags=reply.safety_flags,
            llm_calls=reply.llm_calls,
            errors=reply.errors,
            timings_ms=reply.timings_ms,
        ),
    )


@router.post("/chat", responses={401: {"model": ErrorResponse}, 422: {"model": ErrorResponse}})
async def chat(body: ChatRequest, principal: PrincipalDep, services: ServicesDep) -> ChatResponse:
    """Ask the assistant. Pass ``conversation_id`` from the previous reply to keep context
    (follow-ups like "and last month?", answering "yes" to a confirmation)."""
    conversation_id, conversation = services.conversations.open(body.conversation_id, principal.user_id)
    reply = await services.assistant.respond(
        user_id=principal.user_id,
        message=body.message,
        request_id=request_id_var.get(),
        channel=Channel.TEXT,
        previous=conversation.previous,
        pending=conversation.pending,
    )
    services.conversations.remember(conversation_id, reply)
    await log_turn(
        services.sessions,
        conversation_id=conversation_id,
        user_id=principal.user_id,
        channel=Channel.TEXT,
        user_input=body.message,
        reply=reply,
        response_text=reply.response,
        request_id=reply.request_id,
    )
    return chat_response(conversation_id, reply)
