"""RAG + LLM: retrieval -> passages in the prompt -> LLM answer -> grounding -> sources.

The LLM is scripted, so each test controls exactly what the "model" says - including
the ways a real model gets policy answers wrong: invented fees, a claim borrowed from
another passage, agreeing with a leading question. (``tests/llm`` runs the same flow
against a real local model; poisoned passages are in ``tests/safety``.)

Two of these were real holes found while writing this file (phase 10):
"International transfers are free." passed because *another* passage said own-account
transfers are free, and "the minimum fee is 10 GEL" passed because small integers were
treated as counts. Both are fixed in ``app/agents/grounding.py``.
"""

import pytest
from mcp.server.mcpserver import MCPServer

from app.agents import messages
from app.agents.intents import Intent
from app.llm import LLMUnavailableError
from tests.fakes import ScriptedLLM
from tests.integration.test_agent import ask, assistant

FEE_QUESTION = "What are the fees for international transfers?"
FEE_ANSWER = "The fee is 0.20% of the amount, with a minimum of 15.00 GEL and a maximum of 150.00 GEL."
FEE_PASSAGE = "International transfers: Fees for international transfers (international_transfers.md)"


async def test_the_llm_answers_from_the_retrieved_passages_and_code_adds_the_sources(
    tool_server: MCPServer,
) -> None:
    llm = ScriptedLLM(FEE_ANSWER)

    reply = await ask(assistant(tool_server, llm), FEE_QUESTION)

    assert (reply.intent, reply.tool, reply.response_source) == (
        Intent.BANKING_KNOWLEDGE,
        "search_banking_policy",
        "llm",
    )
    system, user = llm.calls[0]["messages"]
    assert "using ONLY the PASSAGES" in system.content and "Never invent fees" in system.content
    assert user.content.startswith("<customer_message>\n" + FEE_QUESTION)  # the question is data
    passages = [line for line in user.content.splitlines() if line.startswith("- [")]
    assert len(passages) == 4 and "0.20% of the amount" in passages[0]
    # The model wrote the sentence; the code wrote the citations - one per passage it was given.
    body, sources = reply.response.split("\n\nSources: ")
    assert body == FEE_ANSWER
    assert sources.startswith(FEE_PASSAGE) and sources.count("(") == 4


async def test_an_invented_fee_is_replaced_by_the_document(tool_server: MCPServer) -> None:
    reply = await ask(assistant(tool_server, ScriptedLLM("The fee is 0.25% of the amount.")), FEE_QUESTION)

    assert reply.response_source == "template" and reply.errors == ["ungrounded_answer"]
    assert "0.25" not in reply.response
    assert "0.20% of the amount" in reply.response  # the passage itself, quoted
    assert reply.response.endswith(f"Sources: {FEE_PASSAGE}")  # the template cites only what it quotes


async def test_a_claim_borrowed_from_another_passage_is_rejected(tool_server: MCPServer) -> None:
    """Passage [2] says own-account transfers are free; that does not make international ones free."""
    reply = await ask(assistant(tool_server, ScriptedLLM("International transfers are free.")), FEE_QUESTION)

    assert reply.response_source == "template" and reply.errors == ["ungrounded_terms"]
    assert "0.20% of the amount" in reply.response


@pytest.mark.parametrize(
    ("question", "agreeable_answer", "error"),
    [
        (
            "Is the international transfer fee 0.1%?",
            "Yes, the international transfer fee is 0.1%.",
            "ungrounded_answer",
        ),
        (
            "Is the minimum fee for international transfers 10 GEL?",
            "Yes, the minimum fee for international transfers is 10 GEL.",
            "ungrounded_answer",
        ),
        ("Are international transfers free?", "Yes, international transfers are free.", "ungrounded_terms"),
        (
            "Do international transfers arrive instantly?",
            "Yes, international transfers arrive instantly.",
            "ungrounded_terms",
        ),
    ],
)
async def test_a_leading_question_cannot_vouch_for_a_fact(
    tool_server: MCPServer, question: str, agreeable_answer: str, error: str
) -> None:
    """Sycophancy: the model agrees with whatever the customer suggests. The question is not a source."""
    reply = await ask(assistant(tool_server, ScriptedLLM(agreeable_answer)), question)

    assert reply.tool == "search_banking_policy"
    assert reply.response_source == "template" and reply.errors == [error]
    assert agreeable_answer not in reply.response


async def test_a_claim_the_documents_make_is_kept(tool_server: MCPServer) -> None:
    """The check is not a ban on claim words: the compliance passage does say the fee is refunded."""
    answer = "If an international transfer is rejected, the fee is refunded."
    reply = await ask(
        assistant(tool_server, ScriptedLLM(answer)),
        "What happens to the fee if an international transfer is rejected?",
    )

    assert reply.response_source == "llm" and reply.errors == []
    assert reply.response.startswith(answer)


async def test_when_the_documents_do_not_cover_it_the_llm_is_never_asked(tool_server: MCPServer) -> None:
    llm = ScriptedLLM("The XYZ service costs 5 GEL a month.")

    reply = await ask(assistant(tool_server, llm), "What is the bank's policy on the fictional service XYZ?")

    assert llm.calls == []
    assert reply.response == messages.KNOWLEDGE_NOT_AVAILABLE


async def test_without_the_llm_the_document_still_answers(tool_server: MCPServer) -> None:
    reply = await ask(assistant(tool_server, ScriptedLLM(LLMUnavailableError("down"))), FEE_QUESTION)

    assert reply.response_source == "template" and reply.errors == ["llm_unavailable"]
    assert "0.20% of the amount" in reply.response and reply.response.endswith(FEE_PASSAGE)
