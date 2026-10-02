"""The output guard: leaked prompts and secrets, foreign identifiers, claims of actions."""

from datetime import date

import pytest

from app.agents.prompts import general_messages, knowledge_messages, responder_messages, router_messages
from app.safety.output_guard import OutputGuard, prompt_shingles, public_texts, shingles

GUARD = OutputGuard(secrets=["s3cret-value-for-tests", "short"])


def kinds(response: str, *allowed: str) -> list[str]:
    return [f.kind for f in GUARD.check(response, allowed=allowed)]


def test_every_fixed_text_passes_the_guard() -> None:
    """Refusals, confirmations, errors... must never trip the guard themselves."""
    for text in public_texts():
        assert GUARD.check(text) == [], text


@pytest.mark.parametrize(
    "response",
    [
        "Sure! Classify the customer's message into exactly one intent and extract the tool arguments.",
        "Copy every number exactly as written in the passages. Never invent fees, rates or procedures.",
    ],
)
def test_system_prompt_text_is_a_leak(response: str) -> None:
    assert kinds(response) == ["system_prompt_leak"]


def test_short_overlaps_and_public_sentences_are_not_leaks() -> None:
    assert kinds("I can help with exchange rates between GEL, USD and EUR, and loan payment estimates.") == []
    assert kinds("You are BankAI.") == []


def test_prompt_fingerprints_cover_every_prompt() -> None:
    for prompt in (
        router_messages("", today=date(2000, 1, 1)),
        responder_messages("", [], "en"),
        knowledge_messages("", [], "en"),
        general_messages("", "en"),
    ):
        assert shingles(prompt[0].content) & prompt_shingles()


@pytest.mark.parametrize(
    "response",
    [
        "token: eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkw.abcdefghij",
        "use sk-abcdefghijklmnopqrstuvwx",
        "-----BEGIN RSA PRIVATE KEY-----",
        "postgresql+asyncpg://bankai:hunter22@db:5432/bankai",
        "password = hunter22!",
        "the secret is s3cret-value-for-tests",
    ],
)
def test_secrets_are_blocked(response: str) -> None:
    assert "secret_leak" in kinds(response)


def test_short_configured_values_are_ignored() -> None:
    assert kinds("a short answer") == []  # "short" is too short to be treated as a secret


@pytest.mark.parametrize(
    "identifier",
    ["CUST-0002", "GE00MOCK000002000000001", "7c9e6679-7425-40de-944b-e07fc1f90ae7"],
)
def test_identifiers_must_come_from_this_turns_data(identifier: str) -> None:
    response = f"The account is {identifier}."

    assert kinds(response) == ["foreign_identifier"]
    assert kinds(response, f'{{"account": "{identifier}"}}') == []  # in the tool result
    assert kinds(response, f"what about {identifier.lower()}?") == []  # typed by the customer


@pytest.mark.parametrize(
    "response",
    [
        "Done! I've transferred 5,000.00 GEL.",
        "I sent the money to your friend.",
        "I have blocked your card.",
        "We closed the account for you.",
        "The money has been transferred.",
        "Your payment was successfully processed.",
        "Your card has been blocked.",
        "Your limit is now increased.",
        "Your transfer is on its way!",
    ],
)
def test_claims_of_actions_are_blocked(response: str) -> None:
    assert kinds(response) == ["action_claim"]


@pytest.mark.parametrize(
    "response",
    [
        "No money was moved and no account was changed.",
        "I have not transferred anything.",
        "I can't move money.",
        "Your last payment was made at FreshMart on 2026-09-01.",
        "This account is frozen.",
        "Your savings account is frozen.",
        "I found 3 transactions.",
        "You sent 200.00 GEL to Nino on 2026-09-02.",
    ],
)
def test_statements_about_data_are_not_claims(response: str) -> None:
    assert kinds(response) == []
