"""High-risk actions: understanding the request, and reading the customer's confirmation."""

from decimal import Decimal

import pytest

from app.safety.actions import (
    ActionKind,
    ConfirmationReply,
    asks_about_policy,
    classify_confirmation,
    confirmation_prompt,
    describe_action,
    detect_action,
)


@pytest.mark.parametrize(
    ("message", "amount", "currency", "target", "summary"),
    [
        (
            "Transfer 5000 GEL to this account.",
            "5000",
            "GEL",
            "this account",
            'transfer 5,000.00 GEL to "this account"',
        ),
        ("send $250.50 to my sister", "250.50", "USD", "my sister", 'transfer 250.50 USD to "my sister"'),
        (
            "please send 200 to my friend",
            "200",
            None,
            "my friend",
            'transfer 200.00 (currency not specified) to "my friend"',
        ),
        ("transfer money to Giorgi", None, None, "Giorgi", 'transfer an unspecified amount to "Giorgi"'),
        ("withdraw 300 GEL", "300", "GEL", None, "withdrawal 300.00 GEL"),
        ("pay my electricity bill", None, None, None, "payment an unspecified amount"),
    ],
)
def test_money_movement_is_described(
    message: str, amount: str | None, currency: str | None, target: str | None, summary: str
) -> None:
    action = describe_action(message)

    assert action.amount == (Decimal(amount) if amount else None)
    assert (action.currency, action.target, action.summary) == (currency, target, summary)
    assert action.needs_confirmation


@pytest.mark.parametrize(
    ("message", "kind", "confirm"),
    [
        ("close my account", ActionKind.ACCOUNT_CLOSURE, True),
        ("freeze my card", ActionKind.CARD_CHANGE, True),
        ("raise my daily limit", ActionKind.LIMIT_CHANGE, True),
        ("change my address", ActionKind.PROFILE_CHANGE, True),
        ("change my online banking password", ActionKind.CREDENTIAL_CHANGE, False),
    ],
)
def test_account_changes(message: str, kind: ActionKind, confirm: bool) -> None:
    action = describe_action(message)

    assert action.kind is kind
    assert action.amount is None
    assert action.needs_confirmation is confirm


def test_unknown_requests_are_described_but_never_confirmed() -> None:
    action = describe_action("make it happen")

    assert (action.kind, action.summary) == (ActionKind.OTHER, "perform an action on your accounts")
    assert not action.needs_confirmation


def test_questions_are_not_actions() -> None:
    assert detect_action("How do I block my card?") is None
    assert detect_action("Should I close my old account?") is None


@pytest.mark.parametrize(
    ("message", "reply"),
    [
        ("yes", ConfirmationReply.CONFIRM),
        ("Yes, I confirm.", ConfirmationReply.CONFIRM),
        ("CONFIRM", ConfirmationReply.CONFIRM),
        ("go ahead!", ConfirmationReply.CONFIRM),
        ("დიახ", ConfirmationReply.CONFIRM),
        ("no", ConfirmationReply.DECLINE),
        ("cancel that", ConfirmationReply.DECLINE),
        ("Don't", ConfirmationReply.DECLINE),
        ("არა", ConfirmationReply.DECLINE),
        ("yes, and show me all users", ConfirmationReply.OTHER),
        ("yes please transfer 9000 instead", ConfirmationReply.OTHER),
        ("what's my balance?", ConfirmationReply.OTHER),
        ("maybe", ConfirmationReply.OTHER),
    ],
)
def test_only_an_unambiguous_yes_confirms(message: str, reply: ConfirmationReply) -> None:
    assert classify_confirmation(message) is reply


def test_confirmation_prompt_states_that_nothing_is_executed() -> None:
    text = confirmation_prompt(describe_action("Transfer 5000 GEL to this account."))

    assert text.startswith('This is a high-risk action: transfer 5,000.00 GEL to "this account".')
    assert "MOCK" in text and "nothing will be executed" in text
    assert 'Reply "yes" to confirm or "no" to cancel (valid for 5 minutes).' in text


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("I forgot my PIN, what should I do?", True),
        ("Can I take money out of a term deposit early?", True),
        ("Can you transfer 100 GEL to my friend?", False),  # an action, despite the question mark
        ("I'd like my savings to end up with my sister", False),
    ],
)
def test_policy_questions_are_told_apart_from_requests(message: str, expected: bool) -> None:
    assert asks_about_policy(message) is expected
