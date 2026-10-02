"""The deterministic rule router: intents and argument extraction.

These tables double as documentation of what the offline agent understands.
"""

from typing import Any

import pytest

from app.agents.intents import Intent
from app.agents.rules import CONFIDENT, GUESS, RouteDecision, RuleRouter, normalize
from tests.constants import ANCHOR  # Monday 2026-09-28, bank time

ROUTER = RuleRouter(lambda: ANCHOR)


def route(message: str, previous: RouteDecision | None = None) -> RouteDecision:
    decision = ROUTER.route(message, previous=previous)
    assert decision is not None, f"no route for {message!r}"
    return decision


# --- the six example flows from the project specification -----------------------------


@pytest.mark.parametrize(
    ("message", "intent", "arguments"),
    [
        ("What's my current balance?", Intent.ACCOUNT_BALANCE, {}),
        (
            "Show me how much I spent on restaurants this month.",
            Intent.SPENDING_ANALYSIS,
            {"period": "this_month", "category": "restaurants"},
        ),
        ("What are the fees for international transfers?", Intent.BANKING_KNOWLEDGE, {}),
        # Examples 4-6 (other customers, prompt injection, transfers) are decided by the
        # safety layer before routing: see tests/unit/test_input_guard.py.
    ],
)
def test_specification_examples(message: str, intent: Intent, arguments: dict[str, Any]) -> None:
    decision = route(message)
    assert decision.intent is intent
    assert decision.arguments == arguments


# --- informational intents ------------------------------------------------------------


@pytest.mark.parametrize(
    ("message", "intent", "arguments"),
    [
        ("how much money do I have", Intent.ACCOUNT_BALANCE, {}),
        ("My balance in USD please", Intent.ACCOUNT_BALANCE, {"currency": "USD"}),
        ("what's the balance on my savings", Intent.ACCOUNT_BALANCE, {"account_type": "savings"}),
        ("list my accounts", Intent.ACCOUNT_DETAILS, {}),
        ("what is my IBAN?", Intent.ACCOUNT_DETAILS, {}),
        ("show my recent transactions", Intent.TRANSACTIONS, {}),
        ("last 5 transactions", Intent.TRANSACTIONS, {"limit": 5}),
        # Regression (found by the evaluation): "last 7 days" is a period, not a list size.
        ("transaction history for the last 7 days", Intent.TRANSACTIONS, {"period": "last_7_days"}),
        (
            "incoming transactions last month",
            Intent.TRANSACTIONS,
            {"period": "last_month", "direction": "credit"},
        ),
        (
            "did I get my salary this month?",
            Intent.TRANSACTIONS,
            {"period": "this_month", "category": "salary"},
        ),
        (
            "find payments at FreshMart over 50",
            Intent.TRANSACTION_SEARCH,
            {"text": "freshmart", "min_amount": "50"},
        ),
        (
            "transactions above 200 GEL last month",
            Intent.TRANSACTION_SEARCH,
            {"period": "last_month", "min_amount": "200"},
        ),
        ('search for "City Taxi"', Intent.TRANSACTION_SEARCH, {"text": "city taxi"}),
        (
            "payments under 10.50 this week",
            Intent.TRANSACTION_SEARCH,
            {"period": "last_7_days", "max_amount": "10.50"},
        ),
        ("where did my money go last month", Intent.SPENDING_ANALYSIS, {"period": "last_month"}),
        (
            "grocery spending in the last 30 days",
            Intent.SPENDING_ANALYSIS,
            {"period": "last_30_days", "category": "groceries"},
        ),
        ("how much did I spend this year", Intent.SPENDING_ANALYSIS, {"period": "this_year"}),
        (
            "convert 250 usd to gel",
            Intent.EXCHANGE_RATE,
            {"base_currency": "USD", "quote_currency": "GEL", "amount": "250"},
        ),
        (
            "how much is 100 euros in lari",
            Intent.EXCHANGE_RATE,
            {"base_currency": "EUR", "quote_currency": "GEL", "amount": "100"},
        ),
        (
            "what's the dollar exchange rate",
            Intent.EXCHANGE_RATE,
            {"base_currency": "USD", "quote_currency": "GEL"},
        ),
        (
            "monthly payment for a 15000 GEL car loan over 3 years",
            Intent.LOAN_CALCULATION,
            {"product_code": "LOAN_AUTO", "amount": "15000", "term_months": 36},
        ),
        (
            "I want to borrow 5k for 18 months",
            Intent.LOAN_CALCULATION,
            {"product_code": "LOAN_CONSUMER", "amount": "5000", "term_months": 18},
        ),
        ("Can I transfer money abroad?", Intent.BANKING_KNOWLEDGE, {}),
        ("what is the interest rate on the auto loan", Intent.BANKING_KNOWLEDGE, {}),
        ("I lost my card", Intent.BANKING_KNOWLEDGE, {}),
        ("hello!", Intent.GENERAL, {}),
        ("thank you", Intent.GENERAL, {}),
        ("what can you do?", Intent.GENERAL, {}),
    ],
)
def test_intents_and_arguments(message: str, intent: Intent, arguments: dict[str, Any]) -> None:
    decision = route(message)
    assert decision.intent is intent
    assert decision.arguments == arguments


def test_month_name_becomes_a_custom_period() -> None:
    assert route("spending in August").arguments == {
        "period": "custom",
        "start_date": "2026-08-01",
        "end_date": "2026-08-31",
    }


def test_future_month_name_means_last_year() -> None:
    args = route("transactions in December").arguments
    assert (args["start_date"], args["end_date"]) == ("2025-12-01", "2025-12-31")


def test_current_month_name_ends_today() -> None:
    assert route("spending in September").arguments["end_date"] == "2026-09-28"


def test_durations_are_not_amount_filters() -> None:
    """Regression: 'over 3 years' was once read as min_amount=3."""
    decision = route("monthly payment for a 15000 GEL car loan over 3 years")
    assert decision.intent is Intent.LOAN_CALCULATION
    assert "min_amount" not in decision.arguments


@pytest.mark.parametrize(
    ("phrase", "months"), [("four years", 48), ("six months", 6), ("twenty-four months", 24)]
)
def test_loan_terms_in_words(phrase: str, months: int) -> None:
    assert route(f"monthly payment for a 9000 GEL loan over {phrase}").arguments["term_months"] == months


def test_loan_without_amount_or_term_reports_what_is_missing() -> None:
    decision = route("calculate my mortgage payment")
    assert decision.intent is Intent.LOAN_CALCULATION
    assert decision.arguments == {"product_code": "MORTGAGE_STANDARD"}
    assert decision.missing == ("amount", "term_months")


def test_exchange_rate_without_currency_reports_missing() -> None:
    decision = route("what's the exchange rate?")
    assert decision.missing == ("currencies",)


# --- confidence & follow-ups -------------------------------------------------------------


def test_single_match_is_confident_overlap_is_a_guess() -> None:
    assert route("what's my balance").confidence == CONFIDENT
    assert route("what is the minimum balance requirement for savings?").confidence == GUESS


def test_follow_up_reuses_the_previous_intent() -> None:
    previous = route("how much did I spend on restaurants this month")

    follow_up = route("and last month?", previous=previous)

    assert follow_up.intent is Intent.SPENDING_ANALYSIS
    assert follow_up.source == "follow_up"
    assert follow_up.arguments == {"period": "last_month", "category": "restaurants"}


def test_follow_up_replaces_custom_dates() -> None:
    previous = route("spending in August")

    follow_up = route("what about this month", previous=previous)

    assert follow_up.arguments == {"period": "this_month"}


def test_follow_up_with_non_spending_category_is_not_reused_for_spending() -> None:
    previous = route("how much did I spend this month")
    assert ROUTER.route("what about salary?", previous=previous) is None


def test_long_unrelated_message_is_not_a_follow_up() -> None:
    previous = route("how much did I spend this month")
    assert (
        ROUTER.route("tell me a long story about dragons and castles from last month", previous=previous)
        is None
    )


@pytest.mark.parametrize("message", ["what is the weather", "tell me a joke", "asdfgh", "გამარჯობა"])
def test_unknown_requests_return_none(message: str) -> None:
    assert ROUTER.route(message) is None


def test_normalize() -> None:
    assert normalize("  What\N{RIGHT SINGLE QUOTATION MARK}s   MY Balance? ") == "what's my balance?"


def test_outgoing_direction() -> None:
    assert route("outgoing transactions last week").arguments == {
        "period": "last_7_days",
        "direction": "debit",
    }


# --- regressions found in the first live session with a real LLM ----------------------------


def test_where_did_my_cash_go_is_spending() -> None:
    decision = route("where did all my cash go last month?")
    assert (decision.intent, decision.arguments) == (Intent.SPENDING_ANALYSIS, {"period": "last_month"})


@pytest.mark.parametrize("message", ["and last month?", "and what about restaurants?"])
def test_details_the_previous_tool_cannot_use_do_not_create_a_follow_up(message: str) -> None:
    """A balance has no period or category: these must not be forced into the balance tool."""
    previous = route("What's my current balance?")
    assert ROUTER.route(message, previous=previous) is None


def test_applicable_detail_is_a_follow_up() -> None:
    previous = route("What's my current balance?")
    decision = route("and in USD?", previous=previous)
    assert (decision.intent, decision.source, decision.arguments) == (
        Intent.ACCOUNT_BALANCE,
        "follow_up",
        {"currency": "USD"},
    )


def test_message_must_read_like_a_follow_up() -> None:
    previous = route("how much did I spend this month")
    assert ROUTER.route("show my euro stuff for the previous month please", previous=previous) is None
    assert route("what about last month", previous=previous).source == "follow_up"
