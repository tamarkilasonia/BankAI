"""The shared tool contract (input models) - the first line of argument validation."""

from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from app.mcp.schemas import (
    TOOLS,
    AnalyzeSpendingInput,
    CalculateLoanPaymentInput,
    GetExchangeRateInput,
    GetTransactionsInput,
    SearchTransactionsInput,
    describe_validation_error,
)
from app.services.periods import Period

IDENTITY_FIELDS = {"user_id", "customer_id", "customer_number", "principal", "token", "owner"}


def test_registry_has_all_tools() -> None:
    assert set(TOOLS) == {
        "get_account_balance",
        "get_account_details",
        "get_transactions",
        "search_transactions",
        "analyze_spending",
        "get_exchange_rate",
        "calculate_loan_payment",
        "search_banking_policy",
    }


@pytest.mark.parametrize("name", sorted(TOOLS))
def test_no_tool_accepts_an_identity_argument(name: str) -> None:
    """Identity comes from the signed token - never from (LLM-controlled) arguments."""
    assert not set(TOOLS[name].input_model.model_fields) & IDENTITY_FIELDS


@pytest.mark.parametrize("name", sorted(TOOLS))
def test_unknown_arguments_are_rejected(name: str) -> None:
    spec = TOOLS[name]
    valid: dict[str, Any] = {
        "get_exchange_rate": {"base_currency": "USD", "quote_currency": "GEL"},
        "calculate_loan_payment": {"product_code": "LOAN_AUTO", "amount": "5000", "term_months": 24},
        "search_transactions": {"text": "taxi"},
        "search_banking_policy": {"query": "card fees"},
    }.get(name, {})
    spec.input_model.model_validate(valid)  # baseline is valid

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        spec.input_model.model_validate(valid | {"user_id": "00000000-0000-0000-0000-000000000001"})


@pytest.mark.parametrize(
    ("model", "args", "message"),
    [
        (GetTransactionsInput, {"limit": 1000}, "less than or equal to 100"),
        (GetTransactionsInput, {"period": "forever"}, "period"),
        (GetTransactionsInput, {"period": "custom", "start_date": "2026-09-01"}, "requires both"),
        (GetTransactionsInput, {"start_date": "2026-09-01", "end_date": "2026-09-02"}, "only allowed"),
        (
            GetTransactionsInput,
            {"period": "custom", "start_date": "2026-09-02", "end_date": "2026-09-01"},
            "before start_date",
        ),
        (SearchTransactionsInput, {}, "at least one"),
        (SearchTransactionsInput, {"min_amount": "50", "max_amount": "10"}, "greater than max_amount"),
        (SearchTransactionsInput, {"text": "a"}, "at least 2 characters"),
        (SearchTransactionsInput, {"min_amount": "10.555"}, "decimal places"),
        (AnalyzeSpendingInput, {"category": "salary"}, "not a spending category"),
        (AnalyzeSpendingInput, {"currency": "BTC"}, "currency"),
        (GetExchangeRateInput, {"base_currency": "GEL", "quote_currency": "GEL"}, "must differ"),
        (
            GetExchangeRateInput,
            {"base_currency": "USD", "quote_currency": "GEL", "amount": "-5"},
            "greater than 0",
        ),
        (
            CalculateLoanPaymentInput,
            {"product_code": "loan; DROP TABLE", "amount": "1", "term_months": 1},
            "pattern",
        ),
        (
            CalculateLoanPaymentInput,
            {"product_code": "LOAN_AUTO", "amount": "1000", "term_months": 0},
            "greater than",
        ),
    ],
)
def test_invalid_arguments(model: type[BaseModel], args: dict[str, Any], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        model.model_validate(args)


def test_custom_period_is_accepted() -> None:
    args = GetTransactionsInput.model_validate(
        {"period": "custom", "start_date": "2026-07-01", "end_date": "2026-07-31"}
    )
    assert args.period is Period.CUSTOM


def test_validation_summary_names_fields_but_never_echoes_values() -> None:
    secret_looking = "sk-proj-THIS-SHOULD-NOT-BE-ECHOED"
    with pytest.raises(ValidationError) as caught:
        SearchTransactionsInput.model_validate({"text": secret_looking * 10, "limit": 999})

    summary = describe_validation_error(caught.value)
    assert "text:" in summary
    assert "limit:" in summary
    assert secret_looking not in summary
