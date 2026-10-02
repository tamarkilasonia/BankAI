"""Numeric grounding and deterministic rendering of tool results."""

import uuid
from datetime import UTC, date, datetime
from decimal import Decimal

import pytest

from app.agents.grounding import numbers_in, unsupported_numbers, unsupported_terms
from app.agents.rendering import money, render
from app.models.enums import AccountStatus, AccountType, Currency, TransactionCategory, TransactionDirection
from app.services.periods import DateRange, Period
from app.services.schemas import (
    AccountDetails,
    AccountDetailsResult,
    AccountSummary,
    BalanceResult,
    CategorySpending,
    ExchangeRateQuote,
    LoanPaymentQuote,
    MerchantSpending,
    SpendingReport,
    TransactionItem,
    TransactionList,
)

SEPTEMBER = DateRange(period=Period.THIS_MONTH, start_date=date(2026, 9, 1), end_date=date(2026, 9, 28))
NOW = datetime(2026, 9, 28, 18, tzinfo=UTC)


def account(number: str, balance: str, currency: Currency = Currency.GEL, **kw: object) -> AccountSummary:
    values: dict[str, object] = {
        "account_id": uuid.uuid4(),
        "account_number": number,
        "name": f"Checking {currency}",
        "account_type": AccountType.CHECKING,
        "currency": currency,
        "balance": Decimal(balance),
        "status": AccountStatus.ACTIVE,
    }
    return AccountSummary(**(values | kw))  # type: ignore[arg-type]


# --- grounding --------------------------------------------------------------------------


def test_numbers_are_normalized() -> None:
    assert numbers_in("You have 1,234.50 GEL and 3 accounts, 40.0% share") == {
        Decimal("1234.5"),
        Decimal("3"),
        Decimal("40"),
    }


def test_answer_using_only_fact_numbers_is_grounded() -> None:
    facts = "Total spent: 2,275.58 GEL in 61 transactions"
    assert unsupported_numbers("You spent 2,275.58 GEL across 61 payments.", facts) == []


@pytest.mark.parametrize(
    "answer",
    ["You spent about 2,300 GEL.", "You spent 2,275.60 GEL.", "Your fee is 15 GEL.", "The rate is 13.5%."],
)
def test_invented_or_rounded_numbers_are_caught(answer: str) -> None:
    assert unsupported_numbers(answer, "Total spent: 2,275.58 GEL in 61 transactions")


def test_numbers_from_the_question_and_small_counts_are_allowed() -> None:
    answer = "For 24 months you have 2 options."
    assert unsupported_numbers(answer, "no numbers here", "a loan over 24 months") == []


@pytest.mark.parametrize(
    "answer", ["The minimum fee is 10 GEL.", "The fee is 5%.", "It costs $2.", "That is GEL 3.", "5 percent."]
)
def test_small_amounts_and_rates_are_not_counts(answer: str) -> None:
    """Regression (phase 10): "Is the minimum fee 10 GEL?" -> "Yes, it is 10 GEL" passed as a count."""
    assert unsupported_numbers(answer, "The fee is 0.20%, minimum 15.00 GEL.")
    assert unsupported_numbers(answer, f"Facts that do state it: {answer}") == []


# --- rendering ----------------------------------------------------------------------------


def test_money_format() -> None:
    assert money(Decimal("1234.5"), "GEL") == "1,234.50 GEL"


def test_single_balance_is_one_sentence_with_masked_number() -> None:
    result = BalanceResult(
        accounts=[account("GE00MOCK00000100000001", "8687.37")],
        totals_by_currency={Currency.GEL: Decimal("8687.37")},
        as_of=NOW,
    )

    rendered = render(result)

    assert rendered.answer == "Your Checking GEL account (****0001) has a balance of 8,687.37 GEL."
    assert "GE00MOCK00000100000001" not in rendered.answer + " ".join(rendered.facts)


def test_multiple_balances_list_and_never_mix_currencies() -> None:
    result = BalanceResult(
        accounts=[
            account("GE00MOCK00000100000001", "100.00"),
            account("GE00MOCK00000100000003", "50.00", Currency.USD, status=AccountStatus.FROZEN),
        ],
        totals_by_currency={Currency.GEL: Decimal("100.00"), Currency.USD: Decimal("50.00")},
        as_of=NOW,
    )

    answer = render(result).answer

    assert "Total: 100.00 GEL and 50.00 USD." in answer
    assert "(this account is frozen)" in answer


def test_no_accounts() -> None:
    assert "couldn't find" in render(BalanceResult(accounts=[], totals_by_currency={}, as_of=NOW)).answer


def test_details_show_full_account_numbers() -> None:
    details = AccountDetailsResult(
        customer_number="CUST-0001",
        full_name="Nino Beridze",
        accounts=[AccountDetails(**account("GE00MOCK00000100000001", "1").model_dump(), opened_at=NOW)],
    )

    rendered = render(details)

    assert "GE00MOCK00000100000001" in rendered.answer
    assert rendered.answer.startswith("You have 1 account:")


def tx(amount: str, direction: TransactionDirection = TransactionDirection.DEBIT) -> TransactionItem:
    return TransactionItem(
        transaction_id=uuid.uuid4(),
        account_id=uuid.uuid4(),
        booked_at=datetime(2026, 9, 27, 20, 30, tzinfo=UTC),  # 28 Sep in bank time
        direction=direction,
        amount=Decimal(amount),
        currency=Currency.GEL,
        category=TransactionCategory.GROCERIES,
        merchant="FreshMart",
        description="Card payment - FreshMart",
    )


def test_transactions_use_bank_dates_signs_and_truncation_note() -> None:
    result = TransactionList(
        period=SEPTEMBER,
        transactions=[tx("45.20"), tx("1000.00", TransactionDirection.CREDIT)],
        total_matches=37,
        truncated=True,
    )

    answer = render(result).answer

    assert "2026-09-28 | FreshMart | -45.20 GEL | groceries" in answer
    assert "+1,000.00 GEL" in answer
    assert "Showing the 2 most recent of 37" in answer


def test_empty_transaction_list() -> None:
    result = TransactionList(period=SEPTEMBER, transactions=[], total_matches=0, truncated=False)
    assert render(result).answer == "I found no matching transactions between 2026-09-01 and 2026-09-28."


def spending(category_filter: TransactionCategory | None = None, total: str = "537.10") -> SpendingReport:
    return SpendingReport(
        period=SEPTEMBER,
        currency=Currency.GEL,
        category_filter=category_filter,
        total_spent=Decimal(total),
        transaction_count=5 if total != "0" else 0,
        by_category=[]
        if total == "0"
        else [
            CategorySpending(
                category=TransactionCategory.RESTAURANTS,
                total=Decimal(total),
                transaction_count=5,
                share_pct=Decimal("100.0"),
            )
        ],
        top_merchants=[]
        if total == "0"
        else [MerchantSpending(merchant="Sushi Corner", total=Decimal("200.00"), transaction_count=2)],
    )


def test_spending_answers() -> None:
    assert render(spending(TransactionCategory.RESTAURANTS)).answer == (
        "You spent 537.10 GEL on restaurants between 2026-09-01 and 2026-09-28 (5 transactions)."
    )
    overview = render(spending()).answer
    assert overview.startswith("You spent 537.10 GEL between 2026-09-01 and 2026-09-28 in 5 transactions.")
    assert "Top merchant: Sushi Corner (200.00 GEL)." in overview
    assert "I found no spending" in render(spending(total="0")).answer


def test_exchange_rate_is_always_labelled_mock() -> None:
    quote = ExchangeRateQuote(
        base_currency=Currency.USD,
        quote_currency=Currency.GEL,
        rate=Decimal("2.7150"),
        amount=Decimal("250"),
        converted_amount=Decimal("678.75"),
        as_of=date(2026, 9, 1),
        source="MOCK static rate table",
    )

    answer = render(quote).answer

    assert answer.startswith("1 USD = 2.7150 GEL. 250.00 USD = 678.75 GEL.")
    assert "MOCK" in answer


def test_loan_answer_uses_correct_article_and_states_it_is_an_estimate() -> None:
    quote = LoanPaymentQuote(
        product_code="LOAN_AUTO",
        product_name="Auto Loan",
        currency=Currency.GEL,
        principal=Decimal("15000"),
        term_months=36,
        annual_interest_rate_pct=Decimal("11.90"),
        monthly_payment=Decimal("497.50"),
        total_payment=Decimal("17910.00"),
        total_interest=Decimal("2910.00"),
        rate_source="banking_products table (MOCK)",
        note="Not a credit offer.",
    )

    answer = render(quote).answer

    assert answer.startswith("For an Auto Loan of 15,000.00 GEL over 36 months at 11.90% a year")
    assert answer.endswith("Not a credit offer.")


def test_unknown_result_type_is_a_programming_error() -> None:
    with pytest.raises(TypeError, match="no renderer"):
        render(SEPTEMBER)


def test_details_without_accounts() -> None:
    empty = AccountDetailsResult(customer_number="CUST-0001", full_name="Nino Beridze", accounts=[])
    assert render(empty).answer == "I couldn't find any accounts for you."


# --- completeness ------------------------------------------------------------------------------


def test_missing_numbers() -> None:
    from app.agents.grounding import missing_numbers

    required = [Decimal("8388.75"), Decimal("3000.92")]
    assert missing_numbers("Checking: 8,388.75 GEL, savings: 3,000.92 GEL", required) == []
    assert missing_numbers("Your balance is 8,388.75 GEL.", required) == [Decimal("3000.92")]


def test_filtered_spending_facts_carry_no_misleading_share() -> None:
    facts = render(spending(TransactionCategory.RESTAURANTS)).facts
    assert "restaurants: 537.10 GEL (5 transactions)" in facts
    assert not any("% of spending" in fact for fact in facts)
    assert any("% of spending" in fact for fact in render(spending()).facts)


def test_zero_spending_requires_no_number() -> None:
    assert render(spending(total="0")).key_numbers == ()


def test_key_numbers_and_rephrasing_policy() -> None:
    balances = BalanceResult(
        accounts=[account("GE00MOCK00000100000001", "10.00"), account("GE00MOCK00000100000002", "20.00")],
        totals_by_currency={Currency.GEL: Decimal("30.00")},
        as_of=NOW,
    )
    listing = TransactionList(period=SEPTEMBER, transactions=[tx("5.00")], total_matches=1, truncated=False)

    assert render(balances).key_numbers == (Decimal("10.00"), Decimal("20.00"))
    assert render(balances).llm_may_rephrase is True
    assert render(listing).llm_may_rephrase is False
    assert render(spending()).key_numbers == (Decimal("537.10"),)


# --- grounding of words (non-numeric hallucinations) ------------------------------------------

SPENDING_FACTS = [
    "Period: between 2026-09-01 and 2026-09-28",
    "Total spent: 2,275.58 GEL in 61 transactions",
    "groceries: 812.40 GEL (20 transactions, 35.7% of spending)",
    "Top merchant: FreshMart, 300.00 GEL (8 transactions)",
]


def test_answer_using_only_fact_words_is_grounded() -> None:
    answer = "You spent 2,275.58 GEL in September, mostly on groceries. Your top merchant was FreshMart."
    assert unsupported_terms(answer, *SPENDING_FACTS, "how much did I spend this month") == []


@pytest.mark.parametrize(
    ("answer", "problem"),
    [
        ("Most of it went on shopping.", "category:shopping"),
        ("Your top merchant was Carrefour.", "name:Carrefour"),
        ("Carrefour was your top merchant.", "name:Carrefour"),
        ("You spent it in August.", "month:august"),
        ("That is about $800.", "currency:USD"),
        ("Grocery payments are free of charge.", "claim:free"),
        ("There is no monthly fee on this account.", "claim:no_fee"),
        ("Transfers arrive instantly.", "claim:instant"),
        ("Your deposit is insured.", "claim:insured"),
    ],
)
def test_invented_words_are_caught(answer: str, problem: str) -> None:
    assert problem in unsupported_terms(answer, *SPENDING_FACTS)


def test_claims_backed_by_the_documents_are_allowed() -> None:
    passage = (
        "[1] Transfers - Incoming transfers: Incoming transfers are free of charge and arrive instantly."
    )
    assert unsupported_terms("Incoming transfers are free and arrive instantly.", passage) == []


FEE_PASSAGES = [
    "[1] International transfers - Fees for international transfers: The fee is 0.20% of the amount.",
    "[2] Fee schedule - Transfer fees: Transfers between your own accounts and to other BankAI "
    "customers are free. Transfers to other banks in Georgia cost 0.50 GEL.",
    "[3] International transfers - Compliance checks: Transfers may be rejected if information is "
    "missing; in that case the fee is refunded.",
]


def test_a_claim_borrowed_from_another_subject_is_caught() -> None:
    """Regression (phase 10): "free" from the own-accounts sentence backed "international ... are free"."""
    assert unsupported_terms("International transfers are free.", *FEE_PASSAGES) == ["claim:free"]
    assert unsupported_terms("Transfers to other banks are free.", *FEE_PASSAGES) == ["claim:free"]


def test_a_claim_the_documents_make_about_the_same_subject_is_allowed() -> None:
    assert unsupported_terms("Transfers between your own accounts are free.", *FEE_PASSAGES) == []
    # "in that case" -> the passage header says what the sentence is about
    assert (
        unsupported_terms("If an international transfer is rejected, the fee is refunded.", *FEE_PASSAGES)
        == []
    )
    # words the documents never use (paraphrase) don't count against the claim
    assert unsupported_terms("Sending money to other BankAI customers is free.", *FEE_PASSAGES) == []


def test_the_question_cannot_vouch_for_a_claim() -> None:
    answer = "Yes, international transfers are free."
    assert unsupported_terms(answer, *FEE_PASSAGES, question="Are international transfers free?") == [
        "claim:free"
    ]
    assert unsupported_terms("Yes, they are instant.", question="Are transfers instant?") == ["claim:instant"]


def test_words_from_the_question_and_common_words_are_allowed() -> None:
    answer = "Hello! How can I help with your savings in EUR today? May I suggest checking your balance?"
    assert unsupported_terms(answer, question="do you support EUR savings?") == []


def test_sentence_initial_adverbs_are_not_names() -> None:
    """Regression (live session): "Specifically, you spent..." was rejected as an invented name."""
    answer = (
        "Specifically, you spent 2,275.58 GEL. Looking at September, groceries lead.\n"
        "- Interestingly, FreshMart."
    )
    assert unsupported_terms(answer, *SPENDING_FACTS) == []
    assert unsupported_terms("You paid Carrefour, specifically.", *SPENDING_FACTS) == ["name:Carrefour"]
