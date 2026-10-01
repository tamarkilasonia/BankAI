"""Tool results -> FACTS (for the LLM and the grounding check) + a template answer.

Rendering is deterministic code, not an LLM: the template answer is always
correct by construction. It is the answer in offline mode and the fallback
whenever an LLM answer fails the grounding check.

Account numbers are masked (``****0001``) everywhere except in account details,
where the number itself is what the customer asked for.
"""

from dataclasses import dataclass
from decimal import Decimal

from pydantic import BaseModel

from app.agents.messages import KNOWLEDGE_NOT_AVAILABLE
from app.core.time import BANK_TZ
from app.models.enums import AccountStatus, TransactionDirection
from app.rag.schemas import PolicySearchResult
from app.services.periods import DateRange
from app.services.schemas import (
    AccountDetailsResult,
    BalanceResult,
    ExchangeRateQuote,
    LoanPaymentQuote,
    SpendingReport,
    TransactionList,
)

MAX_LISTED_TRANSACTIONS = 10
MAX_LISTED_CATEGORIES = 5


@dataclass(frozen=True)
class Rendered:
    facts: list[str]
    answer: str
    # Numbers an LLM-written answer must contain, or it is replaced by `answer`.
    key_numbers: tuple[Decimal, ...] = ()
    # Lists (transactions, accounts) are shown exactly as rendered: rephrasing a list
    # with an LLM adds nothing and risks dropping rows.
    llm_may_rephrase: bool = True
    # "facts" (data from tools) or "passages" (text retrieved from policy documents).
    prompt_kind: str = "facts"
    # Where the information came from; appended to knowledge answers by code, not by the LLM.
    sources: tuple[str, ...] = ()


def money(amount: Decimal, currency: str) -> str:
    return f"{amount:,.2f} {currency}"


def masked(account_number: str) -> str:
    return f"****{account_number[-4:]}"


def period_text(period: DateRange) -> str:
    if period.start_date == period.end_date:
        return f"on {period.start_date.isoformat()}"
    return f"between {period.start_date.isoformat()} and {period.end_date.isoformat()}"


def _article(noun: str) -> str:
    return "an" if noun[:1].lower() in "aeiou" else "a"


def _frozen_note(status: AccountStatus) -> str:
    return " (this account is frozen)" if status is AccountStatus.FROZEN else ""


def render_balance(result: BalanceResult) -> Rendered:
    facts = [
        f"{a.name} account {masked(a.account_number)}: "
        f"balance {money(a.balance, a.currency)}, status {a.status}"
        for a in result.accounts
    ]
    facts += [
        f"Total balance in {cur}: {money(total, cur)}" for cur, total in result.totals_by_currency.items()
    ]
    key = tuple(a.balance for a in result.accounts)
    if not result.accounts:
        return Rendered(facts, "I couldn't find any of your accounts matching that request.")
    if len(result.accounts) == 1:
        a = result.accounts[0]
        answer = (
            f"Your {a.name} account ({masked(a.account_number)}) has a balance of "
            f"{money(a.balance, a.currency)}{_frozen_note(a.status)}."
        )
        return Rendered(facts, answer, key)
    lines = [
        f"- {a.name} ({masked(a.account_number)}): {money(a.balance, a.currency)}{_frozen_note(a.status)}"
        for a in result.accounts
    ]
    totals = " and ".join(money(total, cur) for cur, total in result.totals_by_currency.items())
    return Rendered(
        facts, "Here are your current balances:\n" + "\n".join(lines) + f"\nTotal: {totals}.", key
    )


def render_details(result: AccountDetailsResult) -> Rendered:
    facts = [f"Customer: {result.full_name} ({result.customer_number})"]
    facts += [
        f"{a.name} account {a.account_number}: {a.account_type}, {a.currency}, status {a.status}, "
        f"opened {a.opened_at.astimezone(BANK_TZ).date().isoformat()}"
        for a in result.accounts
    ]
    if not result.accounts:
        return Rendered(facts, "I couldn't find any accounts for you.", llm_may_rephrase=False)
    lines = [f"- {a.name}: {a.account_number} ({a.account_type}, {a.status})" for a in result.accounts]
    noun = "account" if len(result.accounts) == 1 else "accounts"
    return Rendered(
        facts, f"You have {len(result.accounts)} {noun}:\n" + "\n".join(lines), llm_may_rephrase=False
    )


def render_transactions(result: TransactionList) -> Rendered:
    shown = result.transactions[:MAX_LISTED_TRANSACTIONS]
    when = period_text(result.period)
    facts = [f"Period: {when}", f"Matching transactions: {result.total_matches}"]
    lines = []
    for tx in shown:
        sign = "-" if tx.direction is TransactionDirection.DEBIT else "+"
        day = tx.booked_at.astimezone(BANK_TZ).date().isoformat()
        line = f"{day} | {tx.merchant} | {sign}{money(tx.amount, tx.currency)} | {tx.category}"
        facts.append(line)
        lines.append(f"- {line}")
    if result.total_matches == 0:
        return Rendered(facts, f"I found no matching transactions {when}.", llm_may_rephrase=False)
    more = (
        f"\nShowing the {len(shown)} most recent of {result.total_matches} matching transactions."
        if result.total_matches > len(shown)
        else ""
    )
    if more:
        facts.append(more.strip())
    answer = f"Here are your transactions {when}:\n" + "\n".join(lines) + more
    return Rendered(facts, answer, llm_may_rephrase=False)


def render_spending(result: SpendingReport) -> Rendered:
    when = period_text(result.period)
    total = money(result.total_spent, result.currency)
    facts = [
        f"Period: {when}",
        f"Currency: {result.currency}",
        f"Total spent: {total} in {result.transaction_count} transactions",
    ]
    # A share is only meaningful across categories; filtered to one it is always 100%.
    facts += [
        f"{c.category}: {money(c.total, result.currency)} ({c.transaction_count} transactions"
        + ("" if result.category_filter else f", {c.share_pct}% of spending")
        + ")"
        for c in result.by_category
    ]
    facts += [
        f"Top merchant: {m.merchant}, {money(m.total, result.currency)} ({m.transaction_count} transactions)"
        for m in result.top_merchants
    ]
    label = f"{result.category_filter} " if result.category_filter else ""
    if result.transaction_count == 0:
        # "You haven't spent anything" is a complete answer - no number is required.
        return Rendered(facts, f"I found no {label}spending in {result.currency} {when}.")
    if result.category_filter:
        answer = (
            f"You spent {total} on {result.category_filter} {when} ({result.transaction_count} transactions)."
        )
        return Rendered(facts, answer, (result.total_spent,))
    lines = [
        f"- {c.category}: {money(c.total, result.currency)} ({c.share_pct}%)"
        for c in result.by_category[:MAX_LISTED_CATEGORIES]
    ]
    answer = f"You spent {total} {when} in {result.transaction_count} transactions. Biggest categories:\n"
    answer += "\n".join(lines)
    if result.top_merchants:
        top = result.top_merchants[0]
        answer += f"\nTop merchant: {top.merchant} ({money(top.total, result.currency)})."
    return Rendered(facts, answer, (result.total_spent,))


def render_exchange_rate(result: ExchangeRateQuote) -> Rendered:
    rate = f"1 {result.base_currency} = {result.rate} {result.quote_currency}"
    facts = [f"Exchange rate as of {result.as_of.isoformat()}: {rate}", f"Source: {result.source}"]
    answer = f"{rate}."
    if result.amount is not None and result.converted_amount is not None:
        base_amount = money(result.amount, result.base_currency)
        conversion = f"{base_amount} = {money(result.converted_amount, result.quote_currency)}"
        facts.append(f"Conversion: {conversion}")
        answer += f" {conversion}."
    answer += f" This is a MOCK illustrative rate (as of {result.as_of.isoformat()}), not a real market rate."
    key = (result.rate,) if result.converted_amount is None else (result.converted_amount,)
    return Rendered(facts, answer, key)


def render_loan(result: LoanPaymentQuote) -> Rendered:
    amount = money(result.principal, result.currency)
    monthly = money(result.monthly_payment, result.currency)
    total = money(result.total_payment, result.currency)
    interest = money(result.total_interest, result.currency)
    facts = [
        f"Product: {result.product_name} ({result.product_code})",
        f"Amount: {amount}",
        f"Term: {result.term_months} months",
        f"Annual interest rate: {result.annual_interest_rate_pct}% (source: {result.rate_source})",
        f"Estimated monthly payment: {monthly}",
        f"Total repaid: {total}",
        f"Total interest: {interest}",
        f"Note: {result.note}",
    ]
    answer = (
        f"For {_article(result.product_name)} {result.product_name} of {amount} "
        f"over {result.term_months} months at {result.annual_interest_rate_pct}% a year, "
        f"the estimated monthly payment is {monthly}. "
        f"In total you would repay {total}, of which {interest} is interest. {result.note}"
    )
    return Rendered(facts, answer, (result.monthly_payment, result.annual_interest_rate_pct))


def render_policy(result: PolicySearchResult) -> Rendered:
    if not result.sufficient or not result.passages:
        return Rendered([], KNOWLEDGE_NOT_AVAILABLE, llm_may_rephrase=False)
    facts = [f"[{i}] {p.title} - {p.section}: {p.text}" for i, p in enumerate(result.passages, start=1)]
    sources = tuple(dict.fromkeys(f"{p.title}: {p.section} ({p.source})" for p in result.passages))
    # Offline answer: the best passage verbatim - the documents are written to be quotable.
    return Rendered(facts, result.passages[0].text, prompt_kind="passages", sources=sources)


def render(output: BaseModel) -> Rendered:
    match output:
        case BalanceResult():
            return render_balance(output)
        case AccountDetailsResult():
            return render_details(output)
        case TransactionList():
            return render_transactions(output)
        case SpendingReport():
            return render_spending(output)
        case ExchangeRateQuote():
            return render_exchange_rate(output)
        case LoanPaymentQuote():
            return render_loan(output)
        case PolicySearchResult():
            return render_policy(output)
    raise TypeError(f"no renderer for {type(output).__name__}")
