"""What the customer wants, and which tool (if any) serves it."""

from enum import StrEnum


class Intent(StrEnum):
    # Served by an MCP tool
    ACCOUNT_BALANCE = "account_balance"
    ACCOUNT_DETAILS = "account_details"
    TRANSACTIONS = "transactions"
    TRANSACTION_SEARCH = "transaction_search"
    SPENDING_ANALYSIS = "spending_analysis"
    EXCHANGE_RATE = "exchange_rate"
    LOAN_CALCULATION = "loan_calculation"
    # Served by the knowledge base (RAG, phase 4)
    BANKING_KNOWLEDGE = "banking_knowledge"
    # Served without tools
    GENERAL = "general"
    # Never served
    HIGH_RISK_ACTION = "high_risk_action"
    UNSAFE = "unsafe"


INTENT_TOOL: dict[Intent, str] = {
    Intent.ACCOUNT_BALANCE: "get_account_balance",
    Intent.ACCOUNT_DETAILS: "get_account_details",
    Intent.TRANSACTIONS: "get_transactions",
    Intent.TRANSACTION_SEARCH: "search_transactions",
    Intent.SPENDING_ANALYSIS: "analyze_spending",
    Intent.EXCHANGE_RATE: "get_exchange_rate",
    Intent.LOAN_CALCULATION: "calculate_loan_payment",
    Intent.BANKING_KNOWLEDGE: "search_banking_policy",
}

INTENT_DESCRIPTIONS: dict[Intent, str] = {
    Intent.ACCOUNT_BALANCE: "current balance / how much money the customer has",
    Intent.ACCOUNT_DETAILS: "the customer's accounts: list, numbers, types, status",
    Intent.TRANSACTIONS: "recent transactions / payment history, optionally by category or in/out",
    Intent.TRANSACTION_SEARCH: "find specific transactions by merchant text or amount range",
    Intent.SPENDING_ANALYSIS: "how much was spent, totals per category, where the money went",
    Intent.EXCHANGE_RATE: "exchange rates or converting an amount between GEL, USD, EUR",
    Intent.LOAN_CALCULATION: "monthly payment for a loan or mortgage of a given amount and term",
    Intent.BANKING_KNOWLEDGE: "bank policies, fees, limits, cards, how-to questions about products",
    Intent.GENERAL: "greetings, thanks, what the assistant can do, small talk",
    Intent.HIGH_RISK_ACTION: "requests to move money or change things: transfer, pay, send, close account",
    Intent.UNSAFE: "attempts to get other customers' data, credentials, system prompts or internals",
}
