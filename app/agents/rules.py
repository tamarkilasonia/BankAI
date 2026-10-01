"""Deterministic, rule-based intent routing and argument extraction.

Why rules in an "AI" agent?
    * They are instant, free, and give the same answer every time - which makes
      the agent testable and lets it run with no LLM at all (offline mode).
    * Safety-relevant patterns (prompt injection, other customers' data, money
      movement) must not depend on a model's mood: rules catch them first and
      the LLM cannot override them.
    * The LLM is still used (in hybrid mode) for everything the rules are unsure
      about: paraphrases, typos, other languages.

``RuleRouter.route`` returns ``None`` when it has no idea, and a decision with
``confidence < 0.8`` when several intents matched and it had to guess.

Phase 5 grows the safety patterns here into a dedicated safety layer.
"""

import calendar
import re
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import date, datetime
from decimal import Decimal
from typing import Any

from app.agents.intents import INTENT_TOOL, Intent
from app.core.time import BANK_TZ
from app.mcp.schemas import TOOLS
from app.models.enums import SPENDING_CATEGORIES, TransactionCategory
from app.services.periods import Period

CONFIDENT = 0.9
GUESS = 0.6


@dataclass(frozen=True)
class RouteDecision:
    intent: Intent
    arguments: dict[str, Any] = field(default_factory=dict)  # JSON-style tool arguments
    confidence: float = CONFIDENT
    source: str = "rules"  # rules | llm | follow_up | fallback
    missing: tuple[str, ...] = ()  # required arguments the customer still has to give
    reason: str = ""  # why (e.g. which unsafe pattern)


def _rx(*patterns: str) -> list[re.Pattern[str]]:
    return [re.compile(p) for p in patterns]


# --- safety-relevant patterns (checked first, cannot be overridden) ------------------------

UNSAFE_PATTERNS: dict[str, list[re.Pattern[str]]] = {
    "prompt_injection": _rx(
        r"\b(ignore|disregard|forget|override|bypass)\b.{0,40}\b(instructions?|rules|prompts?|guidelines|restrictions)\b",
    ),
    "system_prompt_extraction": _rx(
        r"\b(system|hidden|initial|secret|internal)\s+(prompt|instructions?|message|rules)\b",
        r"\b(reveal|show|print|repeat|output|tell me|what are)\b.{0,25}\byour\b.{0,15}\b(prompt|instructions|rules)\b",
    ),
    "credentials": _rx(
        r"\b(database|db|admin|server|system|root|your)\b.{0,25}\b(passwords?|credentials?|api[ -]?keys?|secrets?|tokens?)\b",
        r"\b(api[ -]?keys?|access tokens?|secret keys?|connection strings?)\b",
    ),
    "other_customer_data": _rx(
        r"\b(another|other|someone else'?s?|different)\s+(customers?|users?|clients?|persons?|account holders?)\b",
        r"\ball\s+(customers|users|clients)\b",
        r"\bcust-\d{3,}\b",
        r"\b(user|customer|client)\s+[a-z0-9]{1,12}'s\b",
    ),
    "internal_systems": _rx(
        r"\b(dump|export|query|access|list|show)\b.{0,20}\b(the )?(database|db|tables|schema)\b",
        r"\bselect\s+\*|\bdrop\s+table\b|\bunion\s+select\b",
        r"\b(list|show|what are)\b.{0,15}\byour\b.{0,15}\b(tools|functions|endpoints)\b",
    ),
    "impersonation": _rx(
        r"\b(i am|i'm|act as|acting as|pretend to be|pretend you are|you are now)\b.{0,25}"
        r"\b(admin|administrator|developer|system|root|bank employee|manager|support agent)\b",
        r"\b(developer|admin|god|jailbreak|dan)\s+mode\b",
    ),
}

_QUESTION_START = re.compile(
    r"^(how|what|which|when|where|why|is|are|do i|does|can i|could i|am i able|is it possible)\b"
)
HIGH_RISK_PATTERNS = _rx(
    r"^(please |pls |kindly )?(can you |could you |would you |i want to |i'd like to |i would like to |"
    r"help me |let's |lets )?(transfer|send|wire|pay|move|withdraw)\b",
    r"\b(transfer|send|wire|move)\s+(\d|my|the|some|all|money|funds|gel|usd|eur|[$€₾])",
    r"\bpay\s+(my|the|a|this|\d)",
    r"\b(close|delete|cancel|terminate)\s+(my\s+)?(account|card|deposit)\b",
    r"\b(change|reset|increase|raise|update|set)\s+(my\s+)?(pin|card limit|limit|phone number|address)\b",
    r"\b(block|freeze|unfreeze|unblock|activate)\s+(my\s+)?(card|account)\b",
)

# --- informational intents ----------------------------------------------------------------

INTENT_PATTERNS: dict[Intent, list[re.Pattern[str]]] = {
    Intent.ACCOUNT_BALANCE: _rx(
        r"\bbalances?\b",
        r"\bhow much (money |cash )?(do i have|have i got|is (there )?(in|on) my|is left)\b",
        r"\b(available|remaining) (funds|money)\b",
    ),
    Intent.ACCOUNT_DETAILS: _rx(
        r"\b(my|list|show|which|what|all) (bank )?accounts\b",
        r"\baccount (numbers?|details|status|info|information|list)\b",
        r"\biban\b",
        r"\b(frozen|blocked) accounts?\b",
    ),
    Intent.SPENDING_ANALYSIS: _rx(
        r"\b(spent|spend|spending|expenses?|expenditure)\b",
        r"\bhow much did i (pay|use)\b",
        r"\bwhere (did|does|has) (all )?my (money|cash|salary) (go|gone)\b",
        r"\b(breakdown|budget)\b",
    ),
    Intent.TRANSACTIONS: _rx(
        r"\btransactions?\b",
        r"\b(recent|latest|last)\s+(\d+\s+)?(payments|purchases|charges|activity|transfers)\b",
        r"\b(payment|transaction|account) history\b",
        r"\bstatement\b",
        r"\b(incoming|outgoing) (payments|transfers|money)\b",
        r"\bdid i (get|receive)\b.{0,20}\b(salary|paid|money|payment)\b",
        r"\bwhat did i buy\b",
    ),
    Intent.EXCHANGE_RATE: _rx(
        r"\b(exchange|conversion|fx|currency) rates?\b",
        r"\bconvert\b",
        r"\bhow much is\b.{0,25}\b(usd|eur|gel|dollars?|euros?|lari)\b",
        r"\b(usd|eur|gel|dollars?|euros?|lari)\b.{0,15}\b(to|in|into)\b.{0,10}\b(gel|lari|usd|eur|dollars?|euros?)\b",
        r"\brates? (for|of)\b.{0,10}\b(usd|eur|dollars?|euros?)\b",
    ),
    Intent.LOAN_CALCULATION: _rx(
        r"\b(loan|mortgage|borrow|borrowing)\b.{0,60}"
        r"\b(monthly|per month|a month|installments?|payments?|repay|repayment|calculate|cost|afford)\b",
        r"\b(monthly|installments?|repay|repayment|calculate|afford)\b.{0,60}\b(loan|mortgage|borrow)\b",
        r"\bborrow\s+\d",
    ),
    Intent.BANKING_KNOWLEDGE: _rx(
        r"\b(fees?|commissions?|limits?|policy|policies|terms|conditions|requirements?|eligib\w*|documents?)\b",
        r"\binterest rates?\b",
        r"\bhow (do|can|to|should) (i|we|you)\b",
        r"\b(debit|credit)? ?cards?\b",
        r"\b(lost|stolen)\b",
        r"\b(international|abroad|swift|sepa)\b",
        r"\b(send|transfer|wire)(ing)? money\b",
        r"\b(security|phishing|fraud|scam)\b",
    ),
}

# When several informational intents match, the most specific one wins.
PRECEDENCE: tuple[Intent, ...] = (
    Intent.LOAN_CALCULATION,
    Intent.EXCHANGE_RATE,
    Intent.TRANSACTION_SEARCH,
    Intent.SPENDING_ANALYSIS,
    Intent.ACCOUNT_BALANCE,
    Intent.TRANSACTIONS,
    Intent.ACCOUNT_DETAILS,
    Intent.BANKING_KNOWLEDGE,
)

# Overlaps where the precedence winner is reliably right ("find transactions over 100").
# Any other overlap is only a guess, which the LLM resolves in hybrid mode.
COMPATIBLE_MATCHES: tuple[set[Intent], ...] = (
    {Intent.TRANSACTION_SEARCH, Intent.TRANSACTIONS},
    {Intent.SPENDING_ANALYSIS, Intent.TRANSACTIONS},
)

GENERAL_PATTERNS = _rx(
    r"^(hi|hello|hey|hiya|greetings|good (morning|afternoon|evening))\b",
    r"\b(thanks|thank you|thx|cheers)\b",
    r"\b(bye|goodbye|see you)\b",
    r"\b(who are you|what are you|what can you do|what do you do|how can you help)\b",
    r"^help\W*$",
)

# --- argument vocabularies --------------------------------------------------------------

PERIOD_PATTERNS: list[tuple[re.Pattern[str], Period]] = [
    (re.compile(r"\b(today|so far today)\b"), Period.TODAY),
    (re.compile(r"\b(this|current) month\b|\bmonth to date\b"), Period.THIS_MONTH),
    (re.compile(r"\b(last|previous|past) month\b"), Period.LAST_MONTH),
    (re.compile(r"\b(last|past) (7 days|seven days|week)\b|\bthis week\b"), Period.LAST_7_DAYS),
    (re.compile(r"\b(last|past) (30|thirty) days\b"), Period.LAST_30_DAYS),
    (
        re.compile(
            r"\b(last|past) (90|ninety) days\b|\b(last|past) (3|three) months\b|\b(last|past) quarter\b"
        ),
        Period.LAST_90_DAYS,
    ),
    (re.compile(r"\bthis year\b|\byear to date\b"), Period.THIS_YEAR),
]
MONTHS = {name.lower(): number for number, name in enumerate(calendar.month_name) if name}

CATEGORY_WORDS: dict[TransactionCategory, re.Pattern[str]] = {
    TransactionCategory.GROCERIES: re.compile(r"\b(grocer(y|ies)|supermarkets?|food shopping)\b"),
    TransactionCategory.RESTAURANTS: re.compile(r"\b(restaurants?|dining|eating out|cafes?|takeaway)\b"),
    TransactionCategory.TRANSPORTATION: re.compile(r"\b(transport(ation)?|taxis?|fuel|petrol|metro|bus)\b"),
    TransactionCategory.SHOPPING: re.compile(r"\b(shopping|clothes|clothing|electronics)\b"),
    TransactionCategory.UTILITIES: re.compile(
        r"\b(utilit(y|ies)|bills?|electricity|water bill|gas bill|internet)\b"
    ),
    TransactionCategory.SUBSCRIPTIONS: re.compile(r"\b(subscriptions?|streaming)\b"),
    TransactionCategory.HEALTHCARE: re.compile(r"\b(health(care)?|pharmacy|medical|doctor)\b"),
    TransactionCategory.ENTERTAINMENT: re.compile(r"\b(entertainment|cinema|movies?|concerts?|games?)\b"),
    TransactionCategory.SALARY: re.compile(r"\b(salary|salaries|paycheck|wages?)\b"),
    TransactionCategory.TRANSFER: re.compile(r"\btransfers\b"),
    TransactionCategory.DEPOSIT: re.compile(r"\bdeposits\b"),
}

CURRENCY_WORDS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b(usd|dollars?|us\$)|\$"), "USD"),
    (re.compile(r"\b(eur|euros?)\b|€"), "EUR"),
    (re.compile(r"\b(gel|lari|laris)\b|₾"), "GEL"),
]

_NUMBER = r"(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d{1,2}))?"
_AMOUNT = re.compile(rf"(?<![\w.]){_NUMBER}\s*(k\b)?")
_NOT_DURATION = r"(?!\s*(years?|yrs?|months?|mos?|weeks?|days?)\b)"
_MIN_AMOUNT = re.compile(
    rf"\b(over|above|more than|greater than|exceeding|bigger than|larger than|at least)\s+[$€₾]?{_NUMBER}{_NOT_DURATION}"
)
_MAX_AMOUNT = re.compile(
    rf"\b(under|below|less than|smaller than|at most|up to)\s+[$€₾]?{_NUMBER}{_NOT_DURATION}"
)
NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
    "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "fifteen": 15, "eighteen": 18,
    "twenty": 20, "twenty-four": 24, "thirty": 30, "thirty-six": 36,
}  # fmt: skip
_COUNT = r"(\d{1,3}|" + "|".join(sorted(NUMBER_WORDS, key=len, reverse=True)) + ")"
_TERM_MONTHS = re.compile(rf"\b{_COUNT}\s*(months?|mos?)\b")
_TERM_YEARS = re.compile(rf"\b{_COUNT}\s*(years?|yrs?)\b")
_LIMIT = re.compile(r"\b(last|latest|recent|top)\s+(\d{1,3})\b")
_MERCHANT = re.compile(
    r"\b(?:at|from|paid to|payments to)\s+([a-z0-9][\w'&.-]*(?:\s+[a-z0-9][\w'&.-]*){0,3})"
)
_QUOTED = re.compile(r"[\"“']([^\"”']{2,80})[\"”']")
_MERCHANT_STOPWORDS = {
    "this", "last", "in", "on", "during", "for", "over", "above", "under", "below", "since", "the",
    "month", "week", "year", "today", "my", "me", "and", "or", "of", "with", "past", "previous",
}  # fmt: skip
_FOLLOW_UP_START = re.compile(r"^(and|what about|how about|same for|same but|now for|then for|ok and)\b")
_INCOMING = re.compile(r"\b(incoming|received|credits?|income|money in)\b")
_OUTGOING = re.compile(r"\b(outgoing|debits?|money out|payments made)\b")


def normalize(text: str) -> str:
    text = (
        text.replace("\N{RIGHT SINGLE QUOTATION MARK}", "'")
        .replace("\N{LEFT SINGLE QUOTATION MARK}", "'")
        .lower()
    )
    return re.sub(r"\s+", " ", text).strip()


def _count(token: str) -> int:
    return int(token) if token.isdigit() else NUMBER_WORDS[token]


def _decimal(whole: str, fraction: str | None, thousands: str | None = None) -> Decimal:
    """Build a Decimal from regex groups (digits only, so always valid); '5k' -> 5000."""
    value = Decimal(whole.replace(",", "") + (f".{fraction}" if fraction else ""))
    return value * 1000 if thousands else value


class RuleRouter:
    def __init__(self, clock: Callable[[], datetime]) -> None:
        self._clock = clock

    # --- public -----------------------------------------------------------------------

    def route(self, message: str, *, previous: RouteDecision | None = None) -> RouteDecision | None:
        text = normalize(message)

        for reason, patterns in UNSAFE_PATTERNS.items():
            if any(p.search(text) for p in patterns):
                return RouteDecision(Intent.UNSAFE, confidence=0.95, reason=reason)
        if not _QUESTION_START.search(text) and any(p.search(text) for p in HIGH_RISK_PATTERNS):
            return RouteDecision(
                Intent.HIGH_RISK_ACTION, confidence=0.95, reason="money_movement_or_account_change"
            )

        matched = {
            intent for intent, patterns in INTENT_PATTERNS.items() if any(p.search(text) for p in patterns)
        }
        if self._search_filters(text):
            matched.add(Intent.TRANSACTION_SEARCH)

        if matched:
            intent = next(i for i in PRECEDENCE if i in matched)
            confident = len(matched) == 1 or matched in COMPATIBLE_MATCHES
            return self._with_arguments(intent, text, CONFIDENT if confident else GUESS)

        if previous is not None and previous.intent in INTENT_TOOL:
            follow_up = self._follow_up(text, previous)
            if follow_up is not None:
                return follow_up

        if any(p.search(text) for p in GENERAL_PATTERNS):
            return RouteDecision(Intent.GENERAL)
        return None

    # --- arguments ----------------------------------------------------------------------

    def _with_arguments(self, intent: Intent, text: str, confidence: float) -> RouteDecision:
        args: dict[str, Any] = {}
        missing: list[str] = []
        if intent is Intent.ACCOUNT_BALANCE:
            args |= self._currency_filter(text)
            if re.search(r"\bsavings?\b", text):
                args["account_type"] = "savings"
            elif re.search(r"\b(checking|current) account\b", text):
                args["account_type"] = "checking"
        elif intent is Intent.TRANSACTIONS:
            args |= self._period(text) | self._category(text, spending_only=False)
            if _INCOMING.search(text):
                args["direction"] = "credit"
            elif _OUTGOING.search(text):
                args["direction"] = "debit"
            if limit := _LIMIT.search(text):
                args["limit"] = min(int(limit.group(2)), 100)
        elif intent is Intent.TRANSACTION_SEARCH:
            filters = self._search_filters(text)
            # Words inside a merchant name ("City Taxi") must not also set a category.
            rest = text.replace(filters["text"], " ") if "text" in filters else text
            args |= self._period(text) | self._category(rest, spending_only=False) | filters
        elif intent is Intent.SPENDING_ANALYSIS:
            args |= (
                self._period(text) | self._category(text, spending_only=True) | self._currency_filter(text)
            )
        elif intent is Intent.EXCHANGE_RATE:
            args, missing = self._exchange(text)
        elif intent is Intent.LOAN_CALCULATION:
            args, missing = self._loan(text)
        return RouteDecision(intent, args, confidence, missing=tuple(missing))

    def _follow_up(self, text: str, previous: RouteDecision) -> RouteDecision | None:
        """'And last month?' / 'What about groceries?' - reuse the previous intent.

        Only when the message *reads* like a follow-up (starts with "and", "what about"...
        or is very short) and every detail it adds is something the previous tool accepts.
        Otherwise the message goes to the LLM (or the fallback) instead of being forced
        into the previous intent.
        """
        if not (_FOLLOW_UP_START.search(text) or len(text.split()) <= 4):
            return None
        modifiers = (
            self._period(text) | self._category(text, spending_only=False) | self._currency_filter(text)
        )
        accepted = TOOLS[INTENT_TOOL[previous.intent]].input_model.model_fields
        if not modifiers or any(key not in accepted for key in modifiers):
            return None
        if (
            "category" in modifiers
            and previous.intent is Intent.SPENDING_ANALYSIS
            and TransactionCategory(modifiers["category"]) not in SPENDING_CATEGORIES
        ):
            return None
        arguments = {k: v for k, v in previous.arguments.items() if k not in {"start_date", "end_date"}}
        return replace(
            previous, arguments=arguments | modifiers, confidence=0.8, source="follow_up", missing=()
        )

    def _period(self, text: str) -> dict[str, Any]:
        for pattern, period in PERIOD_PATTERNS:
            if pattern.search(text):
                return {"period": period.value}
        month = re.search(r"\b(" + "|".join(MONTHS) + r")\b", text)
        if month and not re.search(r"\bmay i\b", text):
            return self._month_range(MONTHS[month.group(1)])
        return {}

    def _month_range(self, month: int) -> dict[str, Any]:
        """'in August' = the most recent August that has started (custom period)."""
        today = self._clock().astimezone(BANK_TZ).date()
        year = today.year if month <= today.month else today.year - 1
        last_day = date(year, month, calendar.monthrange(year, month)[1])
        return {
            "period": Period.CUSTOM.value,
            "start_date": date(year, month, 1).isoformat(),
            "end_date": min(last_day, today).isoformat(),
        }

    @staticmethod
    def _category(text: str, *, spending_only: bool) -> dict[str, Any]:
        found = [c for c, pattern in CATEGORY_WORDS.items() if pattern.search(text)]
        if spending_only:
            found = [c for c in found if c in SPENDING_CATEGORIES]
        return {"category": found[0].value} if len(found) == 1 else {}

    @staticmethod
    def _currencies(text: str) -> list[str]:
        hits = sorted((m.start(), code) for pattern, code in CURRENCY_WORDS for m in pattern.finditer(text))
        ordered: list[str] = []
        for _, code in hits:
            if code not in ordered:
                ordered.append(code)
        return ordered

    def _currency_filter(self, text: str) -> dict[str, Any]:
        currencies = self._currencies(text)
        return {"currency": currencies[0]} if len(currencies) == 1 else {}

    @staticmethod
    def _search_filters(text: str) -> dict[str, Any]:
        args: dict[str, Any] = {}
        if found := _MIN_AMOUNT.search(text):
            args["min_amount"] = str(_decimal(found.group(2), found.group(3)))
        if found := _MAX_AMOUNT.search(text):
            args["max_amount"] = str(_decimal(found.group(2), found.group(3)))
        if quoted := _QUOTED.search(text):
            args["text"] = quoted.group(1).strip()
        elif merchant := _MERCHANT.search(text):
            words: list[str] = []
            for raw in merchant.group(1).split():
                word = raw.strip(".,?!")
                if word in _MERCHANT_STOPWORDS or word.replace(".", "").isdigit():
                    break
                words.append(word)
            candidate = " ".join(words)
            if len(candidate) >= 2 and candidate not in MONTHS:
                args["text"] = candidate
        return args

    def _exchange(self, text: str) -> tuple[dict[str, Any], list[str]]:
        currencies = self._currencies(text)
        args: dict[str, Any] = {}
        if len(currencies) >= 2:
            args = {"base_currency": currencies[0], "quote_currency": currencies[1]}
        elif len(currencies) == 1 and currencies[0] != "GEL":
            args = {"base_currency": currencies[0], "quote_currency": "GEL"}
        if amount := _AMOUNT.search(text):
            value = _decimal(amount.group(1), amount.group(2), amount.group(3))
            if value > 0:
                args["amount"] = str(value)
        missing = [] if "base_currency" in args else ["currencies"]
        return args, missing

    def _loan(self, text: str) -> tuple[dict[str, Any], list[str]]:
        if re.search(r"\b(car|auto|vehicle)\b", text):
            product = "LOAN_AUTO"
        elif re.search(r"\b(mortgage|home|house|apartment|flat|property)\b", text):
            product = "MORTGAGE_STANDARD"
        else:
            product = "LOAN_CONSUMER"
        args: dict[str, Any] = {"product_code": product}

        term_text = text
        if months := _TERM_MONTHS.search(text):
            args["term_months"] = _count(months.group(1))
            term_text = text.replace(months.group(0), " ")
        elif years := _TERM_YEARS.search(text):
            args["term_months"] = _count(years.group(1)) * 12
            term_text = text.replace(years.group(0), " ")

        amounts = [_decimal(m.group(1), m.group(2), m.group(3)) for m in _AMOUNT.finditer(term_text)]
        candidates = [a for a in amounts if a >= 100]
        if candidates:
            args["amount"] = str(max(candidates))
        missing = [name for name in ("amount", "term_months") if name not in args]
        return args, missing
