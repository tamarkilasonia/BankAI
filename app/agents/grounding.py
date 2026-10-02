"""Numeric grounding: an LLM answer may only contain numbers that appear in the facts.

In banking, the most damaging hallucinations are numbers: a wrong balance, an
invented fee or interest rate. So every LLM-written answer is checked: each
number it contains must occur in the facts we gave it (or in the customer's own
question). If one does not, the answer is discarded and the deterministic
template answer is used instead.

A second check covers the opposite failure - leaving things out: each tool
result declares its *key numbers* (totals, the converted amount, the monthly
payment), and an answer that omits one is also replaced by the template.

The first check is deliberately strict - "about 2,300 GEL" for 2,275.58 GEL fails it.
Small integers (0-12) are allowed because answers naturally count things
("your 3 accounts") and name months - but not as amounts or rates: "10 GEL" and
"5%" must be in the facts like any other amount.

Not every hallucination is a number. ``unsupported_terms`` applies the same idea
to words that carry facts in banking answers:

* **claims** - "free", "no fee", "instant", "guaranteed", "insured"... ("international
  transfers are free" invents a fee without a single digit). A claim must be made by one
  fact sentence about the same subject, and never comes from the customer's question;
* **currencies, spending categories and months** - "you spent it on shopping in August";
* **names** - capitalized words such as merchants or products ("your payment at Carrefour").

Each must occur in the facts or the question. The names check skips a list of common
words and only works for scripts with capital letters (not Georgian) - a known gap.
"""

import calendar
import re
from collections.abc import Iterable
from decimal import Decimal

from app.agents.rules import currencies_in, normalize
from app.models.enums import SPENDING_CATEGORIES

_NUMBER = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?")
_SMALL_INTEGERS = frozenset(Decimal(n) for n in range(13))
# A number next to a currency or a percent sign is an amount or a rate, never a count.
_UNIT_AFTER = re.compile(r"\s?(%|per ?cent\b|(gel|usd|eur|lari|dollars?|euros?)\b|[₾$€])", re.IGNORECASE)
_UNIT_BEFORE = re.compile(r"(\b(gel|usd|eur)|[₾$€])\s?$", re.IGNORECASE)


def _values(text: str) -> list[tuple[str, Decimal, bool]]:
    """(as written, value, is an amount or rate) for every number in ``text``."""
    found = []
    for match in _NUMBER.finditer(text):
        raw = match.group(0).rstrip(",")
        amount = bool(_UNIT_AFTER.match(text, match.end()) or _UNIT_BEFORE.search(text, 0, match.start()))
        # The regex only matches digits, commas and one decimal point: always a valid Decimal.
        found.append((raw, Decimal(raw.replace(",", "")).normalize(), amount))
    return found


def numbers_in(text: str) -> set[Decimal]:
    return {value for _, value, _ in _values(text)}


def unsupported_numbers(answer: str, *sources: str) -> list[str]:
    """Numbers in ``answer`` that none of the ``sources`` contain (small counts excepted)."""
    allowed = set().union(*(numbers_in(source) for source in sources))
    return [
        raw
        for raw, value, amount in _values(answer)
        if value not in allowed and (amount or value not in _SMALL_INTEGERS)
    ]


def missing_numbers(answer: str, required: Iterable[Decimal]) -> list[Decimal]:
    """Key numbers that ``answer`` fails to mention (completeness check)."""
    present = numbers_in(answer)
    return [value for value in required if value.normalize() not in present]


# --- words that carry facts ------------------------------------------------------------------

CLAIMS: dict[str, re.Pattern[str]] = {
    "free": re.compile(r"\bfree( of charge)?\b|\bfee-free\b|\bat no (extra )?cost\b"),
    "no_fee": re.compile(
        r"\bno\s+([\w-]+\s+)?(fees?|charges?|commissions?)\b|\bwithout (any )?(fees?|charges?)\b"
    ),
    "waived": re.compile(r"\bwaive[ds]?\b"),
    "guaranteed": re.compile(r"\bguarantee[ds]?\b"),
    "unlimited": re.compile(r"\bunlimited\b|\bno limits?\b"),
    "instant": re.compile(r"\binstant(ly|aneous(ly)?)?\b|\bimmediately\b|\bin real time\b"),
    "tax_free": re.compile(r"\btax[- ]free\b|\btax[- ]exempt\b"),
    "risk_free": re.compile(r"\brisk[- ]free\b|\bno risk\b"),
    "insured": re.compile(r"\binsured\b|\binsurance\b"),
    "interest_free": re.compile(r"\binterest[- ]free\b|\bno interest\b"),
    "refund": re.compile(r"\brefund(s|ed|able)?\b|\breimburse"),
}
_MONTH_NUMBERS = {number: name.lower() for number, name in enumerate(calendar.month_name) if name}
_MONTH_WORDS = [name for name in _MONTH_NUMBERS.values() if name != "may"]  # "may" is usually a verb
_ISO_DATE = re.compile(r"\b\d{4}-(\d{2})-\d{2}(?!\d)")  # also inside timestamps (2026-08-12T10:00)
_WORDS = re.compile(r"[a-z0-9]+")
_CAPITALIZED = re.compile(r"(?<![\w-])[A-Z][A-Za-z0-9&]*(?:'s)?")
_SENTENCE_START = re.compile(r"(^|[.!?:]\s+|\n\s*([-*•]|\d+[.)])?\s*)$")
_ORDINARY_ENDING = re.compile(r"[a-z]{3,}(ly|ing|ed)$")  # adverbs and verb forms, not names
# Capitalized at the start of a sentence or as plain English; they name nothing in the data.
# (A word list reads better as text, hence the str.split.)
COMMON_WORDS = frozenset(
    """
    a about absolutely according additionally after again all also although always an and any
    anything are as at based be because before best both but by bye can certainly could currently
    day dear did do does during each estimated even evening everything feel finally first for from
    glad good goodbye great happy has have hello help here hey hi how however i if in is it its just
    keep kind last lastly let may me moreover morning most my next nice no not note nothing now of
    okay on once only or our overall perfect please pleased question questions regards remember
    second should since so some something sorry still sure thank thanks that the their then there
    these this those to today total unfortunately unless using via was we welcome were what when
    where which while who why will with within without would yes you your yours
    account accounts available balance balances bank banking card cards checking current customer
    date details exchange fee fees information interest loan loans main monthly mortgage payment
    payments period rate rates savings spending summary transaction transactions
    biggest largest highest lowest smallest top latest recent showing
    bankai mock ok
    """.split()  # noqa: SIM905
)


def _months(text: str) -> set[str]:
    """Months named in ``text`` or implied by its ISO dates (2026-09-01 -> september)."""
    lowered = text.lower()
    found = {m for m in _MONTH_WORDS if re.search(rf"\b{m}\b", lowered)}
    found |= {_MONTH_NUMBERS[int(m)] for m in _ISO_DATE.findall(text) if 1 <= int(m) <= 12}
    return found


_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+|\n+")
_HEADER = re.compile(r"^(.{1,100}?):\s")  # "[1] Title - Section: text" or "Total spent: ..."


def _stem(word: str) -> str:
    return word[:-1] if len(word) > 3 and word.endswith("s") else word  # transfers = transfer


def _content_words(text: str) -> set[str]:
    words = _WORDS.findall(text)
    return {_stem(w) for w in words if len(w) > 2 and not w.isdigit() and w not in COMMON_WORDS}


def _fact_sentences(facts: Iterable[str]) -> list[tuple[str, set[str]]]:
    """Each sentence of the facts with its words, plus the words of its fact's header.

    The header ("International transfers - Compliance checks") says what every sentence
    of the passage is about, even one that only says "in that case the fee is refunded".
    """
    found = []
    for fact in facts:
        lowered = fact.lower()
        header = _HEADER.match(lowered)
        context = _content_words(header.group(1)) if header else set()
        found += [(s, _content_words(s) | context) for s in _SENTENCE_BREAK.split(lowered) if s.strip()]
    return found


def unbacked_claims(answer: str, facts: Iterable[str]) -> list[str]:
    """Claims ("free", "instant", "no fee") that no fact sentence makes about the same subject.

    Matching the word alone is not enough: "Transfers to other BankAI customers are free"
    must not back up "International transfers are free". So a claim counts only if one fact
    sentence contains it together with every word of the answer's sentence that the facts
    know about. Words the facts never use (paraphrases) are ignored. When in doubt the
    answer is rejected and the template is used - a safe, if less fluent, outcome.
    """
    sentences = _fact_sentences(facts)
    vocabulary = set().union(*(words for _, words in sentences))
    problems: list[str] = []
    for sentence in _SENTENCE_BREAK.split(answer.lower()):
        for name, pattern in CLAIMS.items():
            if not pattern.search(sentence):
                continue
            subject = _content_words(pattern.sub(" ", sentence)) & vocabulary
            backed = any(pattern.search(text) and subject <= words for text, words in sentences)
            if not backed and f"claim:{name}" not in problems:
                problems.append(f"claim:{name}")
    return problems


def unsupported_terms(answer: str, *facts: str, question: str = "") -> list[str]:
    """Fact-carrying words in ``answer`` that the ``facts`` don't back up.

    The customer's ``question`` may supply currencies, categories, months and names
    ("do you support EUR savings?") but never a claim: "Are transfers instant?" must not
    let the answer say they are.
    """
    source = " ".join([*facts, question])
    lowered_answer, lowered_source = answer.lower(), source.lower()
    problems = unbacked_claims(answer, facts)
    known_currencies = set(currencies_in(normalize(source)))
    problems += [f"currency:{c}" for c in currencies_in(normalize(answer)) if c not in known_currencies]
    problems += [
        f"category:{c.value}"
        for c in sorted(SPENDING_CATEGORIES)
        if re.search(rf"\b{c.value}\b", lowered_answer) and not re.search(rf"\b{c.value}\b", lowered_source)
    ]
    problems += [f"month:{m}" for m in sorted(_months(answer) - _months(source))]

    source_words = set(_WORDS.findall(lowered_source))
    for match in _CAPITALIZED.finditer(answer):
        word = match.group(0).removesuffix("'s").lower()
        if word in COMMON_WORDS or word in source_words or word in _MONTH_NUMBERS.values():
            continue
        if _SENTENCE_START.search(answer[: match.start()]) and _ORDINARY_ENDING.search(word):
            continue  # "Specifically, ...", "Looking at...": capitalized only because a sentence starts
        if f"name:{match.group(0)}" not in problems:
            problems.append(f"name:{match.group(0)}")
    return problems
