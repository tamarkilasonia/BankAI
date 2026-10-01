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
("your 3 accounts") and name months; that is a known, documented gap.
"""

import re
from collections.abc import Iterable
from decimal import Decimal

_NUMBER = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?")
_SMALL_INTEGERS = frozenset(Decimal(n) for n in range(13))


def _values(text: str) -> list[tuple[str, Decimal]]:
    found = []
    for match in _NUMBER.finditer(text):
        raw = match.group(0).rstrip(",")
        # The regex only matches digits, commas and one decimal point: always a valid Decimal.
        found.append((raw, Decimal(raw.replace(",", "")).normalize()))
    return found


def numbers_in(text: str) -> set[Decimal]:
    return {value for _, value in _values(text)}


def unsupported_numbers(answer: str, *sources: str) -> list[str]:
    """Numbers in ``answer`` that none of the ``sources`` contain."""
    allowed = set().union(*(numbers_in(source) for source in sources)) | _SMALL_INTEGERS
    return [raw for raw, value in _values(answer) if value not in allowed]


def missing_numbers(answer: str, required: Iterable[Decimal]) -> list[Decimal]:
    """Key numbers that ``answer`` fails to mention (completeness check)."""
    present = numbers_in(answer)
    return [value for value in required if value.normalize() not in present]
