"""The output guard: the last check before an answer reaches the customer.

It looks for four things that must never be in an answer, whoever wrote it:

* ``system_prompt_leak`` - eight or more consecutive words from one of the agent's
  system prompts (minus sentences that are public anyway, like the greeting);
* ``secret_leak``        - anything shaped like a credential (JWTs, API keys,
  private keys, URLs with passwords) or equal to a configured secret value;
* ``foreign_identifier`` - an account ID, account number or customer number that
  is not in this turn's tool result or the customer's own message (identifiers
  can only legitimately come from data the customer is allowed to see);
* ``action_claim``       - a claim that money was moved or an account changed. The
  assistant cannot do either, so such a sentence is always false.

LLM answers that fail are replaced by the deterministic template answer; if
that fails too (or there is none) the customer gets a fixed, safe message.
"""

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date
from functools import cache

from app.agents import messages
from app.agents.prompts import general_messages, knowledge_messages, responder_messages, router_messages
from app.safety import actions
from app.safety.policy import DEFAULT_REFUSAL, REFUSALS

SHINGLE_WORDS = 8
MIN_SECRET_CHARS = 8

_WORD = re.compile(r"[a-z0-9]+")
_SECRET_PATTERNS = {
    "jwt": re.compile(r"\beyJ[\w-]{10,}\.[\w-]{10,}\.[\w-]{5,}"),
    "api_key": re.compile(r"\b(sk|pk|rk)-[A-Za-z0-9_-]{16,}"),
    "private_key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "url_with_password": re.compile(r"\b[a-z][a-z0-9+.-]*://[^\s:/@]+:[^\s@/]+@"),
    "assignment": re.compile(r"\b(password|passwd|secret|api[_-]?key|token)\s*[=:]\s*\S{6,}", re.IGNORECASE),
}
_IDENTIFIERS = {
    "account_id": re.compile(
        r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.IGNORECASE
    ),
    "customer_number": re.compile(r"\bCUST-\d{3,}\b", re.IGNORECASE),
    "account_number": re.compile(r"\bGE\d{2}[A-Z0-9]{10,}\b", re.IGNORECASE),
}
_MONEY_VERBS = r"(transferred|sent|paid|wired|moved|withdrawn|executed|initiated|submitted|scheduled)"
_CHANGE_VERBS = (
    r"(blocked|frozen|unblocked|unfrozen|closed|cancell?ed|changed|updated|activated|deactivated|"
    r"increased|raised|lowered|reset)"
)
_THINGS = r"(card|account|limit|pin|password|deposit|loan|phone number|address|e-?mail)"
_FIRST_PERSON = r"\b(i|we)\s*(have\s+|'ve\s+|just\s+|now\s+|successfully\s+|already\s+)*"
# Present perfect / "now": a fresh change. Plain history ("the payment was made on 1 Sep",
# "this account is frozen") is a legitimate statement about data and is not matched.
_DONE = (
    r"(has been|have been|has now been|have now been|has just been|is now|are now|"
    r"was successfully|were successfully)"
)
_ACTION_CLAIMS = [
    re.compile(rf"{_FIRST_PERSON}{_MONEY_VERBS}\b"),  # "I've sent 200 GEL"
    re.compile(rf"{_FIRST_PERSON}{_CHANGE_VERBS}\s+(your|the|this|that)\s+([\w-]+\s+)?{_THINGS}\b"),
    re.compile(
        rf"\b(money|funds|payment|transfer|amount)\s+{_DONE}\s+({_MONEY_VERBS}|processed|completed|made)\b"
    ),
    re.compile(rf"\b(your|the)\s+([\w-]+\s+)?{_THINGS}\s+{_DONE}\s+{_CHANGE_VERBS}\b"),
    re.compile(
        r"\b(your|the)\s+(transfer|payment)\s+(is|has been)\s+"
        r"(complete|completed|successful|done|on its way)\b"
    ),
]
_NEGATED = re.compile(r"\b(no|not|never|nothing|without|cannot|can't|couldn't|didn't)\s+(\w+\s+)?$")


@dataclass(frozen=True)
class OutputFinding:
    kind: str  # system_prompt_leak | secret_leak | foreign_identifier | action_claim
    detail: str  # what matched - for logs; never shown to the customer


def _words(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def shingles(text: str, size: int = SHINGLE_WORDS) -> set[tuple[str, ...]]:
    words = _words(text)
    return {tuple(words[i : i + size]) for i in range(len(words) - size + 1)}


def public_texts() -> list[str]:
    """Texts customers are meant to see; overlap with them is not a leak."""
    fixed = [value for name, value in vars(messages).items() if name.isupper() and isinstance(value, str)]
    fixed += [*messages.TOOL_ERRORS.values(), *messages.CLARIFICATIONS.values()]
    fixed += [*REFUSALS.values(), DEFAULT_REFUSAL]
    fixed += [value for name, value in vars(actions).items() if name.isupper() and isinstance(value, str)]
    return fixed


@cache
def prompt_shingles() -> frozenset[tuple[str, ...]]:
    """Fingerprints of every system prompt (rendered with placeholder inputs)."""
    prompts = [
        router_messages("", today=date(2000, 1, 1)),
        responder_messages("", [], "en"),
        knowledge_messages("", [], "en"),
        general_messages("", "en"),
    ]
    found: set[tuple[str, ...]] = set()
    for prompt in prompts:
        found |= shingles(prompt[0].content)
    for text in public_texts():
        found -= shingles(text)
    return frozenset(found)


def _claims(text: str) -> list[str]:
    lowered = text.lower()
    found = []
    for pattern in _ACTION_CLAIMS:
        for match in pattern.finditer(lowered):
            if not _NEGATED.search(lowered[max(0, match.start() - 25) : match.start()]):
                found.append(match.group(0))
    return found


class OutputGuard:
    def __init__(self, secrets: Iterable[str] = ()) -> None:
        self._secrets = [s for s in secrets if len(s) >= MIN_SECRET_CHARS]

    def check(self, response: str, *, allowed: Sequence[str] = ()) -> list[OutputFinding]:
        """Problems in ``response``. ``allowed`` = texts whose identifiers the customer may see."""
        findings: list[OutputFinding] = []
        if leaked := shingles(response) & prompt_shingles():
            findings.append(OutputFinding("system_prompt_leak", " ".join(sorted(leaked)[0])))
        for name, pattern in _SECRET_PATTERNS.items():
            if pattern.search(response):
                findings.append(OutputFinding("secret_leak", name))
        if any(secret in response for secret in self._secrets):
            findings.append(OutputFinding("secret_leak", "configured_secret"))
        known = " ".join(allowed).lower()
        for name, pattern in _IDENTIFIERS.items():
            for match in pattern.finditer(response):
                if match.group(0).lower() not in known:
                    findings.append(OutputFinding("foreign_identifier", name))
                    break
        for claim in _claims(response):
            findings.append(OutputFinding("action_claim", claim))
        return findings
