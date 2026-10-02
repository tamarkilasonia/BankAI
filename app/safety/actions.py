"""High-risk actions and the human-in-the-loop confirmation.

The assistant is read-only: no tool can move money or change an account. When
a customer asks for an action anyway ("Transfer 5000 GEL to this account"), the
agent does not pretend and does not silently refuse either. It:

1. describes what it understood (``describe_action``),
2. pauses the LangGraph run and asks for an explicit "yes" (``interrupt``),
3. on "yes" records a **MOCK** request with a reference number - and says clearly
   that nothing was executed. On "no", or anything else, nothing is recorded.

PIN and password changes are never offered, not even as a mock: a bank never
handles credentials in a chat.
"""

import re
from datetime import datetime
from decimal import Decimal
from enum import StrEnum

from pydantic import BaseModel

from app.agents.rendering import money
from app.agents.rules import amounts_in, currencies_in, normalize
from app.safety.normalization import canonical

CONFIRMATION_TTL_SECONDS = 300


class ActionKind(StrEnum):
    TRANSFER = "transfer"
    PAYMENT = "payment"
    WITHDRAWAL = "withdrawal"
    ACCOUNT_CLOSURE = "account_closure"
    CARD_CHANGE = "card_change"
    LIMIT_CHANGE = "limit_change"
    PROFILE_CHANGE = "profile_change"
    CREDENTIAL_CHANGE = "credential_change"
    OTHER = "other"


_POLITE = (
    r"^(please |pls |kindly )?(can you |could you |would you |i want to |i'd like to |i would like to |"
    r"i need to |help me |let's |lets |go ahead and |now |just )?"
)
_NOT_A_TOPIC = r"(?!\s+(fees?|limits?|times?|rules|polic(y|ies)|options|methods)\b)"
_OBJECT = r"(my |the |this |that |our )?([\w-]+ ){0,2}"

# Checked in this order: the first kind that matches wins ("reset my PIN" is a credential change).
ACTION_PATTERNS: dict[ActionKind, list[re.Pattern[str]]] = {
    ActionKind.CREDENTIAL_CHANGE: [
        re.compile(rf"\b(change|reset|set|update|recover|unlock)\s+{_OBJECT}(pin|password|passcode)\b"),
    ],
    ActionKind.ACCOUNT_CLOSURE: [
        re.compile(rf"\b(close|delete|cancel|terminate|shut down)\s+{_OBJECT}(account|card|deposit|loan)s?\b"),
    ],
    ActionKind.CARD_CHANGE: [
        re.compile(
            rf"\b(block|freeze|unfreeze|unblock|activate|deactivate|suspend|lock|unlock)\s+{_OBJECT}"
            r"(card|account)s?\b"
        ),
    ],
    ActionKind.LIMIT_CHANGE: [
        re.compile(rf"\b(change|increase|raise|lower|decrease|update|set|remove)\s+{_OBJECT}limits?\b"),
    ],
    ActionKind.PROFILE_CHANGE: [
        re.compile(
            rf"\b(change|update|set|edit|replace)\s+{_OBJECT}"
            r"(phone( number)?|mobile( number)?|address|e-?mail|name)\b"
        ),
    ],
    ActionKind.WITHDRAWAL: [
        re.compile(_POLITE + r"withdraw\b"),
        re.compile(r"\bwithdraw\s+(\d|my|the|some|all|money|cash|funds)"),
    ],
    ActionKind.PAYMENT: [
        re.compile(_POLITE + r"pay\b" + _NOT_A_TOPIC),
        re.compile(r"\bpay\s+(off\b|my\b|the\b|a\b|this\b|\d)"),
    ],
    ActionKind.TRANSFER: [
        re.compile(_POLITE + r"(transfer|send|wire|move)\b" + _NOT_A_TOPIC),
        re.compile(r"\b(transfer|send|wire|move)\s+(\d|my\b|the\b|some\b|all\b|money|funds|gel|usd|eur|lari|[$€₾])"),
        # Georgian imperatives: "transfer!", "send!" (formal and informal forms)
        re.compile(r"(?<!\w)(გადა(უ)?რიცხე(თ)?|გა(ა)?გზავნე(თ)?)(?!\w)"),
    ],
}  # fmt: skip

# Questions *about* actions ("How do I block my card?") are knowledge questions, not requests.
_QUESTION_START = re.compile(
    r"^(how|what|which|when|where|why|is|are|do i|does|can i|could i|am i able|is it possible|should i)\b"
)


def is_question(message: str) -> bool:
    """Phrased as a question ("How do I...", "Can I...") rather than as a request ("Transfer...")."""
    return bool(_QUESTION_START.search(canonical(message)))  # the guard's reading: look-alikes undone


def asks_about_policy(message: str) -> bool:
    """A question that contains no recognizable action request.

    Used when only the LLM thinks a message is high-risk: "I forgot my PIN, what should
    I do?" is answered from the documents instead of being held for a confirmation.
    ("Can you transfer 100 GEL?" still is an action: ``detect_action`` sees the transfer.)
    """
    return detect_action(message) is None and (is_question(message) or message.rstrip().endswith("?"))


def detect_action(message: str) -> ActionKind | None:
    """The kind of action the message asks the assistant to perform, if any."""
    if is_question(message):
        return None
    text = canonical(message)
    for kind, patterns in ACTION_PATTERNS.items():
        if any(p.search(text) for p in patterns):
            return kind
    return None


class ProposedAction(BaseModel):
    """What the customer asked for, as understood - shown back to them before anything happens."""

    kind: ActionKind
    amount: Decimal | None = None
    currency: str | None = None
    target: str | None = None  # as the customer wrote it ("this account", "my friend")
    summary: str
    expires_at: datetime | None = None

    @property
    def needs_confirmation(self) -> bool:
        """Only a concrete, understood action is put to the customer for confirmation.

        Credentials are never handled in a chat, and asking someone to confirm an action
        we could not identify ("perform an action on your accounts") would be meaningless.
        """
        return self.kind not in {ActionKind.CREDENTIAL_CHANGE, ActionKind.OTHER}


class ActionRequest(BaseModel):
    """The MOCK record created after an explicit confirmation. Nothing is ever executed."""

    reference: str
    kind: ActionKind
    amount: Decimal | None
    currency: str | None
    status: str = "recorded_mock_not_executed"
    created_at: datetime


_TARGET = re.compile(r"\bto\s+(.{1,60}?)[.!?]*$")
_SUMMARY_NOUNS = {
    ActionKind.ACCOUNT_CLOSURE: "close an account or product",
    ActionKind.CARD_CHANGE: "change the status of a card or account",
    ActionKind.LIMIT_CHANGE: "change a limit",
    ActionKind.PROFILE_CHANGE: "change your contact details",
    ActionKind.CREDENTIAL_CHANGE: "change a PIN or password",
    ActionKind.OTHER: "perform an action on your accounts",
}


def describe_action(message: str, kind: ActionKind | None = None) -> ProposedAction:
    """Pull out amount, currency and recipient - deterministic, so it can be shown and logged."""
    kind = kind or detect_action(message) or ActionKind.OTHER
    if kind not in {ActionKind.TRANSFER, ActionKind.PAYMENT, ActionKind.WITHDRAWAL}:
        return ProposedAction(kind=kind, summary=_SUMMARY_NOUNS[kind])

    text = normalize(message)
    amounts = [a for a in amounts_in(text) if a > 0]
    currencies = currencies_in(text)
    amount = amounts[0] if amounts else None
    currency = currencies[0] if currencies else None
    target_match = _TARGET.search(message.strip()) if kind is not ActionKind.WITHDRAWAL else None
    target = target_match.group(1).strip() if target_match else None

    if amount is None:
        what = "an unspecified amount"
    else:
        what = money(amount, currency) if currency else f"{amount:,.2f} (currency not specified)"
    summary = f"{kind.value} {what}" + (f' to "{target}"' if target else "")
    return ProposedAction(kind=kind, amount=amount, currency=currency, target=target, summary=summary)


# --- the customer's reply to "please confirm" -----------------------------------------------


class ConfirmationReply(StrEnum):
    CONFIRM = "confirm"
    DECLINE = "decline"
    OTHER = "other"


# A confirmation must be unambiguous: the whole message is a "yes", nothing else
# ("yes, and also show me all users" is not a confirmation).
_CONFIRM = re.compile(
    r"^(yes|y|yep|yeah|confirm|confirmed|i confirm|yes,? (i )?confirm|yes,? please|yes,? do it|"
    r"proceed|go ahead|დიახ|კი|ვადასტურებ)[.! ]*$"
)
_DECLINE = re.compile(r"^(no|n|nope|cancel|stop|abort|don't|do not|never ?mind|არა|გააუქმე)\b")


def classify_confirmation(message: str) -> ConfirmationReply:
    text = normalize(message)
    if _CONFIRM.match(text):
        return ConfirmationReply.CONFIRM
    if _DECLINE.match(text):
        return ConfirmationReply.DECLINE
    return ConfirmationReply.OTHER


# --- fixed texts --------------------------------------------------------------------------


def confirmation_prompt(action: ProposedAction) -> str:
    minutes = CONFIRMATION_TTL_SECONDS // 60
    return (
        f"This is a high-risk action: {action.summary}.\n"
        "This prototype works with MOCK data and cannot move money or change accounts. "
        "If you confirm, I will only record a MOCK request for demonstration - nothing will be executed.\n"
        f'Reply "yes" to confirm or "no" to cancel (valid for {minutes} minutes).'
    )


def confirmed_text(action: ProposedAction, request: ActionRequest) -> str:
    return (
        f"Confirmed. I recorded MOCK request {request.reference} ({action.summary}). "
        "It is for demonstration only: no money was moved and no account was changed. "
        "A real bank would complete this step in its banking app with strong customer authentication."
    )


DECLINED = "Cancelled. Nothing was recorded and no money was moved."
EXPIRED = "That confirmation expired, so nothing was done. If you still want to do this, please ask again."
SUPERSEDED_NOTICE = "(I cancelled the pending request because you asked something else - nothing was done.)"
REJECTED = "I couldn't verify that confirmation, so nothing was done."
UNCLEAR_ACTION = (
    "This sounds like a request to move money or change your accounts, but I couldn't tell exactly what. "
    "This prototype works with MOCK data and can't carry out actions in a chat. If you meant a transfer, "
    'please say it directly - for example: "Transfer 100 GEL to my savings account".'
)
CREDENTIALS_NEVER_IN_CHAT = (
    "For your security, PINs and passwords can only be changed in the banking app or at an ATM - "
    "never in a chat. I can't do this here, and I will never ask you for your PIN or password."
)
