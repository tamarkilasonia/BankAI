"""The input guard: apply the safety policy before the agent does anything.

``check_message`` screens a customer message and returns a ``Verdict``:

* ``refuse``  - a policy rule matched (in the text itself or in a decoded view of it);
* ``confirm`` - the message asks for a high-risk action, which needs a human confirmation;
* ``allow``   - nothing matched; the agent may route the message.

``scan_untrusted`` screens content the customer did not type but the LLM will
read - retrieved policy passages and data returned by tools. Only the
``documents=True`` rules apply there (indirect prompt injection).

Every check is deterministic and runs in well under a millisecond, before any
LLM call: an attack that is caught here costs nothing and cannot be "talked
around", and every refusal names the rule that fired.
"""

from dataclasses import dataclass
from enum import StrEnum

from app.safety.actions import ActionKind, detect_action
from app.safety.normalization import views
from app.safety.policy import DOCUMENT_RULES, RULES, Rule, SafetyCategory


class GuardAction(StrEnum):
    ALLOW = "allow"
    REFUSE = "refuse"
    CONFIRM = "confirm"


@dataclass(frozen=True)
class Verdict:
    action: GuardAction
    category: SafetyCategory | None = None
    rule: str | None = None
    view: str = "plain"  # which reading of the text matched (plain, base64, leet, ...)
    action_kind: ActionKind | None = None

    @property
    def flags(self) -> list[str]:
        """Labels recorded in the agent state (and later in the conversation log)."""
        if self.action is GuardAction.REFUSE and self.category is not None:
            return [self.category.value] + ([f"obfuscated:{self.view}"] if self.view != "plain" else [])
        if self.action is GuardAction.CONFIRM:
            return ["high_risk_action"]
        return []


ALLOW = Verdict(GuardAction.ALLOW)


def _first_match(text: str, rules: tuple[Rule, ...]) -> Verdict | None:
    for view in views(text):
        for rule in rules:
            if rule.pattern.search(view.text):
                return Verdict(GuardAction.REFUSE, rule.category, rule.id, view.name)
    return None


def check_message(message: str) -> Verdict:
    """Screen a customer message (refusals win over high-risk confirmations)."""
    if refused := _first_match(message, RULES):
        return refused
    if (kind := detect_action(message)) is not None:
        return Verdict(GuardAction.CONFIRM, action_kind=kind)
    return ALLOW


def scan_untrusted(text: str) -> Verdict:
    """Screen retrieved or tool-provided text for instructions aimed at the assistant."""
    return _first_match(text, DOCUMENT_RULES) or ALLOW
