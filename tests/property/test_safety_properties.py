"""Metamorphic tests of the input guard, generated with Hypothesis.

The injection suite checks 83 fixed messages. These tests check *relations* instead:

* an attack the guard refuses must still be refused, for the same reason, after any
  disguise the normalizer claims to undo (case, spacing, invisible characters,
  look-alike letters, full-width letters, leetspeak, a spaced-out word, base64, hex,
  URL encoding, ROT13, reversed text) - and after combinations of them;
* a benign question must stay allowed when written in capitals, with odd spacing or
  invisible characters (a false refusal is a failure too).

Hypothesis generates the disguises and, when one gets through, shrinks it to the
smallest failing example. The look-alike table below is written independently of
``normalization.py`` on purpose: it is the attacker's view, not the implementation's.
"""

import base64
import codecs
import json
import re
from urllib.parse import quote

from hypothesis import given
from hypothesis import strategies as st

from app.safety.input_guard import GuardAction, check_message
from app.safety.normalization import canonical, views
from app.safety.policy import RULES, SafetyCategory
from tests.conftest import PROJECT_ROOT

_ROWS = [
    json.loads(line)
    for line in (PROJECT_ROOT / "app/safety/data/prompt_injection.jsonl")
    .read_text(encoding="utf-8")
    .splitlines()
]
# Attacks the guard refuses as plain text (the "encoded" ones are already disguised).
ATTACKS = [r["prompt"] for r in _ROWS if r["expect"] == "refuse" and r["category"] != "encoded"]
BENIGN = [r["prompt"] for r in _ROWS if r["expect"] == "allow"]

INVISIBLE = ["\u200b", "\u200c", "\u200d", "\u2060", "\ufeff", "\u00ad"]  # zero-width, soft hyphen
SPACES = [" ", "  ", "\t", "\n", " \u00a0", "\u3000"]  # incl. no-break and ideographic space
LOOKALIKES = {  # Cyrillic and Greek letters that look like Latin ones
    "a": "аα", "c": "с", "e": "еε", "i": "іι", "j": "ј", "o": "оο", "p": "рρ", "s": "ѕ", "x": "х",
    "y": "у", "h": "һ", "k": "кκ", "d": "ԁ", "v": "ν", "t": "τ",
    "A": "АΑ", "B": "ВΒ", "C": "С", "E": "ЕΕ", "H": "НΗ", "I": "ІΙ", "J": "Ј", "K": "КΚ", "M": "МΜ",
    "N": "Ν", "O": "ОΟ", "P": "РΡ", "S": "Ѕ", "T": "ТΤ", "X": "ХΧ", "Y": "Υ", "Z": "Ζ",
}  # fmt: skip
LEET = {"o": "0", "i": "1", "e": "3", "a": "4", "s": "5", "t": "7"}


# --- disguises -----------------------------------------------------------------------------------


@st.composite
def surface(draw: st.DrawFn, text: str) -> str:
    """Disguises that change characters but not letters: case, spacing, invisible, look-alike, full-width."""
    out = []
    for ch in text:
        if ch.isalpha() and draw(st.booleans()):
            ch = ch.swapcase()
        if ch in LOOKALIKES and draw(st.integers(0, 3)) == 0:
            ch = draw(st.sampled_from(LOOKALIKES[ch]))
        elif ch.isascii() and ch.isalpha() and draw(st.integers(0, 5)) == 0:
            ch = chr(ord(ch) + 0xFEE0)  # full-width form: A -> U+FF21
        if ch == " ":
            ch = draw(st.sampled_from(SPACES))
        out.append(ch)
        if draw(st.integers(0, 6)) == 0:
            out.append(draw(st.sampled_from(INVISIBLE)))
    return "".join(out)


@st.composite
def leetspeak(draw: st.DrawFn, text: str) -> str:
    """1gn0r3: digits for letters, keeping at least one letter in each word (as people write it)."""
    words = []
    for word in text.split(" "):
        letters = [i for i, ch in enumerate(word) if ch.lower() in LEET]
        chosen = draw(st.sets(st.sampled_from(letters))) if letters else set()
        if sum(ch.isalpha() for ch in word) - len(chosen) < 1:
            chosen = set()
        words.append("".join(LEET[ch.lower()] if i in chosen else ch for i, ch in enumerate(word)))
    return " ".join(words)


@st.composite
def spaced_word(draw: st.DrawFn, text: str) -> str:
    """i g n o r e: the letters of one long word spread out."""
    words = text.split(" ")
    long_words = [i for i, w in enumerate(words) if len(w) >= 4 and w.isalpha()]
    if not long_words:
        return text
    i = draw(st.sampled_from(long_words))
    separator = draw(st.sampled_from([" ", ".", "-", "_", "*"]))
    words[i] = separator.join(words[i])
    return " ".join(words)


ENCODINGS = {
    "base64": lambda t: base64.b64encode(t.encode()).decode(),
    "base64url": lambda t: base64.urlsafe_b64encode(t.encode()).decode().rstrip("="),
    "hex": lambda t: t.encode().hex(),
    "hex pairs": lambda t: " ".join(f"{b:02x}" for b in t.encode()),
    "url": lambda t: quote(t, safe=""),
}
TRANSFORMS = {"rot13": lambda t: codecs.encode(t, "rot13"), "reversed": lambda t: t[::-1]}


@st.composite
def disguised_attack(draw: st.DrawFn) -> tuple[str, str]:
    attack = draw(st.sampled_from(ATTACKS))
    how = draw(st.sampled_from(["surface", "leet", "spaced", *ENCODINGS, *TRANSFORMS]))
    if how in ENCODINGS:  # character disguises go inside: the decoded text is normalized again
        return attack, ENCODINGS[how](draw(surface(attack)))
    if how in TRANSFORMS:  # ... or outside: the view is computed from the normalized text
        return attack, draw(surface(TRANSFORMS[how](attack)))
    if how == "leet":
        return attack, draw(surface(draw(leetspeak(attack))))
    if how == "spaced":
        return attack, draw(spaced_word(attack))
    return attack, draw(surface(attack))


# --- properties ------------------------------------------------------------------------------------


def reasons(text: str) -> set[SafetyCategory]:
    """Every category the text violates (an attack can break several rules at once)."""
    return {rule.category for view in views(text) for rule in RULES if rule.pattern.search(view.text)}


def test_the_seeds_are_attacks_the_guard_refuses() -> None:
    assert len(ATTACKS) >= 40 and len(BENIGN) >= 20
    assert all(check_message(a).action is GuardAction.REFUSE for a in ATTACKS)
    assert all(check_message(b).action is GuardAction.ALLOW for b in BENIGN)


@given(disguised_attack())
def test_a_disguised_attack_is_refused_for_the_same_reason(case: tuple[str, str]) -> None:
    attack, disguised = case
    verdict = check_message(disguised)
    assert verdict.action is GuardAction.REFUSE, f"got through: {disguised!r}"
    # The disguise may hide one violation while another still shows ("rul3s" but "CUST-0004"),
    # so the reason may differ - but it must be one of the attack's own violations.
    assert verdict.category in reasons(attack)


@given(st.sampled_from(BENIGN).flatmap(lambda b: st.tuples(st.just(b), surface(b))))
def test_a_benign_question_in_odd_typography_is_not_refused(case: tuple[str, str]) -> None:
    benign, typed = case
    assert check_message(typed).action is GuardAction.ALLOW, f"false refusal: {typed!r} (from {benign!r})"


@given(st.text(max_size=200))
def test_normalizing_twice_changes_nothing(text: str) -> None:
    """Idempotence: if a second pass could still change the text, the first one left a disguise in."""
    once = canonical(text)
    assert canonical(once) == once


@given(st.text(max_size=300))
def test_the_guard_never_crashes(text: str) -> None:
    """Any string at all - control characters, emoji, unpaired surrogates - gets a verdict."""
    assert check_message(text).action in set(GuardAction)


def test_the_fixed_lookalike_table_matches_real_unicode_names() -> None:
    """Guard the guard's test: every look-alike really is Cyrillic or Greek, not the Latin letter."""
    import unicodedata

    for latin, fakes in LOOKALIKES.items():
        for fake in fakes:
            assert fake != latin and re.match(r"(CYRILLIC|GREEK)", unicodedata.name(fake)), (latin, fake)
