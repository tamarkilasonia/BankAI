"""Undo common disguises before the safety rules look at a text.

Attackers rarely write "ignore your instructions" plainly once a filter exists.
They write ``1gn0r3``, ``i g n o r e``, use look-alike Cyrillic letters, hide
zero-width characters inside words, or encode the whole instruction in base64
and ask the model to decode it. Each function here produces one *view* of the
text; the guard runs its rules on every view.

The views are only used for detection - the agent itself always works with
the customer's original message.
"""

import base64
import binascii
import codecs
import re
import unicodedata
from dataclasses import dataclass
from functools import partial
from urllib.parse import unquote

# Letters from other scripts that look like Latin ones (a small, common subset).
_CONFUSABLES = str.maketrans(
    {
        # Cyrillic lowercase
        0x0430: "a", 0x0435: "e", 0x043E: "o", 0x0440: "p", 0x0441: "c", 0x0443: "y", 0x0445: "x",
        0x0456: "i", 0x0458: "j", 0x0455: "s", 0x0501: "d", 0x04BB: "h", 0x043A: "k",
        # Cyrillic uppercase
        0x0410: "A", 0x0412: "B", 0x0415: "E", 0x041A: "K", 0x041C: "M", 0x041D: "H", 0x041E: "O",
        0x0420: "P", 0x0421: "C", 0x0422: "T", 0x0425: "X", 0x0406: "I", 0x0405: "S", 0x0408: "J",
        0x04BA: "H",
        # Greek
        0x03B1: "a", 0x03BF: "o", 0x03C1: "p", 0x03BD: "v", 0x03B5: "e", 0x03B9: "i", 0x03BA: "k",
        0x03C4: "t", 0x0391: "A", 0x0392: "B", 0x0395: "E", 0x0397: "H", 0x0399: "I", 0x039A: "K",
        0x039C: "M", 0x039D: "N", 0x039F: "O", 0x03A1: "P", 0x03A4: "T", 0x03A7: "X", 0x03A5: "Y",
        0x0396: "Z",
    }
)  # fmt: skip
_LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s"})
_IDENTIFIER_LIKE = re.compile(r"\d{4}|[2689]")
_MIXED_TOKEN = re.compile(r"[\w@$']*[a-z][\w@$']*")  # "cu57-0003" is two tokens: "cu57" and "0003"
_SPACED_LETTERS = re.compile(r"\b(?:[^\W\d_][ .\-_*]){2,}[^\W\d_]\b")  # any script: Georgian too
_BASE64 = re.compile(r"(?<![A-Za-z0-9+/=_-])[A-Za-z0-9+/_-]{16,}={0,2}")
_HEX = re.compile(r"\b(?:[0-9a-fA-F]{2}[ :]?){8,}")
_URL_ESCAPE = re.compile(r"%[0-9a-fA-F]{2}")


@dataclass(frozen=True)
class View:
    name: str  # plain | leet | despaced | base64 | hex | url | rot13 | reversed
    text: str


def canonical(text: str) -> str:
    """NFKC, drop invisible format characters, map look-alike letters, lowercase, one space."""
    text = unicodedata.normalize("NFKC", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Cf")  # zero-width, bidi, soft hyphen
    text = text.translate(_CONFUSABLES)
    text = text.replace("\N{RIGHT SINGLE QUOTATION MARK}", "'").replace("\N{LEFT SINGLE QUOTATION MARK}", "'")
    # Casefolding can turn an unlisted capital into a listed small letter (Cyrillic U+0500 -> U+0501),
    # so map once more: canonical(canonical(x)) == canonical(x), a property the tests check.
    return re.sub(r"\s+", " ", text.casefold().translate(_CONFUSABLES)).strip()


def _readable(raw: bytes) -> str | None:
    """Decoded bytes count only if they are mostly readable text (not random binary)."""
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return None
    # Invisible format characters (zero-width etc.) are text, not binary noise: an attacker
    # can hide them inside the encoded message too, so they don't count against it.
    visible = "".join(ch for ch in text if unicodedata.category(ch) != "Cf")
    printable = sum(ch.isprintable() or ch.isspace() for ch in visible)
    letters = sum(ch.isalpha() for ch in visible)
    return text if visible and printable / len(visible) >= 0.95 and letters >= 4 else None


def decode_base64(text: str) -> list[str]:
    decoded = []
    for token in _BASE64.findall(text):
        padded = token.replace("-", "+").replace("_", "/")
        padded += "=" * (-len(padded) % 4)
        try:
            raw = base64.b64decode(padded, validate=True)
        except (binascii.Error, ValueError):
            continue
        if readable := _readable(raw):
            decoded.append(readable)
    return decoded


def decode_hex(text: str) -> list[str]:
    decoded = []
    for match in _HEX.findall(text):
        digits = re.sub(r"[ :]", "", match)  # always whole bytes: the pattern matches digit pairs
        if readable := _readable(bytes.fromhex(digits)):
            decoded.append(readable)
    return decoded


def _unleet(token: re.Match[str]) -> str:
    """ "1gn0r3" -> "ignore", "l157" -> "list"; identifiers stay as they are.

    A token is an identifier ("ge00mock00000100000001") if it has four digits in a row or a
    digit leetspeak never uses (2, 6, 8, 9) - translating it would break rules that need it.
    """
    text = token.group(0)
    return text if _IDENTIFIER_LIKE.search(text) else text.translate(_LEET)


def _join_letters(spaced: re.Match[str], edges: tuple[str, ...]) -> str:
    """ "i g n o r e" -> "ignore". A one-letter word next to the spaced one gets swallowed
    ("my wife's a c c o u n t" -> "...saccount"), so variants keep the first or last letter apart."""
    letters = re.sub(r"[ .\-_*]", "", spaced.group(0))
    if "first" in edges:
        letters = f"{letters[0]} {letters[1:]}"
    if "last" in edges:
        letters = f"{letters[:-1]} {letters[-1]}"
    return letters


def views(text: str) -> list[View]:
    """Every distinct reading of ``text`` the safety rules should check."""
    plain = canonical(text)
    found = [View("plain", plain)]

    def add(name: str, candidate: str) -> None:
        candidate = canonical(candidate)
        if candidate and all(candidate != v.text for v in found):
            found.append(View(name, candidate))

    # Leetspeak only inside tokens that mix letters with digits/symbols ("1gn0r3"), so plain
    # amounts like "1000" stay untouched.
    add("leet", _MIXED_TOKEN.sub(_unleet, plain))
    for edges in [(), ("first",), ("last",), ("first", "last")]:
        add("despaced", _SPACED_LETTERS.sub(partial(_join_letters, edges=edges), plain))
    for decoded in decode_base64(unicodedata.normalize("NFKC", text)):
        add("base64", decoded)
    for decoded in decode_hex(text):
        add("hex", decoded)
    if _URL_ESCAPE.search(text):
        add("url", unquote(text))
    add("rot13", codecs.encode(plain, "rot13"))
    add("reversed", plain[::-1])
    return found
