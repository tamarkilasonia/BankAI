"""Turning a written answer into something pleasant to listen to.

The screen shows the full answer; the voice reads a cleaned-up version: no source
list, no table separators or bullets, masked account numbers read as "ending in
0001", and long answers cut at a sentence boundary with a pointer to the screen.
"""

import re

_MASKED = re.compile(r"\*{4}(\d{4})")
_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")
_SOURCES = "\n\nSources:"
REST_ON_SCREEN = {
    "en": "The full answer is on your screen.",
    "ka": "სრული პასუხი ეკრანზეა.",
}


def speakable(text: str, *, language: str = "en", max_chars: int = 1500) -> str:
    body = text.split(_SOURCES, 1)[0]
    body = _MASKED.sub(r"ending in \1", body).replace(" | ", ", ")
    lines = [_BULLET.sub("", line).strip() for line in body.splitlines()]
    spoken = " ".join(line if line[-1] in ".!?:;," else f"{line}." for line in lines if line)
    if len(spoken) <= max_chars:
        return spoken
    suffix = REST_ON_SCREEN.get(language, REST_ON_SCREEN["en"])
    room = spoken[: max_chars - len(suffix) - 1]
    cut = max(room.rfind(". "), room.rfind("? "), room.rfind("! "))
    return f"{room[: cut + 1] if cut > 0 else room} {suffix}"
