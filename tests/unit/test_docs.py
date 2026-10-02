"""The testing guide (docs/TESTING.md) names tests as evidence for each requirement.

Documentation that points at tests which were renamed or deleted is worse than none,
so every test it names must exist.
"""

import re

from tests.conftest import PROJECT_ROOT

GUIDE = (PROJECT_ROOT / "docs/TESTING.md").read_text(encoding="utf-8")
REFERENCE = re.compile(r"`(tests/[\w/]+\.(?:py|mjs))(?:::(\w+))?`")


def test_every_test_named_in_the_testing_guide_exists() -> None:
    references = REFERENCE.findall(GUIDE)
    missing = []
    for path, name in references:
        file = PROJECT_ROOT / path
        if not file.is_file():
            missing.append(path)
        elif name and not re.search(
            rf"^\s*(async )?def {name}\(", file.read_text(encoding="utf-8"), re.MULTILINE
        ):
            missing.append(f"{path}::{name}")
    assert len(references) >= 50
    assert not missing, missing


def test_the_guide_has_no_unfilled_numbers() -> None:
    assert "@@" not in GUIDE
