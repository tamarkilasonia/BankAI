"""Properties of the numeric grounding check (Hypothesis).

Two promises, over any amount: the way a number is written never matters
("1,234.50" = "1234.5"), and a number the facts don't contain is always caught.
"""

from collections.abc import Callable
from decimal import Decimal

from hypothesis import assume, given
from hypothesis import strategies as st

from app.agents.grounding import missing_numbers, unsupported_numbers

AMOUNTS = st.decimals(min_value=Decimal("0.01"), max_value=Decimal("9999999.99"), places=2)
WRITTEN = [  # how a model or a template might write the same amount
    lambda d: f"{d:,.2f}",
    lambda d: f"{d:.2f}",
    lambda d: f"{d.normalize():f}",
    lambda d: f"{d:,f}",
]


@given(AMOUNTS, st.sampled_from(WRITTEN), st.sampled_from(WRITTEN))
def test_the_same_amount_written_differently_is_grounded(
    amount: Decimal, in_facts: Callable[[Decimal], str], in_answer: Callable[[Decimal], str]
) -> None:
    facts = f"Total spent: {in_facts(amount)} GEL"
    answer = f"You spent {in_answer(amount)} GEL."
    assert unsupported_numbers(answer, facts) == []
    assert missing_numbers(answer, [amount]) == []


@given(AMOUNTS, AMOUNTS)
def test_an_amount_the_facts_do_not_contain_is_always_caught(real: Decimal, invented: Decimal) -> None:
    assume(real.normalize() != invented.normalize())
    answer = f"You spent {invented:,.2f} GEL."
    assert unsupported_numbers(answer, f"Total spent: {real:,.2f} GEL") == [f"{invented:,.2f}"]
    assert missing_numbers(answer, [real]) == [real]


@given(st.integers(0, 12), st.sampled_from([" GEL", " USD", " EUR", "%", " percent"]))
def test_a_small_number_is_a_count_but_never_a_free_amount(n: int, unit: str) -> None:
    assert unsupported_numbers(f"You have {n} accounts.", "no numbers in the facts") == []
    assert unsupported_numbers(f"The fee is {n}{unit}.", "no numbers in the facts") == [str(n)]


@given(st.text(max_size=200), st.text(max_size=200))
def test_grounding_never_crashes(answer: str, facts: str) -> None:
    assert isinstance(unsupported_numbers(answer, facts), list)
