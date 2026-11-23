"""Unit tests for MATH-500 answer parser and benchmark evaluation utilities."""

from __future__ import annotations

import pytest

from weightliftai.weight_programming.math500_evaluator import (
    build_math_prompt,
    detect_cyclic_repetition,
    extract_boxed_answer,
    is_math_equivalent,
    normalize_math_expression,
)


def test_extract_boxed_answer_nested_braces() -> None:
    """Verify extraction handles arbitrary nested LaTeX braces correctly."""
    text1 = r"Hence, the answer is \boxed{\frac{14}{3}}."
    assert extract_boxed_answer(text1) == r"\frac{14}{3}"

    text2 = r"Therefore \boxed{\left( 3, \frac{\pi}{2} \right)}."
    assert extract_boxed_answer(text2) == r"\left( 3, \frac{\pi}{2} \right)"

    text3 = r"First we try \boxed{1}, but ultimately \boxed{42}."
    assert extract_boxed_answer(text3) == "42"


def test_extract_boxed_answer_fallback_phrases() -> None:
    """Verify fallback regex extracts answer when \\boxed{} is absent."""
    text = "After simplifying, the final answer is 125."
    assert extract_boxed_answer(text) == "125"


def test_normalize_math_expression() -> None:
    """Verify LaTeX normalization strips macros, styling, spaces, and delimiters."""
    assert normalize_math_expression(r"$\text{Evelyn}$") == "Evelyn"
    assert normalize_math_expression(r"\( 90^\circ \)") == "90"
    assert normalize_math_expression(r"\boxed{\frac{14}{3}}") == "14/3"
    assert normalize_math_expression(" 1,000 ") == "1000"


@pytest.mark.parametrize(
    "pred,target,expected",
    [
        (r"\frac{14}{3}", r"\frac{14}{3}", True),
        (r"\frac{14}{3}", "14/3", True),
        ("4.0", "4", True),
        ("42", "42", True),
        (r"3\sqrt{13}", r"3\sqrt{13}", True),
        (r"\left( 3, \frac{\pi}{2} \right)", r"(3, \pi/2)", True),
        ("12", "15", False),
        (None, "42", False),
        ("42", None, False),
    ],
)
def test_is_math_equivalent(pred: str | None, target: str | None, expected: bool) -> None:
    """Verify numerical and canonical math equivalence verification."""
    assert is_math_equivalent(pred, target) == expected


def test_detect_cyclic_repetition() -> None:
    """Verify cyclic loop detection detects infinite generation loops."""
    repeating = "Let x = 5. Therefore x = 5. Therefore x = 5. Therefore x = 5."
    assert detect_cyclic_repetition(repeating, min_cycle_len=10, min_repeats=3)

    normal = "Let x = 5. Then 2x = 10. Adding 3 yields 13. The answer is 13."
    assert not detect_cyclic_repetition(normal, min_cycle_len=10, min_repeats=3)


def test_build_math_prompt() -> None:
    """Verify prompt formatting contains instruction and problem text."""
    p = build_math_prompt("Solve for x: 2x = 4")
    assert "Please solve the following math problem step by step." in p
    assert r"\boxed{}" in p
    assert "Solve for x: 2x = 4" in p
