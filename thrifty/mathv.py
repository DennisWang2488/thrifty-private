"""Lightweight math answer verification (stdlib only).

Extract the final answer from a model completion (``\\boxed{...}`` preferred,
falling back to the last number-like token) and compare it to a gold answer up
to normalisation and numeric equivalence. This is deliberately a pragmatic 90%
solution -- for paper-grade grading swap in ``math_verify`` via ``extract_fn``/
``verifier``; the library interfaces don't change.
"""

from __future__ import annotations

import re
from fractions import Fraction

_BOXED = re.compile(r"\\boxed\s*\{")
_NUMBER = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?(?:/\d+)?")


def extract_boxed(text: str) -> str | None:
    """Content of the LAST ``\\boxed{...}`` in ``text`` (brace-balanced)."""
    last = None
    for m in _BOXED.finditer(text):
        depth, start = 1, m.end()
        for i in range(start, len(text)):
            c = text[i]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    last = text[start:i]
                    break
    return last


def extract_answer(text: str) -> str:
    """Final answer of a completion: last \\boxed{} if present, else the last
    number-like token, else the last non-empty line."""
    boxed = extract_boxed(text)
    if boxed is not None:
        return boxed.strip()
    nums = _NUMBER.findall(text)
    if nums:
        return nums[-1]
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def _normalize(ans: str) -> str:
    s = ans.strip()
    s = re.sub(r"\\(left|right|!|,|;)\b", "", s)
    s = s.replace("\\left", "").replace("\\right", "")
    s = re.sub(r"\\text\s*\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\mathrm\s*\{([^}]*)\}", r"\1", s)
    s = s.replace("\\%", "").replace("%", "")
    s = s.replace("$", "").replace(" ", "").replace("^{\\circ}", "").replace("^\\circ", "")
    s = s.replace("dfrac", "frac").replace("tfrac", "frac")
    s = s.rstrip(".")
    # \sqrt2 -> \sqrt{2}
    s = re.sub(r"\\sqrt(\d+)", r"\\sqrt{\1}", s)
    # strip a leading variable assignment: "x=5" -> "5"
    s = re.sub(r"^[a-zA-Z](?:_\d+)?=", "", s)
    return s


def _as_fraction(s: str) -> Fraction | None:
    s = s.replace(",", "")
    m = re.fullmatch(r"\\frac\{(-?\d+)\}\{(-?\d+)\}", s)
    if m:
        try:
            return Fraction(int(m.group(1)), int(m.group(2)))
        except ZeroDivisionError:
            return None
    m = re.fullmatch(r"(-?\d+)/(-?\d+)", s)
    if m:
        try:
            return Fraction(int(m.group(1)), int(m.group(2)))
        except ZeroDivisionError:
            return None
    try:
        return Fraction(s)
    except (ValueError, ZeroDivisionError):
        return None


def math_equal(pred: str, gold: str) -> bool:
    """Equivalence up to normalisation, exact rational value, and (for
    comma-separated solution lists outside parentheses) order."""
    p, g = _normalize(pred), _normalize(gold)
    if p == g:
        return True
    fp, fg = _as_fraction(p), _as_fraction(g)
    if fp is not None and fg is not None:
        return fp == fg
    # unordered comparison for bare solution lists like "1,-2" vs "-2,1"
    # (NOT tuples/intervals wrapped in parens/brackets, where order matters)
    if "," in p and "," in g and p[:1] not in "([{" and g[:1] not in "([{":
        ps, gs = p.split(","), g.split(",")
        if len(ps) == len(gs):
            used = [False] * len(gs)
            for a in ps:
                for i, b in enumerate(gs):
                    if not used[i] and math_equal(a, b):
                        used[i] = True
                        break
                else:
                    return False
            return True
    return False


def make_verifier(gold_by_prompt: dict[str, str]):
    """Prompt-aware verifier ``(prompt, answer) -> bool`` for BudgetRunner."""

    def verify(prompt: str, answer: str) -> bool:
        gold = gold_by_prompt.get(prompt)
        return gold is not None and math_equal(answer, gold)

    return verify
