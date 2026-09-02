"""Difficulty signals: estimate a query's hazard ``nu`` from cheap evidence.

The allocator needs a per-query difficulty to split the budget. The cheapest
signal that needs no extra model is *self-consistency*: draw a few probe samples
and measure how much they agree. High agreement => the model is confident =>
easy (high ``mu``, high ``nu``); scattered answers => hard.

This is a coarse proxy (v0.1). It is deliberately swappable: any callable
``list[str] -> nu`` works, including a learned predictor (v0.2, the LANCER-style
decision-focused difficulty model) -- see ``docs/THEORY.md`` Q4.
"""

from __future__ import annotations

from collections import Counter
from typing import Callable, Sequence

from .core import hazard


def self_consistency_hazard(
    probe_answers: Sequence[str],
    lo: float = 0.05,
    hi: float = 0.95,
) -> float:
    """Estimate ``nu`` from the agreement among probe samples.

    ``mu_hat`` = fraction of probe samples equal to the modal answer, clipped to
    ``[lo, hi]``; returns its hazard. With ``k`` probe samples the resolution is
    ``1/k``, so use ``probe >= 3`` for a usable signal.
    """
    if not probe_answers:
        return hazard(0.5, lo, hi)
    counts = Counter(probe_answers)
    mode_frac = counts.most_common(1)[0][1] / len(probe_answers)
    return hazard(mode_frac, lo, hi)


def pass_rate_hazard(
    probe_answers: Sequence[str],
    verifier: Callable[[str], bool],
    lo: float = 0.02,
    hi: float = 0.98,
) -> float:
    """Estimate ``nu`` from the verifier pass rate among probe samples.

    When a verifier is available, ``mu_hat`` = fraction of probe samples it
    approves -- an *unbiased* estimate of the per-sample success probability
    (unlike self-consistency agreement, which is biased upward for hard queries
    by the pigeonhole floor). This is the preferred signal for verifiable tasks
    and it is free: the probe samples are reused in the final answer.
    """
    if not probe_answers:
        return hazard(0.5, lo, hi)
    rate = sum(verifier(a) for a in probe_answers) / len(probe_answers)
    return hazard(rate, lo, hi)


class OracleHazard:
    """Difficulty oracle for tests/benchmarks: returns the true hazard.

    Lets you measure the *allocation* in isolation, with a perfect difficulty
    signal, separating allocation quality from prediction quality.
    """

    def __init__(self, true_mu: dict[str, float]) -> None:
        self._nu = {p: hazard(m) for p, m in true_mu.items()}

    def __call__(self, prompt: str) -> float:
        return self._nu[prompt]
