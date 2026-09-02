"""Budget-aware allocation of test-time compute across a batch of queries.

The question: given a fixed average sampling budget ``B`` (samples per query) and
a per-query difficulty estimate, how many samples should each query get so that
total expected accuracy is maximised?

We model a query by its per-sample success probability ``mu in (0, 1)``, written
in hazard form ``nu = -log(1 - mu)``. With ``b`` independent samples the chance
that at least one is correct is ``1 - exp(-nu b)`` -- increasing and concave in
``b``. Maximising the sum of these over the batch subject to ``mean(b) <= B`` has
the closed-form water-filling solution

    b*(nu; lam) = max(0, ln(nu / lam) / nu),

where ``lam`` (the shadow price of compute) clears the budget. Hard queries
(``nu <= lam``) are abandoned, easy queries need little, the middle gets the
bulk. The integer version is greedy on marginal gain (this module) and coincides
with the continuous rule; see ``docs/THEORY.md`` for the derivation and for *why*
this index rule is second-order robust to difficulty-estimate error while fixed
thresholds are not -- the robustness property is the library's differentiator.
"""

from __future__ import annotations

import heapq
import math
from typing import Sequence


def hazard(mu: float, lo: float = 1e-4, hi: float = 1 - 1e-4) -> float:
    """Map a per-sample success probability to its hazard ``nu = -log(1 - mu)``."""
    mu = min(max(mu, lo), hi)
    return -math.log1p(-mu)


def success_prob(nu: float, b: int) -> float:
    """Probability at least one of ``b`` samples is correct: ``1 - exp(-nu b)``."""
    return 1.0 - math.exp(-nu * b)


def _marginal_gain(nu: float, b: int) -> float:
    """Increase in success probability from the ``(b+1)``-th sample.

    ``q(nu, b+1) - q(nu, b) = (1 - e^{-nu}) e^{-nu b}`` -- positive, decreasing in
    ``b`` (diminishing returns), which is exactly what makes greedy optimal.
    """
    return (1.0 - math.exp(-nu)) * math.exp(-nu * b)


def uniform_allocate(n_queries: int, budget: float) -> list[int]:
    """Baseline: split the budget evenly, ``floor(B)`` samples each (min 1)."""
    per = max(1, int(round(budget)))
    return [per] * n_queries


def index_allocate(
    nus: Sequence[float],
    budget: float,
    min_per: int = 0,
    max_per: int | None = None,
) -> list[int]:
    """Greedy water-filling allocation under a total budget.

    Hands out the ``round(budget * n)`` samples one at a time, always to the query
    whose *next* sample buys the most expected accuracy. This is the integer
    optimum for the concave production function and reproduces the closed-form
    index rule ``b*(nu; lam)`` with ``lam`` = the marginal gain of the last
    sample handed out.

    Parameters
    ----------
    nus : difficulty estimates (hazards) for each query.
    budget : average samples per query; total pool is ``round(budget * n)``.
    min_per : samples already spent per query (e.g. probe samples) -- the
        allocation starts from here and these are assumed reused downstream.
    max_per : optional per-query cap.
    """
    n = len(nus)
    total = int(round(budget * n))
    b = [min_per] * n
    spent = min_per * n
    if total <= spent:
        return b

    heap: list[tuple[float, int]] = [
        (-_marginal_gain(nus[i], b[i]), i) for i in range(n)
    ]
    heapq.heapify(heap)

    while spent < total and heap:
        neg_gain, i = heapq.heappop(heap)
        if max_per is not None and b[i] >= max_per:
            continue
        b[i] += 1
        spent += 1
        if max_per is None or b[i] < max_per:
            heapq.heappush(heap, (-_marginal_gain(nus[i], b[i]), i))
    return b


def shadow_price(nus: Sequence[float], allocation: Sequence[int]) -> float:
    """The realised shadow price ``lam``: the smallest funded marginal gain.

    Diagnostic -- queries whose first sample's gain is below this were abandoned.
    """
    funded = [
        _marginal_gain(nus[i], allocation[i] - 1)
        for i in range(len(nus))
        if allocation[i] > 0
    ]
    return min(funded) if funded else 0.0
