"""Answer aggregation across the samples drawn for one query.

Two regimes, and they need different allocation laws (this matters -- see the
note in ``runner.py`` and the v0.1 demo):

- ``best_of_n`` (a verifier / reward model can recognise a correct sample): the
  query is solved iff *any* sample is correct, i.e. accuracy = pass@b =
  ``1 - (1-mu)^b``. This is exactly the production function the index allocator
  optimises, so budget-aware allocation is optimal by construction. Covers
  verifiable tasks: code (unit tests), math (answer check), formal proofs,
  reward-model-scored generation.
- ``majority_vote`` (no verifier): the query is solved iff the correct answer is
  the plurality. Its production function is different (it can *decrease* in b for
  mu < 1/2), so the hazard index rule is mis-specified here -- v0.2.
"""

from __future__ import annotations

from collections import Counter
from typing import Callable, Sequence


def majority_vote(answers: Sequence[str]) -> str | None:
    """Most frequent answer (self-consistency). ``None`` if no samples."""
    if not answers:
        return None
    return Counter(answers).most_common(1)[0][0]


def best_of_n(
    answers: Sequence[str], verifier: Callable[[str], bool]
) -> str | None:
    """Return a verifier-approved sample if one exists, else the plurality.

    Models best-of-n selection with a checker/reward model: accuracy = pass@b.
    """
    if not answers:
        return None
    for a in answers:
        if verifier(a):
            return a
    return majority_vote(answers)
