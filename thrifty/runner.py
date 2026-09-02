"""End-to-end orchestration: probe -> estimate difficulty -> allocate -> vote.

The two strategies share the SAME total budget ``round(budget * n)`` samples, so
comparisons are honest. ``uniform`` spends it evenly. ``index`` first probes each
query (those probe samples are REUSED in the final vote, not wasted), estimates
difficulty, then water-fills the remaining pool toward the queries where extra
samples buy the most accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

from .aggregate import best_of_n, majority_vote
from .backends import LLMBackend
from .core import index_allocate, uniform_allocate
from .difficulty import pass_rate_hazard, self_consistency_hazard


@dataclass
class QueryResult:
    prompt: str
    answer: str | None
    n_samples: int


@dataclass
class BatchResult:
    results: list[QueryResult]
    total_samples: int = field(init=False)

    def __post_init__(self) -> None:
        self.total_samples = sum(r.n_samples for r in self.results)

    def answers(self) -> list[str | None]:
        return [r.answer for r in self.results]


class BudgetRunner:
    """Allocate a sampling budget across a batch and aggregate the answers.

    Parameters
    ----------
    backend : the sample source (:class:`~thrifty.backends.LLMBackend`).
    strategy : ``"index"`` (budget-aware, default) or ``"uniform"`` (baseline).
    probe : probe samples per query for the difficulty signal (index only).
    difficulty_fn : ``list[str] -> nu`` (self-consistency by default) OR, if
        ``difficulty_of_prompt`` is given, that ``prompt -> nu`` callable is used
        instead (e.g. a learned predictor or an oracle) and no probe is drawn.
    verifier : optional ``answer -> bool`` checker. If given, answers are
        aggregated by best-of-n (pass@b, the metric the index allocator
        optimises); if ``None``, by majority vote.
    """

    def __init__(
        self,
        backend: LLMBackend,
        strategy: str = "index",
        probe: int = 4,
        difficulty_fn: Callable[[Sequence[str]], float] = self_consistency_hazard,
        difficulty_of_prompt: Callable[[str], float] | None = None,
        verifier: Callable[[str], bool] | None = None,
    ) -> None:
        if strategy not in ("index", "uniform"):
            raise ValueError(f"unknown strategy: {strategy!r}")
        self.backend = backend
        self.strategy = strategy
        self.probe = probe
        self.difficulty_fn = difficulty_fn
        self.difficulty_of_prompt = difficulty_of_prompt
        self.verifier = verifier

    def _aggregate(self, answers: Sequence[str]) -> str | None:
        if self.verifier is not None:
            return best_of_n(answers, self.verifier)
        return majority_vote(answers)

    def run(self, prompts: Sequence[str], budget: float) -> BatchResult:
        if self.strategy == "uniform":
            return self._run_uniform(prompts, budget)
        return self._run_index(prompts, budget)

    def _run_uniform(self, prompts: Sequence[str], budget: float) -> BatchResult:
        alloc = uniform_allocate(len(prompts), budget)
        results = [
            QueryResult(p, self._aggregate(self.backend.sample(p, b)), b)
            for p, b in zip(prompts, alloc)
        ]
        return BatchResult(results)

    def _run_index(self, prompts: Sequence[str], budget: float) -> BatchResult:
        n = len(prompts)
        if self.difficulty_of_prompt is not None:
            # difficulty known without probing (learned predictor / oracle)
            nus = [self.difficulty_of_prompt(p) for p in prompts]
            alloc = index_allocate(nus, budget, min_per=0)
            results = [
                QueryResult(p, self._aggregate(self.backend.sample(p, b)), b)
                for p, b in zip(prompts, alloc)
            ]
            return BatchResult(results)

        # probe-based difficulty: probe samples are reused in the final answer.
        # With a verifier, estimate difficulty from the (unbiased) probe pass
        # rate; otherwise fall back to the self-consistency signal.
        probe = min(self.probe, max(1, int(round(budget))))
        probe_answers = [self.backend.sample(p, probe) for p in prompts]
        if self.verifier is not None:
            nus = [pass_rate_hazard(a, self.verifier) for a in probe_answers]
        else:
            nus = [self.difficulty_fn(a) for a in probe_answers]
        alloc = index_allocate(nus, budget, min_per=probe)

        results = []
        for p, pa, b in zip(prompts, probe_answers, alloc):
            extra = b - probe
            answers = list(pa)
            if extra > 0:
                answers.extend(self.backend.sample(p, extra))
            results.append(QueryResult(p, self._aggregate(answers), b))
        return BatchResult(results)


def accuracy(result: BatchResult, gold: dict[str, str]) -> float:
    """Fraction of queries whose aggregated answer matches the gold answer."""
    if not result.results:
        return 0.0
    correct = sum(r.answer == gold[r.prompt] for r in result.results)
    return correct / len(result.results)
