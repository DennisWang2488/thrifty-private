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
    verifier : optional ``answer -> bool`` checker, or a prompt-aware
        ``(prompt, answer) -> bool`` (auto-detected by arity) for tasks where
        correctness depends on the query, e.g. math with per-problem gold. If
        given, answers are aggregated by best-of-n (pass@b, the metric the
        index allocator optimises); if ``None``, by majority vote.
    """

    def __init__(
        self,
        backend: LLMBackend,
        strategy: str = "index",
        probe: int = 4,
        difficulty_fn: Callable[[Sequence[str]], float] = self_consistency_hazard,
        difficulty_of_prompt: Callable[[str], float] | None = None,
        verifier: Callable[[str], bool] | None = None,
        max_per: int | None = None,
    ) -> None:
        if strategy not in ("index", "uniform"):
            raise ValueError(f"unknown strategy: {strategy!r}")
        self.backend = backend
        self.strategy = strategy
        self.probe = probe
        self.difficulty_fn = difficulty_fn
        self.difficulty_of_prompt = difficulty_of_prompt
        self.verifier = verifier
        self.max_per = max_per
        self._verifier_takes_prompt = False
        if verifier is not None:
            import inspect

            try:
                params = inspect.signature(verifier).parameters
                self._verifier_takes_prompt = len(params) >= 2
            except (TypeError, ValueError):
                pass

    def _bound_verifier(self, prompt: str) -> Callable[[str], bool] | None:
        """The verifier as a one-arg ``answer -> bool`` for this prompt."""
        if self.verifier is None:
            return None
        if self._verifier_takes_prompt:
            return lambda a, _p=prompt: self.verifier(_p, a)
        return self.verifier

    def _aggregate(self, prompt: str, answers: Sequence[str]) -> str | None:
        v = self._bound_verifier(prompt)
        if v is not None:
            return best_of_n(answers, v)
        return majority_vote(answers)

    def run(self, prompts: Sequence[str], budget: float) -> BatchResult:
        if self.strategy == "uniform":
            return self._run_uniform(prompts, budget)
        return self._run_index(prompts, budget)

    def _run_uniform(self, prompts: Sequence[str], budget: float) -> BatchResult:
        alloc = uniform_allocate(len(prompts), budget)
        results = [
            QueryResult(p, self._aggregate(p, self.backend.sample(p, b)), b)
            for p, b in zip(prompts, alloc)
        ]
        return BatchResult(results)

    def _run_index(self, prompts: Sequence[str], budget: float) -> BatchResult:
        n = len(prompts)
        if self.difficulty_of_prompt is not None:
            # difficulty known without probing (learned predictor / oracle)
            nus = [self.difficulty_of_prompt(p) for p in prompts]
            alloc = index_allocate(nus, budget, min_per=0, max_per=self.max_per)
            results = [
                QueryResult(p, self._aggregate(p, self.backend.sample(p, b)), b)
                for p, b in zip(prompts, alloc)
            ]
            return BatchResult(results)

        # probe-based difficulty: probe samples are reused in the final answer.
        # With a verifier, estimate difficulty from the (unbiased) probe pass
        # rate; otherwise fall back to the self-consistency signal.
        probe = min(self.probe, max(1, int(round(budget))))
        probe_answers = [self.backend.sample(p, probe) for p in prompts]
        if self.verifier is not None:
            nus = [
                pass_rate_hazard(a, self._bound_verifier(p))
                for p, a in zip(prompts, probe_answers)
            ]
        else:
            nus = [self.difficulty_fn(a) for a in probe_answers]
        alloc = index_allocate(nus, budget, min_per=probe, max_per=self.max_per)

        results = []
        for p, pa, b in zip(prompts, probe_answers, alloc):
            extra = b - probe
            answers = list(pa)
            if extra > 0:
                answers.extend(self.backend.sample(p, extra))
            results.append(QueryResult(p, self._aggregate(p, answers), b))
        return BatchResult(results)


def accuracy(
    result: BatchResult,
    gold: dict[str, str],
    equal: Callable[[str, str], bool] | None = None,
) -> float:
    """Fraction of queries whose aggregated answer matches the gold answer.

    ``equal(pred, gold) -> bool`` overrides exact string match (e.g.
    :func:`thrifty.mathv.math_equal` for numeric equivalence).
    """
    if not result.results:
        return 0.0
    eq = equal or (lambda a, g: a == g)
    correct = sum(
        r.answer is not None and eq(r.answer, gold[r.prompt]) for r in result.results
    )
    return correct / len(result.results)
