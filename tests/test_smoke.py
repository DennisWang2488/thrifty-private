"""Smoke tests: pipeline correctness, budget conservation, allocation value."""

import random

from thrifty import (
    BudgetRunner,
    MockBackend,
    OracleHazard,
    accuracy,
    index_allocate,
    uniform_allocate,
)
from thrifty.core import hazard, success_prob


def _batch(n=120, seed=0):
    rng = random.Random(seed)
    mus = {f"q{i}": rng.choice([0.05, 0.3, 0.45, 0.85]) for i in range(n)}
    gold = {p: "CORRECT" for p in mus}
    return mus, gold


def test_budget_conservation():
    mus, _ = _batch()
    prompts = list(mus)
    for strat, kw in [("uniform", {}), ("index", {"probe": 4})]:
        runner = BudgetRunner(MockBackend(mus), strategy=strat,
                              verifier=lambda a: a == "CORRECT", **kw)
        res = runner.run(prompts, budget=6.0)
        assert res.total_samples == round(6.0 * len(prompts))


def test_index_allocation_is_hump_shaped():
    # easy and hopeless queries get fewer samples than the doable middle
    nus = [hazard(mu) for mu in [0.02, 0.35, 0.9]]
    alloc = index_allocate(nus, budget=4.0, min_per=0)
    assert alloc[1] >= alloc[0] and alloc[1] >= alloc[2]
    assert sum(alloc) == round(4.0 * 3)


def test_success_prob_monotone_concave():
    nu = hazard(0.3)
    gains = [success_prob(nu, b + 1) - success_prob(nu, b) for b in range(8)]
    assert all(g > 0 for g in gains)
    assert all(gains[i] >= gains[i + 1] - 1e-12 for i in range(len(gains) - 1))


def test_oracle_index_beats_uniform_at_high_budget():
    mus, gold = _batch(n=200)
    prompts = list(mus)
    ver = lambda a: a == "CORRECT"  # noqa: E731
    acc_u = sum(
        accuracy(BudgetRunner(MockBackend(mus, seed=s), strategy="uniform",
                              verifier=ver).run(prompts, 8.0), gold)
        for s in range(6)
    ) / 6
    acc_i = sum(
        accuracy(BudgetRunner(MockBackend(mus, seed=s), strategy="index",
                              verifier=ver,
                              difficulty_of_prompt=OracleHazard(mus))
                 .run(prompts, 8.0), gold)
        for s in range(6)
    ) / 6
    assert acc_i > acc_u


def test_uniform_allocate_basic():
    assert uniform_allocate(5, 3.0) == [3, 3, 3, 3, 3]
