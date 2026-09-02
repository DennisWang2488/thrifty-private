"""Offline demo: does budget-aware allocation reach the same accuracy for fewer
samples than uniform best-of-n? Runs entirely on the MockBackend (no API key).

Synthetic batch: a realistic spread of per-sample success probabilities with a
chunk of near-hopeless queries (the "TACO floor") and a chunk of trivial ones.
We sweep the average budget B and plot accuracy vs total samples for:
  - uniform     : B samples per query
  - index(SC)   : budget-aware, self-consistency difficulty signal (probe=4)
  - index(oracle): budget-aware with a perfect difficulty signal (upper bound)
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thrifty import BudgetRunner, MockBackend, OracleHazard, accuracy  # noqa: E402


def make_batch(n: int, seed: int = 0):
    rng = random.Random(seed)
    mus, gold = {}, {}
    for i in range(n):
        u = rng.random()
        if u < 0.20:
            mu = rng.uniform(0.02, 0.12)      # near-hopeless (hard floor)
        elif u < 0.55:
            mu = rng.uniform(0.20, 0.55)      # the sweet spot: doable with help
        else:
            mu = rng.uniform(0.70, 0.95)      # easy
        p = f"q{i}"
        mus[p] = mu
        gold[p] = "CORRECT"
    return mus, gold


VERIFIER = lambda a: a == "CORRECT"  # noqa: E731 (mock checker: pass@b metric)


def evaluate(strategy, prompts, gold, mus, budget, reps, **kw):
    accs, costs = [], []
    for r in range(reps):
        backend = MockBackend(mus, seed=1000 * r + 7)
        runner = BudgetRunner(backend, strategy=strategy, verifier=VERIFIER, **kw)
        res = runner.run(prompts, budget)
        accs.append(accuracy(res, gold))
        costs.append(res.total_samples)
    return sum(accs) / reps, sum(costs) / reps


def main():
    N, REPS = 300, 12
    mus, gold = make_batch(N)
    prompts = list(mus)
    budgets = [1, 2, 3, 4, 6, 8, 12, 16]

    print(f"{'B':>3} | {'uniform':>18} | {'index(probe)':>18} | "
          f"{'index(oracle)':>18}")
    print(f"{'':>3} | {'acc    @ samples':>18} | {'acc    @ samples':>18} | "
          f"{'acc    @ samples':>18}")
    print("-" * 72)

    rows = []
    for B in budgets:
        ua, uc = evaluate("uniform", prompts, gold, mus, B, REPS)
        ia, ic = evaluate("index", prompts, gold, mus, B, REPS, probe=4)
        oa, oc = evaluate(
            "index", prompts, gold, mus, B, REPS,
            difficulty_of_prompt=OracleHazard(mus),
        )
        rows.append((B, ua, uc, ia, ic, oa, oc))
        print(f"{B:>3} | {ua:5.3f} @ {uc:7.1f} | {ia:5.3f} @ {ic:7.1f} | "
              f"{oa:5.3f} @ {oc:7.1f}")

    # Savings headline: samples needed to reach a target accuracy, and the
    # saving of budget-aware allocation vs uniform. Higher quality bar => more
    # saving (the probe's fixed cost is amortised; allocation has room to work).
    print("\nSamples to reach a target accuracy (saving vs uniform):")
    for target in [0.80, 0.88, 0.93]:
        base = None
        parts = []
        for name, ai, ci in [("uniform", 1, 2), ("index(probe)", 3, 4),
                             ("index(oracle)", 5, 6)]:
            hit = [(r[ci], r[0]) for r in rows if r[ai] >= target]
            if not hit:
                parts.append(f"{name} >max")
                continue
            samples, _ = min(hit)
            if name == "uniform":
                base = samples
                parts.append(f"{name} ~{samples:.0f}")
            else:
                sv = f" ({100 * (base - samples) / base:+.0f}%)" if base else ""
                parts.append(f"{name} ~{samples:.0f}{sv}")
        print(f"  acc>={target:.2f}:  " + "   ".join(parts))


if __name__ == "__main__":
    main()
