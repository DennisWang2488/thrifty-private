"""Error-bar sweep: 5 reps x budgets {2..6} x {uniform, index(probe=2), oracle}.

Reps reuse the disk cache with per-rep shuffled sample order, so only
allocation shortfalls (index concentrating beyond the cached pool) cost fresh
API tokens. Writes results to data/sweep_results.json and a plot to
data/sweep_plot.png (if matplotlib is available).
"""

from __future__ import annotations

import hashlib
import json
import os
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import bench_deepseek as B
from thrifty import BudgetRunner, accuracy
from thrifty.core import hazard
from thrifty.mathv import make_verifier, math_equal

DATA = Path(__file__).parent / "data"
BUDGETS = (2, 3, 4, 5, 6)
REPS = 5


class Args:
    key_file = os.environ.get("DEEPSEEK_KEY_FILE", "")
    model = "deepseek-chat"
    temperature = 0.8
    max_tokens = 1024
    concurrency = 8
    cache_dir = str(DATA / "cache")


def main() -> None:
    rows = [
        json.loads(ln)
        for ln in (DATA / "math500.jsonl").read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    rows = [r for r in rows if int(r["level"]) == 5][:134]
    prompts = [r["problem"] for r in rows]
    gold = {r["problem"]: str(r["answer"]) for r in rows}
    verifier = make_verifier(gold)

    args = Args()
    _, inner = B.build_backend(args)
    cache_dir = Path(args.cache_dir) / args.model

    # oracle mu from the full cached pool (upper-bound difficulty signal)
    def mu_of(p: str) -> float:
        f = cache_dir / (hashlib.sha256(p.encode()).hexdigest()[:24] + ".json")
        ans = json.loads(f.read_text(encoding="utf-8"))
        return sum(verifier(p, a) for a in ans) / len(ans)

    nus = {p: hazard(mu_of(p)) for p in prompts}

    results: dict[str, dict[int, list[float]]] = {}
    for rep in range(REPS):
        backend = B.CachedBackend(inner, cache_dir, shuffle_seed=rep)
        for budget in BUDGETS:
            runs = {
                "uniform": BudgetRunner(backend, strategy="uniform", verifier=verifier),
                "index(probe=2)": BudgetRunner(
                    backend, strategy="index", probe=2, verifier=verifier, max_per=10
                ),
                "oracle": BudgetRunner(
                    backend,
                    strategy="index",
                    verifier=verifier,
                    max_per=10,
                    difficulty_of_prompt=lambda p: nus[p],
                ),
            }
            for name, runner in runs.items():
                res = runner.run(prompts, budget=budget)
                acc = accuracy(res, gold, equal=math_equal)
                results.setdefault(name, {}).setdefault(budget, []).append(acc)
        print(f"rep {rep} done (fresh samples so far: {backend.fresh})", flush=True)

    summary = {
        name: {
            b: {
                "mean": statistics.mean(v),
                "sd": statistics.stdev(v) if len(v) > 1 else 0.0,
                "reps": v,
            }
            for b, v in by_b.items()
        }
        for name, by_b in results.items()
    }
    (DATA / "sweep_results.json").write_text(json.dumps(summary, indent=1))

    print(f"\n{'':16s}" + "".join(f"budget={b:<8}" for b in BUDGETS))
    for name, by_b in summary.items():
        line = "".join(f"{by_b[b]['mean']:.3f}±{by_b[b]['sd']:.3f} " for b in BUDGETS)
        print(f"{name:16s}{line}")
    print(B.cost_line(inner))

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 4))
        for name, by_b in summary.items():
            xs = list(BUDGETS)
            means = [by_b[b]["mean"] for b in xs]
            sds = [by_b[b]["sd"] for b in xs]
            ax.errorbar(xs, means, yerr=sds, marker="o", capsize=3, label=name)
        ax.set_xlabel("budget (avg samples/query)")
        ax.set_ylabel("accuracy (134 MATH-500 level-5, DeepSeek V3)")
        ax.set_title("thrifty: budget-aware allocation vs uniform best-of-n")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(DATA / "sweep_plot.png", dpi=150)
        print(f"plot -> {DATA / 'sweep_plot.png'}")
    except ImportError:
        print("matplotlib not installed; skipped plot")


if __name__ == "__main__":
    main()
