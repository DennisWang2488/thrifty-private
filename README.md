# thrifty

**Budget-aware test-time compute allocation for LLMs.** Reach a target accuracy
with fewer samples by spending your sampling budget where it buys the most:
hard-but-doable queries get more samples, trivial and hopeless ones get fewer.
Backend-agnostic — any OpenAI-compatible API or a local vLLM/SGLang server.

> Status: v0.1, working name. Targets **verifiable** tasks (code, math, anything
> with a checker or reward model) where the metric is best-of-n / pass@b.

## Why

Best-of-n is the default way to spend test-time compute, but it gives every query
the *same* number of samples — wasteful. Easy queries are solved on the first
sample; hopeless ones never get solved; the queries that actually benefit from
more samples are a minority. `thrifty` estimates each query's difficulty and
water-fills the budget toward where extra samples convert to accuracy.

The allocation rule is the closed-form optimum for the pass@b production function
`1 − (1−μ)^b`, not a hand-tuned threshold — and it is *second-order robust* to
errors in the difficulty estimate (a fixed-threshold router is not). See
[`docs/THEORY.md`](docs/THEORY.md).

## Install

```bash
pip install -e .          # zero required dependencies (stdlib only)
pip install -e ".[openai]"  # to use the OpenAI-compatible backend
```

## 30-second tour (offline, no API key)

```python
from thrifty import BudgetRunner, MockBackend, accuracy

mus  = {f"q{i}": mu for i, mu in enumerate([0.05, 0.3, 0.4, 0.85, 0.9])}
gold = {p: "CORRECT" for p in mus}
backend = MockBackend(mus)

runner = BudgetRunner(backend, strategy="index",
                      verifier=lambda a: a == "CORRECT")
res = runner.run(list(mus), budget=4.0)     # avg 4 samples/query
print(accuracy(res, gold), res.total_samples)
```

## Real backend

```python
from thrifty import BudgetRunner, OpenAIBackend

backend = OpenAIBackend(model="gpt-4o-mini")          # or base_url=local vLLM
runner = BudgetRunner(
    backend, strategy="index", probe=4,
    verifier=my_checker,        # e.g. runs unit tests / checks the boxed answer
)
result = runner.run(prompts, budget=8.0)
```

`strategy="uniform"` gives the best-of-n baseline for comparison.

## Does it actually save? (offline benchmark)

`examples/demo_mock.py`, 300 synthetic queries (a realistic difficulty spread
with a hard floor and an easy bulk), best-of-n with a verifier, 12 reps:

| target accuracy | uniform | index (probe signal) | index (oracle signal) |
|---|---|---|---|
| ≥ 0.88 | ~3600 samples | **~2400 (−33%)** | ~1800 (−50%) |
| ≥ 0.93 | not reached @ 4800 | **~2400** | ~2400 |

The `probe` signal is free — it reuses the probe samples in the final answer and,
with a verifier, estimates difficulty from the *unbiased* probe pass rate. At a
high quality bar it nearly matches a perfect difficulty oracle. At a low quality
bar (≤ 0.80) the probe's fixed cost isn't amortised and it degrades to uniform —
that regime is where a cheaper/learned difficulty signal (v0.2) earns its keep.

## How it works (one paragraph)

A query with per-sample success probability `μ` has `pass@b = 1 − (1−μ)^b`. The
budget-constrained optimum is `b*(ν;λ) = max(0, ln(ν/λ)/ν)` with `ν = −ln(1−μ)`
and `λ` (the shadow price of compute) set to exhaust the budget — hump-shaped:
abandon the hopeless, barely touch the trivial, concentrate on the middle. The
integer version is greedy on marginal gain (`core.index_allocate`). The same
shadow-price re-clearing is what makes the rule robust to difficulty-estimate
noise.

## Roadmap

- **v0.2** learned difficulty predictor trained decision-focused (LANCER-style),
  for the low-budget regime and non-verifiable tasks.
- **v0.2** majority-vote production function (no verifier) — a different
  allocation law (`1 − (1−μ)^b` is the wrong objective there).
- **v0.3** model routing (cheap/expensive) as part of the allocation.
- **v0.4** sequential early-stopping within a reasoning trace.
- **v0.5** first-class local vLLM/SGLang integration + token-budget (not just
  sample-count) allocation.

## Limitations (v0.1)

Verifiable tasks only (needs a `verifier`); allocates sample count, not thinking
tokens; the self-consistency difficulty signal is a coarse fallback when no
verifier is available; savings depend on the difficulty spread and quality bar.
