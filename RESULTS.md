# Real-model benchmark results — DeepSeek V3 on MATH-500 level 5

**Date:** 2026-09-01 · **Setup:** `deepseek-chat`, temperature 0.8, max_tokens 1024,
134 level-5 problems, best-of-n with the stdlib `mathv` verifier, single run
(paired samples across strategies via the disk cache). Script:
[`examples/bench_deepseek.py`](examples/bench_deepseek.py). Total API spend ≈ $0.9.

## Accuracy vs budget — 5 reps, mean ± sd (final)

Reps resample which cached generations each strategy sees (per-rep shuffled
serving order in `CachedBackend`), `sweep_reps.py`; plot at
`examples/data/sweep_plot.png`, raw numbers at `examples/data/sweep_results.json`.

| budget | uniform | index (probe=2, max_per=10) | index (oracle μ, max_per=10) |
|---|---|---|---|
| 2 | .684±.011 | .684±.011* | **.742±.004** |
| 3 | .710±.012 | .696±.011 | **.742±.004** |
| 4 | .725±.006 | **.737±.003** | .742±.004 |
| 5 | .731±.005 | **.745±.006** | .746±.005 |
| 6 | .740±.003 | .745±.006 | .749±.004 |

\* identical by construction: at budget 2 the 2-sample probe consumes the
whole budget, so index = uniform.

> **Corrected 2026-11.** The table above and the headline below were produced
> with a replay bug; see [Replay correction](#replay-correction) for the
> corrected table. The ~33% saving survives; the "~3×" oracle headroom was an
> in-sample estimate and is ~2.5× held-out.

**Headline (with error bars, original replay):** index at budget 4 (.737) ≈
uniform at budget 6 (.740) → **~33% fewer samples from the free probe signal**;
oracle at budget 2 (.742) ≈ uniform at budget 6 → **~3× headroom for a learned
difficulty signal** (in-sample; see correction).

### Single-run exploratory sweep (superseded by the table above)

| budget | uniform | index (probe=4) | index (probe=2) | index (oracle μ, max_per=10) |
|---|---|---|---|---|
| 2 | 0.679 | 0.679* | 0.679 | **0.739** |
| 3 | 0.709 | 0.709* | 0.694 | **0.739** |
| 4 | 0.709 | 0.709* | 0.724 | **0.739** |
| 5 | 0.724 | 0.709 | **0.746** | **0.746** |
| 6 | 0.739 | 0.739 | — | — |

\* probe=4 with budget ≤ 4: the probe consumes the whole budget, so the
allocation is uniform *by construction* — the README's "probe cost isn't
amortised at low budget" limitation, observed in the wild.

## Replay correction

`CachedBackend.sample(prompt, n)` used to return the first `n` cached samples on
every call. `strategy="index"` calls it twice per prompt (probe, then top-up), so
the top-up re-served the probe samples: a problem charged `probe + extra`
samples saw only `max(probe, extra)` distinct ones. `uniform` and the oracle call
it once per problem and were unaffected; `index(probe=2)` was handicapped, and
the original index rows above were produced under this bug. `CachedBackend` now
serves from a per-prompt cursor (distinct samples within a run; call `reset()`
between runs), with regression tests in `tests/test_cached_backend.py`.

Same protocol as above (5 reps, per-rep shuffle, `max_per=10`, 134 level-5
problems, no API calls, zero short reads). Mean ± sd:

| budget | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|
| uniform, published | .684±.011 | .710±.012 | .725±.006 | .731±.005 | .740±.003 |
| uniform, corrected | .682±.014 | .710±.012 | .725±.006 | .731±.005 | .742±.004 |
| index probe=2, published | .684±.011 | .696±.011 | .737±.003 | .745±.006 | .745±.006 |
| index probe=2, buggy replay (local) | .682±.014 | .691±.015 | .737±.003 | .746±.007 | .746±.007 |
| index probe=2, corrected | .682±.014 | **.725±.006** | **.743±.004** | .749±.004 | .749±.004 |
| oracle μ, published | .742±.004 | .742±.004 | .742±.004 | .746±.005 | .749±.004 |
| oracle μ, corrected | .749±.004 | .749±.004 | .749±.004 | .749±.004 | .749±.004 |

- Corrected index at budget 4 (.743) ≥ uniform at budget 6 (.742), so the
  **~33% fewer samples** claim holds. At budget 3, index moves from 1.4 points
  below uniform (published) to 1.5 points above.
- Re-running the original (buggy) serving rule locally reproduces the published
  values within about 0.007; the residual is unexplained and may reflect cache
  growth during the original sweep. Raw corrected numbers:
  `examples/data/sweep_results_corrected.json` (the original
  `sweep_results.json` and `sweep_plot.png` are left as published).
- **Oracle headroom.** The oracle μ is estimated from the same samples it is
  scored on (in-sample, >3× at budget 2). With μ estimated on one half of the
  samples and scored on the other (200 splits, problem-clustered bootstrap), the
  oracle gain at budget 2 is +4.2 points [2.0, 6.2], ≈2.5× sample efficiency
  (95% CI 1.6× to >3×). Smoothing μ with a Jeffreys prior so that
  0-observed problems are not abandoned gives ≈1.4×.
- A 1-sample probe captures 69% of the held-out headroom at budget 3 (CI
  31%–145%). Methods, scripts and raw outputs for these figures live in the
  separate `rs-diagnostic` repo (v0.3 go/no-go diagnostic, 2026-10-19).

## Findings

1. **The savings are real but signal-limited.** With the free probe signal
   (probe=2), index reaches 0.746 at budget 5 — better than uniform at budget 6
   (0.739). Roughly one sample/query (~17%) saved at the top of the curve, and
   probe=2 at budget 4 matches uniform at budget 5.
2. **The headroom is large: a perfect difficulty signal is worth ~3× in-sample (~2.5× held-out, see Replay correction).** Oracle-μ
   allocation hits 0.739 at budget 2 (268 samples) — the accuracy uniform needs
   804 samples (budget 6) to reach. This is the strongest argument for the v0.2
   learned difficulty predictor: the allocator is not the bottleneck, the
   signal is.
3. **Probe size is a real hyperparameter.** probe=4 is dead weight below budget
   5 and its 2-sample resolution advantage never paid off; probe=2 dominated it
   everywhere tested. Small probes win when budgets are small.
4. **Dataset difficulty matters more than anything.** V3 on levels 4–5 mixed was
   ~75% solved-at-one-sample (nothing to allocate); level 5 only gave a usable
   spread (22/40 easy, 8/40 hopeless, 10/40 middle). Pilot-first (`--pilot`)
   is the right workflow.
5. **Verifier quality is load-bearing.** Three normalisation bugs (`x=5` vs `5`,
   unordered solution lists, `\sqrt2` vs `\sqrt{2}`) initially misclassified 3
   solvable problems as hopeless — which would have made the allocator abandon
   them. Difficulty estimation inherits every verifier false negative.

## Caveats

One model, one dataset; sample-count budget, not token budget (harder problems
have longer completions, so token accounting would shift the curves somewhat —
counters are in place). Reps share one underlying pool of cached generations
(~6–10 per problem), so they are not fully independent draws; the oracle μ is
estimated from that same pool.

## Project status: FROZEN (2026-09-01)

Wrapped as a portfolio artifact. A 2026 literature scan (see README "Related
work") found the batch-allocation idea published (training-free difficulty
proxies; DSC) and the RL-side analogue done (Pilot-Commit), while labs move
token allocation inside the model via effort controls. The findings here
independently reproduce the field's consensus (middle-difficulty concentration,
~2.5× held-out oracle headroom, probe amortisation boundary) — the right call is to stop,
not extend. Remaining roadmap items (learned difficulty predictor,
token-denominated budgets, cross-query concurrency) are documented but
intentionally not pursued; energy moves to the GRPO/post-training line.
