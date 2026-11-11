"""CachedBackend must serve distinct samples across calls within a run.

Regression test for the v0.2 replay bug: ``strategy="index"`` calls
``sample(p, probe)`` and then ``sample(p, extra)``; the top-up used to re-serve
the probe samples. Offline: no network, no API key.
"""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

from thrifty import BudgetRunner
from thrifty.backends import LLMBackend

_spec = importlib.util.spec_from_file_location(
    "bench_deepseek", Path(__file__).resolve().parents[1] / "examples" / "bench_deepseek.py"
)
bench = importlib.util.module_from_spec(_spec)
sys.modules["bench_deepseek"] = bench
_spec.loader.exec_module(bench)
CachedBackend = bench.CachedBackend


class CountingInner(LLMBackend):
    """Fresh samples are unique strings, so repeats are detectable."""

    def __init__(self):
        self.n_generated = 0

    def sample(self, prompt, n):
        out = [f"fresh{self.n_generated + i}" for i in range(n)]
        self.n_generated += n
        return out


def _seed_cache(cache_dir: Path, prompt: str, answers: list[str]) -> None:
    h = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:24]
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / f"{h}.json").write_text(json.dumps(answers))


def test_successive_calls_are_distinct(tmp_path):
    _seed_cache(tmp_path, "p", [f"a{i}" for i in range(10)])
    be = CachedBackend(CountingInner(), tmp_path)
    first, second = be.sample("p", 2), be.sample("p", 3)
    assert first == ["a0", "a1"] and second == ["a2", "a3", "a4"]
    assert be.fresh == 0 and be.calls == 5


def test_reset_restarts_from_first_sample(tmp_path):
    _seed_cache(tmp_path, "p", [f"a{i}" for i in range(6)])
    be = CachedBackend(CountingInner(), tmp_path)
    be.sample("p", 4)
    be.reset()
    assert be.sample("p", 2) == ["a0", "a1"]


def test_shuffled_order_is_stable_across_calls_and_seeded(tmp_path):
    answers = [f"a{i}" for i in range(8)]
    _seed_cache(tmp_path, "p", answers)
    be = CachedBackend(CountingInner(), tmp_path, shuffle_seed=3)
    got = be.sample("p", 3) + be.sample("p", 5)
    assert sorted(got) == sorted(answers)  # a permutation: no repeats, none missing
    be.reset()
    assert be.sample("p", 3) + be.sample("p", 5) == got  # same seed, same order


def test_shortfall_generates_fresh_without_repeats_or_reshuffle(tmp_path):
    _seed_cache(tmp_path, "p", ["a0", "a1", "a2"])
    inner = CountingInner()
    be = CachedBackend(inner, tmp_path, shuffle_seed=1)
    got = be.sample("p", 2) + be.sample("p", 4)  # needs 3 cached + 3 fresh
    assert len(got) == 6 and len(set(got)) == 6
    assert inner.n_generated == 3 and be.fresh == 3
    # fresh samples were appended to the disk cache for later runs
    h = hashlib.sha256(b"p").hexdigest()[:24]
    assert len(json.loads((tmp_path / f"{h}.json").read_text())) == 6


def test_index_runner_sees_every_sample_it_is_charged_for(tmp_path):
    """The bug: probe + top-up must be probe + extra DISTINCT samples."""
    prompts = [f"q{i}" for i in range(4)]
    for p in prompts:
        _seed_cache(tmp_path, p, [f"{p}-{i}" for i in range(10)])
    seen: dict[str, list[str]] = {p: [] for p in prompts}

    class Recording(CachedBackend):
        def sample(self, prompt, n):
            out = super().sample(prompt, n)
            seen[prompt].extend(out)
            return out

    be = Recording(CountingInner(), tmp_path)
    res = BudgetRunner(
        be, strategy="index", probe=2, max_per=8, verifier=lambda p, a: a.endswith("-0")
    ).run(prompts, budget=5.0)
    for r in res.results:
        assert len(seen[r.prompt]) == r.n_samples
        assert len(set(seen[r.prompt])) == r.n_samples  # no sample served twice
