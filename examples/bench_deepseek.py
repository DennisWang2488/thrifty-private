"""Real-model benchmark: thrifty on MATH-500 via the DeepSeek API.

Setup (no API spend):
    python examples/bench_deepseek.py --download-only     # fetch dataset
    python examples/bench_deepseek.py --dry-run           # show the plan

Pilot (~$0.5): estimate the per-problem difficulty spread first. If most
problems land near mu=0 or mu=1 there is nothing to allocate -- pick harder/
easier levels before spending on the full bench.
    python examples/bench_deepseek.py --pilot --n 40 --levels 4,5

Bench:
    python examples/bench_deepseek.py --n 150 --levels 4,5 --budget 6 \
        --strategies uniform,index

All samples are cached on disk (per prompt, append-only), so repeated runs and
higher budgets only pay for NEW samples. The uniform/index comparison reuses
the same cached samples (paired comparison; lower variance, honest budgets).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thrifty import BudgetRunner, accuracy
from thrifty.backends import LLMBackend, OpenAIBackend
from thrifty.mathv import extract_answer, make_verifier, math_equal

MATH500_URL = (
    "https://huggingface.co/datasets/HuggingFaceH4/MATH-500/resolve/main/test.jsonl"
)
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
SYSTEM_PROMPT = (
    "You are a careful mathematician. Solve the problem step by step, "
    "then give the final answer as \\boxed{...}."
)
# deepseek-chat cache-miss pricing (USD per 1M tokens), for the cost readout
PRICE_IN, PRICE_OUT = 0.27, 1.10


def load_math500(data_dir: Path, levels: set[int]) -> list[dict]:
    path = data_dir / "math500.jsonl"
    if not path.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
        print(f"downloading MATH-500 -> {path}")
        urllib.request.urlretrieve(MATH500_URL, path)
    rows = [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return [r for r in rows if int(r.get("level", 0)) in levels]


class CachedBackend(LLMBackend):
    """Disk-cached wrapper: serves the first ``n`` cached samples per prompt and
    only hits the inner backend for the shortfall. Cache is append-only JSON,
    one file per prompt hash."""

    def __init__(
        self, inner: OpenAIBackend, cache_dir: Path, shuffle_seed: int | None = None
    ) -> None:
        self.inner = inner
        self.cache_dir = cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.calls = 0  # samples served (cached + fresh)
        self.fresh = 0  # samples actually generated this run
        # a seed permutes each prompt's cached samples before serving, so
        # repeated runs draw different (but still cached) subsets -> cheap reps
        self.shuffle_seed = shuffle_seed

    def _path(self, prompt: str) -> Path:
        h = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:24]
        return self.cache_dir / f"{h}.json"

    def sample(self, prompt: str, n: int) -> list[str]:
        if n <= 0:
            return []
        path = self._path(prompt)
        cached: list[str] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
        if self.shuffle_seed is not None and cached:
            import random

            rng = random.Random(f"{self.shuffle_seed}:{path.stem}")
            order = list(range(len(cached)))
            rng.shuffle(order)
            served = [cached[i] for i in order]
            if len(served) >= n:
                self.calls += n
                return served[:n]
            # shortfall: generate fresh, append to the UNshuffled cache
            new = self.inner.sample(prompt, n - len(served))
            self.fresh += len(new)
            cached.extend(new)
            path.write_text(json.dumps(cached, ensure_ascii=False), encoding="utf-8")
            self.calls += n
            return served + new
        if len(cached) < n:
            new = self.inner.sample(prompt, n - len(cached))
            self.fresh += len(new)
            cached.extend(new)
            path.write_text(json.dumps(cached, ensure_ascii=False), encoding="utf-8")
        self.calls += n
        return cached[:n]


def read_key(args) -> str:
    """API key from --key-file, else $DEEPSEEK_KEY_FILE, else $DEEPSEEK_API_KEY."""
    if args.key_file:
        return Path(args.key_file).read_text(encoding="utf-8").strip()
    key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not key:
        raise SystemExit(
            "no API key: pass --key-file PATH, or set DEEPSEEK_KEY_FILE / "
            "DEEPSEEK_API_KEY in the environment"
        )
    return key


def build_backend(args) -> tuple[CachedBackend, OpenAIBackend]:
    key = read_key(args)
    inner = OpenAIBackend(
        model=args.model,
        base_url=DEEPSEEK_BASE_URL,
        api_key=key,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        extract_fn=extract_answer,
        system_prompt=SYSTEM_PROMPT,
        n_per_request=1,  # DeepSeek rejects n>1; concurrent single requests
        max_concurrency=args.concurrency,
    )
    return CachedBackend(inner, Path(args.cache_dir) / args.model), inner


def cost_line(inner: OpenAIBackend) -> str:
    usd = inner.prompt_tokens / 1e6 * PRICE_IN + inner.completion_tokens / 1e6 * PRICE_OUT
    return (
        f"fresh tokens: {inner.prompt_tokens} in / {inner.completion_tokens} out "
        f"(~${usd:.3f} at cache-miss prices)"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--key-file",
        default=os.environ.get("DEEPSEEK_KEY_FILE", ""),
        help="file containing the API key; defaults to $DEEPSEEK_KEY_FILE, "
        "else $DEEPSEEK_API_KEY is used directly",
    )
    ap.add_argument("--model", default="deepseek-chat")
    ap.add_argument("--levels", default="4,5")
    ap.add_argument("--n", type=int, default=50, help="number of problems")
    ap.add_argument("--budget", type=float, default=6.0, help="avg samples/query")
    ap.add_argument("--probe", type=int, default=4)
    ap.add_argument("--strategies", default="uniform,index")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--data-dir", default=str(Path(__file__).parent / "data"))
    ap.add_argument("--cache-dir", default=str(Path(__file__).parent / "data" / "cache"))
    ap.add_argument("--pilot", action="store_true", help="mu histogram only")
    ap.add_argument("--download-only", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    levels = {int(x) for x in args.levels.split(",")}
    rows = load_math500(Path(args.data_dir), levels)
    if args.download_only:
        print(f"dataset ready: {len(rows)} problems at levels {sorted(levels)}")
        return
    rows = rows[: args.n]
    prompts = [r["problem"] for r in rows]
    gold = {r["problem"]: str(r["answer"]) for r in rows}
    verifier = make_verifier(gold)

    if args.dry_run:
        per_call_tokens = args.max_tokens // 2  # rough completion estimate
        total = int(args.budget * len(prompts))
        est = total * per_call_tokens / 1e6 * PRICE_OUT
        print(f"{len(prompts)} problems, budget {args.budget} -> ~{total} samples")
        print(f"rough worst-case output cost: ~${est:.2f} (+ input tokens)")
        print(f"key file: {args.key_file} (exists: {os.path.exists(args.key_file)})")
        return

    backend, inner = build_backend(args)

    if args.pilot:
        k = args.probe
        print(f"pilot: {len(prompts)} problems x {k} samples")
        mus = []
        for p in prompts:
            answers = backend.sample(p, k)
            mus.append(sum(verifier(p, a) for a in answers) / k)
        buckets = [0] * (k + 1)
        for mu in mus:
            buckets[round(mu * k)] += 1
        print("estimated mu histogram (pass rate over probe samples):")
        for i, c in enumerate(buckets):
            print(f"  mu={i/k:.2f}: {'#' * c} ({c})")
        mid = sum(1 for mu in mus if 0.0 < mu < 1.0)
        print(f"problems strictly between 0 and 1: {mid}/{len(mus)} "
              f"(want a healthy middle; if not, change --levels)")
        print(cost_line(inner))
        return

    for strat in args.strategies.split(","):
        strat = strat.strip()
        runner = BudgetRunner(backend, strategy=strat, probe=args.probe, verifier=verifier)
        res = runner.run(prompts, budget=args.budget)
        acc = accuracy(res, gold, equal=math_equal)
        print(f"[{strat:8s}] acc={acc:.3f}  samples={res.total_samples} "
              f"(fresh this run: {backend.fresh})")
    print(cost_line(inner))


if __name__ == "__main__":
    main()
