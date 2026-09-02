"""LLM backends: anything that can draw ``n`` samples for a prompt.

The allocator is backend-agnostic. Implement :class:`LLMBackend` for your endpoint
(an OpenAI-compatible API, a local vLLM server, ...) and the same allocation logic
runs on top. :class:`MockBackend` is a fully offline simulator used for tests,
tutorials, and reproducing the savings curve without spending a cent.
"""

from __future__ import annotations

import random
import threading
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import Sequence


class LLMBackend(ABC):
    """A source of samples. ``sample(prompt, n)`` returns ``n`` extracted answers."""

    @abstractmethod
    def sample(self, prompt: str, n: int) -> list[str]:
        """Return ``n`` answer strings for ``prompt`` (already parsed to the final
        answer, e.g. the boxed value for a math problem)."""


class MockBackend(LLMBackend):
    """Offline simulator: each prompt has a hidden per-sample success prob ``mu``.

    A sample returns the gold token with prob ``mu`` and one of ``n_distractors``
    wrong tokens otherwise (wrong answers are spread, so self-consistency among
    samples tracks ``mu`` -- which is what the difficulty signal exploits). The
    hidden ``mu`` is used ONLY to generate samples; the allocator never sees it.
    """

    def __init__(
        self,
        mus: dict[str, float],
        gold_token: str = "CORRECT",
        n_distractors: int = 4,
        seed: int = 0,
    ) -> None:
        self.mus = mus
        self.gold_token = gold_token
        self.n_distractors = n_distractors
        self._rng = random.Random(seed)
        self.calls = 0  # total samples drawn, for honest cost accounting

    def sample(self, prompt: str, n: int) -> list[str]:
        mu = self.mus[prompt]
        out: list[str] = []
        for _ in range(n):
            if self._rng.random() < mu:
                out.append(self.gold_token)
            else:
                out.append(f"WRONG_{self._rng.randrange(self.n_distractors)}")
        self.calls += n
        return out


class OpenAIBackend(LLMBackend):
    """OpenAI-compatible chat completions backend (best-of-n via ``n`` samples).

    Requires ``openai`` and an API key. ``extract_fn`` parses a raw completion into
    the comparable final answer (default: last non-empty line). Works against any
    OpenAI-compatible server, including a local vLLM/SGLang endpoint, by setting
    ``base_url``.

    Some servers (DeepSeek among them) reject ``n > 1``; set ``n_per_request=1``
    (or leave the default auto-detect) and the backend draws the ``n`` samples as
    concurrent single-sample requests instead. Cumulative token usage is tracked
    in ``prompt_tokens`` / ``completion_tokens`` so cost can be accounted in
    tokens, not just sample counts.
    """

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.8,
        max_tokens: int = 1024,
        extract_fn=None,
        system_prompt: str | None = None,
        n_per_request: int | None = None,
        max_concurrency: int = 8,
        max_retries: int = 3,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as e:  # pragma: no cover - optional dependency
            raise ImportError("pip install openai to use OpenAIBackend") from e
        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.extract_fn = extract_fn or _last_line
        self.system_prompt = system_prompt
        self.n_per_request = n_per_request  # None = auto-detect batched-n support
        self.max_concurrency = max_concurrency
        self.max_retries = max_retries
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.last_usage = None
        """Server-reported token usage for the most recent :meth:`sample` call,
        aggregated over its requests, or ``None`` if the server reported none.
        :class:`~thrifty.cost.CountingBackend` reads this and falls back to
        estimating when it is ``None``."""
        self._usage_lock = threading.Lock()

    def _messages(self, prompt: str) -> list[dict]:
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages

    def _request(self, prompt: str, n: int) -> tuple[list[str], int, int]:
        """One request for ``n`` samples; returns (answers, prompt_tok, compl_tok)."""
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=self._messages(prompt),
                    n=n,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                u = getattr(resp, "usage", None)
                p_tok = (getattr(u, "prompt_tokens", 0) or 0) if u is not None else 0
                c_tok = (getattr(u, "completion_tokens", 0) or 0) if u is not None else 0
                with self._usage_lock:
                    self.prompt_tokens += p_tok
                    self.completion_tokens += c_tok
                    self.calls += n
                answers = [self.extract_fn(c.message.content or "") for c in resp.choices]
                return answers, p_tok, c_tok
            except Exception as e:  # transient server/network errors
                last_err = e
                time.sleep(2**attempt)
        raise last_err  # type: ignore[misc]

    def _finish(self, n: int, p_tok: int, c_tok: int) -> None:
        # Imported here rather than at module scope: cost.py wraps backends, so
        # a top-level import would be circular.
        from .cost import Usage

        self.last_usage = (
            Usage(samples=n, prompt_tokens=p_tok, completion_tokens=c_tok)
            if (p_tok or c_tok)
            else None
        )

    def sample(self, prompt: str, n: int) -> list[str]:
        if n <= 0:
            return []
        if self.n_per_request is None:
            # auto-detect: try one batched-n request, fall back to a loop.
            if n == 1:
                answers, p, c = self._request(prompt, 1)
                self._finish(n, p, c)
                return answers
            try:
                answers, p, c = self._request(prompt, n)
                self.n_per_request = n  # remember that batching works
                self._finish(n, p, c)
                return answers
            except Exception:
                self.n_per_request = 1
        if self.n_per_request >= n:
            answers, p, c = self._request(prompt, n)
            self._finish(n, p, c)
            return answers
        # concurrent requests, each for at most ``n_per_request`` samples
        per = max(1, self.n_per_request)
        chunks = [per] * (n // per) + ([n % per] if n % per else [])
        with ThreadPoolExecutor(max_workers=self.max_concurrency) as ex:
            parts = list(ex.map(lambda k: self._request(prompt, k), chunks))
        answers = [a for part in parts for a in part[0]]
        self._finish(n, sum(p[1] for p in parts), sum(p[2] for p in parts))
        return answers


def _last_line(text: str) -> str:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[-1] if lines else ""
