"""LLM backends: anything that can draw ``n`` samples for a prompt.

The allocator is backend-agnostic. Implement :class:`LLMBackend` for your endpoint
(an OpenAI-compatible API, a local vLLM server, ...) and the same allocation logic
runs on top. :class:`MockBackend` is a fully offline simulator used for tests,
tutorials, and reproducing the savings curve without spending a cent.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
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
        self.calls = 0
        self.last_usage = None
        """Server-reported token usage for the most recent call, or ``None`` if
        the server reported none. :class:`~thrifty.cost.CountingBackend` reads
        this and falls back to estimating when it is ``None``."""

    def sample(self, prompt: str, n: int) -> list[str]:
        if n <= 0:
            return []
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})
        # Most servers support n>1 in one request; fall back to a loop if not.
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            n=n,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        self.calls += n
        # Imported here rather than at module scope: cost.py wraps backends, so
        # a top-level import would be circular.
        from .cost import Usage

        u = getattr(resp, "usage", None)
        self.last_usage = (
            Usage(
                samples=n,
                prompt_tokens=getattr(u, "prompt_tokens", 0) or 0,
                completion_tokens=getattr(u, "completion_tokens", 0) or 0,
            )
            if u is not None
            else None
        )
        return [self.extract_fn(c.message.content or "") for c in resp.choices]


def _last_line(text: str) -> str:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[-1] if lines else ""
