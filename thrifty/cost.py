"""Cost accounting for a sweep: tokens, dollars, wall-clock.

v0.1 of the library counted *samples*, which is only the right cost unit when
every sample is the same length. It is not: the whole point of a reasoning knob
is that samples get longer. This module wraps any
:class:`~thrifty.backends.LLMBackend` and records what each call actually spent,
so a frontier can be plotted against tokens or dollars rather than a sample
count that hides the real spend.

Token counts come from the server when the backend reports them and are
*estimated* otherwise. The distinction is not cosmetic -- an estimated cost axis
should not be published as a measured one -- so :attr:`Usage.estimated` tracks
it and :meth:`CountingBackend.usage` refuses to silently mix the two.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace

from .backends import LLMBackend

# USD per 1M tokens, (prompt, completion). Prices move; this is a convenience
# table for the demo, not a source of truth -- pass your own to `price_of`.
PRICES: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "local": (0.0, 0.0),
}


@dataclass(frozen=True)
class Usage:
    """What a call (or a batch of calls) cost."""

    samples: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    seconds: float = 0.0
    estimated: bool = False
    """True if any token count in this total was estimated rather than reported."""

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            samples=self.samples + other.samples,
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            seconds=self.seconds + other.seconds,
            estimated=self.estimated or other.estimated,
        )

    def usd(self, model: str, prices: dict[str, tuple[float, float]] | None = None) -> float:
        """Dollar cost under a price table. Raises if the model is not priced."""
        table = prices if prices is not None else PRICES
        if model not in table:
            raise KeyError(
                f"no price for {model!r}; pass prices={{{model!r}: (in, out)}} "
                "in USD per 1M tokens"
            )
        p_in, p_out = table[model]
        return (self.prompt_tokens * p_in + self.completion_tokens * p_out) / 1e6

    def as_metrics(self, model: str | None = None, **extra: float) -> dict[str, float]:
        """Flatten into the ``metrics`` dict a :class:`~thrifty.frontier.Config` wants."""
        m: dict[str, float] = {
            "samples": float(self.samples),
            "tokens": float(self.total_tokens),
            "completion_tokens": float(self.completion_tokens),
            "seconds": self.seconds,
        }
        if model is not None:
            m["usd"] = self.usd(model)
        m.update(extra)
        return m


def estimate_tokens(text: str) -> int:
    """Rough token count for backends that report none: ~4 chars per token.

    Deliberately crude. Use it to get a sweep running, not to publish a cost
    axis -- anything derived from it is flagged ``estimated=True``.
    """
    return max(1, round(len(text) / 4))


class CountingBackend(LLMBackend):
    """Wrap a backend and record tokens, wall-clock and sample counts per call.

    Usage is accumulated until :meth:`reset`, which is what a sweep calls between
    knob settings::

        counting = CountingBackend(OpenAIBackend("gpt-4o-mini"), model="gpt-4o-mini")
        for n in (1, 2, 4, 8):
            counting.reset()
            ...  # run the batch at n samples
            metrics = counting.usage().as_metrics(model="gpt-4o-mini")

    If the wrapped backend exposes ``last_usage`` (a :class:`Usage` set by its
    most recent ``sample`` call, as :class:`~thrifty.backends.OpenAIBackend` does
    when the server reports usage), those counts are used verbatim. Otherwise
    token counts are estimated from the returned text and the prompt.
    """

    def __init__(self, inner: LLMBackend, model: str | None = None) -> None:
        self.inner = inner
        self.model = model
        self._usage = Usage()

    def reset(self) -> None:
        self._usage = Usage()

    def usage(self) -> Usage:
        return self._usage

    def sample(self, prompt: str, n: int) -> list[str]:
        if n <= 0:
            return []
        t0 = time.perf_counter()
        answers = self.inner.sample(prompt, n)
        elapsed = time.perf_counter() - t0

        reported: Usage | None = getattr(self.inner, "last_usage", None)
        if reported is not None:
            self._usage += replace(reported, samples=n, seconds=elapsed)
        else:
            self._usage += Usage(
                samples=n,
                prompt_tokens=estimate_tokens(prompt) * n,
                completion_tokens=sum(estimate_tokens(a) for a in answers),
                seconds=elapsed,
                estimated=True,
            )
        return answers
