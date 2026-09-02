"""Pareto frontiers over inference-knob configurations.

The allocator in :mod:`thrifty.core` answers "given a budget, where should it
go?". This module answers the question that comes *first*: "what budgets are even
worth considering?" -- i.e. for a fixed model and task, sweep the inference knobs
(samples ``n``, reasoning length, temperature, verifier on/off, ...) and map the
achievable quality-vs-cost set.

Three operations, in the order you use them:

1. :func:`pareto_front` -- drop every configuration that another configuration
   beats on *all* axes. Exact, no tuning parameter.
2. :func:`epsilon_front` -- thin the exact front down to configurations that are
   *meaningfully* different. A 40-point front separated by measurement noise is
   not a decision aid; a 5-point front is. See the guarantee below.
3. :func:`knee` -- pick one recommended operating point.

Objective directions are explicit: every axis is declared ``"max"`` or ``"min"``,
and everything internally is converted to minimisation.

**What ``epsilon_front`` is, and is not.** It is a *display* rule, not the
epsilon-Pareto approximation of Papadimitriou-Yannakakis (FOCS 2000) -- that is a
relative/multiplicative construction with a coverage guarantee against the true
frontier. This one is additive on a data-dependent scale, so the exact Pareto
front is the result and this is an optional thinning layer on top of it. Do not
describe its output as an epsilon-Pareto approximation set.

**What it does guarantee.** Objectives are first normalised to
``[0, 1]`` over the observed range, so ``eps`` is a fraction of the spread of the
data (``eps=0.05`` = "within 5% of the range on every axis"). A returned set
``A`` satisfies: for every point ``p`` on the exact Pareto front there is an
``a in A`` with ``a_i <= p_i + eps`` on every (minimisation-oriented) axis. This
is the additive ``eps``-Pareto set condition; it follows directly from the
acceptance test in the greedy loop, not from a separate argument.

**What it does not guarantee.** Minimum cardinality -- the greedy is not
size-optimal, and the retained set depends on the sweep order (cost-ascending
here, so ties break toward the cheaper configuration). Normalisation is over the
*observed* points, so adding a new extreme configuration rescales ``eps``.

The relative scale has one consequence worth knowing before you read a thinned
front: ``eps`` cannot express an *absolute* tolerance. A two-point front always
normalises to ``(0, 1)`` and ``(1, 0)`` whatever the raw numbers, so no ``eps``
below 1 will ever merge two configurations that differ by 0.01 accuracy if they
are the only two on the front. Thinning only becomes meaningful once the front
has enough points to establish a spread. If you need "0.01 accuracy is
negligible, full stop", filter on the raw metric before calling this.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

# Objective direction: "max" (quality-like) or "min" (cost-like).
Direction = str


@dataclass(frozen=True)
class Config:
    """One point of the sweep: a knob setting and the metrics it achieved.

    ``knobs`` is what you set (``{"n": 8, "temperature": 0.8}``); ``metrics`` is
    what you measured (``{"accuracy": 0.71, "tokens": 41200, "usd": 0.13}``).
    Kept separate so a frontier over cost can be re-plotted over dollars or
    latency without re-running the sweep.
    """

    knobs: Mapping[str, object]
    metrics: Mapping[str, float]
    label: str = ""

    def name(self) -> str:
        if self.label:
            return self.label
        return ", ".join(f"{k}={v}" for k, v in self.knobs.items())


def _oriented(
    configs: Sequence[Config], objectives: Mapping[str, Direction]
) -> list[tuple[float, ...]]:
    """Project each config onto the objective axes, all oriented to minimisation."""
    for name, direction in objectives.items():
        if direction not in ("max", "min"):
            raise ValueError(
                f"objective {name!r}: direction must be 'max' or 'min', got {direction!r}"
            )
    out = []
    for c in configs:
        missing = [k for k in objectives if k not in c.metrics]
        if missing:
            raise KeyError(f"config {c.name()!r} is missing metric(s) {missing}")
        out.append(
            tuple(
                -float(c.metrics[k]) if d == "max" else float(c.metrics[k])
                for k, d in objectives.items()
            )
        )
    return out


def _dominates(a: Sequence[float], b: Sequence[float]) -> bool:
    """``a`` dominates ``b``: no worse on every axis, strictly better on one."""
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))


def pareto_front(
    configs: Sequence[Config], objectives: Mapping[str, Direction]
) -> list[Config]:
    """The exact non-dominated set, in the input order.

    Duplicated objective vectors are all retained (they are mutually
    non-dominated); de-duplicate upstream if that is not what you want.
    """
    pts = _oriented(configs, objectives)
    keep = []
    for i, p in enumerate(pts):
        if not any(_dominates(q, p) for j, q in enumerate(pts) if j != i):
            keep.append(configs[i])
    return keep


def _normalise(pts: Sequence[Sequence[float]]) -> list[tuple[float, ...]]:
    """Scale each axis to ``[0, 1]`` over its observed range (flat axis -> 0)."""
    if not pts:
        return []
    m = len(pts[0])
    lo = [min(p[i] for p in pts) for i in range(m)]
    hi = [max(p[i] for p in pts) for i in range(m)]
    span = [hi[i] - lo[i] for i in range(m)]
    return [
        tuple(0.0 if span[i] == 0 else (p[i] - lo[i]) / span[i] for i in range(m))
        for p in pts
    ]


def epsilon_front(
    configs: Sequence[Config],
    objectives: Mapping[str, Direction],
    eps: float = 0.05,
) -> list[Config]:
    """Thin the exact Pareto front to configurations that differ by more than ``eps``.

    ``eps`` is in normalised units: a fraction of the observed spread on each
    axis. Points are visited cheapest-first on the *first* declared objective, so
    ties resolve toward the cheaper configuration.

    Returns a subset of :func:`pareto_front` satisfying the additive
    ``eps``-domination condition stated in the module docstring.
    """
    if not 0.0 <= eps < 1.0:
        raise ValueError(f"eps must be in [0, 1), got {eps}")
    front = pareto_front(configs, objectives)
    if not front:
        return []
    # Normalise over the *front* only: eps is a resolution on the frontier, and
    # including dominated interior points would inflate the range and silently
    # coarsen it.
    pts = _normalise(_oriented(front, objectives))
    order = sorted(range(len(front)), key=lambda i: pts[i][0])

    kept_pts: list[tuple[float, ...]] = []
    kept: list[Config] = []
    for i in order:
        p = pts[i]
        covered = any(all(q <= x + eps for q, x in zip(kp, p)) for kp in kept_pts)
        if not covered:
            kept_pts.append(p)
            kept.append(front[i])
    return kept


@dataclass
class Knee:
    """A recommended operating point plus why it was chosen."""

    config: Config
    method: str
    score: float
    front: list[Config] = field(default_factory=list)

    def explain(self) -> str:
        return (
            f"{self.config.name()} (knee by {self.method}, score={self.score:.4f}; "
            f"chosen from a {len(self.front)}-point frontier)"
        )


def knee(
    configs: Sequence[Config],
    objectives: Mapping[str, Direction],
    method: str = "chord",
    min_deviation: float = 1e-6,
) -> Knee | None:
    """Recommend one operating point on the frontier. Two-objective only.

    ``method="chord"`` (default): the front point furthest from the straight line
    joining the two extreme points, in normalised coordinates -- the standard
    "elbow", and the more robust of the two because it is invariant to where the
    sweep happened to place its grid points.

    ``method="marginal"``: the point after which the next unit of cost buys the
    least quality -- i.e. the last point whose incoming quality-per-cost slope is
    above the mean slope of the front. Closer to the "accuracy per token" reading
    but sensitive to grid spacing.

    Returns ``None`` rather than inventing a recommendation when the data does
    not support one: fewer than three front points (no interior point to
    prefer), or a front so close to straight that no point deviates from the
    chord by more than ``min_deviation`` in normalised units. A linear
    quality-cost trade-off genuinely has no elbow -- on a straight front the
    numerical winner is floating-point noise, which is exactly the kind of
    number that should not be reported as a recommendation.
    """
    if len(objectives) != 2:
        raise ValueError(f"knee requires exactly 2 objectives, got {len(objectives)}")
    if method not in ("chord", "marginal"):
        raise ValueError(f"unknown method: {method!r}")

    front = pareto_front(configs, objectives)
    if len(front) < 3:
        return None
    pts = _normalise(_oriented(front, objectives))
    order = sorted(range(len(front)), key=lambda i: (pts[i][0], pts[i][1]))
    xs = [pts[i][0] for i in order]
    ys = [pts[i][1] for i in order]

    if method == "chord":
        x0, y0, x1, y1 = xs[0], ys[0], xs[-1], ys[-1]
        dx, dy = x1 - x0, y1 - y0
        norm = (dx * dx + dy * dy) ** 0.5
        if norm == 0:
            return None
        best_k, best_d = None, -1.0
        for k in range(1, len(order) - 1):
            # perpendicular distance from the chord; both axes are minimisation,
            # so the frontier bows toward the origin and the knee is the point
            # furthest below the chord.
            d = ((x1 - x0) * (y0 - ys[k]) - (x0 - xs[k]) * (y1 - y0)) / norm
            if d > best_d:
                best_k, best_d = k, d
        if best_k is None or best_d <= min_deviation:
            return None
        return Knee(front[order[best_k]], "chord", best_d, front)

    # marginal: slopes along the front (both axes minimisation, so a useful step
    # trades cost up for objective-2 down => negative slope; steeper is better).
    slopes = []
    for k in range(1, len(order)):
        dx = xs[k] - xs[k - 1]
        slopes.append((ys[k] - ys[k - 1]) / dx if dx > 0 else float("-inf"))
    mean_slope = (ys[-1] - ys[0]) / (xs[-1] - xs[0]) if xs[-1] > xs[0] else 0.0
    best_k = 0
    for k, s in enumerate(slopes, start=1):
        if s <= mean_slope + min_deviation:
            best_k = k
    if best_k in (0, len(order) - 1):
        return None
    return Knee(front[order[best_k]], "marginal", slopes[best_k - 1], front)


def to_rows(
    configs: Iterable[Config], metrics: Sequence[str]
) -> list[dict[str, object]]:
    """Flatten configs into plain dicts -- for CSV export or a DataFrame."""
    rows = []
    for c in configs:
        row: dict[str, object] = {"config": c.name()}
        row.update(dict(c.knobs))
        for m in metrics:
            row[m] = c.metrics.get(m)
        rows.append(row)
    return rows
