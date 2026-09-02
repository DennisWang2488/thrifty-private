"""Tests for the frontier / cost-accounting layer."""

from __future__ import annotations

import random

import pytest

from thrifty import (
    Config,
    CountingBackend,
    MockBackend,
    Usage,
    epsilon_front,
    estimate_tokens,
    knee,
    pareto_front,
    to_rows,
)
from thrifty.frontier import _normalise, _oriented

OBJ = {"tokens": "min", "accuracy": "max"}


def cfg(n, tokens, acc):
    return Config(knobs={"n": n}, metrics={"tokens": tokens, "accuracy": acc})


# --------------------------------------------------------------------------- #
# pareto_front
# --------------------------------------------------------------------------- #


def test_pareto_drops_dominated():
    a = cfg(1, 100, 0.50)
    b = cfg(2, 200, 0.60)
    dominated = cfg(3, 300, 0.55)  # costs more than b, scores less
    front = pareto_front([a, b, dominated], OBJ)
    assert dominated not in front
    assert front == [a, b]


def test_pareto_keeps_all_when_no_domination():
    pts = [cfg(1, 100, 0.5), cfg(2, 200, 0.7), cfg(3, 400, 0.9)]
    assert pareto_front(pts, OBJ) == pts


def test_pareto_keeps_duplicate_objective_vectors():
    a, b = cfg(1, 100, 0.5), cfg(2, 100, 0.5)
    assert len(pareto_front([a, b], OBJ)) == 2


def test_pareto_preserves_input_order():
    pts = [cfg(3, 400, 0.9), cfg(1, 100, 0.5), cfg(2, 200, 0.7)]
    assert pareto_front(pts, OBJ) == pts


def test_direction_matters():
    pts = [cfg(1, 100, 0.5), cfg(2, 200, 0.9)]
    # treating accuracy as a cost flips which point survives
    assert pareto_front(pts, {"tokens": "min", "accuracy": "min"}) == [pts[0]]


def test_bad_direction_rejected():
    with pytest.raises(ValueError):
        pareto_front([cfg(1, 1, 1)], {"tokens": "minimise"})


def test_missing_metric_names_the_config():
    bad = Config(knobs={"n": 1}, metrics={"tokens": 10.0})
    with pytest.raises(KeyError, match="accuracy"):
        pareto_front([bad], OBJ)


def test_empty_input():
    assert pareto_front([], OBJ) == []
    assert epsilon_front([], OBJ) == []
    assert knee([], OBJ) is None


# --------------------------------------------------------------------------- #
# epsilon_front -- the guarantee is checked directly, not assumed
# --------------------------------------------------------------------------- #


def _eps_covers(kept, front, objectives, eps):
    """Every exact-front point is within +eps of some kept point on every axis."""
    pts = _normalise(_oriented(front, objectives))
    idx = {id(c): i for i, c in enumerate(front)}
    kept_pts = [pts[idx[id(c)]] for c in kept]
    return all(
        any(all(q <= x + eps + 1e-12 for q, x in zip(kp, p)) for kp in kept_pts)
        for p in pts
    )


def test_epsilon_front_is_a_subset_of_the_exact_front():
    pts = [cfg(i, 100 * i, 0.5 + 0.04 * i) for i in range(1, 12)]
    kept = epsilon_front(pts, OBJ, eps=0.2)
    front = pareto_front(pts, OBJ)
    assert all(c in front for c in kept)


def test_epsilon_front_thins_a_dense_front():
    # 40 points along a smooth frontier -> a handful of meaningfully distinct ones
    pts = [cfg(i, 50 * i, 1 - 1 / (1 + 0.06 * i)) for i in range(1, 41)]
    assert len(pareto_front(pts, OBJ)) == 40
    kept = epsilon_front(pts, OBJ, eps=0.15)
    assert 2 <= len(kept) < 40


def test_epsilon_zero_is_the_exact_front():
    pts = [cfg(i, 100 * i, 0.5 + 0.04 * i) for i in range(1, 8)]
    assert len(epsilon_front(pts, OBJ, eps=0.0)) == len(pareto_front(pts, OBJ))


def test_epsilon_guarantee_holds_on_random_sweeps():
    rng = random.Random(7)
    for trial in range(60):
        pts = [
            cfg(i, rng.uniform(10, 5000), rng.uniform(0.1, 0.95))
            for i in range(rng.randint(3, 30))
        ]
        for eps in (0.02, 0.1, 0.3):
            kept = epsilon_front(pts, OBJ, eps=eps)
            assert kept, f"trial {trial}: eps-front must be non-empty"
            assert _eps_covers(kept, pareto_front(pts, OBJ), OBJ, eps)


def test_epsilon_ties_break_toward_the_cheaper_config():
    # two near-duplicate configs plus a distant one: the cheaper of the pair wins
    cheap, near_dup, distant = cfg(1, 100, 0.80), cfg(2, 140, 0.81), cfg(8, 1000, 0.95)
    kept = epsilon_front([cheap, near_dup, distant], OBJ, eps=0.2)
    assert kept == [cheap, distant]


def test_epsilon_cannot_express_an_absolute_tolerance_on_a_two_point_front():
    # documented limitation: eps is a fraction of the observed spread, and a
    # two-point front always normalises to (0,1) and (1,0) whatever the raw gap.
    cheap, pricey = cfg(1, 100, 0.80), cfg(8, 900, 0.81)
    assert len(epsilon_front([cheap, pricey], OBJ, eps=0.99)) == 2


def test_epsilon_out_of_range_rejected():
    with pytest.raises(ValueError):
        epsilon_front([cfg(1, 1, 1)], OBJ, eps=1.0)
    with pytest.raises(ValueError):
        epsilon_front([cfg(1, 1, 1)], OBJ, eps=-0.1)


# --------------------------------------------------------------------------- #
# knee
# --------------------------------------------------------------------------- #


def test_knee_finds_the_elbow():
    # accuracy saturates after n=4: cheap gains, then a long flat tail
    pts = [
        cfg(1, 100, 0.50),
        cfg(2, 200, 0.68),
        cfg(4, 400, 0.80),
        cfg(8, 800, 0.82),
        cfg(16, 1600, 0.83),
    ]
    k = knee(pts, OBJ)
    assert k is not None
    assert k.config.knobs["n"] == 4
    assert "knee by chord" in k.explain()


def test_knee_needs_an_interior_point():
    assert knee([cfg(1, 100, 0.5), cfg(2, 200, 0.9)], OBJ) is None


def test_knee_on_a_straight_front_declines_to_recommend():
    # perfectly linear trade-off: no elbow exists, so none is invented
    pts = [cfg(i, 100 * i, 0.1 * i) for i in range(1, 6)]
    assert knee(pts, OBJ) is None


def test_knee_marginal_method_runs():
    pts = [
        cfg(1, 100, 0.50),
        cfg(2, 200, 0.68),
        cfg(4, 400, 0.80),
        cfg(8, 800, 0.82),
        cfg(16, 1600, 0.83),
    ]
    k = knee(pts, OBJ, method="marginal")
    assert k is not None and k.method == "marginal"


def test_knee_rejects_wrong_arity_and_method():
    pts = [cfg(1, 100, 0.5)]
    with pytest.raises(ValueError):
        knee(pts, {"tokens": "min", "accuracy": "max", "seconds": "min"})
    with pytest.raises(ValueError):
        knee(pts, OBJ, method="magic")


# --------------------------------------------------------------------------- #
# cost accounting
# --------------------------------------------------------------------------- #


def test_usage_adds_and_propagates_the_estimated_flag():
    measured = Usage(samples=1, prompt_tokens=10, completion_tokens=20)
    guessed = Usage(samples=1, prompt_tokens=5, completion_tokens=5, estimated=True)
    total = measured + guessed
    assert total.samples == 2 and total.total_tokens == 40
    assert total.estimated is True
    assert (measured + measured).estimated is False


def test_usd_uses_the_price_table():
    u = Usage(prompt_tokens=1_000_000, completion_tokens=1_000_000)
    assert u.usd("gpt-4o-mini") == pytest.approx(0.15 + 0.60)
    assert u.usd("local") == 0.0


def test_usd_refuses_an_unpriced_model():
    with pytest.raises(KeyError, match="no price"):
        Usage(prompt_tokens=100).usd("mystery-model-9")
    assert Usage(prompt_tokens=1_000_000).usd(
        "mystery-model-9", prices={"mystery-model-9": (1.0, 2.0)}
    ) == pytest.approx(1.0)


def test_counting_backend_estimates_when_the_server_reports_nothing():
    inner = MockBackend({"q": 0.5})
    counting = CountingBackend(inner)
    counting.sample("q", 6)
    u = counting.usage()
    assert u.samples == 6
    assert u.estimated is True  # MockBackend reports no usage
    assert u.completion_tokens > 0 and u.seconds >= 0.0


def test_counting_backend_prefers_reported_usage():
    class Reporting(MockBackend):
        def sample(self, prompt, n):
            out = super().sample(prompt, n)
            self.last_usage = Usage(prompt_tokens=7 * n, completion_tokens=11 * n)
            return out

    counting = CountingBackend(Reporting({"q": 0.5}))
    counting.sample("q", 3)
    u = counting.usage()
    assert (u.prompt_tokens, u.completion_tokens) == (21, 33)
    assert u.estimated is False


def test_counting_backend_reset_and_zero_n():
    counting = CountingBackend(MockBackend({"q": 0.5}))
    counting.sample("q", 4)
    counting.reset()
    assert counting.usage().samples == 0
    assert counting.sample("q", 0) == []
    assert counting.usage().samples == 0


def test_usage_as_metrics_feeds_a_config():
    u = Usage(samples=4, prompt_tokens=100, completion_tokens=400, seconds=1.5)
    c = Config(knobs={"n": 4}, metrics=u.as_metrics(model="gpt-4o-mini", accuracy=0.7))
    assert c.metrics["tokens"] == 500.0 and c.metrics["accuracy"] == 0.7
    assert pareto_front([c], OBJ) == [c]


def test_estimate_tokens_is_never_zero():
    assert estimate_tokens("") == 1
    assert estimate_tokens("a" * 400) == 100


def test_to_rows_flattens_knobs_and_metrics():
    rows = to_rows([cfg(4, 400, 0.8)], ["tokens", "accuracy", "usd"])
    assert rows[0]["n"] == 4 and rows[0]["accuracy"] == 0.8
    assert rows[0]["usd"] is None  # absent metrics surface as None, not an error
