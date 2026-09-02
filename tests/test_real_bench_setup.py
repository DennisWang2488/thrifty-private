"""Offline tests for the real-benchmark plumbing: math verification and
prompt-aware verifiers. No network, no API key."""

from thrifty import BudgetRunner, MockBackend, accuracy
from thrifty.mathv import extract_answer, extract_boxed, math_equal, make_verifier


def test_extract_boxed():
    assert extract_boxed(r"so the answer is \boxed{42}.") == "42"
    assert extract_boxed(r"\boxed{\frac{1}{2}}") == r"\frac{1}{2}"
    # nested braces, and the LAST box wins
    assert extract_boxed(r"\boxed{a_{1}} then \boxed{x^{2}+1}") == "x^{2}+1"
    assert extract_boxed("no box here") is None


def test_extract_answer_fallbacks():
    assert extract_answer(r"steps... \boxed{7}") == "7"
    assert extract_answer("the result is 3.5, done") == "3.5"
    assert extract_answer("just words\nfinal line") == "final line"


def test_math_equal():
    assert math_equal("42", "42")
    assert math_equal(" 42. ", "42")
    assert math_equal("1/2", r"\frac{1}{2}")
    assert math_equal("0.5", "1/2")
    assert math_equal("1,000", "1000")
    assert math_equal(r"\text{yes}", "yes")
    assert not math_equal("41", "42")
    assert not math_equal("1/3", "0.5")


def test_prompt_aware_verifier_end_to_end():
    # two "problems" with different gold answers; MockBackend's gold token is
    # shared, so map it per prompt via a prompt-aware verifier instead.
    mus = {"p_easy": 0.9, "p_mid": 0.4, "p_hard": 0.05}
    backend = MockBackend(mus)
    gold = {p: "CORRECT" for p in mus}
    verifier = make_verifier(gold)  # (prompt, answer) -> bool

    runner = BudgetRunner(backend, strategy="index", probe=2, verifier=verifier)
    res = runner.run(list(mus), budget=4.0)
    assert res.total_samples == 12
    acc = accuracy(res, gold, equal=math_equal)
    assert 0.0 <= acc <= 1.0
    # easy problem should be solved
    by_prompt = {r.prompt: r for r in res.results}
    assert by_prompt["p_easy"].answer == "CORRECT"


def test_one_arg_verifier_still_works():
    mus = {"a": 0.8, "b": 0.3}
    backend = MockBackend(mus)
    runner = BudgetRunner(
        backend, strategy="index", probe=2, verifier=lambda ans: ans == "CORRECT"
    )
    res = runner.run(list(mus), budget=3.0)
    assert res.total_samples == 6
