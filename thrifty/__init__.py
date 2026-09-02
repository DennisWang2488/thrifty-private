"""thrifty -- budget-aware test-time compute allocation for LLMs.

Reach a target accuracy with fewer tokens by spending your sampling budget where
it buys the most: hard-but-doable queries get more samples, trivial and hopeless
ones get fewer. Backend-agnostic (any OpenAI-compatible API or local vLLM server).

    from thrifty import BudgetRunner, MockBackend

    runner = BudgetRunner(backend, strategy="index")
    result = runner.run(prompts, budget=4.0)   # avg 4 samples/query
"""

from .aggregate import majority_vote
from .backends import LLMBackend, MockBackend, OpenAIBackend
from .core import index_allocate, success_prob, uniform_allocate
from .cost import CountingBackend, Usage, estimate_tokens
from .difficulty import OracleHazard, self_consistency_hazard
from .frontier import Config, Knee, epsilon_front, knee, pareto_front, to_rows
from .mathv import extract_answer, math_equal
from .runner import BatchResult, BudgetRunner, QueryResult, accuracy

__version__ = "0.2.0"

__all__ = [
    "BudgetRunner",
    "BatchResult",
    "QueryResult",
    "accuracy",
    "LLMBackend",
    "MockBackend",
    "OpenAIBackend",
    "self_consistency_hazard",
    "OracleHazard",
    "index_allocate",
    "uniform_allocate",
    "success_prob",
    "majority_vote",
    # frontier / cost profiling
    "Config",
    "pareto_front",
    "epsilon_front",
    "knee",
    "Knee",
    "to_rows",
    "CountingBackend",
    "Usage",
    "estimate_tokens",
    # math answer verification (real-benchmark helpers)
    "extract_answer",
    "math_equal",
]
