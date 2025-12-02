"""Testing utilities for ClopHfit.

This package contains shared utilities for testing and benchmarking fitting algorithms,
including synthetic data generation and comparison tools.
"""

from clophfit.testing.fitter_test_utils import (
    Truth,
    build_fitters,
    k_from_result,
    make_synthetic_ds,
    s_from_result,
)
from clophfit.testing.synthetic import (
    STRESS_SCENARIOS,
    StressScenario,
    TruthParams,
    make_dataset,
    make_realistic_dataset,
    make_simple_dataset,
    make_stress_dataset,
    plot_synthetic_dataset,
)

__all__ = [
    "STRESS_SCENARIOS",
    "StressScenario",
    "Truth",
    "TruthParams",
    "build_fitters",
    "k_from_result",
    "make_dataset",
    "make_realistic_dataset",
    "make_simple_dataset",
    "make_stress_dataset",
    "make_synthetic_ds",
    "plot_synthetic_dataset",
    "s_from_result",
]
