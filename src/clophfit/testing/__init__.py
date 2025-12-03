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
from clophfit.testing.synthetic import TruthParams, make_dataset, make_simple_dataset

__all__ = [
    "Truth",
    "TruthParams",
    "build_fitters",
    "k_from_result",
    "make_dataset",
    "make_simple_dataset",
    "make_synthetic_ds",
    "s_from_result",
]
