"""Test cases for the binding functions module."""

import numpy as np

from clophfit import binding


def test_kd() -> None:
    """It returns int float np.ndarray[float]."""
    assert binding.kd(10, 7, 6.0) == 11
    assert binding.kd(10, 7, 7) == 20
    assert binding.kd(10, 7, 8) == 110.0
    np.testing.assert_array_equal(
        binding.kd(10.0, 7.0, np.array([6.0, 8.0])), np.array([11.0, 110.0])
    )
    np.testing.assert_array_equal(
        binding.kd(10, 7, np.array([7, 6, 8.0])), np.array([20, 11, 110])
    )
