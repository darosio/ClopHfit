"""Test cases for the binding functions module."""
from __future__ import annotations

import numpy as np
import pytest

from clophfit import binding


def test_kd() -> None:
    """It returns int float np.ndarray[float]."""
    assert binding.kd(10, 7, 6.0) == 11
    assert binding.kd(10, 7, 7) == 20
    assert binding.kd(10, 7, 8) == 110.0
    np.testing.assert_allclose(
        binding.kd(10.0, 7.0, np.array([6.0, 8.0])), [11.0, 110.0]
    )
    np.testing.assert_allclose(binding.kd(10, 7, np.array([7, 6, 8.0])), [20, 11, 110])


def test_fit_titration() -> None:
    """It fits pH and Cl titrations."""
    x = [3.0, 5, 7, 9, 11.0]
    y = np.array([1.9991, 1.991, 1.5, 1.009, 1.0009])
    fit = binding.fit_titration("pH", x, y)
    assert abs(fit["K"][0] - 7) < 0.0000000001
    assert abs(fit["SA"][0] - 2) < 0.0001
    assert abs(fit["SB"][0] - 1) < 0.0001
    x = [0, 5.0, 10, 40, 160, 1000]
    y = np.array([2.0, 1.33333333, 1.0, 0.4, 0.11764706, 0.01980198])
    fit = binding.fit_titration("Cl", x, y)
    assert abs(fit["K"][0] - 10) < 0.0000001
    assert abs(fit["SA"][0] - 2) < 0.000000001
    assert abs(fit["SB"][0] - 0) < 0.00000001
    with pytest.raises(NameError, match="kind= pH or Cl"):
        binding.fit_titration("unk", x, y)
