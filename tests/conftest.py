"""
Package-wide test fixtures for clophfit.

If you don't know what this is for, just leave it empty.
Read more about conftest.py under:
- https://docs.pytest.org/en/stable/fixture.html
- https://docs.pytest.org/en/stable/writing_plugins.html
"""

import numpy as np
import pytest

from clophfit.fitting.data_structures import DataArray, Dataset


@pytest.fixture
def ph_dataset() -> Dataset:
    """Create a sample pH dataset."""
    x = np.array([6.0, 7.0, 8.0, 9.0, 10.0])
    # K=7, S0=2, S1=1
    # y = 2 + (1-2)/(1+10^(7-x)) = 2 - 1/(1+10^(7-x))
    # x=6: 2 - 1/(1+10) = 2 - 1/11 = 1.909
    # x=7: 2 - 1/2 = 1.5
    # x=8: 2 - 1/(1+0.1) = 2 - 1/1.1 = 1.09
    y = np.array([1.90909091, 1.5, 1.09090909, 1.00990099, 1.000999])
    da = DataArray(x, y)
    return Dataset({"default": da}, is_ph=True)


@pytest.fixture
def cl_dataset() -> Dataset:
    """Create a sample Cl dataset."""
    x = np.array([0.0, 10.0, 20.0, 30.0])
    # K=10, S0=2, S1=0
    # y = 2 + (0-2) * x / (10 + x) = 2 - 2x/(10+x)
    # x=0: 2
    # x=10: 2 - 20/20 = 1
    # x=20: 2 - 40/30 = 2 - 1.33 = 0.66
    # x=30: 2 - 60/40 = 0.5
    y = np.array([2.0, 1.0, 0.66666667, 0.5])
    da = DataArray(x, y)
    return Dataset({"default": da}, is_ph=False)


@pytest.fixture
def multi_dataset() -> Dataset:
    """Create a multi-label dataset."""
    x = np.array([6.0, 7.0, 8.0])
    # y1: K=7, S0=2, S1=1
    y1 = np.array([1.90909091, 1.5, 1.09090909])
    # y2: example reference curve for K=7, S0=0, S1=1
    y2 = np.array([0.09090909, 0.5, 0.90909091])
    return Dataset({"y1": DataArray(x, y1), "y2": DataArray(x, y2)}, is_ph=True)
