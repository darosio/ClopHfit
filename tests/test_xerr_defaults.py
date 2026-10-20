"""Figures must not borrow the +-1 default of the x_err property.

DataArray.x_err materializes +-1 when no x-error array exists (the ODR and
MCMC paths weight off that); plot_fit used to hand that phantom +-1 to
errorbar, so every x-error-free spectrum fit drew caplines exactly one pH
unit off each point - reported from L1-L4 as "huge delta x".
"""

from __future__ import annotations

import numpy as np
from lmfit import Parameters  # type: ignore[import-untyped]
from matplotlib.figure import Figure

from clophfit.fitting.data_structures import DataArray, Dataset
from clophfit.fitting.plotting import plot_fit


def test_plot_fit_draws_no_x_bars_without_x_err() -> None:
    """The L1-L4 regression: caplines sat exactly 1 pH off every point."""
    x = np.linspace(5.0, 9.0, 12)
    y = 1.0 / (1.0 + np.power(10.0, 7.0 - x))
    da = DataArray(x, y, y_errc=np.full(12, 0.01))
    assert da.x_errc.size == 0
    params = Parameters()
    params.add("K", value=7.0)
    params.add("S0_default", value=0.0)
    params.add("S1_default", value=1.0)
    fig = Figure()
    ax = fig.add_subplot()
    plot_fit(ax, Dataset({"default": da}, is_ph=True), params)
    for line in ax.lines:
        xd = np.asarray(line.get_xdata(), dtype=float)
        if xd.size == x.size:
            assert np.abs(xd - x).max() < 1e-9, "line offset from the data in x"
    assert da.x_errc.size == 0  # plotting must not arm the +-1 materialization
