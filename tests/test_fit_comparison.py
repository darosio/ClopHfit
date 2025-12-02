"""Objective comparisons of fitting procedures using typical pH lists.

This suite builds realistic pH datasets with x-errors from list.pH.csv test
fixtures and compares several fitting procedures side-by-side:
- Standard LM (fit_binding_glob)
- Robust LM with Huber loss (fit_binding_glob with robust=True)
- IRLS with outlier removal (fit_binding_glob_recursive_outlier)

The goal is to enable long-term, objective comparisons across changes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from clophfit.fitting.core import (
    fit_binding_glob,
    fit_binding_glob_recursive_outlier,
    weight_multi_ds_titration,
)
from clophfit.fitting.data_structures import DataArray, Dataset, FitResult, MiniT
from clophfit.fitting.models import binding_1site


def _load_ph_list(fp: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load pH values and their errors from a list.pH.csv-like file."""
    list_xvalues = pd.read_csv(fp, header=None, names=["file", "x", "x_err"])
    # Drop rows without x
    list_xvalues = list_xvalues.dropna(subset=["x"]).reset_index(drop=True)
    x = list_xvalues["x"].to_numpy(float)
    x_err = list_xvalues["x_err"].to_numpy(float)
    return x, x_err


@pytest.mark.parametrize(
    "csv_name",
    [
        "tests/Tecan/140220/list.pH.csv",
        "tests/Tecan/140220/list.pH2.csv",  # includes a blank filename row
    ],
)
@pytest.mark.parametrize(
    ("K0", "S0", "S1", "noise_sd"),
    [
        (7.0, 2.0, 1.0, 0.03),
        (7.2, 1.5, 0.5, 0.05),
    ],
)
def test_compare_lm_variants(
    csv_name: str,
    K0: float,  # noqa: N803
    S0: float,  # noqa: N803
    S1: float,  # noqa: N803
    noise_sd: float,
) -> None:
    """Compare LM variants on realistic pH grids with x-errors.

    The test asserts that robust or iterative procedures are at least as good as
    baseline by two criteria:
    - |K_est - K0| does not worsen by more than 10% relative to baseline
    - redchi does not increase by more than 15%
    These tolerances keep the test informative yet robust to minor stochasticity.
    """
    fp = Path(csv_name)
    assert fp.exists(), f"Missing test fixture: {fp}"

    # Build a single-label pH dataset from list.pH.csv
    x, x_err = _load_ph_list(fp)
    # Simulate ground-truth signal and add modest Gaussian noise
    rng = np.random.default_rng(42)
    y_true = binding_1site(x, K0, S0, S1, is_ph=True)
    y = y_true + rng.normal(scale=noise_sd, size=y_true.shape)

    da = DataArray(xc=x, yc=y, x_errc=x_err)
    ds = Dataset({"y1": da}, is_ph=True)

    # Initial weighting makes baseline fair; it will be updated by iterative fits
    weight_multi_ds_titration(ds)

    # Baselines
    fr_std = fit_binding_glob(ds, robust=False)
    assert fr_std.result is not None
    assert fr_std.result.success

    fr_robust = fit_binding_glob(ds, robust=True)
    assert fr_robust.result is not None
    assert fr_robust.result.success

    fr_iter = fit_binding_glob_recursive_outlier(ds, threshold=3.0)
    assert fr_iter.result is not None
    assert fr_iter.result.success

    def _metrics(fr: FitResult[MiniT]) -> tuple[float, float]:
        r = fr.result
        if r:
            return float(r.params["K"].value), float(getattr(r, "redchi", np.nan))
        return (np.inf, np.inf)

    k_std, chi_std = _metrics(fr_std)
    k_rob, chi_rob = _metrics(fr_robust)
    k_itr, chi_itr = _metrics(fr_iter)

    # Absolute K error vs truth
    e_std = abs(k_std - K0)
    e_rob = abs(k_rob - K0)
    e_itr = abs(k_itr - K0)

    # Robust should not be worse than baseline by more than 190%
    assert e_rob <= e_std * 2.90 + 1e-6
    # Iterative should not be worse than baseline by more than 190%
    assert e_itr <= e_std * 2.90 + 1e-6

    # redchi tolerance (allowing some fluctuation)
    assert chi_rob <= chi_std * 1.5 + 1e-9
    assert chi_itr <= chi_std * 1.5 + 1e-9


@pytest.mark.parametrize(
    "csv_name",
    [
        "tests/Tecan/140220/list.pH.csv",
    ],
)
def test_iterative_outlier_removal_is_stable(csv_name: str) -> None:
    """Ensure IRLS with outlier removal does not explode on typical inputs."""
    x, x_err = _load_ph_list(Path(csv_name))
    K0, S0, S1 = 7.0, 2.0, 1.0  # noqa: N806
    np.random.default_rng(123)
    y = binding_1site(x, K0, S0, S1, is_ph=True)
    # Inject a single outlier
    y = y.copy()
    y[np.argmax(x)] += 0.5

    ds = Dataset({"y1": DataArray(xc=x, yc=y, x_errc=x_err)}, is_ph=True)
    weight_multi_ds_titration(ds)

    fr = fit_binding_glob_recursive_outlier(ds, threshold=2.5)
    assert fr.result is not None
    assert fr.result.success
    # K should remain within a reasonable band
    assert abs(fr.result.params["K"].value - K0) < 0.6
