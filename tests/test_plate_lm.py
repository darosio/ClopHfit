"""Tests for the multi-well classical fit with plate-wide noise scales."""

from __future__ import annotations

import numpy as np
import pytest

from clophfit.fitting.data_structures import DataArray, Dataset
from clophfit.fitting.models import binding_1site
from clophfit.fitting.plate_lm import (
    PlateLMResult,
    fit_plate_lm,
    fit_plate_lm_screened,
)


def make_plate(
    ks: dict[str, float], noise: dict[str, float], seed: int = 0
) -> dict[str, Dataset]:
    """Build one synthetic plate with known K per well and known noise per label."""
    rng = np.random.default_rng(seed)
    x = np.linspace(5.0, 9.0, 7)
    out = {}
    for well, k in ks.items():
        arrays = {}
        for lbl, sd in noise.items():
            y = binding_1site(x, k, 1000.0, 200.0, is_ph=True)
            y += rng.normal(0.0, sd, size=len(x))
            arrays[lbl] = DataArray(x, y, y_errc=np.ones_like(y))
        out[well] = Dataset(arrays, is_ph=True)
    return out


def test_recovers_k_per_well() -> None:
    """Each well's K comes back, with no control grouping in play."""
    ks = {"A01": 6.5, "A02": 7.0, "A03": 7.5}
    result = fit_plate_lm(make_plate(ks, {"1": 5.0}), groups={})
    for well, truth in ks.items():
        assert result.k[well] == pytest.approx(truth, abs=0.05)


def test_control_group_shares_one_k() -> None:
    """Wells in a control group are fitted with a single shared K, not an average."""
    ks = {"A01": 7.0, "A02": 7.0, "B01": 6.0}
    result = fit_plate_lm(make_plate(ks, {"1": 5.0}), groups={"ctrl": ["A01", "A02"]})
    assert result.k["A01"] == result.k["A02"]  # identical, not merely close
    assert result.k["A01"] == pytest.approx(7.0, abs=0.05)
    assert result.k["B01"] == pytest.approx(6.0, abs=0.05)


def test_profiles_the_per_label_noise_scale() -> None:
    """ye_mag is recovered from the residuals, since least squares cannot fit it.

    y_err is supplied as 1.0 everywhere while the real scatter differs per label,
    so an honest scale has to come from the residuals themselves. This is what
    scale_covar does with a single global factor, generalised per label.

    Enough wells that the realised scatter approaches the nominal sigma: with
    three wells the two differ by more than the quantity being tested.
    """
    ks = {f"A{i:02d}": 6.6 + 0.05 * i for i in range(1, 25)}
    result = fit_plate_lm(make_plate(ks, {"1": 8.0, "2": 2.0}), groups={})
    assert result.ye_mag["1"] == pytest.approx(8.0, rel=0.25)
    assert result.ye_mag["2"] == pytest.approx(2.0, rel=0.25)
    assert result.ye_mag["1"] / result.ye_mag["2"] == pytest.approx(4.0, rel=0.3)


def test_k_stays_inside_the_ph_range() -> None:
    """The solver cannot wander K outside the pH scale.

    Unbounded, least_squares walks K to a few hundred below the data while the
    plateaus are still poorly seeded. That produced overflow warnings from the
    model on every plate, and a K of -320 is not a hypothesis worth exploring.
    lmfit's per-well path has bounded K at [3, 11] all along.
    """
    ks = {"A01": 6.5, "A02": 7.0, "A03": 7.5}
    result = fit_plate_lm(make_plate(ks, {"1": 5.0}), groups={})
    for well in ks:
        assert 3.0 <= result.k[well] <= 11.0


def test_result_carries_the_plateaus_not_only_k() -> None:
    """K alone cannot be plotted or inspected; the curve needs its plateaus.

    ``--plate-fit`` wrote a CSV of K and its error and nothing else: no K plot
    and no per-well fit figures, while every other fitter in the same run
    produced both. The plateaus are computed by the fit anyway, so exporting
    them lets the plate fit reuse the plotting path instead of being a
    second-class output.
    """
    ks = {"A01": 6.5, "A02": 7.0, "A03": 7.5}
    result = fit_plate_lm(make_plate(ks, {"1": 5.0, "2": 3.0}), groups={})
    for well in ks:
        p = result.params[well]
        assert p["K"] == pytest.approx(result.k[well])
        for lbl in ("1", "2"):
            # Plateaus bracket the data the fit saw.
            assert np.isfinite(p[f"S0_{lbl}"])
            assert np.isfinite(p[f"S1_{lbl}"])
        assert np.isfinite(p["sK"])


def test_screening_refit_drops_a_planted_outlier_and_recovers_k() -> None:
    """Fit, mask on the fit's own residuals, refit - the classical two-step.

    Geometric masking works on the shape of the raw trace and cannot see
    whether a point disagrees with the *fit*; on this campaign it changed pKa
    accuracy not at all. A z-score screen uses the fitted residuals, which is
    the quantity that actually says a point is wrong.
    """
    ks = {f"A{i:02d}": 6.4 + 0.15 * i for i in range(1, 13)}
    datasets = make_plate(ks, {"1": 4.0, "2": 3.0})
    # One point of one well, moved far off its curve.
    da = datasets["A05"]["1"]
    spoiled = np.asarray(da.yc, dtype=float).copy()
    spoiled[3] += 400.0
    datasets["A05"] = Dataset(
        {
            "1": DataArray(
                np.asarray(da.xc, dtype=float),
                spoiled,
                y_errc=np.asarray(da.y_errc, dtype=float),
            ),
            "2": datasets["A05"]["2"],
        },
        is_ph=True,
    )
    plain = fit_plate_lm(datasets, groups={})
    screened = fit_plate_lm_screened(datasets, groups={}, threshold=3.0)

    assert screened.n_excluded >= 1, "the planted outlier was not screened"
    # The spoiled well's K is recovered better after screening.
    assert abs(screened.k["A05"] - ks["A05"]) < abs(plain.k["A05"] - ks["A05"])
    # And the clean wells are left where they were.
    for well in ("A01", "A12"):
        assert screened.k[well] == pytest.approx(plain.k[well], abs=0.05)


def test_screening_refit_leaves_a_clean_plate_alone() -> None:
    """With nothing to screen, the two-step must return the one-pass answer."""
    ks = {f"A{i:02d}": 6.5 + 0.12 * i for i in range(1, 13)}
    datasets = make_plate(ks, {"1": 4.0, "2": 3.0})
    plain = fit_plate_lm(datasets, groups={})
    screened = fit_plate_lm_screened(datasets, groups={}, threshold=4.0)
    for well in ks:
        assert screened.k[well] == pytest.approx(plain.k[well], abs=0.02)


def test_robust_first_pass_finds_the_outlier_least_squares_hides() -> None:
    """Screen on a robust fit, not on one the outlier has already bent.

    Ordinary least squares drags the curve toward an outlier, which shrinks that
    point's own residual and inflates its neighbours' - masking and swamping.
    The profiled scale makes it worse: it is the RMS of the residuals, so the
    outlier also inflates the ruler the z-scores are measured against. The
    screen then decides with a bent curve and a stretched ruler, which is how a
    threshold ends up behaving erratically.
    """
    ks = {f"A{i:02d}": 6.4 + 0.15 * i for i in range(1, 13)}
    datasets = make_plate(ks, {"1": 3.0, "2": 3.0})
    da = datasets["A05"]["1"]
    y = np.asarray(da.yc, dtype=float).copy()
    y[3] += 250.0  # one point, far off its curve
    datasets["A05"] = Dataset(
        {
            "1": DataArray(
                np.asarray(da.xc, dtype=float),
                y,
                y_errc=np.asarray(da.y_errc, dtype=float),
            ),
            "2": datasets["A05"]["2"],
        },
        is_ph=True,
    )

    def z_of_planted(result: PlateLMResult) -> float:
        rows = [
            r
            for r in result.residuals
            if r["well"] == "A05" and str(r["label"]) == "1" and r["raw_i"] == 3
        ]
        return abs(float(rows[0]["std_res"]))

    plain = fit_plate_lm(datasets, groups={})
    robust = fit_plate_lm(datasets, groups={}, loss="huber")
    # The robust pass is not dragged, so the bad point stands out further.
    assert z_of_planted(robust) > z_of_planted(plain)
    # And the well's K is closer to truth without any screening at all.
    assert abs(robust.k["A05"] - ks["A05"]) < abs(plain.k["A05"] - ks["A05"])


def test_screening_uses_a_robust_first_pass_by_default() -> None:
    """The screen's first pass must be the robust one, or it screens blind."""
    ks = {f"A{i:02d}": 6.5 + 0.12 * i for i in range(1, 13)}
    datasets = make_plate(ks, {"1": 3.0, "2": 3.0})
    da = datasets["A07"]["1"]
    y = np.asarray(da.yc, dtype=float).copy()
    y[2] += 250.0
    datasets["A07"] = Dataset(
        {
            "1": DataArray(
                np.asarray(da.xc, dtype=float),
                y,
                y_errc=np.asarray(da.y_errc, dtype=float),
            ),
            "2": datasets["A07"]["2"],
        },
        is_ph=True,
    )
    screened = fit_plate_lm_screened(datasets, groups={}, threshold=3.0)
    assert screened.n_excluded >= 1
    assert screened.k["A07"] == pytest.approx(ks["A07"], abs=0.1)
