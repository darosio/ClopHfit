"""Fix any subset of the three noise terms, so single-term models are fittable.

``fit_noise_model_nnls`` could fix the floor, or fix alpha, but not fix gain,
and refused to fix two at once. That left three of the four cases the noise
comparison needs unreachable:

===================  ==================  ============================
floor                zeroed term         reachable before
===================  ==================  ============================
free                 alpha = 0           yes, via ``rel_error_fixed``
free                 gain = 0            no
fixed                alpha = 0           no, refused as "both fixed"
fixed                gain = 0            no
===================  ==================  ============================

Fixing a term means subtracting its known contribution from the squared
residual and fitting only what remains, so the four cases are the same
estimator with different columns in the design matrix. Zero is an ordinary
fixed value, not a disabled term: fixing alpha at 0 subtracts nothing and
leaves floor and gain to be fitted.

The point of reaching them is iteration. ``fit_plate_lm`` refits its noise
model until the terms stop moving, and until now it could only do that with
all three free -- where gain and alpha are collinear over one plate's signal
range, so the pair wanders along a ridge instead of converging.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from clophfit.fitting.noise_calibration import fit_noise_model_nnls

FLOOR, GAIN, ALPHA = 3.0, 0.5, 0.04
N = 4000


@pytest.fixture
def residuals() -> pd.DataFrame:
    """Residuals drawn from a known floor + gain + alpha model."""
    rng = np.random.default_rng(11)
    yhat = np.exp(rng.uniform(np.log(50.0), np.log(5000.0), N))
    var = FLOOR**2 + GAIN * yhat + (ALPHA * yhat) ** 2
    return pd.DataFrame({
        "label": "1",
        "yhat": yhat,
        "raw_res": rng.normal(0.0, np.sqrt(var)),
    })


def test_fixing_gain_leaves_floor_and_alpha_to_be_fitted(
    residuals: pd.DataFrame,
) -> None:
    """The case that had no route at all: gain pinned, the rest estimated."""
    floors, gains, alphas = fit_noise_model_nnls(residuals, gain_fixed={"1": GAIN})
    assert gains["1"] == pytest.approx(GAIN)
    assert alphas["1"] == pytest.approx(ALPHA, rel=0.25)
    # The floor is not asserted. Over 50-5000 counts it contributes under a
    # quarter of the variance even at the dimmest point, so raw-residual NNLS
    # cannot pin it -- the same unidentifiability that makes the floor a
    # supplied measurement everywhere else in this work.
    assert floors["1"] >= 0.0


def test_gain_fixed_at_zero_is_a_value_not_a_disabled_term(
    residuals: pd.DataFrame,
) -> None:
    """With gain held at zero the remaining terms absorb the variance."""
    floors, gains, alphas = fit_noise_model_nnls(residuals, gain_fixed={"1": 0.0})
    assert gains["1"] == 0.0
    # alpha must rise above its true value to cover the missing shot term.
    assert alphas["1"] > ALPHA
    assert floors["1"] >= 0.0


def test_floor_and_alpha_can_both_be_fixed(residuals: pd.DataFrame) -> None:
    """Two fixed terms was refused outright; it is the single-term fit."""
    floors, gains, alphas = fit_noise_model_nnls(
        residuals, sigma_floor_fixed={"1": FLOOR}, rel_error_fixed={"1": ALPHA}
    )
    assert floors["1"] == pytest.approx(FLOOR)
    assert alphas["1"] == pytest.approx(ALPHA)
    assert gains["1"] == pytest.approx(GAIN, rel=0.3)


def test_floor_and_gain_can_both_be_fixed(residuals: pd.DataFrame) -> None:
    """The mirror case: only alpha is estimated."""
    floors, gains, alphas = fit_noise_model_nnls(
        residuals, sigma_floor_fixed={"1": FLOOR}, gain_fixed={"1": GAIN}
    )
    assert floors["1"] == pytest.approx(FLOOR)
    assert gains["1"] == pytest.approx(GAIN)
    assert alphas["1"] == pytest.approx(ALPHA, rel=0.3)


def test_fixing_all_three_returns_them_unchanged(residuals: pd.DataFrame) -> None:
    """Nothing left to fit is not an error; it is the supplied model."""
    floors, gains, alphas = fit_noise_model_nnls(
        residuals,
        sigma_floor_fixed={"1": FLOOR},
        gain_fixed={"1": GAIN},
        rel_error_fixed={"1": ALPHA},
    )
    assert (floors["1"], gains["1"], alphas["1"]) == pytest.approx((FLOOR, GAIN, ALPHA))


def test_nothing_fixed_still_fits_all_three(residuals: pd.DataFrame) -> None:
    """The unconstrained path must keep working."""
    floors, gains, alphas = fit_noise_model_nnls(residuals)
    total = floors["1"] ** 2 + gains["1"] * 1000 + (alphas["1"] * 1000) ** 2
    truth = FLOOR**2 + GAIN * 1000 + (ALPHA * 1000) ** 2
    # The split is not identifiable over one signal range; the total is.
    assert total == pytest.approx(truth, rel=0.2)


def test_a_label_absent_from_a_fixed_mapping_is_left_free() -> None:
    """Per-label constraints, so one channel can be pinned and the other not."""
    rng = np.random.default_rng(3)
    y = np.exp(rng.uniform(np.log(50.0), np.log(5000.0), N))
    frames = [
        pd.DataFrame({
            "label": lbl,
            "yhat": y,
            "raw_res": rng.normal(0.0, np.sqrt(FLOOR**2 + GAIN * y + (ALPHA * y) ** 2)),
        })
        for lbl in ("1", "2")
    ]
    floors, gains, alphas = fit_noise_model_nnls(
        pd.concat(frames, ignore_index=True), gain_fixed={"1": 0.0}
    )
    # Label 1 is held at the supplied value; label 2 is estimated, and what it
    # lands on is NNLS's business -- over one signal range gain and alpha are
    # collinear, so a free gain may well be parked at the boundary. What the
    # constraint must not do is leak across labels.
    assert gains["1"] == 0.0
    assert alphas["1"] > 0.0
    total = floors["2"] ** 2 + gains["2"] * 1000 + (alphas["2"] * 1000) ** 2
    truth = FLOOR**2 + GAIN * 1000 + (ALPHA * 1000) ** 2
    assert total == pytest.approx(truth, rel=0.25)


def test_the_variance_regression_is_weighted() -> None:
    """The target is a squared residual, whose own variance goes as sigma^4.

    Fitting it unweighted lets the brightest points dominate: the estimate stays
    unbiased but its scatter is enormous, 0.46 +/- 0.31 for a true gain of 0.5
    over six seeds. Weighting by 1/var**2 -- the inverse variance of a squared
    Gaussian residual -- gives 0.497 +/- 0.016, a nineteen-fold reduction. That
    is what lets an iterative refit converge instead of wandering.
    """
    spread = []
    for seed in range(6):
        rng = np.random.default_rng(seed)
        y = np.exp(rng.uniform(np.log(50.0), np.log(5000.0), N))
        var = FLOOR**2 + GAIN * y + (ALPHA * y) ** 2
        df = pd.DataFrame({
            "label": "1",
            "yhat": y,
            "raw_res": rng.normal(0.0, np.sqrt(var)),
        })
        _, gains, _ = fit_noise_model_nnls(
            df, sigma_floor_fixed={"1": FLOOR}, rel_error_fixed={"1": ALPHA}
        )
        spread.append(gains["1"])
    arr = np.asarray(spread)
    assert arr.mean() == pytest.approx(GAIN, rel=0.2)
    assert arr.std() < 0.08  # unweighted gives 0.31
