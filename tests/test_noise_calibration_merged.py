"""Pooled noise calibration: one gain and alpha per label, from every plate.

The single-plate estimator parks a term at a boundary whenever one plate's own
signal range cannot separate ``y`` from ``y**2`` - which it usually cannot. The
merged estimator concatenates the residual tables and fits once, so a plate that
carries no information about the split borrows the population's instead of
answering zero.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from clophfit.fitting.noise_calibration import fit_noise_model_merged

GAIN = 0.5
ALPHA = 0.03
# Two decades of signal, so y and y**2 are actually distinguishable.
Y_LO, Y_HI = 50.0, 5000.0
N_PER_PLATE = 4000


def _plate(
    name: str, label: str, floor: float, *, gain_amp: float = 1.0, seed: int = 0
) -> pd.DataFrame:
    """One plate's residual table drawn from the known noise model."""
    rng = np.random.default_rng(seed)
    yhat = np.exp(rng.uniform(np.log(Y_LO), np.log(Y_HI), N_PER_PLATE))
    var = floor**2 + GAIN * gain_amp * yhat + (ALPHA * yhat) ** 2
    return pd.DataFrame({
        "plate": name,
        "label": label,
        "yhat": yhat,
        "raw_res": rng.normal(0.0, np.sqrt(var)),
        "sigma_floor": floor,
        "gain_amp": gain_amp,
    })


@pytest.fixture
def merged() -> pd.DataFrame:
    """Three plates of one label, each with its own floor."""
    return pd.concat(
        [
            _plate("L1", "1", floor=1.0, seed=1),
            _plate("L2", "1", floor=3.0, seed=2),
            _plate("L3", "1", floor=8.0, seed=3),
        ],
        ignore_index=True,
    )


def test_recovers_the_shared_gain_and_alpha(merged: pd.DataFrame) -> None:
    """One fit over three plates returns the values they were drawn from."""
    out = fit_noise_model_merged(merged)
    row = out.loc[out["label"] == "1"].iloc[0]
    assert row["gain"] == pytest.approx(GAIN, rel=0.30)
    assert row["alpha"] == pytest.approx(ALPHA, rel=0.15)
    assert row["n_plates"] == 3


def test_reports_one_row_per_label() -> None:
    """Labels are fitted independently, not pooled with one another."""
    df = pd.concat(
        [_plate("L1", "1", floor=3.0, seed=4), _plate("L1", "2", floor=0.4, seed=5)],
        ignore_index=True,
    )
    out = fit_noise_model_merged(df)
    assert sorted(out["label"]) == ["1", "2"]


def test_gain_amplification_defaults_to_one(merged: pd.DataFrame) -> None:
    """A label with no Gain dependence needs no column and is not scaled.

    Label 2 has no Gain dependence over its narrow range, so its floor is
    supplied unscaled. Scaling its *gain* term while its floor stays unscaled
    quotes the two at different references, which is the bug this pins.
    """
    without = fit_noise_model_merged(merged.drop(columns="gain_amp"))
    with_ones = fit_noise_model_merged(merged)
    assert without["gain"].to_numpy() == pytest.approx(with_ones["gain"].to_numpy())
    assert without["alpha"].to_numpy() == pytest.approx(with_ones["alpha"].to_numpy())


def test_gain_is_quoted_at_unit_amplification() -> None:
    """The returned gain is the coefficient of ``gain_amp * y``.

    A caller that passes an amplification therefore reads back a gain at
    ``gain_amp == 1``, not at the plates' own setting - which is what makes it
    comparable across plates run at different reader Gain.
    """
    amp = 4.0
    scaled = pd.concat(
        [
            _plate("L1", "1", floor=1.0, gain_amp=amp, seed=6),
            _plate("L2", "1", floor=3.0, gain_amp=amp, seed=7),
        ],
        ignore_index=True,
    )
    out = fit_noise_model_merged(scaled)
    assert out.iloc[0]["gain"] == pytest.approx(GAIN, rel=0.30)


def test_each_plate_floor_is_removed_before_pooling() -> None:
    """Fitting against a zero floor pushes the signal-dependent terms up.

    2.0 to 9.6 is what Gain-scaling the quoted floor produces across these
    eleven plates. Leaving each plate's own floor in the remainder would let it
    be absorbed by gain and alpha, so the same table fitted against a zero floor
    must predict more variance than one fitted against the true floors.
    """
    df = pd.concat(
        [_plate("L1", "1", floor=2.0, seed=8), _plate("L2", "1", floor=9.6, seed=9)],
        ignore_index=True,
    )
    subtracted = fit_noise_model_merged(df).iloc[0]
    unsubtracted = fit_noise_model_merged(df.assign(sigma_floor=0.0)).iloc[0]
    at_floor = 60.0  # where the floor is still an appreciable share of sigma
    assert _sigma(unsubtracted, at_floor) > _sigma(subtracted, at_floor)


@pytest.mark.parametrize("seeds", [(8, 9), (11, 12), (13, 14)])
def test_total_sigma_survives_a_split_that_does_not(seeds: tuple[int, int]) -> None:
    """Two plates pin the total noise but not how it divides between the terms.

    This is the campaign's central finding reproduced on synthetic data: over
    one plate's signal range ``y`` and ``y**2`` are near-collinear, so which of
    gain and alpha carries the variance is set by whatever constrains it. Only
    the total is worth quoting. Changing nothing but the seed moves gain across
    its whole range while sigma stays put, so an estimator that reported a
    stable-looking gain here would be reporting its prior.
    """
    df = pd.concat(
        [
            _plate("L1", "1", floor=2.0, seed=seeds[0]),
            _plate("L2", "1", floor=9.6, seed=seeds[1]),
        ],
        ignore_index=True,
    )
    row = fit_noise_model_merged(df).iloc[0]
    truth = np.sqrt(GAIN * 1000.0 + (ALPHA * 1000.0) ** 2)
    assert _sigma(row, 1000.0) == pytest.approx(truth, rel=0.15)


def _sigma(row: pd.Series, y: float) -> float:
    """Signal-dependent sigma the fitted row predicts at *y*."""
    return float(np.sqrt(row["gain"] * y + (row["alpha"] * y) ** 2))


def test_rejects_a_table_without_a_supplied_floor(merged: pd.DataFrame) -> None:
    """The floor is supplied, never fitted: three free terms are collinear."""
    with pytest.raises(ValueError, match="sigma_floor"):
        fit_noise_model_merged(merged.drop(columns="sigma_floor"))
