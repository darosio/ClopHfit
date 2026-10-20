"""The van't Hoff joint fit against a hand-computed two-temperature case."""

from __future__ import annotations

import numpy as np

from clophfit.prenspire.thermo import LN10, R_GAS, fit_vanthoff_model

T20, T37 = 293.15, 310.15
DH_TRUE = 41_000.0  # J/mol
DS_TRUE = 35.0  # J/(mol K)


def pka_at(temp_k: float) -> float:
    """pKa(T) = dH/(2.303 R T) - dS/(2.303 R): the exact model being fitted."""
    return DH_TRUE / (LN10 * R_GAS * temp_k) - DS_TRUE / (LN10 * R_GAS)


def curve(x: np.ndarray, k: float, s0: float, s1: float) -> np.ndarray:
    """Henderson-Hasselbalch signal: plateaus s0 (protonated) -> s1 (deprotonated)."""
    return s0 + (s1 - s0) / (1.0 + np.power(10.0, k - x))


def test_recovers_a_known_thermodynamic_pair() -> None:
    """Synthetic 2-band x 2-temperature data reproduces dH, dS and both pKa."""
    rng = np.random.default_rng(7)
    x = np.linspace(4.0, 9.5, 15)
    data = {}
    for band, (s0, s1) in (("exc_anionic", (0.05, 1.0)), ("em_exc420", (0.30, 1.15))):
        for t, temp_k in (("20", T20), ("37", T37)):
            y = curve(x, pka_at(temp_k), s0, s1) + rng.normal(0.0, 0.01, x.size)
            data[band, t] = (x, y)
    fit = fit_vanthoff_model(data)
    assert fit.ok
    k20, se20 = fit.k_by_temp["20"]
    k37, se37 = fit.k_by_temp["37"]
    assert abs(k20 - pka_at(T20)) < 0.05
    assert abs(k37 - pka_at(T37)) < 0.05
    assert se20 < 0.05
    assert se37 < 0.05
    assert abs(fit.dh - DH_TRUE) < 0.1 * DH_TRUE
    assert abs(fit.ds - DS_TRUE) < 10.0
    # the sign convention: pKa falls with T only if deprotonation is endothermic
    assert fit.dh > 0
    assert k37 < k20


def test_identical_curves_give_a_flat_enthalpy() -> None:
    """Data with no temperature dependence must return dH close to zero."""
    rng = np.random.default_rng(11)
    x = np.linspace(4.0, 9.5, 15)
    y0 = curve(x, 7.5, 0.1, 1.0)
    data = {
        ("exc_anionic", t): (x, y0 + rng.normal(0.0, 0.01, x.size))
        for t in ("20", "37")
    }
    fit = fit_vanthoff_model(data)
    assert fit.ok
    assert abs(fit.dh) < 4_000.0
    assert all(abs(k - 7.5) < 0.05 for k, _ in fit.k_by_temp.values())


def test_one_temperature_is_not_a_pair() -> None:
    """With a single temperature the enthalpy is unidentifiable: refuse."""
    x = np.linspace(4.0, 9.5, 15)
    fit = fit_vanthoff_model({("exc_anionic", "20"): (x, curve(x, 7.5, 0.1, 1.0))})
    assert not fit.ok


def test_ligand_pair_fits_pkd_and_reports_kd() -> None:
    """For Cl data pKd(T) is what is linear in 1/T; Kd comes out in mM."""
    rng = np.random.default_rng(2)
    x20 = np.array([0.0, 5.0, 10.0, 20.0, 40.0, 80.0, 160.0, 320.0, 500.0])
    temps = {"20": 293.15, "37": 310.15}
    a_true = 1.4  # pKd at the centred mean 1/T
    b_true = -30.0  # Kd rises slightly with T (endo dissociation => pKd falls)
    inv = {t: 1.0 / T for t, T in temps.items()}
    m = float(np.mean(list(inv.values())))
    data = {}
    for t, xv in (("20", x20), ("37", x20 * 1.0)):
        kd = np.power(10.0, a_true + b_true * (inv[t] - m))
        y = 0.1 + 0.9 * xv / (xv + kd)
        data["em_exc420", t] = (xv, y + rng.normal(0.0, 0.005, xv.size))
    fit = fit_vanthoff_model(data, is_ph=False)
    assert fit.ok
    kd20 = 10.0 ** (a_true + b_true * (inv["20"] - m))
    assert abs(fit.k_by_temp["20"][0] - kd20) / kd20 < 0.15
    # Kd on the mM scale and a positive slope b_true<0 in pKd means endo binding...
    assert fit.k_by_temp["37"][0] > fit.k_by_temp["20"][0]
