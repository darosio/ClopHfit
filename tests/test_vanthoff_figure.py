"""The van't Hoff figure must show the titration it actually fitted.

A pH titration is a Henderson-Hasselbalch curve in pH space with a pKa
midpoint; a chloride titration is a Langmuir curve in mM space with a Kd
midpoint. Drawing the pH form for a chloride sample, or titling the
Henderson-Hasselbalch curve with a Kd, puts the model on an axis it does
not belong to.
"""

from __future__ import annotations

import numpy as np
import pytest

from clophfit.prenspire.thermo import VanthoffModel
from clophfit.prenspire.thermo_plot import (
    vanthoff_axes,
    vanthoff_curve,
    vanthoff_enthalpy,
    vanthoff_right_line,
    vanthoff_right_point,
)

TEMPS = ("20", "37")
CL_X = np.array([0.0, 5.0, 10.0, 20.0, 40.0, 80.0, 160.0, 320.0, 500.0])
PH_X = np.linspace(4.0, 9.5, 15)


def _model(*, is_ph: bool) -> VanthoffModel:
    """Build a two-temperature model: pKa = 7.2 for pH, Kd = 25 mM for ligand."""
    inv = {t: 1.0 / (float(t) + 273.15) for t in TEMPS}
    m = float(np.mean(list(inv.values())))
    if is_ph:
        a, b = 7.2, 40.0
        x = PH_X
        k_by_temp = {t: (a + b * (inv[t] - m), 0.05) for t in TEMPS}
    else:
        # pKd is what is linear in 1/T
        a, b = np.log10(25.0), -30.0
        x = CL_X
        k_by_temp = {t: (10.0 ** (a + b * (inv[t] - m)), 1.0) for t in TEMPS}
    curves = {}
    for band in ("exc_anionic",):
        for t in TEMPS:
            k = k_by_temp[t][0]
            if is_ph:
                y = 0.1 + 0.9 / (1.0 + np.power(10.0, k - x))
            else:
                y = 0.1 + 0.9 * x / (x + k)
            curves[band, t] = (x, y)
    return VanthoffModel(
        a=a,
        b=b,
        inv=inv,
        m=m,
        k_by_temp=k_by_temp,
        plateaus={(band, t): (0.1, 1.0) for band in ("exc_anionic",) for t in TEMPS},
        curves=curves,
        is_ph=is_ph,
    )


def test_a_ph_curve_is_drawn_in_ph_space() -> None:
    """A pH sample: the curve is Henderson-Hasselbalch on the pH axis."""
    model = _model(is_ph=True)
    k = model.k_by_temp["20"][0]
    y = vanthoff_curve(model, "20", np.array([k]), (0.1, 1.0))
    # the midpoint of the curve is the pKa, on the pH axis
    assert y[0] == pytest.approx(0.55, abs=0.01)


def test_a_ligand_curve_is_drawn_in_ligand_space() -> None:
    """A chloride sample: the curve is Langmuir in mM, half-height at Kd."""
    model = _model(is_ph=False)
    kd = model.k_by_temp["20"][0]
    y = vanthoff_curve(model, "20", np.array([kd]), (0.1, 1.0))
    # the midpoint of the curve is the Kd, on the mM axis
    assert y[0] == pytest.approx(0.55, abs=0.01)
    # a Henderson-Hasselbalch curve drawn with the same numbers would have
    # its half-height at pH = pKd = 1.4, i.e. nowhere near the Kd
    assert abs(kd - np.log10(kd)) > 1.0


def test_the_enthalpy_sign_follows_the_titration() -> None:
    """A pH sample's dH is dH = +ln10 R b; a ligand sample's flips sign.

    For a ligand the fitted quantity is pKd, and d ln Kd / d(1/T) = -dH/R,
    so the same b carries the opposite sign of enthalpy.
    """
    ph = _model(is_ph=True)
    cl = _model(is_ph=False)
    assert vanthoff_enthalpy(ph) == pytest.approx(ph.b * 2.302585093 * 8.314462618)
    assert vanthoff_enthalpy(cl) == pytest.approx(-cl.b * 2.302585093 * 8.314462618)


def test_the_right_panel_plots_the_constant_the_table_reports() -> None:
    """A ligand's panel must show the same Kd in mM the table reports.

    The figure is read next to the CSV, so the two must not disagree about
    the number. Kd is exponential in 1/T, so the panel is log-scaled rather
    than plotting pKd: pKd would need a different label from the table.
    """
    cl = _model(is_ph=False)
    values = [vanthoff_right_point(cl, t)[0] for t in TEMPS]
    assert values == pytest.approx([cl.k_by_temp[t][0] for t in TEMPS])
    assert vanthoff_axes(cl).right_log is True
    # the fitted line follows Kd = 10^(a + b (1/T - m))
    line = vanthoff_right_line(cl, np.array([cl.inv["20"], cl.inv["37"]]))
    assert line == pytest.approx([
        10.0 ** (cl.a + cl.b * (v - cl.m)) for v in (cl.inv["20"], cl.inv["37"])
    ])

    ph = _model(is_ph=True)
    assert [vanthoff_right_point(ph, t)[0] for t in TEMPS] == pytest.approx([
        ph.k_by_temp[t][0] for t in TEMPS
    ])
    assert vanthoff_axes(ph).right_log is False
    ph_line = vanthoff_right_line(ph, np.array([ph.inv["20"], ph.inv["37"]]))
    assert ph_line == pytest.approx([
        ph.a + ph.b * (v - ph.m) for v in (ph.inv["20"], ph.inv["37"])
    ])


def test_the_axis_labels_name_the_constant_that_was_fitted() -> None:
    """A pH sample labels its axes pKa and pH; a ligand sample Kd and mM."""
    ph = vanthoff_axes(_model(is_ph=True))
    assert ph.left_x == "pH"
    assert ph.right_y == "pKa"
    assert ph.right_log is False

    cl = vanthoff_axes(_model(is_ph=False))
    assert "mM" in cl.left_x
    assert "Kd" in cl.right_y
    assert cl.right_log is True
    assert cl.left_log is True
