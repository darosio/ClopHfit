"""Joint van't Hoff fit of an EnSpire titration across temperatures.

Instead of fitting pKa at each temperature independently, fit every well of
every temperature as one model tied by the van't Hoff relation::

    pKa(T) = dH / (2.303 R T) - dS / (2.303 R)      (T in kelvin)

The plateaus ``S0, S1`` stay free per (band, temperature) - brightness is not
comparable across temperatures - while the two thermodynamic parameters are
shared by all bands and all temperatures. Both reported pKa values therefore
come from one fit over all the data, and ``dH`` of chromophore deprotonation
falls out as a fitted quantity, not a two-point difference.

``pKa(T) = A + B (1/T - m)`` with the inverse-temperature axis centred on its
mean ``m`` decorrelates the fitted pair; the reported constants are
``dH = 2.303 R B`` and ``dS = -2.303 R (A - B m)``.

Hill slope is fixed to 1 (Henderson-Hasselbalch); see
:func:`clophfit.prenspire.bands.band_rho` for the readout gate.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import lmfit  # type: ignore[import-untyped]
import numpy as np
import pandas as pd
from lmfit import Parameters

from clophfit.prenspire.bands import (
    DIRECT_BANDS,
    band_rho,
    band_values,
    blank_rows,
    classify,
    subtract_buffer,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from clophfit.clophfit_types import ArrayF
    from clophfit.prenspire.prenspire import EnspireFile

R_GAS = 8.314462618  # J mol-1 K-1
LN10 = 2.302585093
ZERO_C = 273.15
_MIN_TEMPS = 2  # one enthalpy needs a pair of temperatures


def _kelvin(temp: str | float) -> float:
    return float(temp) + ZERO_C


class VanthoffFit(NamedTuple):
    """The joint fit of one sample: pKa per temperature plus the pair behind it."""

    ok: bool
    """Whether the fit converged."""
    k_by_temp: dict[str, tuple[float, float]]
    """Temperature label to (pKa, standard error)."""
    a: float
    b: float
    """Centred coefficients, pKa = A + B (1/T - m)."""
    dh: float
    ds: float
    """Enthalpy (J/mol) and entropy (J/mol/K) of deprotonation."""
    se_a: float
    se_b: float
    n: int
    """Wells fitted."""
    plateaus: dict[tuple[str, str], tuple[float, float]]
    """Fitted (S0, S1) per (band, temperature), for plotting."""
    inv: dict[str, float]
    m: float
    """Inverse temperatures [1/K] and their centring mean."""


class VanthoffModel(NamedTuple):
    """Everything needed to redraw one sample's joint fit."""

    a: float
    b: float
    inv: dict[str, float]
    m: float
    k_by_temp: dict[str, tuple[float, float]]
    plateaus: dict[tuple[str, str], tuple[float, float]]
    curves: dict[tuple[str, str], tuple[ArrayF, ArrayF]]


def fit_vanthoff_model(
    data: dict[tuple[str, str], tuple[ArrayF, ArrayF]],
    *,
    is_ph: bool = True,
) -> VanthoffFit:
    """Fit (band, temp) curves jointly; pKa(T) = A + B (1/T - m).

    Parameters
    ----------
    data : dict[tuple[str, str], tuple[ArrayF, ArrayF]]
        Band name and temperature label to its (x, y) curve, y already
        normalised and scaled by the caller.
    is_ph : bool
        pH titration (fit pKa(T) against a Henderson-Hasselbalch fraction)
        or ligand titration (fit pKd(T) against a Langmuir fraction and
        report Kd on the x scale).

    Returns
    -------
    VanthoffFit
        The joint fit; ``ok=False`` when it did not converge or fewer than
        two temperatures were given.
    """
    temps = sorted({t for _, t in data})
    if len(temps) < _MIN_TEMPS or not data:
        return VanthoffFit(
            ok=False,
            k_by_temp={},
            a=float("nan"),
            b=float("nan"),
            dh=float("nan"),
            ds=float("nan"),
            se_a=float("nan"),
            se_b=float("nan"),
            n=0,
            plateaus={},
            inv={},
            m=float("nan"),
        )
    inv = {t: 1.0 / _kelvin(t) for t in temps}
    m = float(np.mean(list(inv.values())))
    keys = sorted(data)

    params = Parameters()
    a0 = (
        8.0
        if is_ph
        else float(
            np.log10(
                np.median(
                    np.concatenate([np.atleast_1d(data[k][0]) for k in sorted(data)])
                )
                + 1.0
            )
        )
    )
    params.add("A", value=a0)
    params.add("B", value=0.0)
    for band, t in keys:
        y = data[band, t][1]
        params.add(f"s0_{band}_{t}", value=float(np.nanmin(y)))
        params.add(f"s1_{band}_{t}", value=float(np.nanmax(y)))

    def residual(pars: Parameters) -> np.ndarray:
        parts = []
        for band, t in keys:
            x, y = data[band, t]
            k = pars["A"].value + pars["B"].value * (inv[t] - m)
            s0 = pars[f"s0_{band}_{t}"].value
            s1 = pars[f"s1_{band}_{t}"].value
            if is_ph:
                frac = 1.0 / (1.0 + np.power(10.0, k - x))
            else:
                # ligand: k is pKd, the fraction bound is Langmuir in Kd
                frac = x / (x + np.power(10.0, k))
            parts.append(s0 + (s1 - s0) * frac - y)
        return np.concatenate(parts)

    res = lmfit.Minimizer(residual, params).minimize()
    if not getattr(res, "success", False) or res.covar is None:
        return VanthoffFit(
            ok=False,
            k_by_temp={},
            a=float("nan"),
            b=float("nan"),
            dh=float("nan"),
            ds=float("nan"),
            se_a=float("nan"),
            se_b=float("nan"),
            n=0,
            plateaus={},
            inv={},
            m=float("nan"),
        )
    a: float = res.params["A"].value
    b: float = res.params["B"].value
    cov = np.asarray(res.covar, dtype=float)
    ia = list(res.params.keys()).index("A")
    ib = list(res.params.keys()).index("B")
    var_a, var_b, cov_ab = cov[ia, ia], cov[ib, ib], cov[ia, ib]
    k_by_temp = {}
    for t in temps:
        d = inv[t] - m
        pk = a + b * d
        se_pk = float(np.sqrt(max(var_a + 2 * d * cov_ab + d * d * var_b, 0.0)))
        if is_ph:
            k_by_temp[t] = (pk, se_pk)
        else:  # report Kd on its native (mM) scale; log->linear error propagation
            kd = float(np.power(10.0, pk))
            k_by_temp[t] = (
                kd,
                kd * LN10 * se_pk if np.isfinite(se_pk) else float("nan"),
            )
    n = int(sum(len(np.atleast_1d(data[k][0])) for k in keys))
    plateaus = {
        (band, t): (
            float(res.params[f"s0_{band}_{t}"].value),
            float(res.params[f"s1_{band}_{t}"].value),
        )
        for band, t in keys
    }
    # pH: Ka = 10^(-pKa) -> dH = +ln10 R b. Ligand: Kd itself is modelled,
    # ln Kd = ln10 (a - b m) + ln10 b / T against d lnK/d(1/T) = -dH/R, so
    # both thermodynamic signs flip relative to the pKa form.
    sign = 1.0 if is_ph else -1.0
    return VanthoffFit(
        ok=True,
        k_by_temp=k_by_temp,
        a=a,
        b=b,
        dh=sign * LN10 * R_GAS * b,
        ds=-sign * LN10 * R_GAS * (a - b * m),
        se_a=float(np.sqrt(var_a)),
        se_b=float(np.sqrt(var_b)),
        n=n,
        plateaus=plateaus,
        inv=inv,
        m=m,
    )


def _curves_by_temp(  # ruff: ignore[too-many-arguments]
    ef: EnspireFile,
    samples: pd.DataFrame,
    buffers: pd.DataFrame,
    *,
    is_ph: bool,
    buffer: bool,
    wanted: Sequence[str],
    min_rho: float,
) -> dict[str, dict[str, tuple[ArrayF, ArrayF]]]:
    """Corrected, scaled curves per (temperature, band), as the band path does."""
    out: dict[str, dict[str, tuple[ArrayF, ArrayF]]] = {}
    for temp, group in samples.groupby("Temp", sort=True, dropna=False):
        t = str(temp).split(".")[0]
        one = group.drop_duplicates("Well")
        wells = one["Well"].tolist()
        titrant = one["pH"] if is_ph else one["Cl"]
        x = np.array(titrant.astype(float), dtype=np.float64)
        readouts = classify(ef, t, is_ph=is_ph)
        if not readouts:
            continue
        bufw = (
            buffers[np.isclose(buffers["Temp"].astype(float), float(t))][
                "Well"
            ].tolist()
            if len(buffers)
            else []
        )
        probe = band_values(ef, readouts[0], wells)
        blanks = blank_rows(probe)
        if blanks:
            found = [w for w in wells if w[0] in blanks]
            keep = [i for i, w in enumerate(wells) if w[0] not in blanks]
            wells, x = [wells[i] for i in keep], x[keep]
            bufw = bufw or found
        store: dict[str, tuple[ArrayF, ArrayF]] = {}
        for r in readouts:
            if r.name not in wanted:
                continue
            y = band_values(ef, r, wells)
            if bufw and buffer:
                y = subtract_buffer(y, band_values(ef, r, bufw))
            arr = np.array(y, dtype=np.float64)
            top = float(np.nanmax(np.abs(arr)))
            if not np.isfinite(top) or top <= 0.0:
                continue
            arr /= top
            if np.isfinite(min_rho) and abs(band_rho(x, arr, wells)) < min_rho:
                continue  # a flat band cannot inform K
            store[r.name] = (x, arr)
        if store:
            out[t] = store
    return out


def vanthoff_titration(  # ruff: ignore[too-many-arguments]
    ef: EnspireFile,
    note: pd.DataFrame,
    *,
    buffer: bool = True,
    is_ph: bool = True,
    wanted: Sequence[str] = DIRECT_BANDS,
    min_rho: float = 0.8,
) -> tuple[pd.DataFrame, dict[str, VanthoffModel]]:
    """Joint van't Hoff fit per sample of the note; one row per (sample, temp).

    Parameters
    ----------
    ef : EnspireFile
        The parsed EnSpire export.
    note : pd.DataFrame
        Normalised note with ``Well``, ``pH``, ``Cl``, ``Name`` and ``Temp``.
    buffer : bool
        Subtract the buffer wells of the same plate column.
    is_ph : bool
        pH titration (else chloride): which column is the titrant.
    wanted : Sequence[str]
        Band names entering the joint fit; default the two direct readouts.
    min_rho : float
        Monotonicity gate below which a band is dropped, as in the band path.

    Returns
    -------
    tuple[pd.DataFrame, dict[str, VanthoffModel]]
        One row per (sample, temperature): ``K``, ``se``, plus per-sample
        ``A``, ``B``, ``dH_J_mol``, ``dS_J_molK``, ``n``; and per sample the
        fitted pieces needed to redraw the joint model.

    Notes
    -----
    pH titrations: pKa is the fitted midpoint (linear in 1/T). Ligand
    titrations (``is_ph=False``): pKd = -log10 Kd is what is linear in 1/T,
    so the model carries pKd(T) internally and the table reports Kd on its
    native mM scale (errors propagated log -> linear). The enthalpy is that
    of deprotonation / ligand dissociation respectively.
    """
    samples = note[note["Name"] != "buffer"]
    buffers = note[note["Name"] == "buffer"]
    rows: list[dict[str, object]] = []
    models: dict[str, VanthoffModel] = {}
    for name, group in samples.groupby("Name", sort=True):
        curves = _curves_by_temp(
            ef,
            group,
            buffers,
            is_ph=is_ph,
            buffer=buffer,
            wanted=wanted,
            min_rho=min_rho,
        )
        flat = {
            (b, t): pair for t, store in curves.items() for b, pair in store.items()
        }
        fit = fit_vanthoff_model(flat, is_ph=is_ph)
        if not fit.ok:
            rows.append({
                "sample": name,
                "temp": "+".join(sorted(curves)) or "none",
                "K": np.nan,
                "se": np.nan,
                "A": np.nan,
                "B": np.nan,
                "dH_J_mol": np.nan,
                "dS_J_molK": np.nan,
                "n": fit.n,
            })
            continue
        models[str(name)] = VanthoffModel(
            a=fit.a,
            b=fit.b,
            inv=fit.inv,
            m=fit.m,
            k_by_temp=fit.k_by_temp,
            plateaus=fit.plateaus,
            curves=flat,
        )
        for t, (k, se) in sorted(fit.k_by_temp.items()):
            rows.append({
                "sample": name,
                "temp": t,
                "K": k,
                "se": se,
                "A": fit.a,
                "B": fit.b,
                "dH_J_mol": fit.dh,
                "dS_J_molK": fit.ds,
                "n": fit.n,
            })
    return pd.DataFrame(rows), models
