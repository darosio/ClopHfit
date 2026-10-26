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
import pymc as pm  # type: ignore[import-untyped]
from lmfit import Parameters

from clophfit.prenspire.bands import (
    DIRECT_BANDS,
    band_rho,
    band_values,
    blank_rows,
    classify,
    subtract_buffer,
)
from clophfit.prenspire.prenspire import EnspireFile, merge_enspire_files

if TYPE_CHECKING:
    from collections.abc import Sequence

    import arviz as az  # type: ignore[import-untyped]

    from clophfit.clophfit_types import ArrayF

R_GAS = 8.314462618  # J mol-1 K-1
LN10 = 2.302585093
ZERO_C = 273.15
_MIN_TEMPS = 2  # one enthalpy needs a pair of temperatures


def _kelvin(temp: str | float) -> float:
    return float(temp) + ZERO_C


def titrant_kind(note: pd.DataFrame) -> dict[str, bool]:
    """Decide per sample whether its titrant axis is pH or chloride.

    The titrant is the column that varies within a sample: a pH titration
    steps the pH and leaves the chloride alone, a chloride titration the
    other way round. Deciding once for the whole note would fit a chloride
    sample against its constant pH column and report a meaningless K.

    Parameters
    ----------
    note : pd.DataFrame
        Normalised note with ``Well``, ``pH``, ``Cl``, ``Name`` and ``Temp``.

    Returns
    -------
    dict[str, bool]
        Sample name to True when the sample is a pH titration.
    """
    kinds: dict[str, bool] = {}
    for name, group in note.groupby("Name", sort=True):
        one = group.drop_duplicates("Well")
        kinds[str(name)] = bool(one["pH"].nunique() > one["Cl"].nunique())
    return kinds


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
    is_ph: bool = True
    """Whether the sample was fitted as a pH (else ligand) titration."""


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


def fit_vanthoff_model_bayes(  # ruff: ignore[too-many-arguments]
    data: dict[tuple[str, str], tuple[ArrayF, ArrayF]],
    *,
    is_ph: bool = True,
    x_sigma: float | None = None,
    seed: int = 0,
    draws: int = 2000,
    chains: int = 2,
    cores: int | None = None,
    target_accept: float = 0.9,
    tune: int | None = None,
) -> az.InferenceData:
    """Posterior of the joint van't Hoff fit over every temperature at once.

    The Bayesian counterpart of :func:`fit_vanthoff_model`: the same model -
    ``pKa(T) = A + B (1/T - m)`` shared by every band and temperature, the
    plateaus free per ``(band, temperature)`` - but the interval is a highest
    density interval read off a NUTS posterior instead of an LM covariance,
    and ``dH``/``dS`` get intervals of their own rather than being propagated
    from two correlated coefficients.

    The priors are seeded from the least-squares joint fit: ``A`` and ``B``
    are centered on it with a width of four least-squares standard errors
    (floored so a degenerate least-squares error cannot collapse the prior),
    the plateaus on their least-squares values with a width of a tenth of the
    normalised signal axis, and the noise scale of each curve on a half-normal
    at 0.05. This is the same documented empirical-Bayes reuse the band-level
    and whole-spectrum Bayesian fitters make; the floor is what keeps it a
    start rather than a pin.

    Parameters
    ----------
    data : dict[tuple[str, str], tuple[ArrayF, ArrayF]]
        Band name and temperature label to its ``(x, y)`` curve, y already
        normalised and scaled by the caller - the shape
        :func:`vanthoff_curves` produces. Every band of one temperature must
        carry the same titrant axis, since they were read off the same wells.
    is_ph : bool
        pH titration (fit pKa(T) against a Henderson-Hasselbalch fraction) or
        ligand titration (fit pKd(T) against a Langmuir fraction; the
        reported ``K`` is Kd on the x scale).
    x_sigma : float | None
        Prior SD on each well's true titrant value,
        ``x_true ~ N(note value, x_sigma)``, per well and temperature.
        ``None`` keeps the axis fixed at the note values, as the
        least-squares fit does.
    seed : int
        NUTS seed; a fixed seed reproduces the posterior.
    draws : int
        Posterior draws per chain.
    chains : int
        Number of NUTS chains; two is the minimum that leaves R-hat defined.
    cores : int | None
        Chain parallelism; None chains in parallel.
    target_accept : float
        NUTS target acceptance.
    tune : int | None
        Tuning draws; None keeps ``draws // 2``.

    Returns
    -------
    az.InferenceData
        Posterior draws for ``A``, ``B``, ``dH_J_mol``, ``dS_J_molK``, the
        per-temperature constants ``pK_<t>`` (the fitted pKa or pKd) and
        ``K_<t>`` (the same pKa, or Kd on the native x scale), the per-curve
        plateaus ``s0_<band>_<t>``/``s1_<band>_<t>`` and noise scales, and
        the latent ``x_true_<t>`` when a prior width was given.

    Raises
    ------
    ValueError
        With fewer than two temperatures, a non-positive ``x_sigma``, bands of
        one temperature that disagree on their axis, or a least-squares joint
        fit that did not converge (no posterior is started then).
    """
    temps = sorted({t for _band, t in data})
    if not data or len(temps) < _MIN_TEMPS:
        msg = f"a joint van't Hoff posterior needs at least {_MIN_TEMPS} temperatures"
        raise ValueError(msg)
    if x_sigma is not None and not (np.isfinite(x_sigma) and x_sigma > 0.0):
        msg = "x_sigma must be positive and finite, or None"
        raise ValueError(msg)
    init = fit_vanthoff_model(data, is_ph=is_ph)
    if not init.ok:
        msg = "the least-squares van't Hoff fit did not converge; no posterior started"
        raise ValueError(msg)
    keys = sorted(data)
    # every band of one temperature was read off the same wells, so the curves
    # must agree on that temperature's titrant axis before one is sampled
    axis: dict[str, np.ndarray] = {}
    for t in temps:
        per_band = [np.asarray(data[k][0], dtype=np.float64) for k in keys if k[1] == t]
        first = per_band[0]
        for other in per_band[1:]:
            if other.shape != first.shape or not np.allclose(other, first):
                msg = f"the bands of {t} C disagree on their titrant axis"
                raise ValueError(msg)
        axis[t] = first
    # pH: Ka = 10^(-pKa) -> dH = +ln10 R B. Ligand: Kd itself is modelled,
    # ln Kd = ln10 (a - b m) + ln10 b / T against d ln Kd/d(1/T) = -dH/R, so
    # both thermodynamic signs flip relative to the pKa form, as in the LM fit.
    sign = 1.0 if is_ph else -1.0
    m = init.m
    with pm.Model():
        a = pm.Normal("A", mu=init.a, sigma=max(4.0 * init.se_a, 0.5))
        b = pm.Normal("B", mu=init.b, sigma=max(4.0 * init.se_b, 200.0))
        pm.Deterministic("dH_J_mol", sign * LN10 * R_GAS * b)
        pm.Deterministic("dS_J_molK", -sign * LN10 * R_GAS * (a - b * m))
        xd: dict[str, np.ndarray | pm.math.TensorVariable] = {}
        for t in temps:
            if x_sigma is None:
                xd[t] = axis[t]
            else:
                xd[t] = pm.Normal(
                    f"x_true_{t}", mu=axis[t], sigma=x_sigma, shape=axis[t].size
                )
            pk = a + b * (init.inv[t] - m)
            pm.Deterministic(f"pK_{t}", pk)
            kd = pm.math.exp(LN10 * pk)
            pm.Deterministic(f"K_{t}", pk if is_ph else kd)
            frac = (
                1.0 / (1.0 + pm.math.exp(LN10 * (pk - xd[t])))
                if is_ph
                else xd[t] / (xd[t] + kd)
            )
            for band in sorted({band for band, t_ in keys if t_ == t}):
                s0_ls, s1_ls = init.plateaus[band, t]
                s0 = pm.Normal(f"s0_{band}_{t}", mu=s0_ls, sigma=0.1)
                s1 = pm.Normal(f"s1_{band}_{t}", mu=s1_ls, sigma=0.1)
                sigma = pm.HalfNormal(f"sigma_{band}_{t}", sigma=0.05)
                pm.Normal(
                    f"obs_{band}_{t}",
                    mu=s0 + (s1 - s0) * frac,
                    sigma=sigma,
                    observed=np.asarray(data[band, t][1], dtype=np.float64),
                )
        return pm.sample(
            draws,
            tune=draws // 2 if tune is None else tune,
            chains=chains,
            cores=cores,
            target_accept=target_accept,
            random_seed=seed,
            initvals={"A": init.a, "B": init.b},
            return_inferencedata=True,
            progressbar=False,
        )


class _TempCurves(NamedTuple):
    """One temperature's curves and the wells they were read off."""

    wells: list[str]
    store: dict[str, tuple[ArrayF, ArrayF]]


def _as_single_export(
    ef: EnspireFile | Sequence[EnspireFile],
) -> EnspireFile:
    """One export, merging the list when the titration spans several files."""
    if isinstance(ef, EnspireFile):
        return ef
    return merge_enspire_files(list(ef))


def _curves_by_temp(  # ruff: ignore[too-many-arguments]
    ef: EnspireFile,
    samples: pd.DataFrame,
    buffers: pd.DataFrame,
    *,
    is_ph: bool,
    buffer: bool,
    wanted: Sequence[str],
    min_rho: float,
) -> dict[str, _TempCurves]:
    """Corrected, scaled curves per (temperature, band), as the band path does."""
    out: dict[str, _TempCurves] = {}
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
            # the wells are the ones that survived the blank and finite-signal
            # gates, so a caller mapping a per-well vector back to well names
            # must use this list and not the note's
            out[t] = _TempCurves(wells=wells, store=store)
    return out


class VanthoffCurves(NamedTuple):
    """One sample's corrected curves and the titrant kind they were read on."""

    is_ph: bool
    """Whether the sample was read as a pH (else ligand) titration."""
    curves: dict[tuple[str, str], tuple[ArrayF, ArrayF]]
    """``(band, temperature)`` to its ``(x, y)`` curve, y normalised per band."""
    wells: dict[str, list[str]]
    """Temperature label to the wells its curves were read off, in order."""


def vanthoff_curves(  # ruff: ignore[too-many-arguments]
    ef: EnspireFile | Sequence[EnspireFile],
    note: pd.DataFrame,
    *,
    buffer: bool = True,
    is_ph: bool | None = None,
    wanted: Sequence[str] = DIRECT_BANDS,
    min_rho: float = 0.8,
) -> dict[str, VanthoffCurves]:
    """Corrected curves of every sample, as both van't Hoff fitters consume them.

    The assembly is shared so that the least-squares joint fit
    (:func:`fit_vanthoff_model`) and its posterior counterpart
    (:func:`fit_vanthoff_model_bayes`) read the same wells, the same bands and
    the same titrant axis: a comparison between the two is then a comparison
    of the inference, not of the preprocessing.

    Parameters
    ----------
    ef : EnspireFile | Sequence[EnspireFile]
        The parsed EnSpire export, or several when the titration is split over
        files (one per temperature); they are merged, and they must carry
        disjoint measurement labels.
    note : pd.DataFrame
        Normalised note with ``Well``, ``pH``, ``Cl``, ``Name`` and ``Temp``.
    buffer : bool
        Subtract the buffer wells of the same plate column.
    is_ph : bool | None
        Force the titrant column for every sample. ``None`` decides per sample
        with :func:`titrant_kind`.
    wanted : Sequence[str]
        Band names entering the joint fit; default the two direct readouts.
    min_rho : float
        Monotonicity gate below which a band is dropped, as in the band path.

    Returns
    -------
    dict[str, VanthoffCurves]
        Sample name to its titrant kind and its ``(band, temperature)``
        curves, the signal already normalised per band.
    """
    ef = _as_single_export(ef)
    samples = note[note["Name"] != "buffer"]
    buffers = note[note["Name"] == "buffer"]
    kinds = titrant_kind(samples)
    out: dict[str, VanthoffCurves] = {}
    for name, group in samples.groupby("Name", sort=True):
        sample_is_ph = kinds[str(name)] if is_ph is None else is_ph
        curves = _curves_by_temp(
            ef,
            group,
            buffers,
            is_ph=sample_is_ph,
            buffer=buffer,
            wanted=wanted,
            min_rho=min_rho,
        )
        out[str(name)] = VanthoffCurves(
            is_ph=sample_is_ph,
            curves={
                (band, t): pair
                for t, one in curves.items()
                for band, pair in one.store.items()
            },
            wells={t: one.wells for t, one in curves.items()},
        )
    return out


def vanthoff_titration(  # ruff: ignore[too-many-arguments]
    ef: EnspireFile | Sequence[EnspireFile],
    note: pd.DataFrame,
    *,
    buffer: bool = True,
    is_ph: bool | None = None,
    wanted: Sequence[str] = DIRECT_BANDS,
    min_rho: float = 0.8,
) -> tuple[pd.DataFrame, dict[str, VanthoffModel]]:
    """Joint van't Hoff fit per sample of the note; one row per (sample, temp).

    Parameters
    ----------
    ef : EnspireFile | Sequence[EnspireFile]
        The parsed EnSpire export, or several when the titration is split over
        files (one per temperature); they are merged, and they must carry
        disjoint measurement labels.
    note : pd.DataFrame
        Normalised note with ``Well``, ``pH``, ``Cl``, ``Name`` and ``Temp``.
    buffer : bool
        Subtract the buffer wells of the same plate column.
    is_ph : bool | None
        Force the titrant column for every sample. ``None`` decides per
        sample with :func:`titrant_kind`, which is what a note mixing a pH
        and a chloride titration needs.
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
    rows: list[dict[str, object]] = []
    models: dict[str, VanthoffModel] = {}
    for name, info in vanthoff_curves(
        ef, note, buffer=buffer, is_ph=is_ph, wanted=wanted, min_rho=min_rho
    ).items():
        sample_is_ph = info.is_ph
        flat = info.curves
        fit = fit_vanthoff_model(flat, is_ph=sample_is_ph)
        if not fit.ok:
            rows.append({
                "sample": name,
                "temp": "+".join(sorted({t for _band, t in flat})) or "none",
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
            is_ph=sample_is_ph,
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
