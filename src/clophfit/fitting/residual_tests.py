"""Formal residual diagnostics: the four assumptions a least-squares fit makes.

Each assumption has a picture that shows it failing and a test that measures it:

======================  ===============================  ===============================
Assumption              What failure looks like          Test
======================  ===============================  ===============================
Homoscedasticity        funnel / fan against prediction  Breusch-Pagan (Koenker), White
Independence            waves along the titration        Durbin-Watson, lag-1 r, runs
Normality               curved QQ, skewed histogram      Shapiro-Wilk, KS, Anderson-Darling
No outliers             isolated extreme points          externally studentized t, PRESS
======================  ===============================  ===============================

Two things make these tests easy to misread on titration plates, and every
function here reports accordingly:

* **They are pooled over hundreds of points**, so any real effect, however
  small, reaches p ~ 0. Each test therefore also returns an effect size - R^2,
  the DW statistic, W, D, a rate or a ratio - and those, not the p-values, are
  what compare models.
* **The residuals belong to a fit.** A plate spends several parameters per well,
  so residuals shrink and pick up induced correlation even when the noise is
  independent and calibrated. Durbin-Watson is not expected at exactly 2, and
  the Kolmogorov-Smirnov test against N(0, 1) mixes shape with that shrinkage
  (see ``residuals.qq_fit``); the Lilliefors form, with location and scale
  estimated, tests shape alone.

The outlier and PRESS measures need each point's leverage, computed per well
from the weighted Jacobian of :func:`~clophfit.fitting.models.binding_1site` at
the supplied parameters. For a Bayesian multi-well fit those are posterior
means and the latent per-well pH offset is not counted, so leverage is an
approximation there; for a classical per-well fit it is exact.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from scipy import stats

from clophfit.fitting.models import ACID_SCALE, binding_1site
from clophfit.fitting.utils import bonferroni_threshold, studentized_scores

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = [
    "breusch_pagan",
    "durbin_watson",
    "durbin_watson_null",
    "normality_tests",
    "press_statistics",
    "residual_tests",
    "runs_test",
    "runs_test_null",
    "studentized_outliers",
    "well_leverage",
    "white_test",
]

_MIN_POINTS = 8  # below this a per-label test says nothing
_OUTLIER_ALPHA = 0.05


def _lm_test(z: np.ndarray, design: np.ndarray) -> dict[str, float]:
    """Koenker's studentized LM test: n R^2 of z^2 on the design, chi^2 with k dof.

    Parameters
    ----------
    z : np.ndarray
        Standardised residuals.
    design : np.ndarray
        ``(n, k)`` regressors, without the constant.

    Returns
    -------
    dict[str, float]
        ``lm``, ``df``, ``p`` and ``r2``.
    """
    ok = np.isfinite(z) & np.isfinite(design).all(axis=1)
    z, design = z[ok], design[ok]
    n, k = design.shape
    if n <= k + 1:
        return {"lm": np.nan, "df": float(k), "p": np.nan, "r2": np.nan}
    # Regressors standardised so a squared signal of 1e6 does not wreck the fit.
    sd = design.std(axis=0)
    design = (design - design.mean(axis=0)) / np.where(sd > 0, sd, 1.0)
    target = z**2
    a = np.column_stack([np.ones(n), design])
    coef, *_ = np.linalg.lstsq(a, target, rcond=None)
    resid = target - a @ coef
    ss_tot = float(np.sum((target - target.mean()) ** 2))
    r2 = 1.0 - float(resid @ resid) / ss_tot if ss_tot > 0 else 0.0
    lm = n * r2
    return {"lm": lm, "df": float(k), "p": float(stats.chi2.sf(lm, k)), "r2": r2}


def breusch_pagan(z: np.ndarray, yhat: np.ndarray, x: np.ndarray) -> dict[str, float]:
    """Test whether the squared residual follows the prediction or x (Breusch-Pagan).

    The studentized form, ``n R^2`` of ``z^2`` regressed on the regressors, is
    used rather than the original, which assumes normal errors and rejects on
    kurtosis alone.

    Parameters
    ----------
    z : np.ndarray
        Standardised residuals.
    yhat : np.ndarray
        Predictions.
    x : np.ndarray
        Titration coordinate (pH or concentration).

    Returns
    -------
    dict[str, float]
        ``lm``, ``df`` (2), ``p`` and ``r2`` - the fraction of the squared
        residuals' variance the fan explains, the effect size to compare.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> y = rng.uniform(100, 1000, 800)
    >>> fan = breusch_pagan(rng.normal(0, y / 500), y, rng.uniform(5, 9, 800))
    >>> fan["p"] < 1e-6
    True
    """
    design = np.column_stack([np.asarray(yhat, float), np.asarray(x, float)])
    return _lm_test(np.asarray(z, float), design)


def white_test(z: np.ndarray, yhat: np.ndarray, x: np.ndarray) -> dict[str, float]:
    """White's test: squared residual on ŷ, x, their squares and cross product.

    Catches a variance that bends with the signal - high at both ends, say -
    which the linear Breusch-Pagan design cannot see.

    Parameters
    ----------
    z : np.ndarray
        Standardised residuals.
    yhat : np.ndarray
        Predictions.
    x : np.ndarray
        Titration coordinate.

    Returns
    -------
    dict[str, float]
        ``lm``, ``df`` (5), ``p`` and ``r2``.
    """
    y, xx = np.asarray(yhat, float), np.asarray(x, float)
    design = np.column_stack([y, xx, y**2, xx**2, y * xx])
    return _lm_test(np.asarray(z, float), design)


def durbin_watson(table: pd.DataFrame, *, z_col: str = "std_res") -> dict[str, float]:
    """Durbin-Watson pooled within series: each well's label ordered along the titration.

    ``DW = sum((e_t - e_{t-1})^2) / sum(e_t^2)`` with the differences taken only
    inside a (well, label) series, so a jump between wells is never counted.
    Well below its null, neighbouring steps share an error (a wave, or a
    misplaced x); above it, they alternate. The null is **not 2** for series this
    short: independent *errors* give ``2 (n - 1) / n`` per series - 1.71 for a
    seven-step titration - which ``dw_expected`` reports. Fitted *residuals* are
    not independent even then: the fit makes neighbours alternate, and for a
    binding curve that lifts the null well above 2. Use
    :func:`durbin_watson_null` for the null of the fitted residuals.

    Parameters
    ----------
    table : pd.DataFrame
        Residual table with ``well``, ``label``, ``step`` (or ``x``) and *z_col*.
    z_col : str
        The residual column.

    Returns
    -------
    dict[str, float]
        ``dw``, ``dw_expected`` (its value for independent errors of these series
        lengths), the pooled lag-1 correlation ``lag1_r``, ``n_pairs`` and ``p``,
        a two-sided normal-approximation p-value for ``lag1_r`` against 0.
    """
    order = "step" if "step" in table else "x"
    num = den = 0.0
    n_points = n_diffs = 0
    a: list[np.ndarray] = []
    b: list[np.ndarray] = []
    for _, g in table.groupby(["well", "label"], sort=False):
        e = g.sort_values(order)[z_col].to_numpy(dtype=float)
        e = e[np.isfinite(e)]
        if e.size < 2:  # ruff: ignore[magic-value-comparison] - a difference needs two points
            continue
        num += float(np.sum(np.diff(e) ** 2))
        den += float(np.sum(e**2))
        n_points += e.size
        n_diffs += e.size - 1
        a.append(e[:-1])
        b.append(e[1:])
    if den <= 0 or not a:
        return {
            "dw": np.nan,
            "dw_expected": np.nan,
            "lag1_r": np.nan,
            "n_pairs": 0.0,
            "p": np.nan,
        }
    lo, hi = np.concatenate(a), np.concatenate(b)
    r = float(np.corrcoef(lo, hi)[0, 1]) if lo.size > 2 else np.nan  # ruff: ignore[magic-value-comparison]
    n = float(lo.size)
    p = float(2 * stats.norm.sf(abs(r) * np.sqrt(n))) if np.isfinite(r) else np.nan
    return {
        "dw": num / den,
        "dw_expected": 2.0 * n_diffs / n_points,
        "lag1_r": r,
        "n_pairs": n,
        "p": p,
    }


def normality_tests(
    z: np.ndarray, *, n_mc: int = 999, seed: int = 0
) -> dict[str, float]:
    """Shapiro-Wilk, Kolmogorov-Smirnov against N(0, 1), Lilliefors and Anderson-Darling.

    * Shapiro-Wilk ``W`` measures shape only; it is the most powerful of the
      three, so on hundreds of points it rejects small departures - read W.
    * KS against N(0, 1) tests shape *and* scale: a calibrated fit still fails it
      when its residuals are shrunk by the parameters it spent.
    * Lilliefors is KS with location and scale estimated, p by Monte Carlo
      (``scipy.stats.goodness_of_fit``): shape only, like Shapiro-Wilk, but
      weighted towards the centre rather than the tails.
    * Anderson-Darling, also with location and scale estimated and p by Monte
      Carlo: shape only, weighted towards the tails, where a Student-t or
      mixture likelihood is meant to differ from the Normal.

    Parameters
    ----------
    z : np.ndarray
        Standardised residuals.
    n_mc : int
        Monte Carlo samples for the Lilliefors and Anderson-Darling p-values.
    seed : int
        Seed for those samples, so a report is reproducible.

    Returns
    -------
    dict[str, float]
        ``sw_w``, ``sw_p``, ``ks_n01_d``, ``ks_n01_p``, ``lillie_d``, ``lillie_p``,
        ``ad_a2``, ``ad_p``.
    """
    v = np.asarray(z, dtype=float)
    v = v[np.isfinite(v)]
    out = dict.fromkeys(
        [
            "sw_w",
            "sw_p",
            "ks_n01_d",
            "ks_n01_p",
            "lillie_d",
            "lillie_p",
            "ad_a2",
            "ad_p",
        ],
        np.nan,
    )
    if v.size < _MIN_POINTS:
        return out
    sw = stats.shapiro(v)
    ks = stats.kstest(v, "norm")
    lil = stats.goodness_of_fit(
        stats.norm,
        v,
        statistic="ks",
        n_mc_samples=n_mc,
        rng=np.random.default_rng(seed),
    )
    ad = stats.goodness_of_fit(
        stats.norm,
        v,
        statistic="ad",
        n_mc_samples=n_mc,
        rng=np.random.default_rng(seed),
    )
    out.update(
        sw_w=float(sw.statistic),
        sw_p=float(sw.pvalue),
        ks_n01_d=float(ks.statistic),
        ks_n01_p=float(ks.pvalue),
        lillie_d=float(lil.statistic),
        lillie_p=float(lil.pvalue),
        ad_a2=float(ad.statistic),
        ad_p=float(ad.pvalue),
    )
    return out


def _well_design(
    g: pd.DataFrame, params: Mapping[str, float], *, is_ph: bool
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Weighted residuals and weighted Jacobian of one well's binding fit.

    Parameters
    ----------
    g : pd.DataFrame
        The well's residual rows (both labels): ``label``, ``x``, ``raw_res``, ``sigma``.
    params : Mapping[str, float]
        ``K`` and ``S0_<label>``/``S1_<label>`` for the labels present, and
        ``acid_scale`` when the fit had the acid-step factor (applied to the rows
        whose ``acid_step`` column is true).
    is_ph : bool
        pH titration (else concentration).

    Returns
    -------
    tuple[np.ndarray, np.ndarray, list[str]]
        Weighted residuals, the ``(n, p)`` weighted Jacobian, and the parameter
        names its columns follow.
    """
    labels = [str(lbl) for lbl in dict.fromkeys(g["label"].astype(str))]
    names = ["K"] + [f"{s}_{lbl}" for lbl in labels for s in ("S0", "S1")]
    x = g["x"].to_numpy(dtype=float)
    lab = g["label"].astype(str).to_numpy()
    w = 1.0 / g["sigma"].to_numpy(dtype=float)
    # A fit with the acid-step factor spends one more parameter, shared by the
    # labels' acid rows; leaving it out would understate their leverage.
    acid = (
        g["acid_step"].to_numpy(dtype=bool)
        if ACID_SCALE in params and "acid_step" in g
        else np.zeros(x.size, dtype=bool)
    )
    if acid.any():
        names.append(ACID_SCALE)

    def predict(values: Mapping[str, float]) -> np.ndarray:
        out = np.empty_like(x)
        for lbl in labels:
            m = lab == lbl
            out[m] = binding_1site(
                x[m], values["K"], values[f"S0_{lbl}"], values[f"S1_{lbl}"], is_ph=is_ph
            )
        if acid.any():
            out[acid] *= values[ACID_SCALE]
        return out

    base = {n: float(params[n]) for n in names}
    jac = np.empty((x.size, len(names)))
    for j, name in enumerate(names):
        step = 1e-6 * max(abs(base[name]), 1.0)
        up, down = dict(base), dict(base)
        up[name] += step
        down[name] -= step
        jac[:, j] = (predict(up) - predict(down)) / (2 * step)
    r_w = g["raw_res"].to_numpy(dtype=float) * w
    return r_w, jac * w[:, None], names


def _well_linearised(
    g: pd.DataFrame, params: Mapping[str, float], *, is_ph: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Leverage and least-squares residuals of one well, linearised at *params*.

    The closed-form leave-one-out formulas assume residuals of the least-squares
    solution, which are orthogonal to the Jacobian. Posterior-mean residuals of
    a Bayesian fit are not, and neither are residuals at any other supplied
    parameters; fed in raw, a large point can drive the leave-one-out variance
    negative and its score to zero. Projecting through ``I - H`` gives the
    residuals of the one-step (Gauss-Newton) refit at those parameters: exactly
    the input when it already is a least-squares fit, and the right thing to
    test otherwise.

    Parameters
    ----------
    g : pd.DataFrame
        The well's residual rows.
    params : Mapping[str, float]
        Its fitted parameters.
    is_ph : bool
        pH titration (else concentration).

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None
        ``h_ii``, the projected weighted residuals, the weights ``1 / sigma``
        and the weighted Jacobian; ``None`` when the well cannot be linearised.
    """
    try:
        r_w, jac, _ = _well_design(g, params, is_ph=is_ph)
    except KeyError:
        return None
    if (
        not (np.isfinite(r_w).all() and np.isfinite(jac).all())
        or jac.shape[0] <= jac.shape[1]
    ):
        return None
    q, _ = np.linalg.qr(jac)
    h = np.clip(np.einsum("ij,ij->i", q, q), 0.0, 1.0 - 1e-12)
    r_ls = r_w - q @ (q.T @ r_w)
    w = 1.0 / g["sigma"].to_numpy(dtype=float)
    return h, r_ls, w, jac


def durbin_watson_null(
    table: pd.DataFrame,
    params: Mapping[str, Mapping[str, float]],
    *,
    is_ph: bool = True,
) -> pd.DataFrame:
    """Durbin-Watson and lag-1 correlation expected of *fitted* residuals, per label.

    With independent, calibrated errors the weighted residuals of a
    least-squares fit have covariance ``I - H``, ``H`` the hat matrix of the
    well's weighted Jacobian (both labels, one shared K). So, per series,
    ``E[sum (e_t - e_{t-1})^2] = tr(D'D (I - H))`` and ``E[sum e_t^2] = tr(I - H)``
    with ``D`` the first-difference operator along the steps; pooling those
    traces over wells gives the null DW of the pooled statistic, and the summed
    lag-1 covariances over the summed variances of the leading and trailing
    points the null lag-1 correlation. Plateau points carry the most leverage,
    so the series ends are the least variable, which is why the two differ.
    A multi-well Bayesian fit spends more than these parameters (the latent pH
    axis), so its true null lies further from the independent-error value.

    Parameters
    ----------
    table : pd.DataFrame
        Residual table with ``well``, ``label``, ``step`` (or ``x``), ``x``,
        ``raw_res`` and ``sigma``.
    params : Mapping[str, Mapping[str, float]]
        Well to fitted ``K``/``S0_<label>``/``S1_<label>``.
    is_ph : bool
        pH titration (else concentration).

    Returns
    -------
    pd.DataFrame
        Indexed by label: ``dw_expected_fit`` and ``lag1_expected``.

    Examples
    --------
    A straight line fitted to seven independent points: its residuals alternate,
    so the null is above the independent-error 12/7.

    >>> import numpy as np, pandas as pd
    >>> x = np.linspace(0, 1, 7)
    >>> q, _ = np.linalg.qr(np.column_stack([np.ones(7), x]))
    >>> s = np.eye(7) - q @ q.T
    >>> d = np.diff(np.eye(7), axis=0)
    >>> round(float(np.trace(d.T @ d @ s) / np.trace(s)), 2)
    2.36
    """
    order = "step" if "step" in table else "x"
    t = table.assign(label=table["label"].astype(str))
    finite = np.isfinite(t["raw_res"].to_numpy(float)) & np.isfinite(
        t["sigma"].to_numpy(float)
    )
    t = t[finite].sort_values(["well", "label", order])
    # Per label: DW numerator, DW denominator, lag-1 covariance, and the
    # variances of the leading and trailing points of each pair.
    acc: dict[str, np.ndarray] = {}
    for well, g in t.groupby("well", sort=False):
        p = params.get(str(well))
        lin = None if p is None else _well_linearised(g, p, is_ph=is_ph)
        if lin is None:
            continue
        q, _ = np.linalg.qr(lin[3])
        s = np.eye(len(g)) - q @ q.T
        lab = g["label"].to_numpy()
        for lbl in dict.fromkeys(lab):
            idx = np.flatnonzero(lab == lbl)
            if idx.size < 2:  # ruff: ignore[magic-value-comparison] - a difference needs two points
                continue
            sub = s[np.ix_(idx, idx)]
            d = np.diff(np.eye(idx.size), axis=0)
            var = np.diag(sub)
            parts = np.array([
                np.trace(d.T @ d @ sub),
                var.sum(),
                np.trace(sub, offset=1),
                var[:-1].sum(),
                var[1:].sum(),
            ])
            acc[lbl] = acc.get(lbl, np.zeros(5)) + parts
    rows = {
        lbl: {
            "dw_expected_fit": float(v[0] / v[1]) if v[1] > 0 else np.nan,
            "lag1_expected": float(v[2] / np.sqrt(v[3] * v[4]))
            if v[3] * v[4] > 0
            else np.nan,
        }
        for lbl, v in acc.items()
    }
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("label")


def _runs_moments(e: np.ndarray) -> tuple[float, float, float]:
    """Observed runs of signs and their Wald-Wolfowitz mean and variance.

    Parameters
    ----------
    e : np.ndarray
        One series of residuals in order; zeros carry no sign and are dropped.

    Returns
    -------
    tuple[float, float, float]
        Runs, expected runs and variance, given the counts of each sign.
    """
    s = np.sign(e[np.isfinite(e) & (e != 0)])
    n_pos, n_neg = float(np.sum(s > 0)), float(np.sum(s < 0))
    n = n_pos + n_neg
    if n == 0:
        return 0.0, 0.0, 0.0
    runs = 1.0 + float(np.sum(s[1:] != s[:-1]))
    if n_pos == 0 or n_neg == 0:
        return runs, 1.0, 0.0
    two = 2.0 * n_pos * n_neg
    return runs, two / n + 1.0, two * (two - n) / (n**2 * (n - 1.0))


def runs_test(table: pd.DataFrame, *, z_col: str = "std_res") -> dict[str, float]:
    """Wald-Wolfowitz runs test on residual signs, pooled within series.

    A run is a stretch of residuals with one sign along a (well, label) series.
    Too few runs means neighbouring steps share their error, which is the wave
    Durbin-Watson also detects. The runs test uses signs only, so a few large
    residuals, which dominate DW's sums of squares, cannot drive it. It suits
    Student-t and mixture fits, where such points are expected. Runs, their
    expectations and variances (conditional on each series' sign counts) are
    summed over series, so short titrations still pool into a usable test.
    Like DW, the textbook null assumes independent *errors*. Fitted residuals
    alternate more, so a correct fit shows *more* runs than
    ``runs_expected``. Use :func:`runs_test_null` for the fitted-residual null.

    Parameters
    ----------
    table : pd.DataFrame
        Residual table with ``well``, ``label``, ``step`` (or ``x``) and *z_col*.
    z_col : str
        The residual column.

    Returns
    -------
    dict[str, float]
        ``runs`` (observed), ``runs_expected`` and ``runs_z`` under independent
        errors, ``runs_p`` (two-sided, normal approximation) and ``n_series``.

    Examples
    --------
    Twenty series that each change sign once have 40 runs, far fewer than
    independent signs would give:

    >>> import pandas as pd
    >>> rows = [
    ...     {"well": w, "label": "1", "step": i, "std_res": 1.0 if i < 4 else -1.0}
    ...     for w in range(20)
    ...     for i in range(8)
    ... ]
    >>> out = runs_test(pd.DataFrame(rows))
    >>> out["runs"], out["runs_p"] < 1e-6
    (40.0, True)
    """
    order = "step" if "step" in table else "x"
    runs = mean = var = 0.0
    n_series = 0
    for _, g in table.groupby(["well", "label"], sort=False):
        e = g.sort_values(order)[z_col].to_numpy(dtype=float)
        r, m, v = _runs_moments(e)
        if m == 0:
            continue
        runs, mean, var, n_series = runs + r, mean + m, var + v, n_series + 1
    if var <= 0:
        return {
            "runs": runs,
            "runs_expected": mean,
            "runs_z": np.nan,
            "runs_p": np.nan,
            "n_series": float(n_series),
        }
    z = (runs - mean) / np.sqrt(var)
    return {
        "runs": runs,
        "runs_expected": mean,
        "runs_z": float(z),
        "runs_p": float(2 * stats.norm.sf(abs(z))),
        "n_series": float(n_series),
    }


def runs_test_null(
    table: pd.DataFrame,
    params: Mapping[str, Mapping[str, float]],
    *,
    is_ph: bool = True,
    n_mc: int = 999,
    seed: int = 0,
) -> pd.DataFrame:
    """Distribution of the pooled runs count for *fitted* residuals, per label.

    With independent, calibrated errors, a least-squares fit leaves weighted
    residuals ``e = (I - H) u`` with ``u ~ N(0, I)`` and ``H`` the hat matrix of
    the well's weighted Jacobian (both labels, one shared K), as in
    :func:`durbin_watson_null`. The runs count of that projection has no closed
    form, so it is simulated. Each Monte Carlo draw projects fresh noise
    through every well and sums the runs per label, which gives the null
    distribution of the statistic :func:`runs_test` reports.

    Parameters
    ----------
    table : pd.DataFrame
        Residual table with ``well``, ``label``, ``step`` (or ``x``), ``x``,
        ``raw_res`` and ``sigma``.
    params : Mapping[str, Mapping[str, float]]
        Well to fitted ``K``/``S0_<label>``/``S1_<label>``.
    is_ph : bool
        pH titration (else concentration).
    n_mc : int
        Monte Carlo draws.
    seed : int
        Seed for the draws.

    Returns
    -------
    pd.DataFrame
        Indexed by label: ``runs_expected_fit`` and ``runs_sd_fit`` (mean and SD
        of the simulated pooled runs), and ``runs_null`` (the simulated counts,
        as an array, for the p-value in :func:`residual_tests`).
    """
    order = "step" if "step" in table else "x"
    t = table.assign(label=table["label"].astype(str))
    finite = np.isfinite(t["raw_res"].to_numpy(float)) & np.isfinite(
        t["sigma"].to_numpy(float)
    )
    t = t[finite].sort_values(["well", "label", order])
    rng = np.random.default_rng(seed)
    acc: dict[str, np.ndarray] = {}
    for well, g in t.groupby("well", sort=False):
        p = params.get(str(well))
        lin = None if p is None else _well_linearised(g, p, is_ph=is_ph)
        if lin is None:
            continue
        q, _ = np.linalg.qr(lin[3])
        u = rng.standard_normal((len(g), n_mc))
        e = u - q @ (q.T @ u)
        lab = g["label"].to_numpy()
        for lbl in dict.fromkeys(lab):
            sign = np.sign(e[lab == lbl])
            runs = 1.0 + np.sum(sign[1:] != sign[:-1], axis=0)
            acc[lbl] = acc.get(lbl, np.zeros(n_mc)) + runs
    rows = {
        lbl: {
            "runs_expected_fit": float(v.mean()),
            "runs_sd_fit": float(v.std(ddof=1)),
            "runs_null": v,
        }
        for lbl, v in acc.items()
    }
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("label")


def well_leverage(
    table: pd.DataFrame,
    params: Mapping[str, Mapping[str, float]],
    *,
    is_ph: bool = True,
) -> pd.Series:
    """Leverage ``h_ii`` of every residual, from its well's weighted binding fit.

    Parameters
    ----------
    table : pd.DataFrame
        Residual table with ``well``, ``label``, ``x``, ``raw_res``, ``sigma``.
    params : Mapping[str, Mapping[str, float]]
        Well to its fitted ``K``, ``S0_<label>``, ``S1_<label>``.
    is_ph : bool
        pH titration (else concentration).

    Returns
    -------
    pd.Series
        ``h_ii`` aligned with *table*'s index; NaN for wells without parameters.
    """
    h = pd.Series(np.nan, index=table.index, dtype=float)
    for well, g in table.groupby("well", sort=False):
        p = params.get(str(well))
        lin = None if p is None else _well_linearised(g, p, is_ph=is_ph)
        if lin is not None:
            h.loc[g.index] = lin[0]
    return h


def studentized_outliers(
    table: pd.DataFrame,
    params: Mapping[str, Mapping[str, float]],
    *,
    is_ph: bool = True,
    alpha: float = _OUTLIER_ALPHA,
) -> pd.DataFrame:
    """Externally studentized residual of every point, and whether it is an outlier.

    Each well is its own regression: ``t_i`` follows Student-t with
    ``n - p - 1`` degrees of freedom, and the cutoff is Bonferroni-corrected
    for the well's ``n`` points, so a plate of clean wells flags ~alpha of its
    *wells*, not of its points.

    Parameters
    ----------
    table : pd.DataFrame
        Residual table with ``well``, ``label``, ``x``, ``raw_res``, ``sigma``.
    params : Mapping[str, Mapping[str, float]]
        Well to its fitted parameters.
    is_ph : bool
        pH titration (else concentration).
    alpha : float
        Family-wise level per well.

    Returns
    -------
    pd.DataFrame
        *table*'s index with ``t_abs``, ``dof``, ``cutoff`` and ``outlier``.
    """
    out = pd.DataFrame(
        {"t_abs": np.nan, "dof": np.nan, "cutoff": np.nan, "outlier": False},
        index=table.index,
    )
    for well, g in table.groupby("well", sort=False):
        p = params.get(str(well))
        lin = None if p is None else _well_linearised(g, p, is_ph=is_ph)
        if lin is None:
            continue
        _, r_ls, _, jac = lin
        t_abs, dof = studentized_scores(r_ls, jac)
        cutoff = bonferroni_threshold(len(t_abs), dof, alpha=alpha)
        out.loc[g.index, "t_abs"] = t_abs
        out.loc[g.index, "dof"] = float(dof)
        out.loc[g.index, "cutoff"] = cutoff
        out.loc[g.index, "outlier"] = t_abs > cutoff
    return out


def press_statistics(
    table: pd.DataFrame,
    params: Mapping[str, Mapping[str, float]],
    *,
    is_ph: bool = True,
) -> dict[str, float]:
    """PRESS: the prediction error of every point when it is left out of its well's fit.

    ``e_(i) = e_i / (1 - h_ii)`` on each well's linearised least-squares
    residuals (see :func:`well_leverage`), exact for a linear fit and first-order
    for a nonlinear one, so no refits are needed. Reported on the raw scale, in
    signal counts, which makes it comparable across noise models fitted to the
    same points: a lower PRESS is a mean model that predicts better.

    Parameters
    ----------
    table : pd.DataFrame
        Residual table with ``well``, ``label``, ``x``, ``raw_res``, ``sigma``.
    params : Mapping[str, Mapping[str, float]]
        Well to its fitted parameters.
    is_ph : bool
        pH titration (else concentration).

    Returns
    -------
    dict[str, float]
        ``press`` (sum of squared deleted residuals), ``press_rms`` (its root
        mean, in counts), ``press_ratio`` (PRESS / SSE; 1 would mean no point
        pulls its own fit) and ``max_leverage``.
    """
    return _press_summary(_press_parts(table, params, is_ph=is_ph))


def _press_parts(
    table: pd.DataFrame,
    params: Mapping[str, Mapping[str, float]],
    *,
    is_ph: bool,
) -> pd.DataFrame:
    """Per-point deleted residual, fitted residual (counts) and leverage.

    Each well is linearised whole, so a label's leverage is that of the joint
    fit sharing one K - the same design the studentized test uses.

    Parameters
    ----------
    table : pd.DataFrame
        Residual table with ``well``, ``label``, ``x``, ``raw_res``, ``sigma``.
    params : Mapping[str, Mapping[str, float]]
        Well to its fitted parameters.
    is_ph : bool
        pH titration (else concentration).

    Returns
    -------
    pd.DataFrame
        *table*'s index with ``deleted``, ``fitted`` and ``h``; NaN where the
        well has no parameters.
    """
    out = pd.DataFrame(
        {"deleted": np.nan, "fitted": np.nan, "h": np.nan}, index=table.index
    )
    for well, g in table.groupby("well", sort=False):
        p = params.get(str(well))
        lin = None if p is None else _well_linearised(g, p, is_ph=is_ph)
        if lin is None:
            continue
        h, r_ls, w, _ = lin
        e = r_ls / w
        out.loc[g.index, "deleted"] = e / (1.0 - h)
        out.loc[g.index, "fitted"] = e
        out.loc[g.index, "h"] = h
    return out


def _press_summary(parts: pd.DataFrame) -> dict[str, float]:
    """Aggregate :func:`_press_parts` rows into PRESS, its rms, ratio and max leverage.

    Parameters
    ----------
    parts : pd.DataFrame
        ``deleted``, ``fitted`` and ``h`` per point.

    Returns
    -------
    dict[str, float]
        ``press``, ``press_rms``, ``press_ratio`` and ``max_leverage``.
    """
    parts = parts.dropna()
    if parts.empty:
        return dict.fromkeys(
            ["press", "press_rms", "press_ratio", "max_leverage"], np.nan
        )
    d = parts["deleted"].to_numpy(float)
    e = parts["fitted"].to_numpy(float)
    h = parts["h"].to_numpy(float)
    press = float(np.sum(d**2))
    sse = float(np.sum(e**2))
    return {
        "press": press,
        "press_rms": float(np.sqrt(press / d.size)),
        "press_ratio": press / sse if sse > 0 else np.nan,
        "max_leverage": float(h.max()),
    }


def _runs_fit_columns(observed: float, null: Mapping[str, Any]) -> dict[str, float]:
    """Compare an observed pooled runs count with its simulated fitted-residual null.

    Parameters
    ----------
    observed : float
        Pooled runs from :func:`runs_test`.
    null : Mapping[str, Any]
        One label's row of :func:`runs_test_null`.

    Returns
    -------
    dict[str, float]
        ``runs_expected_fit``, ``runs_sd_fit`` and ``runs_p_fit``, the two-sided
        Monte Carlo p-value ``(1 + #{|R* - m| >= |R - m|}) / (1 + n_mc)``.
    """
    sim = np.asarray(null["runs_null"], dtype=float)
    m = float(null["runs_expected_fit"])
    extreme = int(np.sum(np.abs(sim - m) >= abs(observed - m)))
    return {
        "runs_expected_fit": m,
        "runs_sd_fit": float(null["runs_sd_fit"]),
        "runs_p_fit": (1.0 + extreme) / (1.0 + sim.size),
    }


def residual_tests(
    table: pd.DataFrame,
    params: Mapping[str, Mapping[str, float]] | None = None,
    *,
    is_ph: bool = True,
    n_mc: int = 999,
    seed: int = 0,
) -> pd.DataFrame:
    """All four assumption checks, one row per label.

    Parameters
    ----------
    table : pd.DataFrame
        Canonical residual table: ``well``, ``label``, ``step`` or ``x``,
        ``yhat``, ``raw_res``, ``sigma``, ``std_res``.
    params : Mapping[str, Mapping[str, float]] | None
        Well to fitted ``K``/``S0_<label>``/``S1_<label>``. Without it the
        leverage-based columns (studentized outliers, PRESS) are left NaN.
    is_ph : bool
        pH titration (else concentration).
    n_mc : int
        Monte Carlo samples for the Lilliefors and Anderson-Darling p-values
        and the fitted-residual runs null.
    seed : int
        Seed for those samples.

    Returns
    -------
    pd.DataFrame
        Indexed by label. Columns: ``n``; ``bp_r2``/``bp_p`` and
        ``white_r2``/``white_p`` (homoscedasticity); ``dw``, ``dw_expected``,
        ``lag1_r``, ``dw_p``, and with *params* the fitted-residual nulls
        ``dw_expected_fit`` and ``lag1_expected``; ``runs``, ``runs_expected``,
        ``runs_z``, ``runs_p``, and with *params* ``runs_expected_fit``,
        ``runs_sd_fit`` and the Monte Carlo ``runs_p_fit`` (independence);
        ``sw_w``, ``sw_p``, ``ks_n01_d``, ``ks_n01_p``, ``lillie_d``,
        ``lillie_p``, ``ad_a2``, ``ad_p`` (normality); ``t_max``,
        ``outlier_rate``, ``wells_with_outlier`` (studentized outliers);
        ``press``, ``press_rms``, ``press_ratio``, ``max_leverage`` (PRESS).
    """
    t = table.copy()
    t["label"] = t["label"].astype(str)
    stud = studentized_outliers(t, params, is_ph=is_ph) if params else None
    press = _press_parts(t, params, is_ph=is_ph) if params else None
    dw_null: dict[Any, dict[Any, Any]] = (
        durbin_watson_null(t, params, is_ph=is_ph).to_dict(orient="index")
        if params
        else {}
    )
    runs_null: dict[Any, dict[Any, Any]] = (
        runs_test_null(t, params, is_ph=is_ph, n_mc=n_mc, seed=seed).to_dict(
            orient="index"
        )
        if params
        else {}
    )
    rows: dict[str, dict[str, Any]] = {}
    for lbl, g in t.groupby("label", sort=True):
        z = g["std_res"].to_numpy(dtype=float)
        row: dict[str, Any] = {"n": int(np.isfinite(z).sum())}
        yhat, x = g["yhat"].to_numpy(dtype=float), g["x"].to_numpy(dtype=float)
        bp, wh = breusch_pagan(z, yhat, x), white_test(z, yhat, x)
        dw = durbin_watson(g)
        row |= {
            "bp_r2": bp["r2"],
            "bp_p": bp["p"],
            "white_r2": wh["r2"],
            "white_p": wh["p"],
        }
        row |= {
            "dw": dw["dw"],
            "dw_expected": dw["dw_expected"],
            "lag1_r": dw["lag1_r"],
            "dw_p": dw["p"],
        }
        row |= {str(k): float(v) for k, v in dw_null.get(str(lbl), {}).items()}
        runs = runs_test(g)
        row |= {k: v for k, v in runs.items() if k != "n_series"}
        if str(lbl) in runs_null:
            row |= _runs_fit_columns(runs["runs"], runs_null[str(lbl)])
        row |= normality_tests(z, n_mc=n_mc, seed=seed)
        if stud is not None and press is not None:
            s = stud.loc[g.index]
            tested = s["t_abs"].notna()
            row["t_max"] = (
                float(s.loc[tested, "t_abs"].max()) if tested.any() else np.nan
            )
            row["outlier_rate"] = (
                float(s.loc[tested, "outlier"].mean()) if tested.any() else np.nan
            )
            per_well = s.loc[tested].groupby(g.loc[tested, "well"])["outlier"].any()
            row["wells_with_outlier"] = (
                float(per_well.mean()) if len(per_well) else np.nan
            )
            row |= _press_summary(press.loc[g.index])
        rows[str(lbl)] = row
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("label")
