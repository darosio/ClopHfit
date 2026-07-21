"""Comprehensive comparison of plate-fitting pipelines on real plates.

Scores each pipeline on the two metrics that matter for choosing one:

1. **Residual Gaussianity** -- Shapiro / D'Agostino / Anderson on ``std_res``.
   That column is the probability-integral transform back to a Normal scale, so
   it is comparable across robust and non-robust likelihoods alike.
2. **Control consistency** -- each control well's K against the inverse-variance
   weighted mean of the rest of its group, summarized as the fraction inside a
   ROPE and the median |dK|.

Reported alongside them, because either metric can be gamed by discarding data:
points excluded, wells converged, and the median reported ``sK``.

Usage::

    python scripts/compare_plate_pipelines.py --tier classical
    python scripts/compare_plate_pipelines.py --tier mcmc --out results/
"""

from __future__ import annotations

import argparse
import logging
import time
import traceback
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
logging.getLogger("clophfit").setLevel(logging.ERROR)

from clophfit.fitting.bayes import fit_binding_pymc_multi  # noqa: E402
from clophfit.fitting.bayes_config import (  # noqa: E402
    InitConfig,
    NoiseConfig,
    RobustConfig,
    SamplerConfig,
)
from clophfit.fitting.ctr_validation import classical_ctr_holdout_rows  # noqa: E402
from clophfit.fitting.model_validation import (  # noqa: E402
    ResidualTail,
    apply_exclusions,
    mark_outliers,
)
from clophfit.prtecan import Titration  # noqa: E402

DEFAULT_RAW = Path("/home/dati/arslanbaeva/data/raw")
PLATES = ("L2", "L3", "L4")
SCREENS = {
    "none": None,
    "mad3.5": "mad:3.5:5",
    "studentized": "studentized:0.05:5",
}


def load_plate(raw: Path, folder: str) -> Titration:
    """Load one titration plate with its additions and scheme.

    Parameters
    ----------
    raw : Path
        Directory holding the plate folders.
    folder : str
        Plate folder name, e.g. ``"L2"``.

    Returns
    -------
    Titration
        The loaded titration.
    """
    tit = Titration.fromlistfile(raw / folder / "list.pH.csv", is_ph=True)
    tit.load_additions(raw / folder / "additions.pH")
    tit.load_scheme(raw / folder / "scheme.txt")
    return tit


def normality_scores(residuals: pd.DataFrame, column: str = "std_res") -> dict[str, float]:
    """Score how Normal a residual column is, pooled and per label.

    Parameters
    ----------
    residuals : pd.DataFrame
        Canonical residual table.
    column : str
        Column to test; ``std_res`` is on a Normal scale by construction.

    Returns
    -------
    dict[str, float]
        Anderson A^2, D'Agostino K^2, Shapiro p, plus mean/sd of the column.
    """
    out: dict[str, float] = {}
    for tag, frame in [("pooled", residuals), *[
        (f"lbl{lbl}", grp) for lbl, grp in residuals.groupby("label", observed=True)
    ]]:
        x = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        x = x[np.isfinite(x)]
        if x.size < 8:
            continue
        out[f"anderson_A2_{tag}"] = float(stats.anderson(x, dist="norm").statistic)
        out[f"dagostino_K2_{tag}"] = float(stats.normaltest(x).statistic)
        out[f"shapiro_p_{tag}"] = float(stats.shapiro(x[:5000]).pvalue)
        out[f"res_mean_{tag}"] = float(np.mean(x))
        out[f"res_sd_{tag}"] = float(np.std(x, ddof=1))
    return out


def bulk_sd_scores(residuals: pd.DataFrame, column: str = "std_res") -> dict[str, float]:
    """Robust SD of the residual bulk (IQR/1.349), pooled and per label.

    Target is ~1. Below 1 means the model's sigma is too large for the bulk
    (scale over-inflation); ~1 with heavy Anderson tails means a likelihood
    shape mismatch, not scale.

    Parameters
    ----------
    residuals : pd.DataFrame
        Canonical residual table.
    column : str
        Column to summarize; ``std_res`` is on a Normal scale by construction.

    Returns
    -------
    dict[str, float]
        ``bulk_sd_pooled`` and ``bulk_sd_lbl{label}`` per label.
    """
    def _bulk_sd(values: pd.Series) -> float:
        x = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
        x = x[np.isfinite(x)]
        if x.size < 8:
            return float("nan")
        q25, q75 = np.percentile(x, [25, 75])
        return float((q75 - q25) / 1.349)

    out = {"bulk_sd_pooled": _bulk_sd(residuals[column])}
    for lbl, grp in residuals.groupby("label", observed=True):
        out[f"bulk_sd_lbl{lbl}"] = _bulk_sd(grp[column])
    return out


def ctr_scores(results: dict[str, Any], scheme: Any, *, rope: float = 0.10) -> dict[str, float]:
    """Score agreement among replicate control wells.

    Parameters
    ----------
    results : dict[str, Any]
        Per-well fit results.
    scheme : Any
        Plate scheme naming the control groups.
    rope : float
        Region of practical equivalence on K, in pH units.

    Returns
    -------
    dict[str, float]
        Fraction of controls inside the ROPE, median |dK| and |z|, and the
        pooled within-group spread of K.
    """
    rows = classical_ctr_holdout_rows(results, scheme, trace_id="cmp", rope=rope)
    out: dict[str, float] = {"n_ctr": float(len(rows))}
    if rows.empty:
        return out
    delta = pd.to_numeric(rows["delta_k_abs_mean"], errors="coerce")
    out["ctr_median_absdK"] = float(np.nanmedian(delta))
    out["ctr_max_absdK"] = float(np.nanmax(delta))
    out["ctr_frac_in_rope"] = float(np.nanmean(rows["p_abs_delta_k_lt_rope"]))
    z = pd.to_numeric(rows["z_delta_k"], errors="coerce").abs()
    out["ctr_median_absz"] = float(np.nanmedian(z))
    # Pooled within-group spread of K: replicates should agree.
    spreads = []
    for _group, wells in scheme.names.items():
        ks = [
            results[w].result.params["K"].value
            for w in wells
            if w in results and results[w].result is not None
        ]
        if len(ks) > 1:
            spreads.append(np.std(ks, ddof=1))
    if spreads:
        out["ctr_within_group_sd"] = float(np.mean(spreads))
    return out


def score_pipeline(tit: Titration, res: Any, meta: dict[str, Any]) -> dict[str, Any]:
    """Collect every metric for one fitted plate.

    Parameters
    ----------
    tit : Titration
        The plate, for its scheme.
    res : Any
        The fit results container.
    meta : dict[str, Any]
        Identifying fields merged into the returned row.

    Returns
    -------
    dict[str, Any]
        One row of the comparison table.
    """
    row: dict[str, Any] = dict(meta)
    good = {w: fr for w, fr in res.results.items() if fr.result is not None}
    row["n_wells"] = len(res.results)
    row["n_fitted"] = len(good)
    row["frac_converged"] = float(
        np.mean([bool(getattr(fr.result, "success", True)) for fr in good.values()])
    ) if good else np.nan
    row["n_excluded"] = int(sum(
        int((~da.mask & np.isfinite(da.yc)).sum())
        for fr in good.values() if fr.dataset is not None
        for da in fr.dataset.values()
    ))
    ks = np.array([fr.result.params["K"].value for fr in good.values()], dtype=float)
    sks = np.array([
        fr.result.params["K"].stderr if fr.result.params["K"].stderr else np.nan
        for fr in good.values()
    ], dtype=float)
    row["medK"] = float(np.nanmedian(ks)) if ks.size else np.nan
    row["iqrK"] = float(np.nanpercentile(ks, 75) - np.nanpercentile(ks, 25)) if ks.size else np.nan
    row["med_sK"] = float(np.nanmedian(sks)) if sks.size else np.nan
    try:
        row.update(normality_scores(res.residuals))
    except Exception as exc:  # noqa: BLE001
        row["normality_error"] = str(exc)[:120]
    try:
        row.update(bulk_sd_scores(res.residuals))
    except Exception as exc:  # noqa: BLE001
        row["bulk_sd_error"] = str(exc)[:120]
    try:
        row.update(ctr_scores(good, tit.scheme))
    except Exception as exc:  # noqa: BLE001
        row["ctr_error"] = str(exc)[:120]
    return row


def run_classical(raw: Path, plates: tuple[str, ...]) -> list[dict[str, Any]]:
    """Run every classical pipeline on every plate.

    Parameters
    ----------
    raw : Path
        Raw-data directory.
    plates : tuple[str, ...]
        Plate folder names.

    Returns
    -------
    list[dict[str, Any]]
        One scored row per (plate, fitter, screen).
    """
    rows: list[dict[str, Any]] = []
    for plate in plates:
        tit = load_plate(raw, plate)
        for fitter in ("lm", "huber", "odr", "fgls"):
            for screen, spec in SCREENS.items():
                meta = {"plate": plate, "fitter": fitter, "screen": screen}
                t0 = time.time()
                try:
                    if fitter == "odr" and screen == "studentized":
                        continue  # fit_binding_odr has no studentized path
                    if fitter == "fgls":
                        if spec is not None:
                            continue  # FGLS screening is a separate two-pass concern
                        res = tit.fgls_fit_plate(sigma_floor=tit.bg_noise)
                    else:
                        kw = {"remove_outliers": spec} if spec else {}
                        res = tit.fit_plate(method=fitter, **kw)
                    rows.append({**score_pipeline(tit, res, meta), "secs": time.time() - t0})
                    print(f"  ok  {plate:3} {fitter:6} {screen:12} {time.time()-t0:6.1f}s", flush=True)
                except Exception:  # noqa: BLE001
                    rows.append({**meta, "error": traceback.format_exc(limit=1)[:200]})
                    print(f"  FAIL {plate:3} {fitter:6} {screen:12}", flush=True)
    return rows


INIT = InitConfig(
    strategy="data_priors",
    edge_points=2,
    k_prior="midpoint_truncnorm",
    k_bounds=(4.5, 9.0),
    k_sigma=1.5,
    signal_sigma_scale=0.5,
)


def mcmc_grid(tit: Titration) -> list[dict[str, Any]]:
    """Build the MCMC pipeline grid for one plate.

    Crosses likelihood family, per-well ye_mags, and noise parameterization,
    including an FGLS-calibrated structured noise variant.

    Parameters
    ----------
    tit : Titration
        The plate, used for its measured background noise floor.

    Returns
    -------
    list[dict[str, Any]]
        One entry per configuration, each with a name plus kwargs.
    """
    floor = tit.bg_noise
    robusts = {
        "studentt_nu1": RobustConfig(enabled=True, likelihood="student_t", nu=1),
        "mixture": RobustConfig(
            enabled=True, likelihood="mixture",
            contamination_frac_prior={"1": 0.15, "2": 0.015},
        ),
    }
    noises = {
        "structured": NoiseConfig.structured(
            floor=floor, gain=0.5, alpha=0.02, floor_mode="centered",
            gain_mode="free", alpha_mode="free",
            shared_alpha=False, shared_gain=False, learn_ye_mags=True,
        ),
        "yemag": NoiseConfig.ye_mag(),
    }
    grid = []
    for rname, rob in robusts.items():
        for nname, noi in noises.items():
            for pwym in (True, False):
                grid.append({
                    "name": f"{rname}|{nname}|pwym{int(pwym)}",
                    "robust": rob, "noise": noi, "per_well_ye_mags": pwym,
                })
    return grid


def run_mcmc(  # noqa: PLR0913
    raw: Path,
    plates: tuple[str, ...],
    *,
    n_samples: int,
    n_tune: int,
    nuts_sampler: str,
    target_accept: float,
    screen_threshold: float,
    out: Path,
) -> list[dict[str, Any]]:
    """Run the MCMC pipeline grid, plus FGLS-seeded and screened variants.

    Parameters
    ----------
    raw : Path
        Raw-data directory.
    plates : tuple[str, ...]
        Plate folder names.
    n_samples : int
        Sampler draws.
    n_tune : int
        Tuning steps.
    nuts_sampler : str
        NUTS backend, e.g. ``"nutpie"``.
    target_accept : float
        NUTS target acceptance.
    screen_threshold : float
        ``|likelihood_res|`` cutoff for the screened second pass.
    out : Path
        Directory for incremental result writes.

    Returns
    -------
    list[dict[str, Any]]
        One scored row per (plate, config, pass).
    """
    sampler = SamplerConfig(
        nuts_sampler=nuts_sampler, target_accept=target_accept,
        n_tune=n_tune, n_samples=n_samples, compute_log_likelihood=True,
    )
    common = {
        "n_sd": 7, "n_xerr": 1.0, "x_error_model": "per_well",
        "ctr_free_k": True, "init": INIT, "sampler": sampler,
    }
    rows: list[dict[str, Any]] = []
    for plate in plates:
        tit = load_plate(raw, plate)
        base_ds = tit.create_dataset_dict()

        # FGLS-calibrated noise as an extra candidate: learn gain/alpha first.
        fgls_noise = None
        try:
            fgls_res = tit.fgls_fit_plate(sigma_floor=tit.bg_noise)
            nm = fgls_res.noise_model
            if nm is not None:
                fgls_noise = NoiseConfig.structured(
                    noise_model=nm, floor_mode="fixed",
                    gain_mode="centered", alpha_mode="centered",
                    learn_ye_mags=True,
                )
                print(f"  {plate}: FGLS noise gain={nm.gain} alpha={nm.alpha}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"  {plate}: FGLS calibration failed: {exc}", flush=True)

        grid = mcmc_grid(tit)
        if fgls_noise is not None:
            grid.append({
                "name": "studentt_nu1|fgls_structured|pwym1",
                "robust": RobustConfig(enabled=True, likelihood="student_t", nu=1),
                "noise": fgls_noise, "per_well_ye_mags": True,
            })

        for cfg in grid:
            name = cfg["name"]
            meta = {"plate": plate, "fitter": "mcmc_multi", "screen": "none",
                    "config": name}
            t0 = time.time()
            try:
                screen = fit_binding_pymc_multi(
                    base_ds, tit.scheme,
                    per_well_ye_mags=cfg["per_well_ye_mags"],
                    robust=cfg["robust"], noise=cfg["noise"], **common,
                )
                rows.append({**score_pipeline(tit, screen, meta),
                             "secs": time.time() - t0})
                print(f"  ok  {plate} {name:42} {time.time()-t0:7.1f}s", flush=True)

                # Screened second pass: mark on |likelihood_res|, refit Normal.
                t1 = time.time()
                marked = mark_outliers(
                    screen.residuals,
                    ResidualTail(residual_col="likelihood_res",
                                 threshold=screen_threshold,
                                 allowed_tail_fraction=0.0,
                                 min_allowed_tail_count=0),
                )
                masked = apply_exclusions(base_ds, marked, min_keep=5)
                meta2 = {**meta, "screen": f"likres>{screen_threshold}"}
                final = fit_binding_pymc_multi(
                    masked, tit.scheme,
                    per_well_ye_mags=cfg["per_well_ye_mags"],
                    robust=RobustConfig(enabled=False), noise=cfg["noise"], **common,
                )
                rows.append({**score_pipeline(tit, final, meta2),
                             "secs": time.time() - t1})
                print(f"  ok  {plate} {name:42} +refit {time.time()-t1:7.1f}s", flush=True)
            except Exception:  # noqa: BLE001
                rows.append({**meta, "error": traceback.format_exc(limit=1)[:200]})
                print(f"  FAIL {plate} {name}", flush=True)
            pd.DataFrame(rows).to_csv(out / "comparison_mcmc_partial.csv", index=False)
    return rows


def loo_and_rhat(trace: Any) -> dict[str, Any]:
    """Compute LOO and convergence diagnostics from a trace.

    Parameters
    ----------
    trace : Any
        PyMC/ArviZ trace carrying a ``log_likelihood`` group.

    Returns
    -------
    dict[str, Any]
        ELPD-LOO with standard error, effective parameters, Pareto-k counts,
        worst r_hat, and how many variables exceed 1.01.
    """
    import arviz as az

    out: dict[str, Any] = {}
    try:
        rh = az.rhat(trace)
        vals = np.concatenate([np.ravel(v.values) for v in rh.data_vars.values()])
        vals = vals[np.isfinite(vals)]
        out["rhat_max"] = float(np.max(vals))
        out["n_rhat_gt_1.01"] = int(np.sum(vals > 1.01))
        out["n_rhat_gt_1.05"] = int(np.sum(vals > 1.05))
        worst = max(
            ((str(n), float(np.nanmax(v.values))) for n, v in rh.data_vars.items()),
            key=lambda kv: kv[1], default=("", np.nan))
        out["rhat_worst_var"] = worst[0]
    except Exception as exc:  # noqa: BLE001
        out["rhat_error"] = str(exc)[:100]
    try:
        ess = az.ess(trace)
        e = np.concatenate([np.ravel(v.values) for v in ess.data_vars.values()])
        out["ess_min"] = float(np.nanmin(e))
    except Exception:  # noqa: BLE001
        pass
    # One log-likelihood array per label, so LOO is computed per label and the
    # ELPDs summed: the observation groups are disjoint, so the total expected
    # log predictive density is the sum of theirs.
    try:
        names = list(trace.log_likelihood.data_vars)
        elpd = se2 = p_loo = 0.0
        k_bad = 0
        k_max = -np.inf
        for name in names:
            loo = az.loo(trace, pointwise=True, var_name=name)
            # arviz >= 1.0 renamed ELPDData.elpd_loo/p_loo to .elpd/.p
            elpd += float(loo.elpd)
            se2 += float(loo.se) ** 2
            p_loo += float(loo.p)
            k = np.asarray(loo.pareto_k.values).ravel()
            good_k = float(getattr(loo, "good_k", 0.7) or 0.7)
            k_bad += int(np.sum(k > good_k))
            k_max = max(k_max, float(np.nanmax(k)))
            out[f"elpd_loo_{name}"] = float(loo.elpd)
        out["elpd_loo"] = elpd
        out["se_loo"] = float(np.sqrt(se2))
        out["p_loo"] = p_loo
        out["pareto_k_gt_0.7"] = k_bad
        out["pareto_k_max"] = k_max
    except Exception as exc:  # noqa: BLE001
        out["loo_error"] = str(exc)[:100]
    return out


def run_nugrid(  # noqa: PLR0913
    raw: Path, plates: tuple[str, ...], *, nus: tuple[float, ...],
    n_samples: int, n_tune: int, nuts_sampler: str, target_accept: float,
    out: Path,
) -> list[dict[str, Any]]:
    """Sweep Student-t nu on the winning noise model, with LOO and r_hat.

    Parameters
    ----------
    raw : Path
        Raw-data directory.
    plates : tuple[str, ...]
        Plate folder names.
    nus : tuple[float, ...]
        Student-t degrees of freedom to sweep.
    n_samples : int
        Sampler draws.
    n_tune : int
        Tuning steps.
    nuts_sampler : str
        NUTS backend.
    target_accept : float
        NUTS target acceptance.
    out : Path
        Directory for incremental writes.

    Returns
    -------
    list[dict[str, Any]]
        One scored row per (plate, nu, per_well_ye_mags).
    """
    sampler = SamplerConfig(
        nuts_sampler=nuts_sampler, target_accept=target_accept,
        n_tune=n_tune, n_samples=n_samples, compute_log_likelihood=True)
    common = {"n_sd": 7, "n_xerr": 1.0, "x_error_model": "per_well",
              "ctr_free_k": True, "init": INIT, "sampler": sampler}
    rows: list[dict[str, Any]] = []
    for plate in plates:
        tit = load_plate(raw, plate)
        base_ds = tit.create_dataset_dict()
        for nu in nus:
            for pwym in (False, True):
                name = f"studentt_nu{nu}|yemag|pwym{int(pwym)}"
                meta = {"plate": plate, "fitter": "mcmc_multi", "screen": "none",
                        "config": name, "nu": nu, "pwym": int(pwym)}
                t0 = time.time()
                try:
                    fit = fit_binding_pymc_multi(
                        base_ds, tit.scheme, per_well_ye_mags=pwym,
                        robust=RobustConfig(enabled=True, likelihood="student_t", nu=nu),
                        noise=NoiseConfig.ye_mag(), **common)
                    row = score_pipeline(tit, fit, meta)
                    row.update(loo_and_rhat(fit.trace))
                    row["secs"] = time.time() - t0
                    rows.append(row)
                    print(f"  ok {plate} nu={nu:<4} pwym={int(pwym)} "
                          f"A2={row.get('anderson_A2_pooled', float('nan')):8.2f} "
                          f"elpd={row.get('elpd_loo', float('nan')):10.1f} "
                          f"rhat={row.get('rhat_max', float('nan')):.4f} "
                          f"{row['secs']:6.0f}s", flush=True)
                except Exception:  # noqa: BLE001
                    rows.append({**meta, "error": traceback.format_exc(limit=1)[:200]})
                    print(f"  FAIL {plate} nu={nu} pwym={int(pwym)}", flush=True)
                pd.DataFrame(rows).to_csv(out / "comparison_nugrid_partial.csv", index=False)
    return rows


def main() -> None:
    """Parse arguments and run the requested tier."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    ap.add_argument("--plates", nargs="+", default=list(PLATES))
    ap.add_argument("--tier", choices=("classical", "mcmc", "nugrid", "all"), default="classical")
    ap.add_argument("--nus", nargs="+", type=float, default=[0.1, 0.5, 1.0, 2.0, 3.0])
    ap.add_argument("--out", type=Path, default=Path("pipeline_comparison"))
    ap.add_argument("--n-samples", type=int, default=2000)
    ap.add_argument("--n-tune", type=int, default=1000)
    ap.add_argument("--nuts-sampler", default="nutpie")
    ap.add_argument("--target-accept", type=float, default=0.98)
    ap.add_argument("--screen-threshold", type=float, default=4.0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    if args.tier in {"classical", "all"}:
        print("=== classical tier ===", flush=True)
        rows += run_classical(args.raw, tuple(args.plates))

    if args.tier in {"mcmc", "all"}:
        print("=== mcmc tier ===", flush=True)
        rows += run_mcmc(
            args.raw, tuple(args.plates),
            n_samples=args.n_samples, n_tune=args.n_tune,
            nuts_sampler=args.nuts_sampler, target_accept=args.target_accept,
            screen_threshold=args.screen_threshold, out=args.out,
        )

    if args.tier == "nugrid":
        print("=== nu grid (production sampler) ===", flush=True)
        rows += run_nugrid(
            args.raw, tuple(args.plates), nus=tuple(args.nus),
            n_samples=args.n_samples, n_tune=args.n_tune,
            nuts_sampler=args.nuts_sampler, target_accept=args.target_accept,
            out=args.out)

    df = pd.DataFrame(rows)
    df.to_csv(args.out / f"comparison_{args.tier}.csv", index=False)
    cols = [c for c in ("plate","fitter","config","screen","n_excluded","medK","iqrK","med_sK",
                        "anderson_A2_pooled","shapiro_p_pooled","ctr_median_absdK",
                        "ctr_frac_in_rope","ctr_within_group_sd","secs") if c in df.columns]
    print()
    print(df[cols].to_string(index=False))
    print("\nwrote", args.out / f"comparison_{args.tier}.csv")


if __name__ == "__main__":
    main()
