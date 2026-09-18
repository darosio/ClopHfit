"""A dissociation constant stays in a physical range, and a non-binder is a result.

Found refitting the chloride titrations of the library plates:

1. The multi-well Kd prior was an unbounded Normal, so a library well came out
   at Kd = -2.7 mM, and least squares let Kd fall to 2e-16 mM. Every fitter now
   keeps Kd between 1 mM and 100 times the highest concentration titrated.
2. V224Q does not bind chloride. Its wells were discarded before fitting (no
   finite preliminary Kd) and then dropped from the multi-well fit, so the
   construct vanished from the results; a library well that does not bind
   went the same way. Such wells are now fitted and reported "does not bind".
3. The control-replicate Kd prior reused pH rules - an SD capped at 0.6 and a
   pH-at-mid-fluorescence fallback - which mean nothing for a Kd in mM.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pymc as pm  # type: ignore[import-untyped]
import pytest
from click.testing import CliRunner

from clophfit.__main__ import ppr
from clophfit.fitting import bayes, plate_lm
from clophfit.fitting.bayes_config import SamplerConfig
from clophfit.fitting.core import fit_binding_glob
from clophfit.fitting.data_structures import DataArray, Dataset, FitResult
from clophfit.fitting.models import KD_MIN, binding_1site, kd_bounds
from clophfit.fitting.odr import fit_binding_odr
from clophfit.prtecan import PlateScheme, export

DATA = Path(__file__).parent / "Tecan" / "140220"
CL = np.array([0.0, 17.5, 35.0, 70.0, 105.0, 140.0, 164.0])
X_MAX = float(CL[-1])


class _StopBuildError(Exception):
    """Raised by a patched step to end the model build once it has been seen."""


def _cl_dataset(kd: float | None, seed: int = 0, drift: float = 0.4) -> Dataset:
    """Two-label chloride titration; ``kd=None`` does not bind.

    A non-binder rises linearly by *drift* over the titration, as V224Q does on
    the library plates (+30-85%): a response with no curvature, which puts Kd
    beyond the titration. A perfectly flat curve (``drift=0``) fixes no Kd at
    all - any Kd fits with S1 = S0 - and is undetermined, not "does not bind".
    """
    rng = np.random.default_rng(seed)
    labels = {}
    for lbl, (s0, s1) in {"1": (1000.0, 300.0), "2": (200.0, 150.0)}.items():
        y = (
            s0 * (1 + drift * CL / CL[-1])
            if kd is None
            else binding_1site(CL, kd, s0, s1, is_ph=False)
        )
        da = DataArray(CL, y + rng.normal(0.0, 5.0, CL.size))
        da.y_err = np.full(CL.size, 5.0)
        labels[lbl] = da
    return Dataset(labels, is_ph=False)


# -- 1. bounds -----------------------------------------------------------------


def test_kd_bounds_scale_with_the_titration() -> None:
    """Floor 1 mM; ceiling 100x the top concentration, open when that is unknown."""
    lo, hi = kd_bounds(164.0)
    assert lo == KD_MIN
    assert hi == pytest.approx(16400.0)
    assert kd_bounds(float("nan"))[1] == np.inf


def test_least_squares_kd_never_below_the_floor() -> None:
    """Data saturating below the first point pin Kd at 1 mM, not near zero."""
    fr = fit_binding_glob(_cl_dataset(0.05))
    assert fr.result is not None
    assert fr.result.params["K"].value >= KD_MIN - 1e-9


def test_least_squares_kd_stops_at_the_ceiling_for_a_non_binder() -> None:
    """A response without curvature drives Kd up, to the ceiling and no further."""
    fr = fit_binding_glob(_cl_dataset(None))
    assert fr.result is not None
    ceiling = kd_bounds(X_MAX)[1]
    assert 0.99 * ceiling <= fr.result.params["K"].value <= ceiling + 1e-6


def test_odr_kd_never_below_the_floor() -> None:
    """ODR has its own solver, and needs the bound passed separately."""
    fr = fit_binding_odr(_cl_dataset(0.05))
    assert fr.result is not None
    assert fr.result.params["K"].value >= KD_MIN - 1e-9


def test_plate_fit_kd_floor() -> None:
    """The plate fitters share one K bound: 1 mM for a Kd, [3, 11] for a pKa."""
    lo, _ = plate_lm._k_bounds(2, 5, is_ph=False)  # ruff: ignore[private-member-access] - the bound under test
    assert lo[:2].tolist() == [KD_MIN, KD_MIN]
    assert np.isneginf(lo[2:]).all()


# -- the log-Kd prior -----------------------------------------------------------


def test_prior_from_a_preliminary_kd_is_its_delta_method_image() -> None:
    """Centre log K, SD n_sd * sK / K, as the pH prior is n_sd * sK."""
    mu, sigma = bayes.log_kd_prior_params(8.0, 1.0, 5.0, (1.0, 16400.0))
    assert mu == pytest.approx(np.log(8.0))
    assert sigma == pytest.approx(5.0 / 8.0)


@pytest.mark.parametrize(
    ("k", "se"), [(None, None), (np.inf, 1.0), (8.0, None), (-2.0, 1.0)]
)
def test_no_usable_preliminary_kd_spans_the_range(
    k: float | None, se: float | None
) -> None:
    """A non-binder's prior neither favours binding nor its absence."""
    mu, sigma = bayes.log_kd_prior_params(k, se, 5.0, (1.0, 16400.0))
    assert mu == pytest.approx(np.log(16400.0) / 2)
    assert sigma == pytest.approx(np.log(16400.0))


def _draw(mu: float, sigma: float, n: int = 4000) -> np.ndarray:
    with pm.Model():
        k = bayes.log_kd_prior("K", mu, sigma, (1.0, 16400.0))
        return np.asarray(pm.draw(k, draws=n, random_seed=1))


def test_log_kd_prior_stays_in_range_and_centres_on_the_estimate() -> None:
    """A seeded prior is centred on the preliminary Kd with about its width."""
    draws = _draw(float(np.log(8.0)), 0.3)
    assert draws.min() >= 1.0
    assert draws.max() <= 16400.0
    assert np.median(draws) == pytest.approx(8.0, rel=0.05)
    assert np.std(np.log(draws)) == pytest.approx(0.3, rel=0.25)


def test_wide_log_kd_prior_does_not_pile_onto_the_bounds() -> None:
    """The flattest single-peaked shape: little mass within 1% of either end."""
    log_draws = np.log(_draw(float(np.log(16400.0) / 2), float(np.log(16400.0))))
    frac = (log_draws - 0.0) / np.log(16400.0)
    assert ((frac < 0.01) | (frac > 0.99)).mean() < 0.02
    assert 0.3 < np.median(frac) < 0.7


def test_log_kd_prior_puts_the_sampler_on_a_standard_normal() -> None:
    """Jitter of +/-1 at the start is then one prior SD, not a factor of ten."""
    with pm.Model() as model:
        bayes.log_kd_prior("K", 2.0, 0.5, (1.0, 16400.0))
    assert [rv.name for rv in model.free_RVs] == ["z_K"]
    assert "K" in {d.name for d in model.deterministics}


def test_single_well_bayes_builds_the_bounded_kd_for_chloride() -> None:
    """``create_parameter_priors`` gets the log-Kd prior when given a Kd range."""
    fr = fit_binding_glob(_cl_dataset(8.0))
    assert fr.result is not None
    with pm.Model() as model:
        bayes.create_parameter_priors(fr.result.params, 5.0, kd_range=(1.0, 16400.0))
    assert "z_K" in {rv.name for rv in model.free_RVs}


# -- 2. non-binders are fitted -------------------------------------------------


def _fit_results() -> dict[str, FitResult]:
    ok, flat = fit_binding_glob(_cl_dataset(8.0)), fit_binding_glob(_cl_dataset(None))
    return {"A01": ok, "A02": fit_binding_glob(_cl_dataset(8.0, 1)), "A03": flat,
            "A04": fit_binding_glob(_cl_dataset(None, 1)), "B05": ok}  # fmt: skip


def _scheme() -> PlateScheme:
    scheme = PlateScheme()
    scheme.names = {"E2GFP": {"A01", "A02"}, "V224Q": {"A03", "A04"}}
    return scheme


@pytest.mark.parametrize("ctr_free_k", [True, False])
def test_multi_fit_keeps_a_chloride_control_that_does_not_bind(
    monkeypatch: pytest.MonkeyPatch, *, ctr_free_k: bool
) -> None:
    """V224Q's wells reach the K priors, in log Kd over the chloride range."""
    seen: dict[str, Any] = {}

    def stop(_fr: object, wells_list: list[str], *_a: object, **kw: object) -> None:
        seen.update(wells=list(wells_list), kd_range=kw.get("kd_range"))
        raise _StopBuildError

    monkeypatch.setattr(bayes, "_free_k_init", stop)
    with pytest.raises(_StopBuildError):
        bayes.fit_binding_pymc_multi(
            _fit_results(),
            _scheme(),
            n_xerr=0.0,
            sampler=SamplerConfig(n_samples=2, n_tune=1),
            ctr_free_k=ctr_free_k,
        )
    assert sorted(seen["wells"]) == ["A01", "A02", "A03", "A04", "B05"]
    assert seen["kd_range"] == kd_bounds(X_MAX)


def test_kd_free_init_ignores_the_ph_rules() -> None:
    """Each well's own preliminary Kd, in log Kd; no 0.6 cap, no group mean."""
    results = _fit_results()
    free, mu, sigma, _ = bayes._free_k_init(  # ruff: ignore[private-member-access] - the seam under test
        results, list(results), _scheme(), {"E2GFP": (8.0, 0.1)}, 5.0,
        ctr_free_k=True, kd_range=(1.0, 16400.0),
    )  # fmt: skip
    pars = results["A01"].result.params["K"]  # type: ignore[union-attr]
    i = free.index("A01")
    assert mu[i] == pytest.approx(np.log(pars.value))
    assert sigma[i] == pytest.approx(max(5.0 * pars.stderr / pars.value, 0.05))
    j = free.index("A03")  # V224Q: no usable preliminary Kd, not the group's
    assert sigma[j] == pytest.approx(np.log(16400.0))


def test_shared_mode_builds_a_kd_for_a_group_without_an_estimate() -> None:
    """Pooled chloride controls: the non-binder gets its own, range-wide, Kd."""
    with pm.Model():
        k_params, _ = bayes._build_ctr_k_params(  # ruff: ignore[private-member-access] - the seam under test
            _scheme(), {"E2GFP": (8.0, 0.5)}, {"A01", "A02", "A03", "A04"},
            ctr_free_k=False, kd_range=(1.0, 16400.0),
        )  # fmt: skip
    assert set(k_params) == {"E2GFP", "V224Q"}


@pytest.mark.slow
def test_multi_fit_reports_a_non_binder_beyond_the_titration() -> None:
    """Sampled end to end: Kd in range, K reported, the sampler's z not."""
    fit = bayes.fit_binding_pymc_multi(
        _fit_results(),
        _scheme(),
        n_xerr=0.0,
        sampler=SamplerConfig(nuts_sampler="pymc", n_samples=200, n_tune=200),
        ctr_free_k=True,
    )
    k = fit.trace["posterior"]["K_free"]
    assert float(k.min()) >= 1.0
    assert float(k.max()) <= 16400.0
    for fr in fit.results.values():
        assert fr.result is not None
        assert not any(name.startswith("z_") for name in fr.result.params)
    table = pd.DataFrame(
        [
            {"well": w, "K": fr.result.params["K"].value,  # type: ignore[union-attr]
             "sK": fr.result.params["K"].stderr,  # type: ignore[union-attr]
             "Khdi03": fr.result.params["K"].min}  # type: ignore[union-attr]
            for w, fr in fit.results.items()
        ]
    ).set_index("well")  # fmt: skip
    verdict = export.k_verdicts(table, is_ph=False, max_k_se=0.3, x_max=X_MAX)
    assert verdict.loc[["A03", "A04"], "no_binding"].all()
    assert not verdict.loc[["A01", "A02", "B05"]].any(axis=None)


# -- 3. verdicts -----------------------------------------------------------------


def _table(
    k: list[float], sk: list[float | None], hdi: list[float] | None = None
) -> pd.DataFrame:
    cols: dict[str, Any] = {"K": k, "sK": sk}
    if hdi is not None:
        cols["Khdi03"] = hdi
    return pd.DataFrame(
        cols, index=pd.Index([f"A{i:02d}" for i in range(1, len(k) + 1)], name="well")
    )


def test_no_binding_needs_the_lower_bound_beyond_the_titration() -> None:
    """HDI lower edge above the top concentration; a weak binder is not one."""
    fit = _table([16330.0, 400.0, 41.0], [530.0, 300.0, 8.5], [14964.0, 90.0, 27.5])
    assert export.no_binding_k(fit, x_max=X_MAX).tolist() == [True, False, False]


def test_no_binding_for_a_least_squares_kd_pinned_at_the_ceiling() -> None:
    """Lm reports a meaningless SE there (1e6 mM on a test plate); the pin decides."""
    ceiling = kd_bounds(X_MAX)[1]
    fit = _table([ceiling, ceiling / 2, 300.0], [3e6, 1e6, None])
    assert export.no_binding_k(fit, x_max=X_MAX).tolist() == [True, False, True]


def test_chloride_verdicts_are_exclusive() -> None:
    """A non-binder is a result, not an undetermined well."""
    fit = _table([16330.0, 3.0, 8.0], [530.0, 2.7, 0.5], [14964.0, 1.07, 7.2])
    v = export.k_verdicts(fit, is_ph=False, max_k_se=0.3, x_max=X_MAX)
    assert v["no_binding"].tolist() == [True, False, False]
    assert v["undetermined"].tolist() == [False, True, False]


def test_ph_verdicts_have_no_binding_column() -> None:
    """A pKa has no 'does not bind'."""
    v = export.k_verdicts(_table([7.0], [0.1]), is_ph=True, max_k_se=0.3, x_max=9.0)
    assert list(v.columns) == ["undetermined"]


@pytest.mark.slow
def test_ppr_chloride_fits_and_reports_v224q(tmp_path: Path) -> None:
    """End to end: V224Q is fitted, flagged, and listed; nothing below 1 mM."""
    args = ["--out", str(tmp_path), "tecan", str(DATA / "list.cl.csv"), "--fit"]
    args += ["--sch", str(DATA / "scheme.txt"), "--add", str(DATA / "additions.cl")]
    args += ["--bg", "--cl", "1000.0"]
    result = CliRunner().invoke(ppr, args)
    assert result.exit_code == 0, result.output
    fit_dir = next(tmp_path.rglob("ffit0.csv")).parent
    tables = sorted(fit_dir.glob("ffit*.csv"))
    v224q = ["A12", "B01", "G01"]
    for table in tables:
        fit = pd.read_csv(table, index_col="well")
        assert set(v224q) <= set(fit.index)
        assert fit["K"].min() >= KD_MIN - 1e-6
        assert {"undetermined", "no_binding"} <= set(fit.columns)
    lm = pd.read_csv(tables[0], index_col="well")  # pinned at the ceiling
    assert lm.loc[v224q, "no_binding"].all()
    lines = (fit_dir / "discarded_wells.txt").read_text(encoding="utf-8").splitlines()
    assert "# no_binding" in lines
