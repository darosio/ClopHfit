"""Acid-step factor: one free brightness factor on the most acidic step, shared by labels."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pymc as pm  # type: ignore[import-untyped]
import pytest
from click.testing import CliRunner

from clophfit.__main__ import ppr
from clophfit.fitting import bayes
from clophfit.fitting.bayes_config import SamplerConfig
from clophfit.fitting.core import ACID_SCALE, fit_binding_glob
from clophfit.fitting.data_structures import DataArray, Dataset
from clophfit.fitting.models import binding_1site
from clophfit.fitting.residual_tests import well_leverage
from clophfit.prtecan import PlateScheme

TECAN = Path(__file__).parent / "Tecan"
X = np.array([9.0, 8.4, 7.8, 7.1, 6.5, 5.9, 5.0])


def _dimmed(k: float, factor: float, seed: int) -> Dataset:
    rng = np.random.default_rng(seed)
    y1 = binding_1site(X, k, 900.0, 2400.0, is_ph=True)
    y2 = binding_1site(X, k, 2000.0, 120.0, is_ph=True)
    y1[-1] *= factor
    y2[-1] *= factor
    return Dataset(
        {
            "1": DataArray(X, y1 + rng.normal(0, 15, X.size)),
            "2": DataArray(X, y2 + rng.normal(0, 4, X.size)),
        },
        is_ph=True,
    )


def test_default_fit_has_no_acid_parameter() -> None:
    """Without the option the parameter set is the plain two-state one."""
    res = fit_binding_glob(_dimmed(7.3, 1.0, 0))
    assert res.result is not None
    assert ACID_SCALE not in res.result.params


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_acid_scale_recovers_k_and_factor(seed: int) -> None:
    """A 15 % loss at the last step is absorbed and K comes back unbiased."""
    ds = _dimmed(7.3, 0.85, seed)
    plain = fit_binding_glob(ds).result
    scaled = fit_binding_glob(ds, acid_scale=True).result
    assert plain is not None
    assert scaled is not None
    assert scaled.params[ACID_SCALE].value == pytest.approx(0.85, abs=0.03)
    assert (
        abs(scaled.params["K"].value - 7.3) < abs(plain.params["K"].value - 7.3) + 0.01
    )
    assert scaled.params["K"].value == pytest.approx(7.3, abs=0.05)


def test_acid_scale_works_with_outlier_screen() -> None:
    """The screen scores residuals of the scaled model, so it keeps the acid step."""
    res = fit_binding_glob(
        _dimmed(7.3, 0.8, 3), acid_scale=True, remove_outliers="mad:3.5:5"
    )
    assert res.dataset is not None
    assert all(bool(da.mask[-1]) for da in res.dataset.values())


def test_acid_scale_rejects_ligand_titration() -> None:
    """A chloride titration has no acid step."""
    ds = Dataset(
        {"1": DataArray(np.array([0.0, 10, 50, 200]), np.array([2.0, 1.5, 1.0, 0.6]))},
        is_ph=False,
    )
    with pytest.raises(ValueError, match="pH titrations only"):
        fit_binding_glob(ds, acid_scale=True)


class _StopBuildError(RuntimeError):
    """Raised in place of sampling once the model is built."""


def test_multi_model_adds_per_well_acid_factor(monkeypatch: pytest.MonkeyPatch) -> None:
    """--mcmc multi gets one acid_scale per well, and none without the option."""
    seen: dict[str, object] = {}

    def capture(*_args: object, **_kwargs: object) -> None:
        model = pm.modelcontext(None)
        seen["vars"] = set(model.named_vars)
        seen["dims"] = model.named_vars_to_dims.get(ACID_SCALE)
        raise _StopBuildError

    monkeypatch.setattr(pm, "sample", capture)
    scheme = PlateScheme()
    wells = {
        w: copy.deepcopy(_dimmed(7.3, 0.85, i))
        for i, w in enumerate(("A01", "A02", "B01"))
    }
    for acid, expected in ((True, True), (False, False)):
        with pytest.raises(_StopBuildError):
            bayes.fit_binding_pymc_multi(
                copy.deepcopy(wells),
                scheme,
                n_xerr=0.0,
                acid_scale=acid,
                sampler=SamplerConfig(n_samples=2, n_tune=1),
            )
        assert (ACID_SCALE in seen["vars"]) is expected  # type: ignore[operator]
    wells_cl = {
        "A01": Dataset(
            {
                "1": DataArray(
                    np.array([0.0, 10, 50, 200]), np.array([2.0, 1.5, 1.0, 0.6])
                )
            },
            is_ph=False,
        )
    }
    with pytest.raises(ValueError, match="pH titrations only"):
        bayes.fit_binding_pymc_multi(
            wells_cl,
            scheme,
            acid_scale=True,
            sampler=SamplerConfig(n_samples=2, n_tune=1),
        )


def test_cli_flag_reaches_spec_and_is_refused_for_single_mcmc(tmp_path: Path) -> None:
    """The flag is echoed in the run spec, and refused with a per-well MCMC model."""
    runner = CliRunner()
    list_f, scheme_f = (
        str(TECAN / "L2" / "list.pH.csv"),
        str(TECAN / "L2" / "scheme.txt"),
    )
    res = runner.invoke(
        ppr,
        [
            "--out",
            str(tmp_path),
            "tecan",
            list_f,
            "--sch",
            scheme_f,
            "--acid-scale",
            "--print-spec",
        ],
    )
    assert res.exit_code == 0, res.output
    assert "acid_scale" in res.output
    res = runner.invoke(
        ppr,
        [
            "--out",
            str(tmp_path),
            "tecan",
            list_f,
            "--sch",
            scheme_f,
            "--acid-scale",
            "--mcmc",
            "single",
        ],
    )
    assert res.exit_code != 0
    assert "--acid-scale" in res.output


def test_residual_table_applies_acid_factor() -> None:
    """Residuals of an acid-scaled fit use the scaled prediction on the acid step."""
    ds = _dimmed(7.3, 0.8, 4)
    plain = fit_binding_glob(ds).residual_table(well="A01")
    scaled = fit_binding_glob(ds, acid_scale=True).residual_table(well="A01")
    assert "acid_step" not in plain.columns
    acid = scaled[scaled["acid_step"]]
    assert set(acid["step"]) == {X.size - 1}
    assert len(acid) == 2  # one row per label
    # the fitted factor absorbs the loss: acid-step residuals shrink
    plain_acid = plain[plain["step"] == X.size - 1]
    assert acid["std_res"].abs().max() < plain_acid["std_res"].abs().max()


def test_acid_factor_raises_acid_step_leverage() -> None:
    """With the factor as a parameter, the acid rows' leverage goes up."""
    ds = _dimmed(7.3, 0.8, 5)
    fits = {
        "plain": fit_binding_glob(ds),
        "acid": fit_binding_glob(ds, acid_scale=True),
    }
    lev = {}
    for name, fr in fits.items():
        assert fr.result is not None
        table = fr.residual_table(well="A01")
        params = {"A01": {k: float(v.value) for k, v in fr.result.params.items()}}
        h = well_leverage(table, params, is_ph=True)
        lev[name] = float(h[table["step"] == X.size - 1].mean())
    assert lev["acid"] > lev["plain"]
