"""Three gaps found fitting the chloride titrations, and the sampler options production needed.

1. ``ppr`` could not set the number of NUTS chains or a seed, so a six-chain,
   reproducible production fit needed a wrapper around the CLI.
2. Diagnostic plots aborted chloride runs after their fits had succeeded: an
   empty residual table raised inside matplotlib ("Number of columns must be
   a positive integer, not 0"), and so did a K without standard error in the
   K plot ("'xerr' must not contain None").
3. A control group with no finite preliminary K -- V224Q in a chloride
   titration, which does not bind, so its Kd is infinite -- raised in
   ``weighted_stats`` and aborted the whole multi-well fit.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pymc as pm  # type: ignore[import-untyped]
import pytest
from click.testing import CliRunner

import clophfit.__main__ as cli
from clophfit.__main__ import ppr
from clophfit.fitting import bayes
from clophfit.fitting.bayes import weighted_stats
from clophfit.fitting.bayes_config import SamplerConfig
from clophfit.fitting.core import fit_binding_glob
from clophfit.prtecan import PlateScheme, export
from clophfit.prtecan.titration import TitrationResults

if TYPE_CHECKING:
    from clophfit.fitting.data_structures import Dataset

LIST_PH = str(Path(__file__).parent / "Tecan" / "140220" / "list.pH.csv")


class _StopBuildError(Exception):
    """Raised by a patched step to end the model build once it has been seen."""


# -- 1. chains and seed from the CLI --------------------------------------------


def test_cli_chains_and_seed_reach_the_sampler(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """--mcmc-chains and --mcmc-seed must land in the SamplerConfig ppr builds."""
    seen: dict[str, Any] = {}

    def fake_export(_tit: object, _config: object, spec: object, *_a: object) -> None:
        seen["spec"] = spec

    monkeypatch.setattr(cli, "export_data_fit", fake_export)
    result = CliRunner().invoke(
        ppr,
        [
            "-o",
            str(tmp_path / "out"),
            "tecan",
            LIST_PH,
            "--mcmc",
            "multi",
            "--mcmc-chains",
            "6",
            "--mcmc-seed",
            "20260914",
        ],
    )
    assert result.exit_code == 0, result.output
    sampler = seen["spec"].sampler
    assert sampler.chains == 6
    assert sampler.random_seed == 20260914


def test_cli_leaves_chains_and_seed_to_the_sampler_by_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Unset, both stay None, so existing runs sample exactly as before."""
    seen: dict[str, Any] = {}
    monkeypatch.setattr(
        cli, "export_data_fit", lambda _t, _c, spec, *_a: seen.update(spec=spec)
    )
    result = CliRunner().invoke(
        ppr, ["-o", str(tmp_path / "out"), "tecan", LIST_PH, "--mcmc", "multi"]
    )
    assert result.exit_code == 0, result.output
    assert seen["spec"].sampler.chains is None
    assert seen["spec"].sampler.random_seed is None


def test_chains_and_seed_move_the_signature() -> None:
    """A different chain count or seed is a different run and must sign as one."""

    def sig(*extra: str) -> str:
        out = (
            CliRunner()
            .invoke(ppr, ["tecan", LIST_PH, "--print-spec", "--mcmc", "multi", *extra])
            .output
        )
        return next(ln for ln in out.splitlines() if ln.startswith("signature:"))

    assert sig("--mcmc-chains", "6") != sig()
    assert sig("--mcmc-seed", "1") != sig("--mcmc-seed", "2")


# -- 2. an empty residual table is not an error ----------------------------------


def test_export_residuals_writes_nothing_for_an_empty_table(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """No residuals: return quietly, write no files, raise nothing."""
    monkeypatch.setattr(
        export, "residuals_from_fit_results", lambda *_a, **_k: pd.DataFrame()
    )
    export.export_residuals(tmp_path, {}, 4)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("missing", ["A02", "B05"])  # a control well, a sample well
def test_plot_k_draws_a_well_without_stderr(missing: str) -> None:
    """A K without standard error is drawn without an error bar, not raised on."""
    se: dict[str, float | None] = {"A01": 0.1, "A02": 0.1, "B05": 0.2}
    se[missing] = None
    df = pd.DataFrame(
        {"K": [7.0, 7.1, 6.9], "sK": list(se.values()), "n_labels": [2, 2, 2]},
        index=pd.Index(list(se), name="well"),
    )
    scheme = PlateScheme()
    scheme.ctrl = ["A01", "A02"]
    scheme.names = {"E2GFP": {"A01", "A02"}}
    res = TitrationResults(scheme=scheme, fit_keys=set(se), _dataframe=df)
    assert res.plot_k() is not None


# -- 3. a control that does not bind ---------------------------------------------


def test_weighted_stats_skips_a_control_with_no_finite_k() -> None:
    """A non-binding control is left out, with a warning naming it, not raised on."""
    values: dict[str, list[float | None]] = {
        "E2GFP": [7.0, 7.2],
        "V224Q": [np.inf, None],
    }
    stderr: dict[str, list[float | None]] = {
        "E2GFP": [0.1, 0.1],
        "V224Q": [np.nan, None],
    }
    with pytest.warns(UserWarning, match="V224Q"):
        result = weighted_stats(values, stderr)
    assert set(result) == {"E2GFP"}


def test_shared_control_k_is_built_only_for_groups_with_an_estimate() -> None:
    """Pooled-K mode must not index a group that has no K estimate."""
    scheme = PlateScheme()
    scheme.names = {"E2GFP": {"A01", "A02"}, "V224Q": {"A03", "A04"}}
    with pm.Model():
        k_params, _ = bayes._build_ctr_k_params(  # ruff: ignore[private-member-access] - the seam under test
            scheme, {"E2GFP": (7.0, 0.1)}, {"A01", "A02"}, ctr_free_k=False
        )
    assert set(k_params) == {"E2GFP"}


@pytest.mark.parametrize("ctr_free_k", [True, False])
def test_multi_fit_leaves_out_a_control_group_without_k(
    monkeypatch: pytest.MonkeyPatch, multi_dataset: Dataset, *, ctr_free_k: bool
) -> None:
    """The non-binder's wells never reach the K priors; the other wells do."""
    ok = fit_binding_glob(multi_dataset)
    flat = copy.deepcopy(ok)
    assert flat.result is not None
    flat.result.params["K"].stderr = None  # no usable preliminary K, as for V224Q
    scheme = PlateScheme()
    scheme.names = {"E2GFP": {"A01", "A02"}, "V224Q": {"A03", "A04"}}
    seen: dict[str, Any] = {}

    def stop(
        _fit_results: object, wells_list: list[str], *_a: object, **_k: object
    ) -> None:
        seen["wells"] = list(wells_list)
        raise _StopBuildError

    monkeypatch.setattr(bayes, "_free_k_init", stop)
    with pytest.warns(UserWarning, match="V224Q"), pytest.raises(_StopBuildError):
        bayes.fit_binding_pymc_multi(
            {
                "A01": ok,
                "A02": copy.deepcopy(ok),
                "A03": flat,
                "A04": copy.deepcopy(flat),
            },
            scheme,
            n_xerr=0.0,
            sampler=SamplerConfig(n_samples=2, n_tune=1),
            ctr_free_k=ctr_free_k,
        )
    assert sorted(seen["wells"]) == ["A01", "A02"]
