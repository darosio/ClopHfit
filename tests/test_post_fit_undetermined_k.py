"""Wells whose fitted K is undetermined are reported as such, not plotted as results.

The only K check ppr ran was pre-fit: a cheap fit judged against the whole x
span (~4 pH). A library well passed it with a preliminary SE of 2.2 and came out
of the final multi-well fit at pK 14 +/- 9, drawn in the K plot beside the
determined wells and setting its x-limits. The post-fit screen that used to
catch it had been retired, and its rule (``sK / K > 0.3``) never fires on a pKa.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

import clophfit.__main__ as cli
from clophfit.__main__ import ppr
from clophfit.prtecan import PlateScheme, export
from clophfit.prtecan.titration import TecanConfig, TitrationResults

DATA = Path(__file__).parent / "Tecan" / "140220"


def _fit(k: list[float], sk: list[float | None], **extra: list[float]) -> pd.DataFrame:
    wells = [f"A{i:02d}" for i in range(1, len(k) + 1)]
    return pd.DataFrame({"K": k, "sK": sk, **extra}, index=pd.Index(wells, name="well"))


# -- the rule -------------------------------------------------------------------


def test_ph_cut_is_absolute_se_in_ph() -> None:
    """A pKa is undetermined above the SE limit, whatever its value."""
    fit = _fit([7.0, 7.1, 14.1, 5.2], [0.06, 0.30, 8.6, 0.31])
    flag = export.undetermined_k(fit, is_ph=True, max_k_se=0.30)
    assert flag.tolist() == [False, False, True, True]


def test_ph_cut_catches_what_a_ratio_misses() -> None:
    """A ratio of 1.26 / 7.13 = 0.18 passes sK/K > 0.3; 1.26 pH is no pKa."""
    fit = _fit([7.13], [1.26])
    assert 1.26 / 7.13 < 0.3
    assert export.undetermined_k(fit, is_ph=True, max_k_se=0.30).item()


@pytest.mark.parametrize("sk", [None, np.nan, np.inf])
def test_missing_or_infinite_se_is_undetermined(sk: float | None) -> None:
    """No standard error means the fit could not locate K."""
    flag = export.undetermined_k(_fit([7.0], [sk]), is_ph=True, max_k_se=0.30)
    assert flag.item()


def test_a_fit_table_without_k_flags_every_well() -> None:
    """A table with no K column has nothing determined in it."""
    fit = pd.DataFrame({"S0_1": [1.0]}, index=pd.Index(["A01"], name="well"))
    assert export.undetermined_k(fit, is_ph=True, max_k_se=0.30).item()


def test_chloride_cut_is_relative() -> None:
    """Kd: non-positive, an SE larger than itself, or an HDI through zero."""
    fit = _fit(
        [8.0, -2.7, 3.4, 9.0, 40.0],
        [1.0, 0.6, 3.3, 10.0, 30.0],
        Khdi03=[6.2, -3.8, 0.1, 0.5, -5.0],
    )
    flag = export.undetermined_k(fit, is_ph=False, max_k_se=0.30)
    assert flag.tolist() == [False, True, False, True, True]


def test_chloride_ignores_the_ph_limit() -> None:
    """A Kd of 8 +/- 1 mM is determined, though 1 > 0.30."""
    assert not export.undetermined_k(
        _fit([8.0], [1.0]), is_ph=False, max_k_se=0.3
    ).item()


# -- what is done with it -------------------------------------------------------


def _plate(df: pd.DataFrame) -> TitrationResults:
    scheme = PlateScheme()
    scheme.ctrl = ["A01", "A02"]
    scheme.names = {"E2GFP": {"A01", "A02"}}
    return TitrationResults(scheme=scheme, fit_keys=set(df.index), _dataframe=df)


def test_plot_k_leaves_excluded_wells_off_and_says_so() -> None:
    """The excluded well is not drawn, nor does it set the x-limits."""
    df = _fit([7.0, 7.1, 6.9, 14.1], [0.1, 0.1, 0.2, 8.6], n_labels=[2, 2, 2, 1])
    fig = _plate(df).plot_k(exclude=["A04"])
    labels = [t.get_text() for ax in fig.axes for t in ax.get_yticklabels()]
    assert not any(lbl.startswith("A04") for lbl in labels)
    assert "1 undetermined well(s) not shown" in fig.texts[0].get_text()
    assert max(ax.get_xlim()[1] for ax in fig.axes) < 8


def test_plot_k_survives_every_well_excluded() -> None:
    """A plate with no determined K still gets its (empty) plot, not an abort."""
    df = _fit([7.0, 7.1, 14.1], [0.4, 0.5, 8.6], n_labels=[1, 1, 1])
    fig = _plate(df).plot_k(exclude=list(df.index))
    assert "3 undetermined well(s) not shown" in fig.texts[0].get_text()


def test_plot_k_without_exclusions_is_unchanged() -> None:
    """No exclusion, no note: existing plots stay as they were."""
    df = _fit([7.0, 7.1, 6.9], [0.1, 0.1, 0.2], n_labels=[2, 2, 2])
    fig = _plate(df).plot_k(title="t")
    assert fig.texts[0].get_text() == "t"


def test_undetermined_wells_are_appended_under_their_own_heading(
    tmp_path: Path,
) -> None:
    """After the pre-fit sections, so the discards above stay discards."""
    path = tmp_path / "discarded_wells.txt"
    path.write_text("D06\n\n# low_signal\nD09\n", encoding="utf-8")
    export.record_undetermined(tmp_path, ["E12", "D09"])
    assert path.read_text(encoding="utf-8").splitlines() == [
        "D06",
        "",
        "# low_signal",
        "D09",
        "",
        "# undetermined_k",
        "D09",
        "E12",
    ]


def test_nothing_is_written_when_every_k_is_determined(tmp_path: Path) -> None:
    """No undetermined wells, no file and no empty heading."""
    export.record_undetermined(tmp_path, [])
    assert not (tmp_path / "discarded_wells.txt").exists()


# -- the CLI --------------------------------------------------------------------


def _config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *extra: str
) -> TecanConfig:
    seen: dict[str, TecanConfig] = {}
    monkeypatch.setattr(cli, "export_data_fit", lambda _t, c, *_a: seen.update(c=c))
    args = ["-o", str(tmp_path / "out"), "tecan", str(DATA / "list.pH.csv"), *extra]
    result = CliRunner().invoke(ppr, args)
    assert result.exit_code == 0, result.output
    return seen["c"]


def test_cli_max_k_se_reaches_the_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """--max-k-se sets the limit; unset it is 0.30 pH."""
    assert _config(monkeypatch, tmp_path).max_k_se == pytest.approx(0.30)
    assert _config(monkeypatch, tmp_path, "--max-k-se", "0.2").max_k_se == 0.2


def test_cli_rejects_a_non_positive_limit(tmp_path: Path) -> None:
    """A limit of zero would call every well undetermined."""
    args = ["-o", str(tmp_path), "tecan", str(DATA / "list.pH.csv")]
    result = CliRunner().invoke(ppr, [*args, "--max-k-se", "0"])
    assert result.exit_code != 0


@pytest.mark.slow
def test_ppr_reports_undetermined_wells_end_to_end(tmp_path: Path) -> None:
    """Every ffit table carries the flag; the reported fit's wells are listed."""
    args = ["--out", str(tmp_path), "tecan", str(DATA / "list.pH.csv")]
    args += ["--fit", "--sch", str(DATA / "scheme.txt"), "--bg", "--max-k-se", "0.05"]
    result = CliRunner().invoke(ppr, args)
    assert result.exit_code == 0, result.output
    fit_dir = next(tmp_path.rglob("ffit0.csv")).parent
    tables = sorted(fit_dir.glob("ffit*.csv"))
    assert tables
    for table in tables:
        fit = pd.read_csv(table, index_col="well")
        assert fit["undetermined"].dtype == bool
        expected = export.undetermined_k(fit, is_ph=True, max_k_se=0.05)
        assert fit["undetermined"].equals(expected)
    last = pd.read_csv(tables[-1], index_col="well")
    flagged = sorted(last.index[last["undetermined"]])
    assert flagged, "a 0.05 pH limit should flag some wells on this plate"
    lines = (fit_dir / "discarded_wells.txt").read_text(encoding="utf-8").splitlines()
    section = lines[lines.index("# undetermined_k") + 1 :]
    assert section == flagged
