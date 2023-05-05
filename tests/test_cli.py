"""Test ``clop`` cli."""
from __future__ import annotations

import filecmp
from pathlib import Path

import pytest
from click.testing import CliRunner
from matplotlib.testing.compare import compare_images  # type: ignore
from matplotlib.testing.exceptions import ImageComparisonFailure  # type: ignore

from clophfit.__main__ import clop


def test_eq1() -> None:
    """It runs XXX pr.tecan and generates correct results and graphs."""
    runner = CliRunner()
    result = runner.invoke(clop, ["eq1", "2", "2", "2"])
    assert result.exit_code == 0
    assert "4." in result.output


def test_prenspire(tmp_path: Path) -> None:
    """Run cli with actual data."""
    expected = Path(__file__).parent / "EnSpire" / "data" / "output"
    out = tmp_path / "E"
    out.mkdir()
    runner = CliRunner()
    result = runner.invoke(
        clop,
        ["prenspire", "tests/EnSpire/data/NTT_37C_pKa.csv", "--out", str(out)],
    )
    assert result.exit_code == 0
    # validate output files
    assert (out / "NTT_37C_pKa_A.csv").exists()
    assert (out / "NTT_37C_pKa_B.csv").exists()
    assert (out / "NTT_37C_pKa_A.png").exists()
    assert (out / "NTT_37C_pKa_B.png").exists()
    # validate output file contents
    assert filecmp.cmp(out / "NTT_37C_pKa_A.csv", expected / "NTT_37C_pKa_A.csv")
    assert filecmp.cmp(out / "NTT_37C_pKa_B.csv", expected / "NTT_37C_pKa_B.csv")
    # validate graph
    for f in ["NTT_37C_pKa_A.png", "NTT_37C_pKa_B.png"]:
        msg = compare_images(out / f, expected / f, 0.0001)
        if msg:
            raise ImageComparisonFailure(msg)


@pytest.mark.filterwarnings("ignore:OVER")
def test_prtecan(tmp_path: Path) -> None:
    """It runs XXX pr.tecan and generates correct results and graphs."""
    out = tmp_path / "out3"
    out.mkdir()
    runner = CliRunner()
    result = runner.invoke(
        clop,
        [
            "prtecan",
            "tests/Tecan/140220/list.pH",
            "--out",
            str(out),
            "--fit",
            "--scheme",
            "tests/Tecan/140220/scheme.txt",
            "--bg",
        ],
    )
    assert result.exit_code == 0
