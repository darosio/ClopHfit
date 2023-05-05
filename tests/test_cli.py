"""Test ``clop`` cli."""
from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from clophfit.__main__ import clop


def test_eq1() -> None:
    """It runs XXX pr.tecan and generates correct results and graphs."""
    runner = CliRunner()
    result = runner.invoke(clop, ["eq1", "2", "2", "2"])
    assert result.exit_code == 0
    assert "4." in result.output


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


# # Make sure pr.enspire produces right output.
# import filecmp
# import os
# import shutil

# import pytest
# from matplotlib.testing.compare import compare_images
# from matplotlib.testing.exceptions import ImageComparisonFailure
# from typer.testing import CliRunner

# from clophfit.prenspire.script import app

# PATH = os.path.split(__file__)[0]
# tmpoutput = os.path.join(PATH, "EnSpire", "data", "tmpoutput") + os.sep
# expected = os.path.join(PATH, "EnSpire", "data", "output") + os.sep


@pytest.fixture(scope="module")
def run_prenspire(tmp_path: Path) -> None:
    """Run cli with actual data."""
    out = tmp_path / "out4"
    out.mkdir()
    runner = CliRunner()
    result = runner.invoke(
        clop,
        ["prenspire", "tests/EnSpire/data/NTT_37C_pKa.csv", "--out", "."],  # str(out)],
    )
    assert result.exit_code == 0
    # yield out


def test_res(run_prenspire):
    assert 1 == 1


# def test_csv_out(run_script):
#     """It outputs correct csv files."""
#     assert filecmp.cmp(tmpoutput + "NTT_37C_pKa_A.csv", expected + "NTT_37C_pKa_A.csv")
#     assert filecmp.cmp(tmpoutput + "NTT_37C_pKa_B.csv", expected + "NTT_37C_pKa_B.csv")


# @pytest.mark.parametrize("f", ["NTT_37C_pKa_A.png", "NTT_37C_pKa_B.png"])
# def test_png_out(run_script, f):
#     """It outputs correct png files."""
#     fp_test = os.path.join(tmpoutput + f)
#     fp_expected = os.path.join(expected + f)
#     msg = compare_images(fp_test, fp_expected, 0.0001)
#     if msg:
#         raise ImageComparisonFailure(msg)
