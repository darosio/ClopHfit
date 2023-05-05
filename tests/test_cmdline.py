"""Test cli.

Make sure pr.enspire produces right output.

"""
import filecmp
import os
import shutil

import pytest
from matplotlib.testing.compare import compare_images
from matplotlib.testing.exceptions import ImageComparisonFailure
from typer.testing import CliRunner

from clophfit.prenspire.script import app

PATH = os.path.split(__file__)[0]
tmpoutput = os.path.join(PATH, "EnSpire", "data", "tmpoutput") + os.sep
expected = os.path.join(PATH, "EnSpire", "data", "output") + os.sep


@pytest.fixture
def runner():
    """Runner."""
    return CliRunner()


@pytest.fixture(scope="module")
def run_script():
    """Run cli with actual data."""
    os.chdir(os.path.join(PATH, "data"))
    runner = CliRunner()
    result = runner.invoke(app, ["NTT_37C_pKa.csv", "--out", "tmpoutput"])
    assert result.exit_code == 0  # this can fail; like a test
    yield tmpoutput
    if os.path.exists(tmpoutput):
        shutil.rmtree(tmpoutput)


def test_version(runner):
    """Version number."""
    result = runner.invoke(app, ["--version"])
    assert "" == result.stdout


def test_csv_out(run_script):
    """It outputs correct csv files."""
    assert filecmp.cmp(tmpoutput + "NTT_37C_pKa_A.csv", expected + "NTT_37C_pKa_A.csv")
    assert filecmp.cmp(tmpoutput + "NTT_37C_pKa_B.csv", expected + "NTT_37C_pKa_B.csv")


@pytest.mark.parametrize("f", ["NTT_37C_pKa_A.png", "NTT_37C_pKa_B.png"])
def test_png_out(run_script, f):
    """It outputs correct png files."""
    fp_test = os.path.join(tmpoutput + f)
    fp_expected = os.path.join(expected + f)
    msg = compare_images(fp_test, fp_expected, 0.0001)
    if msg:
        raise ImageComparisonFailure(msg)
