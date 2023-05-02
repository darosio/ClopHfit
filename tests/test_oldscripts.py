"""Test cases for the old scripts."""
from __future__ import annotations

import subprocess  # nosec B404
import sys
import typing
from pathlib import Path
from typing import Any
from typing import Iterator
from typing import List
from typing import Tuple

import matplotlib.testing.compare as mpltc  # type: ignore
import pytest
from matplotlib.testing.exceptions import ImageComparisonFailure  # type: ignore

_data = Path(__file__).parent / "data"
tmpoutput = _data / "_tmpoutput"
_expected = _data / "output"

Rscript = Tuple[Tuple[str, str, List[str]], Any]


@pytest.mark.skipif(sys.platform == "win32", reason="does not run on windows")
class TestTitrationFit:
    """Test the old ``fit_titration.py`` script."""

    note_fp = "./NTT-A04-Cl_note"
    csv_files = ["./Meas/A04 Cl_A.csv", "./Meas/A04 Cl_B.csv"]
    res_svd = [
        "K =  17.035\nsK =  0.666\nSA =  -0.0\nsSA =  0.007\nSB =  -0.275\nsSB =  0.002",
        "K =  14.824\nsK =  0.708\nSA =  0.019\nsSA =  0.004\nSB =  -0.274\nsSB =  0.002",
    ]
    res_band = [
        "K =  15.205\nsK =  0.549\nSA =  205506.748\nsSA =  1862.684\nSB =  1781.387\nsSB =  1249.163",
        "K =  15.015\nsK =  0.701\nSA =  253789.494\nsSA =  3055.602\nSB =  -4381.091\nsSB =  2040.904",
    ]

    @pytest.fixture(
        scope="class",
        params=[
            (csv_files[0], res_svd[0], ["-m", "svd"]),
            (csv_files[0], res_band[0], ["-m", "band", "-b", "480", "530"]),
            (csv_files[1], res_svd[1], ["-m", "svd"]),
            (csv_files[1], res_band[1], ["-m", "band", "-b", "480", "530"]),
        ],
    )
    def run_script(self, request: pytest.FixtureRequest) -> Iterator[Rscript]:
        """Run the script as class fixture."""
        cli = Path("../../src/clophfit/old/fit_titration.py")
        csv_file = request.param[0]
        with subprocess.Popen(  # nosec B603
            [cli, csv_file, self.note_fp, "-t", "cl", "-d", "_tmpoutput"]  # noqa: S603
            + request.param[2],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            cwd=_data,
        ) as process:
            yield request.param, process.communicate()
            if (_data / "bs.txt").exists():
                (_data / "bs.txt").unlink()
            for fp in _expected.glob("*_pdf.png"):
                fp.unlink()

    def test_stdout(self, run_script: Rscript) -> None:
        """It print out results."""
        expected = run_script[0][1]
        assert expected in run_script[1][0]

    @pytest.mark.xfail(reason="Deprecation from dependency.")
    def test_stderr_svd(self, run_script: Rscript) -> None:
        """Test stderr for svd."""
        if run_script[0][2][1] == "svd":
            assert not run_script[1][1]
        else:
            raise AssertionError

    def test_stderr_band(self, run_script: Rscript) -> None:
        """Test stderr for band."""
        if run_script[0][2][1] == "band":
            assert not run_script[1][1]

    def test_pdf(self, run_script: Rscript) -> None:
        """It saves pdf file."""
        csv_ = run_script[0][0].split("/")[-1].split(".")[0]
        f = "_".join([run_script[0][2][1], csv_, "NTT-A04-Cl_note.pdf"])
        fp_test = tmpoutput / f
        fp_expected = _expected / f
        msg = mpltc.compare_images(fp_test, fp_expected, 80.0)
        if msg:
            raise ImageComparisonFailure(msg)  # pragma: no cover


@pytest.mark.skipif(sys.platform == "win32", reason="does not run on windows")
@typing.no_type_check
class TestTitrationFitGlobal:
    """It test the old ``fit_titration_global.py`` script."""

    dat_files = [
        "./global/pH/D05.dat",
        "./global/Cl/B05-20130628-cor.dat",
    ]
    res = [
        """SA1   683.357   714.804   740.747   767.245   800.925
SB1   246.164   299.394   338.212   374.611   417.296
SA2   11.0487   47.6023   76.2874     104.3   138.128
SB2    491.64   534.936   571.809   611.045   664.602
  K   7.34376   7.51152   7.65166   7.80003   7.99971
bootstrap:""",
        """SA1   26910.8   27072.2   27216.7   27361.3   27522.7
SB1   916.697   1037.35   1144.64    1251.2   1369.35
SA2   3813.97   3970.34   4110.42   4250.52   4406.96
SB2      -INF   24.2754   87.1125   149.801   219.626
  K   7.51099   7.78702   8.03881   8.29508   8.58663
bootstrap:""",
    ]

    @pytest.fixture(
        scope="class",
        params=[
            (dat_files[0], res[0], ["--boot", "3"]),
            (dat_files[1], res[1], ["--boot", "3", "-t", "cl"]),
        ],
    )
    def run_script(self, request: pytest.FixtureRequest) -> Iterator[Rscript]:
        """Run the script as class fixture."""
        cli = "../../src/clophfit/old/fit_titration_global.py"
        dat_file = request.param[0]
        with subprocess.Popen(  # nosec B603
            [cli, dat_file, "_tmpoutput"] + request.param[2],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            cwd=_data,
            shell=False,  # noqa: S603
        ) as process:
            yield request.param, process.communicate()

    def test_stdout(self, run_script: Rscript) -> None:
        """It print out results."""
        expected = run_script[0][1]
        assert expected in run_script[1][0]

    @pytest.mark.xfail(reason="Deprecation from dependency.")
    def test_stderr(self, run_script: Rscript) -> None:
        """Stderr is empty."""
        assert not run_script[1][1]

    @pytest.mark.xfail(reason="Image sizes do not match.")
    def test_png(self, run_script: Rscript) -> None:
        """It saves pdf file."""
        f = ".".join([run_script[0][0], "png"])
        fp_test = tmpoutput / f.rsplit("/", maxsplit=1)[-1]
        fp_expected = _expected / f.lstrip("./")
        msg = mpltc.compare_images(fp_test, fp_expected, 1.01)
        if msg:  # pragma: no cover
            raise ImageComparisonFailure(msg)
