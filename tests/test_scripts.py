"""Test cases for the old scripts."""
import glob
import os
import subprocess
from typing import Any, Generator, List, Tuple

import matplotlib.testing.compare as mpltc
import pytest
from matplotlib.testing.exceptions import ImageComparisonFailure

PATH = os.path.split(__file__)[0]


tmpoutput = os.path.join(PATH, "data", "_tmpoutput") + os.sep
expected = os.path.join(PATH, "data", "output") + os.sep

os.chdir(os.path.join(PATH, "data"))


class Test_fit_titration:
    """It test the old ``fit_titration.py`` script."""

    note_fp = "./NTT-A04-Cl_note"
    csv_files = ["./Meas/A04 Cl_A.csv", "./Meas/A04 Cl_B.csv"]
    res_svd = [
        "K =  17.035\nsK =  0.666\nSA =  -0.0\nsSA =  0.007\n"
        + "SB =  -0.275\nsSB =  0.002",
        """K =  14.824\nsK =  0.708\nSA =  0.019\nsSA =  0.004
SB =  -0.274\nsSB =  0.002\n""",
    ]
    res_band = [
        "K =  15.205\nsK =  0.549\nSA =  205506.748\nsSA =  1862.684\n"
        + "SB =  1781.387\nsSB =  1249.163",
        "K =  15.015\nsK =  0.701\nSA =  253789.494\nsSA =  3055.602\n"
        + "SB =  -4381.091\nsSB =  2040.904",
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
    def run_script(self, request: Any) -> Generator[Tuple[Tuple, Tuple], List, None]:
        """Run the script as class fixture."""
        cli = "../../src/clophfit/old/fit_titration.py"
        csv_file = request.param[0]
        process = subprocess.Popen(
            [cli, csv_file, self.note_fp, "-t", "cl", "-d", "_tmpoutput"]
            + request.param[2],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
        yield request.param, process.communicate()
        if os.path.exists("./bs.txt"):
            os.remove("bs.txt")
        for fp in glob.glob("output/*_pdf.png"):
            os.remove(fp)

    def test_stdout(self, run_script: Tuple) -> None:
        """It print out results."""
        expected = run_script[0][1]
        assert expected in run_script[1][0]

    @pytest.mark.xfail(reason="Deprecation from dependency.")
    def test_stderr_svd(self, run_script: Tuple[Tuple, Tuple]) -> None:
        """Test stderr for svd."""
        if run_script[0][2][1] == "svd":
            assert run_script[1][1] == ""
        else:
            assert 1 == 2

    def test_stderr_band(self, run_script: Tuple) -> None:
        """Test stderr for band."""
        if run_script[0][2][1] == "band":
            assert run_script[1][1] == ""
        else:
            assert 1 == 1

    def test_pdf(self, run_script: Tuple[Tuple, Tuple]) -> None:
        """It saves pdf file."""
        csv_ = run_script[0][0].split("/")[-1].split(".")[0]
        f = "_".join([run_script[0][2][1], csv_, "NTT-A04-Cl_note.pdf"])
        fp_test = os.path.join(tmpoutput, f)
        fp_expected = os.path.join(expected, f)
        msg = mpltc.compare_images(fp_test, fp_expected, 80.0)
        if msg:
            raise ImageComparisonFailure(msg)


class Test_fit_titration_global:
    """It test the old ``fit_titration_global.py`` script."""

    dat_files = [
        "./global/pH/D05.dat",
        "./global/Cl/B05-20130628-cor.dat",
    ]
    res = [
        """SA1   683.357   714.804   740.747   767.249   800.923
SB1   246.166   299.394   338.214   374.611   417.297
SA2   11.0488   47.6024   76.2864     104.3   138.128
SB2   491.637   534.936   571.809   611.045   664.602
  K   7.34381   7.51146   7.65166   7.80003   7.99974
bootstrap:""",
        """SA1     26911     27072   27216.7   27361.5   27522.5
SB1   916.693   1037.35   1144.64   1251.21   1369.35
SA2   3813.97   3970.38   4110.42   4250.56   4406.97
SB2      -INF   24.2754   87.1121   149.801   219.629
  K   7.51098     7.787   8.03882   8.29506   8.58665
bootstrap:""",
    ]

    @pytest.fixture(
        scope="class",
        params=[
            (dat_files[0], res[0], ["--boot", "3"]),
            (dat_files[1], res[1], ["--boot", "3", "-t", "cl"]),
        ],
    )
    def run_script(self, request: Any) -> Generator[Tuple[Tuple, Tuple], List, None]:
        """Run the script as class fixture."""
        cli = "../../src/clophfit/old/fit_titration_global.py"
        dat_file = request.param[0]
        process = subprocess.Popen(
            [cli, dat_file, "_tmpoutput"] + request.param[2],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
        yield request.param, process.communicate()
        # shutil.rmtree(tmpoutput)

    def test_stdout(self, run_script: Tuple) -> None:
        """It print out results."""
        expected = run_script[0][1]
        assert expected in run_script[1][0]

    @pytest.mark.xfail(reason="Deprecation from dependency.")
    def test_stderr(self, run_script: Tuple[Tuple, Tuple]) -> None:
        """Stderr is empty."""
        assert run_script[1][1] == ""

    @pytest.mark.xfail(reason="Image sizes do not match.")
    def test_png(self, run_script: Tuple[Tuple, Tuple]) -> None:
        """It saves pdf file."""
        f = ".".join([run_script[0][0], "png"])
        fp_test = os.path.join(tmpoutput, f.split("/")[-1])
        fp_expected = os.path.join(expected, f.lstrip("./"))
        msg = mpltc.compare_images(fp_test, fp_expected, 1.01)
        if msg:
            raise ImageComparisonFailure(msg)
