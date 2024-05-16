"""Test ``clop`` cli."""

import csv
import filecmp
import re
import tempfile
import warnings
from pathlib import Path
from typing import IO, cast

import pytest
from click.testing import CliRunner
from matplotlib.testing.compare import compare_images
from matplotlib.testing.exceptions import ImageComparisonFailure

from clophfit.__main__ import clop, fit_titration, note2csv, ppr

# tests path
tpath = Path(__file__).parent


@pytest.fixture()
def runner() -> CliRunner:
    """Fixture for invoking command-line interfaces."""
    return CliRunner()


def create_temp_tsv_file(content: str) -> IO[str]:
    """Create a temporary TSV file and populate it with content."""
    temp_file = tempfile.NamedTemporaryFile(delete=False, mode="w+", suffix=".tsv")
    temp_file.write(content)
    temp_file.seek(0)
    return cast(IO[str], temp_file)


def test_default_case(runner: CliRunner) -> None:
    """Test default case for note2csv function."""
    temp_file = create_temp_tsv_file("Well\tpH\tCl\tName\nD01\t9.15\t0\tNTT-G10\n")

    result = runner.invoke(note2csv, [temp_file.name])

    assert result.exit_code == 0

    # Read the output CSV and check its contents
    output_path = Path(temp_file.name).with_suffix(".csv")
    with output_path.open("r") as f:
        reader = csv.reader(f)
        header = next(reader)
        assert header == ["Well", "pH", "Cl", "Name", "Temp", "Labels"]
        row = next(reader)
        assert row == ["D01", "9.15", "0", "NTT-G10", "37.0", "A B"]


def test_custom_output(runner: CliRunner) -> None:
    """Test custom case for note2csv function."""
    temp_file = create_temp_tsv_file("Well\tpH\tCl\tName\nD01\t9.15\t0\tNTT-G10\n")

    with tempfile.NamedTemporaryFile(
        delete=False, mode="w+", suffix=".csv"
    ) as output_file:
        output_file.write(
            ",".join(["Well", "pH", "Cl", "Name", "Temp", "Labels"]) + "\n"
        )
        output_file.seek(0)  # Important to rewind the file to the beginning
        result = runner.invoke(
            note2csv,
            [temp_file.name, "-o", output_file.name, "-t", "22.3", "-l", "A F G"],
        )

    assert result.exit_code == 0

    # Read the output CSV and check its contents
    output_path = Path(output_file.name)
    with output_path.open("r") as f:
        reader = csv.reader(f)
        header = next(reader)
        assert header == ["Well", "pH", "Cl", "Name", "Temp", "Labels"]
        row = next(reader)
        assert row == ["D01", "9.15", "0", "NTT-G10", "22.3", "A F G"]


def test_eq1() -> None:
    """It runs XXX pr.tecan and generates correct results and graphs."""
    runner = CliRunner()
    result = runner.invoke(clop, ["eq1", "2", "2", "2"])
    assert result.exit_code == 0
    assert "4." in result.output


def test_prenspire(tmp_path: Path) -> None:
    """Test prenspire command with actual data and validate output."""
    expected = tpath / "EnSpire" / "cli" / "output"
    input_csv = tpath / "EnSpire" / "cli" / "NTT_37C_pKa.csv"
    out = tmp_path / "E"
    out.mkdir()
    runner = CliRunner()
    result = runner.invoke(ppr, ["--out", str(out), "enspire", str(input_csv)])
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
        msg = compare_images(str(out / f), str(expected / f), 0.007)
        if msg:  # pragma: no cover
            raise ImageComparisonFailure(msg)


@pytest.mark.filterwarnings("ignore:OVER")
def test_prtecan(tmp_path: Path) -> None:
    """Test prtecan command with actual data."""
    list_f = str(tpath / "Tecan" / "140220" / "list.pH")
    scheme_f = str(tpath / "Tecan" / "140220" / "scheme.txt")
    out = tmp_path / "out3"
    out.mkdir()
    runner = CliRunner()
    with warnings.catch_warnings():
        # Suppress the UserWarnings related to insufficient data points and cleaning
        warnings.simplefilter("ignore", category=UserWarning)
        result = runner.invoke(
            ppr,
            ["--out", str(out), "tecan", list_f, "--fit", "--scheme", scheme_f, "--bg"],
        )
    assert result.exit_code == 0


@pytest.mark.filterwarnings("ignore:OVER")
def test_prtecan_cl(tmp_path: Path) -> None:
    """Test prtecan command with actual data."""
    list_f = str(tpath / "Tecan" / "140220" / "list.cl")
    scheme_f = str(tpath / "Tecan" / "140220" / "scheme.txt")
    adds_f = str(tpath / "Tecan" / "140220" / "additions.cl")
    out = tmp_path / "out4"
    out.mkdir()
    runner = CliRunner()
    base_args = ["--out", str(out), "tecan", list_f, "--fit", "--scheme", scheme_f]
    base_args.extend(["--dil", adds_f, "--bg", "--no-is-ph"])
    result = runner.invoke(ppr, base_args)
    assert result.exit_code == 0


@pytest.mark.parametrize(
    ("csv_file", "output", "opts"),
    [
        ("A04 Cl_A.csv", " K :  13.64  14.28  14.86  15.44  16.04  16.71  17.51", None),
        ("A04 Cl_A.csv", " K :  13.52  14.11  14.66  15.20", ["-b", "480", "530"]),
        ("A04 Cl_B.csv", " K :  12.73  13.46  14.14  14.82  15.54  16.35  17.32", None),
        ("A04 Cl_B.csv", " K :  12.93  13.66  14.34  15.02", ["-b", "480", "530"]),
    ],
)
def test_fit_titration(
    csv_file: str, output: str, opts: list[str], tmp_path: Path
) -> None:
    """Test the old ``fit_titration.py`` script with actual data and validate output."""
    note_fp = tpath / "spec" / "NTT-A04-Cl_note"
    csv_fp = tpath / "spec" / "Meas" / csv_file
    out = tmp_path / "tit_fit"
    out.mkdir()
    runner = CliRunner()
    base_args = ["--no-is-ph", "-o", str(out), "spec", str(csv_fp), str(note_fp)]
    if opts:
        base_args.extend(opts)
        sbands = f"({opts[1]}, {opts[2]})"
    else:
        sbands = "None"
    result = runner.invoke(fit_titration, base_args)
    assert result.exit_code == 0
    expected_output = re.sub(" ", r"\\s+", output.strip())
    # Assert that the pattern appears in the output
    assert re.search(expected_output, result.output) is not None
    # Asserting that PDF is created
    expected_pdf_filename = out / f"{csv_fp.stem}_{sbands}_{note_fp.stem}.pdf"
    assert expected_pdf_filename.exists()


@pytest.mark.parametrize(
    ("dat_f", "ph_opt", "output", "opts"),
    [
        ("pH_D05.dat", "--is-ph", "K: 7.5313", ["-b", "78"]),
        ("pH_D05.dat", "--is-ph", "K: 7.65166", ["--no-weight"]),
        ("Cl_B05-20130628-cor.dat", "--no-is-ph", "K: 6.4537", None),
    ],
)
def test_fit_titration_glob(dat_f: str, ph_opt: str, output: str, opts: str) -> None:
    """Fit result for K is correct and png are generated as with old ``fit_titration_global.py`` script."""
    dat_fp = tpath / "glob" / dat_f
    base_args = ["-v", ph_opt, "glob", str(dat_fp)]
    if opts:
        base_args.extend(opts)
    runner = CliRunner()
    result = runner.invoke(fit_titration, base_args)
    assert result.exit_code == 0
    expected_output = re.sub(" ", r"\\s+", output.strip())
    assert re.search(expected_output, result.output) is not None
    # assert that the png files are generated
    pngs = [dat_fp.with_suffix(".png")]
    if opts and "-b" in opts:
        pngs.append(dat_fp.with_stem(dat_fp.stem + "-emcee").with_suffix(".png"))
        if "--is-ph" in ph_opt:
            pngs.append(
                dat_fp.with_stem(dat_fp.stem + "-emc-ratios").with_suffix(".png")
            )
    for file in pngs:
        assert file.exists(), f"{file} does not exist!"
    # delete the png files
    for file in pngs:
        file.unlink()
