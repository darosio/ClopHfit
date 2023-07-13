"""Test ``clop`` cli."""

import filecmp
import re
from pathlib import Path

import pytest
from click.testing import CliRunner
from matplotlib.testing.compare import compare_images  # type: ignore
from matplotlib.testing.exceptions import ImageComparisonFailure  # type: ignore

from clophfit.__main__ import clop, enspire, tecan

# tests path
tpath = Path(__file__).parent


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
    result = runner.invoke(enspire, [str(input_csv), "--out", str(out)])
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
    result = runner.invoke(
        tecan, [list_f, "--out", str(out), "--fit", "--scheme", scheme_f, "--bg"]
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
    base_args = [list_f, "--out", str(out), "--fit", "--scheme", scheme_f]
    base_args.extend(["--dil", adds_f, "--bg", "--no-is-ph"])
    result = runner.invoke(tecan, base_args)
    assert result.exit_code == 0


@pytest.mark.parametrize(
    ("csv_file", "expected_output", "additional_params"),
    [
        ("A04 Cl_A.csv", " K :  13.64  14.28  14.86  15.44  16.04  16.71  17.51", None),
        ("A04 Cl_A.csv", " K :  13.52  14.11  14.66  15.20", ["-b", "480", "530"]),
        ("A04 Cl_B.csv", " K :  12.73  13.46  14.14  14.82  15.54  16.35  17.32", None),
        ("A04 Cl_B.csv", " K :  12.93  13.66  14.34  15.02", ["-b", "480", "530"]),
    ],
)
def test_fit_titration(
    csv_file: str, expected_output: str, additional_params: list[str], tmp_path: Path
) -> None:
    """Test the old ``fit_titration.py`` script with actual data and validate output."""
    note_fp = tpath / "data" / "NTT-A04-Cl_note"
    csv_fp = tpath / "data" / "Meas" / csv_file
    out = tmp_path / "tit_fit"
    out.mkdir()
    runner = CliRunner()
    base_args = ["fit_titration", str(csv_fp), str(note_fp)]
    base_args.extend(["-t", "cl", "-d", str(out)])
    if additional_params:
        base_args.extend(additional_params)
        sbands = f"({additional_params[1]}, {additional_params[2]})"
    else:
        sbands = "None"
    result = runner.invoke(clop, base_args)
    assert result.exit_code == 0
    expected_output = re.sub(" ", r"\\s+", expected_output.strip())
    # Assert that the pattern appears in the output
    assert re.search(expected_output, result.output) is not None
    # Asserting that PDF is created
    expected_pdf_filename = out / f"{csv_fp.stem}_{sbands}_{note_fp.stem}.pdf"
    assert expected_pdf_filename.exists()
