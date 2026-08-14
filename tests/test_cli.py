"""Test ``clop`` cli."""

import csv
import filecmp
import logging
import re
import tempfile
from pathlib import Path
from typing import IO, cast

import pytest
from click.testing import CliRunner
from PIL import Image

from clophfit.__main__ import clop, fit_titration, note2csv, ppr

# tests path
tpath = Path(__file__).parent


@pytest.fixture(name="runner")
def runner_setup() -> CliRunner:
    """Fixture for invoking command-line interfaces."""
    return CliRunner()


def create_temp_tsv_file(content: str) -> IO[str]:
    """Create a temporary TSV file and populate it with content."""
    with tempfile.NamedTemporaryFile(
        encoding="utf-8", delete=False, mode="w+", suffix=".tsv"
    ) as temp_file:
        temp_file.write(content)
        temp_file.seek(0)
        return cast("IO[str]", temp_file)


def test_default_case(runner: CliRunner) -> None:
    """Test default case for note2csv function."""
    temp_file = create_temp_tsv_file("Well\tpH\tCl\tName\nD01\t9.15\t0\tNTT-G10\n")
    result = runner.invoke(note2csv, [temp_file.name])
    assert result.exit_code == 0
    # Read the output CSV and check its contents
    output_path = Path(temp_file.name).with_suffix(".csv")
    with output_path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        assert header == ["Well", "pH", "Cl", "Name", "Temp", "Labels"]
        row = next(reader)
        assert row == ["D01", "9.15", "0", "NTT-G10", "37.0", "A B"]


def test_custom_output(runner: CliRunner) -> None:
    """Test custom case for note2csv function."""
    temp_file = create_temp_tsv_file("Well\tpH\tCl\tName\nD01\t9.15\t0\tNTT-G10\n")

    with tempfile.NamedTemporaryFile(
        encoding="utf-8", delete=False, mode="w+", suffix=".csv"
    ) as output_file:
        output_file.write("Well,pH,Cl,Name,Temp,Labels" + "\n")
        output_file.seek(0)  # Important to rewind the file to the beginning
        result = runner.invoke(
            note2csv,
            [temp_file.name, "-o", output_file.name, "-t", "22.3", "-l", "A F G"],
        )

    assert result.exit_code == 0

    # Read the output CSV and check its contents
    output_path = Path(output_file.name)
    with output_path.open("r", encoding="utf-8") as f:
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
    # validate output images
    for stem in ["NTT_37C_pKa_A", "NTT_37C_pKa_B"]:
        fp = out / f"{stem}.png"
        assert fp.exists()
        img = Image.open(fp)
        assert img.width > 0
        assert img.height > 0
        assert fp.stat().st_size > 1000
    # validate output file contents
    assert filecmp.cmp(out / "NTT_37C_pKa_A.csv", expected / "NTT_37C_pKa_A.csv")
    assert filecmp.cmp(out / "NTT_37C_pKa_B.csv", expected / "NTT_37C_pKa_B.csv")


@pytest.mark.slow
def test_prtecan(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """Test prtecan command with actual data."""
    list_f = str(tpath / "Tecan" / "140220" / "list.pH.csv")
    scheme_f = str(tpath / "Tecan" / "140220" / "scheme.txt")
    out = str(tmp_path)
    runner = CliRunner()
    # Ensure we're capturing warnings at the appropriate level
    with caplog.at_level(logging.WARNING):
        result = runner.invoke(
            ppr,
            ["--out", out, "tecan", list_f, "--fit", "--sch", scheme_f, "--bg"],
        )
    assert any("OVER" in record.message for record in caplog.records)
    assert result.exit_code == 0


def test_prtecan_raw_dir(tmp_path: Path, runner: CliRunner) -> None:
    """It finds Tecan files via --raw-dir when they are not next to the list file."""
    data_dir = tpath / "Tecan" / "140220"
    list_f = tmp_path / "list.pH.csv"
    list_f.write_text((data_dir / "list.pH.csv").read_text())
    args = ["--out", str(tmp_path / "out"), "tecan", str(list_f), "--dry-run"]
    result = runner.invoke(ppr, [*args, "--raw-dir", str(data_dir)])
    assert result.exit_code == 0
    assert f"✓ Tecan files found: 7 in {data_dir}" in result.output


def test_prtecan_missing_tecan_files(tmp_path: Path, runner: CliRunner) -> None:
    """It blames the missing Tecan files, not the list file, and suggests --raw-dir."""
    list_f = tmp_path / "list.pH.csv"
    list_f.write_text((tpath / "Tecan" / "140220" / "list.pH.csv").read_text())
    args = ["--out", str(tmp_path / "out"), "tecan", str(list_f)]
    for extra in ([], ["--dry-run"]):
        result = runner.invoke(ppr, [*args, *extra])
        assert result.exit_code != 0
        assert "pH9.1_200214.xls" in result.output
        assert "--raw-dir" in result.output
        assert "List file not found" not in result.output


@pytest.mark.slow
def test_prtecan_cl(tmp_path: Path) -> None:
    """Test prtecan command with actual data."""
    list_f = str(tpath / "Tecan" / "140220" / "list.cl.csv")
    scheme_f = str(tpath / "Tecan" / "140220" / "scheme.txt")
    adds_f = str(tpath / "Tecan" / "140220" / "additions.cl")
    out = tmp_path / "out4"
    out.mkdir()
    runner = CliRunner()
    base_args = ["--out", str(out), "tecan", list_f, "--no-fit", "--sch", scheme_f]
    base_args.extend(["--add", adds_f, "--bg", "--cl", "1000.0"])
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
    expected_output = re.sub(r" ", r"\\s+", output.strip())
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
    """Fit K correctly, generate png files as the old ``fit_titration_global.py``."""
    dat_fp = tpath / "glob" / dat_f
    base_args = ["-v", ph_opt, "glob", str(dat_fp)]
    if opts:
        base_args.extend(opts)
    runner = CliRunner()
    result = runner.invoke(fit_titration, base_args)
    assert result.exit_code == 0
    expected_output = re.sub(r" ", r"\\s+", output.strip())
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


def test_prtecan_accepts_multi_mcmc(tmp_path: Path, runner: CliRunner) -> None:
    """--mcmc multi is offered again, now that it is wired to the joint fit.

    It was retired because it had been a no-op since 01735f12: accepted and
    silently doing nothing. It is back because it now routes to
    fit_binding_pymc_multi, which is the model a shared control K requires and
    which the CLI previously had no way to express.
    """
    list_f = str(tpath / "Tecan" / "140220" / "list.pH.csv")
    result = runner.invoke(
        ppr,
        [
            "--out",
            str(tmp_path / "out"),
            "tecan",
            list_f,
            "--mcmc",
            "multi",
            "--dry-run",
        ],
    )
    assert "not a valid choice" not in result.output.lower()


def test_prtecan_offers_odr_and_ye_mag_knobs(runner: CliRunner) -> None:
    """The axes the science actually varies must be reachable from the CLI.

    ODR is the best classical arm on held-out controls, and the per-well
    ye_mag structure is what the noise factorial turned on; neither could be
    expressed here before.
    """
    out = runner.invoke(ppr, ["tecan", "--help"]).output
    assert "[lm|huber|irls|odr]" in out
    assert "--per-well-ye-mags" in out
    assert "[centered|hierarchical|separable]" in out


def test_odr_reaches_the_global_fit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, runner: CliRunner
) -> None:
    """--fit-method odr must select ODR, not fall through to lm.

    The mapping from params.fit_method to the plate fit used to collapse
    anything unrecognised to "lm", so a newly offered method would be accepted
    and silently ignored - the failure that left --mcmc multi a no-op for
    months. This captures the method the plate fit actually receives, rather
    than asserting the flag parses.
    """
    from clophfit.prtecan.titration import Titration  # noqa: PLC0415

    methods: list[str] = []
    real = Titration.fit_plate

    def spy(self: Titration, *args: object, **kwargs: object) -> object:
        methods.append(str(kwargs.get("method", "")))
        return real(self, *args, **kwargs)

    monkeypatch.setattr(Titration, "fit_plate", spy)
    runner.invoke(
        ppr,
        [
            "--out",
            str(tmp_path / "out"),
            "tecan",
            str(tpath / "Tecan" / "140220" / "list.pH.csv"),
            "--fit-method",
            "odr",
            "--no-png",
        ],
    )
    assert methods, "the plate fit was never reached"
    assert "odr" in methods, f"odr never reached the plate fit; saw {set(methods)}"


def test_prtecan_rejects_retired_mcmc_modes(tmp_path: Path, runner: CliRunner) -> None:
    """The multi-noise modes were no-ops after 01735f12; they stay unoffered."""
    list_f = str(tpath / "Tecan" / "140220" / "list.pH.csv")
    for mode in ("multi-noise", "multi-noise-xrw"):
        result = runner.invoke(
            ppr,
            [
                "--out",
                str(tmp_path / "out"),
                "tecan",
                list_f,
                "--mcmc",
                mode,
                "--dry-run",
            ],
        )
        assert result.exit_code != 0, f"{mode} was accepted"
        assert "single-refit" in result.output


def test_prtecan_rejects_retired_ctr_free_k(tmp_path: Path, runner: CliRunner) -> None:
    """--ctr-free-k was written to params and read by nobody; it must not be offered."""
    list_f = str(tpath / "Tecan" / "140220" / "list.pH.csv")
    result = runner.invoke(
        ppr,
        ["--out", str(tmp_path / "out"), "tecan", list_f, "--ctr-free-k", "--dry-run"],
    )
    assert result.exit_code != 0
    assert "no such option" in result.output.lower()


def test_print_spec_signs_the_resolved_analysis(runner: CliRunner) -> None:
    """The signature depends on resolved values, not on how flags were typed.

    The option surface is large enough that two invocations can describe one
    analysis while looking different, and different analyses can look alike.
    Signing resolved values answers "have I run this before" without anyone
    having to remember the flags.
    """
    list_f = str(tpath / "Tecan" / "140220" / "list.pH.csv")

    def sig(*extra: str) -> str:
        out = runner.invoke(ppr, ["tecan", list_f, "--print-spec", *extra]).output
        return next(ln for ln in out.splitlines() if ln.startswith("signature:"))

    reordered = sig("--mcmc", "multi", "--fit-method", "odr")
    assert sig("--fit-method", "odr", "--mcmc", "multi") == reordered
    assert sig("--fit-method", "lm", "--mcmc", "multi") != reordered
    assert (
        sig("--fit-method", "odr", "--mcmc", "multi", "--per-well-ye-mags") != reordered
    )
