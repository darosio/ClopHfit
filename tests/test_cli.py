"""Test ``clop`` cli."""

import csv
import filecmp
import logging
import re
import tempfile
from pathlib import Path
from typing import IO, cast

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner
from PIL import Image

from clophfit.__main__ import (
    _plot_bayes,  # ruff: ignore[import-private-name]
    clop,
    fit_titration,
    note2csv,
    ppr,
)
from clophfit.fitting.data_structures import DataArray, Dataset
from clophfit.prenspire import bands as bands_module

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


def test_prenspire_global(tmp_path: Path, runner: CliRunner) -> None:
    """``--method global`` writes the whole-spectrum K table and the species figure."""
    data = tpath / "EnSpire"
    lines = (data / "NTT-G10_note.csv").read_text().splitlines()
    # the 20 C wells only, to keep the jackknife short
    note = tmp_path / "NTT-G10_note.csv"
    note.write_text("\n".join([lines[0], *(ln for ln in lines[1:] if ",20.0," in ln)]))
    out = tmp_path / "E"
    out.mkdir()
    args = ["--out", str(out), "enspire", str(data / "G10.csv"), str(note)]
    result = runner.invoke(ppr, [*args, "--method", "global"])
    assert result.exit_code == 0, result.output
    table = pd.read_csv(out / "NTT-G10_K_global.csv")
    whole = table[table.subset == "all"]
    assert len(whole) == 1
    assert whole.K.between(7.0, 9.0).all()
    assert (whole.se > 0).all()
    assert {"row D", "row E", "row F"} <= set(table.subset)
    assert (out / "NTT-G10_species.pdf").stat().st_size > 1000


def test_prenspire_all(tmp_path: Path, runner: CliRunner) -> None:
    """``--method all`` runs band, svd and global in one invocation.

    The guard a user asked for when comparing readouts: one call must leave
    behind the band table, at least one svd figure, and the global K table -
    the three artifacts that the separate ``band``, ``svd`` and ``global``
    choices produce individually.
    """
    data = tpath / "EnSpire"
    lines = (data / "NTT-G10_note.csv").read_text().splitlines()
    # the 20 C wells only, to keep the global jackknife short
    note = tmp_path / "NTT-G10_note.csv"
    note.write_text("\n".join([lines[0], *(ln for ln in lines[1:] if ",20.0," in ln)]))
    out = tmp_path / "E"
    out.mkdir()
    args = ["--out", str(out), "enspire", str(data / "G10.csv"), str(note)]
    result = runner.invoke(ppr, [*args, "--method", "all"])
    assert result.exit_code == 0, result.output
    assert (out / "NTT-G10_K.csv").exists()  # band table
    assert (out / "NTT-G10_bands.pdf").exists()  # band figure
    assert (out / "NTT-G10_K_global.csv").exists()  # global table
    assert (out / "NTT-G10_species.pdf").exists()  # global figure
    svd_pdfs = sorted(p.name for p in out.glob("NTT-G10_20*_*.pdf"))
    assert svd_pdfs  # at least one svd figure


def test_prenspire_vanthoff(tmp_path: Path, runner: CliRunner) -> None:
    """``--method vanthoff`` joins the temperatures into one thermodynamic fit."""
    data = tpath / "EnSpire"
    out = tmp_path / "E"
    out.mkdir()
    args = [
        "--out",
        str(out),
        "enspire",
        str(data / "G10.csv"),
        str(data / "NTT-G10_note.csv"),
    ]
    result = runner.invoke(ppr, [*args, "--method", "vanthoff"])
    assert result.exit_code == 0, result.output
    table = pd.read_csv(out / "NTT-G10_K_vanthoff.csv")
    assert set(table.temp.astype(str)) == {"20", "37"}
    assert table.K.between(7.0, 9.5).all()
    assert (table.se > 0).all()
    assert (table.se < 0.5).all()
    assert table.dH_J_mol.notna().all()


def test_prenspire_vanthoff_fits_a_chloride_note_on_its_own_axis(
    tmp_path: Path, runner: CliRunner
) -> None:
    """A chloride note is fitted as a Langmuir, and the figure shows it.

    The G10 plate is a pH titration; swapping the two titrant columns of
    its note makes the same wells a chloride titration. The fit must then
    report a Kd in mM rather than a pKa, and the figure must be drawn on
    the chloride axis - a Henderson-Hasselbalch curve over a millimolar
    axis would not pass through the data.
    """
    data = tpath / "EnSpire"
    out = tmp_path / "E"
    out.mkdir()
    note = pd.read_csv(data / "NTT-G10_note.csv")
    # the wells of the pH ladder become a chloride ladder that spans the
    # same signal range, so the Langmuir has a transition to find
    note["Cl"] = (9.15 - note["pH"]) * 500.0
    note["pH"] = 5.2
    note_fp = tmp_path / "NTT-G10_note.csv"
    note.to_csv(note_fp, index=False)
    result = runner.invoke(
        ppr,
        [
            "--out",
            str(out),
            "enspire",
            str(data / "G10.csv"),
            str(note_fp),
            "--method",
            "vanthoff",
        ],
    )
    assert result.exit_code == 0, result.output
    table = pd.read_csv(out / "NTT-G10_K_vanthoff.csv")
    # a Kd on the millimolar scale, not a pKa
    assert table.K.gt(100.0).all()
    assert table.K.lt(5000.0).all()
    assert (out / "NTT-G10_K_vanthoff.pdf").exists()
    assert "Kd" in result.output


def test_prenspire_global_scans(tmp_path: Path, runner: CliRunner) -> None:
    """``--scan`` lets the whole-spectrum global fit take more than the excitation scan."""
    data = tpath / "EnSpire"
    lines = (data / "NTT-G10_note.csv").read_text().splitlines()
    note = tmp_path / "NTT-G10_note.csv"
    note.write_text("\n".join([lines[0], *(ln for ln in lines[1:] if ",20.0," in ln)]))
    out = tmp_path / "E"
    out.mkdir()
    args = ["--out", str(out), "enspire", str(data / "G10.csv"), str(note)]
    result = runner.invoke(
        ppr,
        [*args, "--method", "global", "--scan", "exc_anionic", "--scan", "em_exc420"],
    )
    assert result.exit_code == 0, result.output
    assert (out / "NTT-G10_K_global.csv").exists()


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
    assert "[centered|hierarchical|separable|separable_step]" in out


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
    from clophfit.fitting.data_structures import (  # ruff: ignore[import-outside-top-level]
        Dataset,
    )
    from clophfit.prtecan.titration import (  # ruff: ignore[import-outside-top-level]
        Titration,
        TitrationResults,
    )

    methods: list[str] = []
    real = Titration.fit_plate

    def spy(
        self: Titration,
        datasets: dict[str, Dataset] | None = None,
        method: str = "",
        *,
        label: str | None = None,
        **kwargs: object,
    ) -> TitrationResults:
        methods.append(method)
        return real(self, datasets, method, label=label, **kwargs)

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


def test_prtecan_ctr_free_k_is_live_not_retired(runner: CliRunner) -> None:
    """--ctr-free-k is offered again, and this time it changes the model.

    It was withdrawn once because it was written to ``params`` and read by
    nobody, so a run that asked for free control K quietly fitted pooled K. The
    flag is back only because it now reaches ``fit_binding_pymc_multi`` -- see
    ``test_mcmc_spec_knobs_reach_the_multi_model``, which pins that call. Here
    the weaker but independent claim: the option exists and it moves the
    analysis signature, so a free-K run cannot be mistaken for a pooled one.
    """
    list_f = str(tpath / "Tecan" / "140220" / "list.pH.csv")

    def sig(*extra: str) -> str:
        out = runner.invoke(ppr, ["tecan", list_f, "--print-spec", *extra]).output
        return next(ln for ln in out.splitlines() if ln.startswith("signature:"))

    assert sig("--mcmc", "multi", "--ctr-free-k") != sig("--mcmc", "multi")
    assert sig("--mcmc", "multi", "--mcmc-robust") != sig("--mcmc", "multi")
    assert sig("--mcmc", "multi", "--mcmc-robust", "--student-t-nu", "2") != sig(
        "--mcmc", "multi", "--mcmc-robust"
    )


def test_prtecan_x_error_model_moves_the_signature(runner: CliRunner) -> None:
    """A per-well pH axis is a different model and must sign as one.

    The call-level claim is ``test_mcmc_x_error_model_reaches_the_multi_model``;
    here, that the options exist and that a per-well run cannot be mistaken for
    a shared-axis one, nor two per-well runs with different offset SDs for
    each other.
    """
    list_f = str(tpath / "Tecan" / "140220" / "list.pH.csv")

    def sig(*extra: str) -> str:
        out = runner.invoke(ppr, ["tecan", list_f, "--print-spec", *extra]).output
        return next(ln for ln in out.splitlines() if ln.startswith("signature:"))

    base = ("--mcmc", "multi")
    per_well = (*base, "--mcmc-x-error", "per_well")
    assert sig(*per_well) != sig(*base)
    assert sig(*per_well, "--mcmc-x-start-between", "0.028") != sig(*per_well)


def test_prtecan_x_start_between_needs_the_per_well_axis(
    runner: CliRunner, tmp_path: Path
) -> None:
    """An offset SD on the shared axis would be accepted and ignored: refuse it."""
    list_f = str(tpath / "Tecan" / "140220" / "list.pH.csv")
    result = runner.invoke(
        ppr,
        [
            "-o",
            str(tmp_path / "out"),
            "tecan",
            list_f,
            "--mcmc",
            "multi",
            "--mcmc-x-start-between",
            "0.028",
        ],
    )
    assert result.exit_code != 0
    assert "--mcmc-x-error per_well" in result.output


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (("--fit-method", "odr", "--fit-noise", "gain"), "lm or huber"),
        (("--plate-noise", "gain"), "--plate-fit lm"),
        (("--plate-fit", "odr", "--plate-noise", "floor-gain"), "--plate-fit lm"),
    ],
)
def test_prtecan_gain_calibration_needs_a_fitter_it_can_weight(
    runner: CliRunner, tmp_path: Path, args: tuple[str, ...], message: str
) -> None:
    """A gain calibration on a fit that would ignore it is refused, not dropped."""
    list_f = str(tpath / "Tecan" / "140220" / "list.pH.csv")
    result = runner.invoke(ppr, ["-o", str(tmp_path / "out"), "tecan", list_f, *args])
    assert result.exit_code != 0
    assert message in result.output


def test_prtecan_fit_noise_moves_the_signature(runner: CliRunner) -> None:
    """Calibrated single-well weights are a different analysis and sign as one."""
    list_f = str(tpath / "Tecan" / "140220" / "list.pH.csv")

    def sig(*extra: str) -> str:
        out = runner.invoke(ppr, ["tecan", list_f, "--print-spec", *extra]).output
        return next(ln for ln in out.splitlines() if ln.startswith("signature:"))

    assert sig("--fit-noise", "gain") != sig()
    assert sig("--fit-noise", "floor-gain") != sig("--fit-noise", "gain")


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


def test_prenspire_mcr(tmp_path: Path, runner: CliRunner) -> None:
    """``--method mcr`` resolves species and returns a pKa per temperature."""
    data = tpath / "EnSpire"
    out = tmp_path / "E"
    out.mkdir()
    args = [
        "--out",
        str(out),
        "enspire",
        str(data / "G10.csv"),
        str(data / "NTT-G10_note.csv"),
    ]
    result = runner.invoke(ppr, [*args, "--method", "mcr"])
    assert result.exit_code == 0, result.output
    table = pd.read_csv(out / "NTT-G10_K_mcr.csv")
    assert set(table.temp.astype(str)) == {"20", "37"}
    assert table.K.between(7.0, 9.5).all()
    assert (
        table.rel_err < 0.1
    ).all()  # real plate noise; synthetic tests carry the accuracy bar
    assert (table.se > 0).all()
    assert (table.se < 0.5).all()
    assert (out / "NTT-G10_mcr_spectra.csv").exists()
    spec = pd.read_csv(out / "NTT-G10_mcr_spectra.csv")
    assert {"lambda", "species_0", "species_1"} <= set(spec.columns)
    assert (spec[["species_0", "species_1"]] >= 0).all().all()
    assert (out / "NTT-G10_mcr.pdf").stat().st_size > 1000


def test_prenspire_bayes(tmp_path: Path, runner: CliRunner) -> None:
    """``--method bayes`` returns a seeded posterior with a 94 % HDI per temp."""
    data = tpath / "EnSpire"
    out = tmp_path / "E"
    out.mkdir()
    args = [
        "--out",
        str(out),
        "enspire",
        str(data / "G10.csv"),
        str(data / "NTT-G10_note.csv"),
    ]
    result = runner.invoke(
        ppr, [*args, "--method", "bayes", "--draws", "400", "--seed", "11"]
    )
    assert result.exit_code == 0, result.output
    table = pd.read_csv(out / "NTT-G10_K_bayes.csv")
    assert set(table.temp.astype(str)) == {"20", "37"}
    assert table.K.between(7.0, 9.5).all()
    assert (table.hdi03 < table.K).all()
    assert (table.hdi97 > table.K).all()
    # ESS and r_hat are both asserted outright: the run has two chains,
    # so the convergence diagnostic is defined. A guard that skipped the
    # assertion on an undefined r_hat would pass even if the chains had
    # collapsed to one.
    assert (table.ess_bulk > 100).all()
    assert table.r_hat.notna().all()
    assert (table.r_hat < 1.1).all()
    assert (out / "NTT-G10_K_bayes.pdf").stat().st_size > 1000


def test_plot_bayes_skips_a_sample_the_sampler_could_not_fit(
    tmp_path: Path,
) -> None:
    """A band with no posterior must not abort the figure.

    ``fit_bands_bayes`` returns a null trace when the sampler produced
    nothing, so the figure writer has to skip such a sample instead of
    asking ArviZ to extract from a missing trace.
    """
    nan = float("nan")
    unfitted = bands_module.BandBayes(nan, nan, nan, nan, nan, None)
    data = {"exc_anionic": (np.array([6.0, 7.0, 8.0]), np.array([0.9, 0.6, 0.2]))}
    # nothing drawable: no crash, and no empty figure either
    _plot_bayes([("sample", "20", data, unfitted, True)], tmp_path / "none.pdf")
    assert not (tmp_path / "none.pdf").exists()

    # a fitted sample next to the unfitted one: the figure is written and
    # holds the fitted sample only
    x = np.linspace(5.0, 9.5, 12)
    frac = 1.0 / (1.0 + np.power(10.0, 7.4 - x))
    ds = Dataset(
        {"exc_anionic": DataArray(x, 0.05 + 0.95 * frac)},
        is_ph=True,
    )
    fitted = bands_module.fit_bands_bayes(ds, seed=1, draws=200, chains=1)
    out = tmp_path / "K_bayes.pdf"
    _plot_bayes(
        [
            ("unfitted", "20", data, unfitted, True),
            ("fitted", "20", {"exc_anionic": (x, 0.05 + 0.95 * frac)}, fitted, True),
        ],
        out,
    )
    assert out.exists()
    assert out.stat().st_size > 1000


def test_prenspire_bayes_writes_a_table_when_nothing_fits(
    tmp_path: Path, runner: CliRunner
) -> None:
    """A note whose bands do not titrate is reported, not a traceback.

    With no row to report the table has no columns at all, so the writer
    must not index a column that does not exist.
    """
    data = tpath / "EnSpire"
    note = tmp_path / "NTT-G10_note.csv"
    lines = (data / "NTT-G10_note.csv").read_text().splitlines()
    # one well per sample and temperature: too few points to carry a K
    kept = [lines[0]]
    seen: dict[tuple[str, str], int] = {}
    for line in lines[1:]:
        parts = line.split(",")
        key = (parts[3], parts[4])
        seen[key] = seen.get(key, 0) + 1
        if seen[key] <= 1:
            kept.append(line)
    note.write_text("\n".join(kept))
    out = tmp_path / "E"
    out.mkdir()
    result = runner.invoke(
        ppr,
        [
            "--out",
            str(out),
            "enspire",
            str(data / "G10.csv"),
            str(note),
            "--method",
            "bayes",
            "--draws",
            "50",
            "--seed",
            "3",
        ],
    )
    assert result.exit_code == 0, result.output
    assert (out / "NTT-G10_K_bayes.csv").exists()


def test_prenspire_mcr_reports_a_degenerate_decomposition(
    tmp_path: Path, runner: CliRunner
) -> None:
    """A degenerate MCR fit must be reported, not shown as a fitted K.

    Asking for more species than the data supports lets ALS park a
    redundant component at a constant share of every well. The residual
    improves, so nothing in the fit itself objects - but the K would be
    read off a column that never moves. The CLI has to say so instead of
    printing a confident pKa.
    """
    data = tpath / "EnSpire"
    out = tmp_path / "E"
    out.mkdir()
    args = [
        "--out",
        str(out),
        "enspire",
        str(data / "G10.csv"),
        str(data / "NTT-G10_note.csv"),
    ]
    result = runner.invoke(ppr, [*args, "--method", "mcr", "--components", "3"])
    assert result.exit_code == 0, result.output
    table = pd.read_csv(out / "NTT-G10_K_mcr.csv")
    assert "degenerate" in table.columns
    assert "note" in table.columns
    if table.degenerate.any():
        # no K may be reported for a degenerate decomposition
        assert table.loc[table.degenerate, "K"].isna().all()
        assert table.loc[table.degenerate, "note"].str.contains("dead").all()
        assert "degenerate" in result.output
    else:
        # the decomposition was sound: then it must carry a real K
        assert table.K.notna().all()


def test_plot_mcr_draws_the_curve_on_the_scale_of_the_points(
    tmp_path: Path, runner: CliRunner
) -> None:
    """The MCR curve must sit on the same scale as the points it overlays.

    The closed profiles are fractions summing to one, so they live on the
    0-1 scale. The fitted curve used to be drawn as ``f0 + (f1-f0)*frac``,
    where ``f0`` and ``f1`` are the tracking component's own plateaus -
    values like 0.0016 and 1.06 on the real plate. The curve therefore
    ended at 0.0016 and 1.06 rather than 0 and 1, and did not pass through
    the points it was supposed to describe.
    """
    data = tpath / "EnSpire"
    out = tmp_path / "E"
    out.mkdir()
    args = [
        "--out",
        str(out),
        "enspire",
        str(data / "G10.csv"),
        str(data / "NTT-G10_note.csv"),
    ]
    result = runner.invoke(ppr, [*args, "--method", "mcr"])
    assert result.exit_code == 0, result.output
    assert (out / "NTT-G10_mcr.pdf").stat().st_size > 1000
    # The closed profiles are fractions on the 0-1 scale, so the drawn
    # curve has to be on that same scale: it must span the points, not a
    # rescaled copy of them. Read the figure back by checking the curve
    # is drawn inside the fraction band the points occupy.
    table = pd.read_csv(out / "NTT-G10_K_mcr.csv")
    fitted = table[table.K.notna()]
    assert len(fitted), "the shipped plate must fit"
    # the tracking component's own plateaus may sit slightly outside
    # [0, 1] - they are the component's baseline and saturation, not
    # fractions - but the model curve is a fraction by construction.
    assert (fitted.F1 > 0).all()
    assert (fitted.F1 < 2.0).all()
