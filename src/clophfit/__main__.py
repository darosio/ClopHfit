"""Command-line interface."""

from __future__ import annotations

import csv
import logging
import pprint
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

import click
import lmfit  # type: ignore[import-untyped]
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from click import Context, Path as cPath

from clophfit import __enspire_out_dir__, __tecan_out_dir__, configure_logging, fitting
from clophfit.fitting.data_structures import DataArray, Dataset
from clophfit.fitting.errors import (
    DataValidationError,
    MissingDependencyError,
)
from clophfit.prenspire import EnspireFile, Note
from clophfit.prtecan import TecanConfig, Titration, calculate_conc

if TYPE_CHECKING:
    from clophfit.fitting.data_structures import FitResult, MiniT


@click.group()
@click.version_option(message="%(version)s")
def clop() -> None:  # pragma: no cover
    """Group command."""


@clop.command()
@click.argument("kd1", type=float)
@click.argument("pka", type=float)
@click.argument("ph", type=float)
def eq1(kd1: float, pka: float, ph: float) -> None:
    """Model Kd dependence on pH."""
    click.echo(fitting.models.kd(kd1=kd1, pka=pka, ph=ph))


@click.group()
@click.pass_context
@click.version_option(message="%(version)s")
@click.option("--verbose", "-v", count=True, help="Increase verbosity: -v for INFO, -vv for DEBUG. Default is WARNING.")  # fmt: skip
@click.option("--quiet", "-q", is_flag=True, help="Silence terminal output; show only ERROR messages.")  # fmt: skip
@click.option("--out", "-o", type=cPath(), help="Output folder.")
def ppr(ctx: Context, verbose: int, quiet: bool, out: str) -> None:  # pragma: no cover
    """Parse Plate Reader `ppr` group command."""
    ctx.ensure_object(dict)
    ctx.obj["VERBOSE"] = verbose
    ctx.obj["QUIET"] = quiet
    if out:
        ctx.obj["OUT"] = out


######################################
# pr.tecan                           #
######################################
@ppr.command()
@click.pass_context
@click.argument("list_file", type=cPath(exists=True))
@click.option("--cl", type=float, help="Cl stock concentration (mM) of added aliquots.")
@click.option("--bg", is_flag=True, help="Whether to subtract buffer (from scheme.txt).")  # fmt: skip
@click.option("--bg-adj", is_flag=True, help="Whether to heuristically adjust negative background values.")  # fmt: skip
@click.option("--dil", is_flag=True, help="Whether to apply dilution correction.")
@click.option("--nrm", is_flag=True, help="Whether to normalize using metadata.")
@click.option("--bg-mth", default="mean", show_default=True, help="Method for background calculation.")  # fmt: skip
@click.option("--sch", type=cPath(exists=True), help="Path to plate scheme file (buffers and controls).")  # fmt: skip
@click.option("--add", type=cPath(exists=True), help="Path to additions file (initial volume + additions).")  # fmt: skip
@click.option("--all", "comb", is_flag=True, help="Whether to export all data combinations.")  # fmt: skip
@click.option("--lim", type=(float, float), help="Range MIN, MAX of plot_K.")
@click.option("--title", "-t", type=str, default="", help="Title for plots.")
@click.option("--fit/--no-fit", default=True, show_default=True, help="Whether to perform fitting.")  # fmt: skip
@click.option("--png/--no-png", default=True, show_default=True, help="Whether to export PNG files.")  # fmt: skip
@click.option("--mcmc", type=click.Choice(["None", "multi", "single"], case_sensitive=False), default="None", show_default=True, help="Run MCMC sampling: None, multi, or single.")  # fmt: skip
@click.option("--dry-run", is_flag=True, help="Validate inputs without processing data.")  # fmt: skip
def tecan(  # noqa: C901,PLR0912,PLR0913,PLR0915
    ctx: Context,  # Click context object.
    list_file: str,
    cl: float,
    bg: bool,
    bg_adj: bool,
    dil: bool,
    nrm: bool,
    bg_mth: str,
    sch: str | None,
    add: str | None,
    comb: bool,
    lim: tuple[float, float] | None,
    title: str,
    fit: bool,
    png: bool,
    mcmc: str,
    dry_run: bool,
) -> None:
    """Convert a list of Tecan-exported excel files into titrations.

    LIST_FILE : Path to file containing Tecan files and concentration values.

    Saves titrations as .dat files and fits all wells using 2 labels. The
    function produces:

    - K plot

    - csv tables for all labelblocks and global fittings.

    Buffer is always subtracted if scheme indicates buffer well positions.
    """
    out = ctx.obj.get("OUT", __tecan_out_dir__)
    verbose = ctx.obj.get("VERBOSE", 0)
    quiet = ctx.obj.get("QUIET", 0)
    configure_logging(verbose=verbose, quiet=quiet, log_file="ppr_tecan_cli.log")
    logger = logging.getLogger("clophfit.cli.ppr_tecan")
    logger.debug("CLI started")
    out_fp = Path(out) / "Cl" if cl else Path(out) / "pH"
    out_fp.mkdir(parents=True, exist_ok=True)
    # Options validation with clear error messages
    try:
        _validate_tecan_options(cl, bg, dil, add, sch, comb)
    except (DataValidationError, MissingDependencyError) as e:
        raise click.ClickException(str(e)) from e

    # Dry run mode: validate inputs and exit
    if dry_run:
        click.echo("🔍 Dry run mode: Validating inputs...\n")
        _dry_run_validation(list_file, sch, add, cl, out_fp)
        click.echo("\n✅ Validation successful! All inputs are valid.")
        click.echo("   Remove --dry-run flag to process data.")
        return

    # Config
    tecan_config = TecanConfig(out_fp, comb, lim, title, fit, png)

    # Load titration with error handling
    list_fp = Path(list_file)
    logger.info("Titration list: %s", list_fp.resolve())
    logger.info("%s", tecan_config)

    try:
        tit = Titration.fromlistfile(list_fp, is_ph=not cl)
    except FileNotFoundError as e:
        msg = (
            f"List file not found: {list_fp}\n"
            f"Please check that the file exists and the path is correct."
        )
        raise click.ClickException(msg) from e
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as e:
        msg = (
            f"Error parsing list file: {list_fp}\n"
            f"Expected format: CSV with columns for file paths and concentrations.\n"
            f"Details: {e}"
        )
        raise click.ClickException(msg) from e
    except Exception as e:
        msg = (
            f"Error loading titration data from {list_fp}: {e}\n"
            f"Please check the file format and contents."
        )
        raise click.ClickException(msg) from e

    tit.params.bg = bg
    tit.params.bg_adj = bg_adj
    tit.params.dil = dil
    tit.params.nrm = nrm
    tit.params.bg_mth = bg_mth
    tit.params.mcmc = mcmc
    logger.info("%s", tit.params)

    # Load additions file with error handling
    if add:
        try:
            tit.load_additions(Path(add))
            logger.info("Additions: %s", tit.additions)
        except FileNotFoundError:
            msg = (
                f"Additions file not found: {add}\n"
                f"This file is required when using --cl or --dil options."
            )
            raise click.ClickException(msg) from None
        except Exception as e:
            msg = (
                f"Error loading additions file {add}: {e}\n"
                f"Expected format: Initial volume and addition volumes."
            )
            raise click.ClickException(msg) from e

    if cl and tit.additions:
        try:
            tit.x = calculate_conc(tit.additions, cl)
            logger.info("%s", tit.x)
        except Exception as e:
            msg = (
                f"Error calculating chloride concentrations: {e}\n"
                f"Please check additions file format and --cl value."
            )
            raise click.ClickException(msg) from e

    # Load scheme file with error handling
    if sch:
        try:
            tit.load_scheme(Path(sch))
            f = tit.buffer.plot(title=title)
            f.savefig(out_fp / "buffer.png")
            f = tit.buffer.plot(nrm=True, title=title)
            f.savefig(out_fp / "buffer_norm.png")
            logger.info("%s", tit.scheme)
        except FileNotFoundError:
            msg = (
                f"Scheme file not found: {sch}\n"
                f"This file is required when using --bg option."
            )
            raise click.ClickException(msg) from None
        except Exception as e:
            msg = (
                f"Error loading scheme file {sch}: {e}\n"
                f"Expected format: Tab-separated file with well positions and sample names."
            )
            raise click.ClickException(msg) from e

    # Export metadata and plots with error handling
    try:
        with (out_fp / "metadata-labels.txt").open("w", encoding="utf-8") as fp:
            for lbg in tit.labelblocksgroups.values():
                pprint.pprint(lbg.metadata, stream=fp)
        f = tit.plot_temperature(title=title)
        f.savefig(out_fp / "temperatures.png")
    except PermissionError as e:
        msg = (
            f"Permission denied writing to output directory: {out_fp}\n"
            f"Please check directory permissions."
        )
        raise click.ClickException(msg) from e
    except OSError as e:
        msg = f"Error writing output files to {out_fp}: {e}"
        raise click.ClickException(msg) from e

    # Output and export with error handling
    try:
        tit.export_data_fit(tecan_config)
    except Exception as e:
        msg = (
            f"Error during data export and fitting: {e}\n"
            f"Check the log file for more details: ppr_tecan_cli.log"
        )
        raise click.ClickException(msg) from e


def _validate_tecan_options(  # noqa: PLR0913
    cl: float | None,
    bg: bool,
    dil: bool,
    add: str | None,
    sch: str | None,
    comb: bool,
) -> None:
    """Validate tecan command options.

    Parameters
    ----------
    cl : float | None
        Chloride concentration option.
    bg : bool
        Background subtraction flag.
    dil : bool
        Dilution correction flag.
    add : str | None
        Additions file path.
    sch : str | None
        Scheme file path.
    comb : bool
        All combinations flag.

    Raises
    ------
    MissingDependencyError
        If required files are not specified.
    DataValidationError
        If option combinations are invalid.
    """
    if cl and not add:
        raise MissingDependencyError(
            missing_file="additions file (--add)",
            required_by="--cl option",
            reason="Chloride titrations require addition volumes to calculate concentrations.",
        )
    if dil and not add:
        raise MissingDependencyError(
            missing_file="additions file (--add)",
            required_by="--dil option",
            reason="Dilution correction requires addition volumes.",
        )
    if bg and not sch:
        raise MissingDependencyError(
            missing_file="scheme file (--sch)",
            required_by="--bg option",
            reason="Buffer subtraction requires a plate scheme to identify buffer wells.",
        )
    if comb and not (bg and sch and dil):
        msg = "All combinations mode requires --bg, --sch, and --dil to be specified."
        raise DataValidationError(
            msg,
            suggestions=[
                "Add --bg --sch scheme.txt --dil flags",
                "Or remove --all flag if you don't need all combinations",
            ],
        )


def _dry_run_validation(
    list_file: str,
    sch: str | None,
    add: str | None,
    cl: float | None,
    out_fp: Path,
) -> None:
    """Perform dry-run validation of input files.

    Parameters
    ----------
    list_file : str
        Path to list file.
    sch : str | None
        Path to scheme file.
    add : str | None
        Path to additions file.
    cl : float | None
        Chloride concentration.
    out_fp : Path
        Output directory.

    Raises
    ------
    click.ClickException
        If validation fails.
    """
    list_fp = Path(list_file)

    # Validate list file
    click.echo(f"✓ List file exists: {list_fp}")
    try:
        df = pd.read_csv(list_fp)
        click.echo(f"  - Contains {len(df)} entries")
        if len(df) == 0:
            msg = "List file is empty"
            raise click.ClickException(msg)  # noqa: TRY301 #FIXME:
    except click.ClickException:
        raise
    except Exception as e:
        msg = f"Error reading list file: {e}"
        raise click.ClickException(msg) from e

    # Validate scheme file
    if sch:
        sch_fp = Path(sch)
        click.echo(f"✓ Scheme file exists: {sch_fp}")
        try:
            sch_df = pd.read_csv(sch_fp, sep="\t")
            click.echo(f"  - Contains {len(sch_df)} well definitions")
        except Exception as e:
            msg = f"Error reading scheme file: {e}"
            raise click.ClickException(msg) from e

    # Validate additions file
    if add:
        add_fp = Path(add)
        click.echo(f"✓ Additions file exists: {add_fp}")
        try:
            with add_fp.open(encoding="utf-8") as f:
                lines = f.readlines()
            click.echo(f"  - Contains {len(lines)} lines")
            if cl:
                click.echo(f"  - Will calculate [Cl] with {cl} mM stock")
        except Exception as e:
            msg = f"Error reading additions file: {e}"
            raise click.ClickException(msg) from e

    # Validate output directory
    click.echo(f"✓ Output directory: {out_fp}")
    if not out_fp.exists():
        click.echo(f"  - Will create: {out_fp}")
    else:
        click.echo("  - Already exists (files may be overwritten)")


########################################
# pr.enspire                           #
########################################
@ppr.command()
@click.pass_context
@click.argument("csv_f", type=cPath(exists=True, path_type=str))
@click.argument("note_f", type=cPath(exists=True), required=False)
@click.option(
    "-b",
    "bands",
    multiple=True,
    default=None,
    nargs=3,
    type=(str, int, int),
    help="Label and band interval (format: LABEL LOWER UPPER)",
)
def enspire(ctx: Context, csv_f: str, note_f: str | None, bands: tuple[Any]) -> None:
    """Save spectra as csv tables from EnSpire xls file."""
    verbose = ctx.obj.get("VERBOSE", 0)
    out = ctx.obj.get("OUT", __enspire_out_dir__)
    ef = EnspireFile(Path(csv_f), verbose=verbose)
    ef.export_measurements(Path(out))
    if note_f is not None:
        fit_enspire(ef, Path(note_f), Path(out), list(bands), verbose)


# TODO: Simplify this function
def fit_enspire(  # noqa: C901,PLR0912
    ef: EnspireFile,
    note_fp: Path,
    out_dir: Path,
    bands: list[tuple[str, int, int]],
    verbose: int,
) -> None:
    """Fit prenspire titration (all labels, temp, mutant, titrations)."""
    note = Note(note_fp, verbose=verbose)
    note.build_titrations(ef)
    dbands = {label: (ini, fin) for label, ini, fin in bands} if bands else {}
    ds_data: dict[str, DataArray] = {}
    for name, d_name in note.titrations.items():
        for temp, d_temp in d_name.items():
            for tit, d_tit in d_temp.items():
                if tit.split("_")[0] == "pH":
                    is_ph = False
                elif tit.split("_")[0] == "Cl":
                    is_ph = True
                else:
                    msg = "Unknown titration type."
                    raise ValueError(msg)
                for label, data in d_tit.items():
                    band = dbands.get(label)
                    fit_result = fitting.core.analyze_spectra(
                        data, is_ph=is_ph, band=band
                    )
                    if fit_result.is_valid() and fit_result.mini:
                        userargs = fit_result.mini.userargs[0]["default"]
                        ds_data[label] = DataArray(userargs.x, userargs.y)
                        pdf_file = out_dir / f"{name}_{temp}_{label}_{tit}_{band}.pdf"
                        if fit_result.figure:
                            fit_result.figure.savefig(pdf_file)
                    _print_result(fit_result, pdf_file, str(band))
                # Global spectra analysis with more than 1 label.
                if (
                    len(d_tit.keys() - dbands.keys()) > 1  # svd > 1
                    or len(dbands.keys() & d_tit.keys()) > 1  # bands > 1
                ):
                    ds = Dataset(ds_data, is_ph=is_ph)
                    spectra_gres = fitting.core.analyze_spectra_glob(d_tit, ds, dbands)
                    if spectra_gres.svd and spectra_gres.svd.is_valid():
                        pdf_file = out_dir / f"{name}_{temp}_all_{tit}_SVD.pdf"
                        if spectra_gres.svd.figure:
                            spectra_gres.svd.figure.savefig(pdf_file)
                        _print_result(spectra_gres.svd, pdf_file, "")
                    if spectra_gres.gsvd and spectra_gres.gsvd.is_valid():
                        pdf_file = out_dir / f"{name}_{temp}_g_{tit}_SVD.pdf"
                        if spectra_gres.gsvd.figure:
                            spectra_gres.gsvd.figure.savefig(pdf_file)
                        _print_result(spectra_gres.gsvd, pdf_file, "")
                    if spectra_gres.bands and spectra_gres.bands.is_valid():
                        keys = dbands.keys() & d_tit.keys()
                        lname = [f"{k}({dbands[k][0]},{dbands[k][1]})" for k in keys]
                        bands_str = "".join(lname)
                        pdf_file = out_dir / f"{name}_{temp}_all_{tit}_{bands_str}.pdf"
                        if spectra_gres.bands.figure:
                            spectra_gres.bands.figure.savefig(pdf_file)
                        _print_result(spectra_gres.bands, pdf_file, bands_str)


def _print_result(fit_result: FitResult[MiniT], pdf_file: Path, band_str: str) -> None:
    print(str(pdf_file))
    print(f"Best fit using '{band_str}' band:\n")
    ci = lmfit.conf_interval(fit_result.mini, fit_result.result)
    print(lmfit.ci_report(ci, ndigits=2, with_offset=False))
    print(f"\n Plot saved in '{pdf_file}'.\n")


#############################
#  fit_titration_global     #
#############################
@click.group()
@click.pass_context
@click.version_option(message="%(version)s")
@click.option("--verbose", "-v", count=True, help="Verbosity of messages.")
@click.option("--out", "-o", type=cPath(), help="Output folder.")
@click.option(
    "--is-ph/--no-is-ph", default=True, show_default=True, help="Concentrations are pH."
)
def fit_titration(
    ctx: Context, verbose: int, out: str, is_ph: bool
) -> None:  # pragma: no cover
    """Fit Titration group command."""
    ctx.ensure_object(dict)
    ctx.obj["VERBOSE"] = verbose
    ctx.obj["OUT"] = out
    ctx.obj["IS_PH"] = is_ph


@fit_titration.command()
@click.pass_context
@click.argument("csv_f", type=click.Path(exists=True))
@click.argument("note_f", type=click.Path(exists=True))
@click.option(
    "-b", "--band", nargs=2, type=int, help="Integration interval from <1> to <2>"
)
def spec(ctx: Context, csv_f: str, note_f: str, band: tuple[int, int] | None) -> None:
    """Update old svd or band fit of titration spectra."""
    verbose = ctx.obj.get("VERBOSE", 0)
    is_ph = ctx.obj.get("IS_PH", True)
    out = Path(ctx.obj.get("OUT", "."))

    note_df = pd.read_csv(note_f, sep="\t")
    csv_df = pd.read_csv(csv_f)
    # Ignore buffer wells! SVD will use differences between spectra.
    note_df = note_df[note_df["mutant"] != "buffer"]

    class Notes(NamedTuple):
        wells: list[str]
        conc: list[float]

    titration_type = "pH" if is_ph else "Cl"
    note = Notes(list(note_df["well"]), list(note_df[titration_type]))
    spectra = csv_df[note.wells]
    spectra.index = csv_df["lambda"]
    spectra.columns = np.array(note.conc)
    if verbose:
        print(csv_df)
        click.echo(note_f)
        print(note)
        print("DataFrame\n", spectra)
    is_ph = titration_type == "pH"
    fit_result = fitting.core.analyze_spectra(spectra, is_ph=is_ph, band=band)
    # output
    out.mkdir(parents=True, exist_ok=True)
    pdf_file = out / f"{Path(csv_f).stem}_{band}_{Path(note_f).stem}.pdf"
    if fit_result.figure is not None:
        fit_result.figure.savefig(pdf_file)
    _print_result(fit_result, pdf_file, str(band))


@fit_titration.command()
@click.pass_context
@click.argument("file", type=click.Path(exists=True))
@click.option("-b", "--boot", type=int, help="Number of booting iterations.")
@click.option(
    "--weight/--no-weight", default=True, show_default=True, help="Use residue weights."
)
def glob(ctx: Context, file: str, boot: int, weight: bool) -> None:
    """Update old glob fit of multiple datasets."""
    verbose = ctx.obj.get("VERBOSE", 0)
    is_ph = ctx.obj.get("IS_PH", True)
    file_df = pd.read_csv(file)
    fp = Path(file)
    min_correl_to_print = 0.65
    burn = 75
    if verbose:
        click.echo(file_df)
    x = file_df["x"].to_numpy().astype(float)
    ds_data = {
        lbl: DataArray(x, file_df[lbl].to_numpy().astype(float))
        for lbl in file_df.columns[1:]
    }
    ds = Dataset(ds_data, is_ph=is_ph)
    if weight:
        fitting.core.weight_multi_ds_titration(ds)
    f_res = fitting.core.fit_binding_glob(ds)
    params = f_res.result.params if f_res.result else lmfit.Parameters()
    # Figure
    figure, ax = plt.subplots()
    fitting.plotting.plot_fit(
        ax, ds, params, nboot=30, pp=fitting.plotting.PlotParameters(is_ph)
    )
    lmfit.printfuncs.report_fit(f_res.result, min_correl=min_correl_to_print)
    figure.savefig(Path(file).with_suffix(".png"))
    if boot and f_res.mini:
        # Emcee
        samples = f_res.mini.emcee(burn=burn, steps=boot).flatchain
        fig = fitting.plotting.plot_emcee(samples)
        fig.savefig(fp.with_suffix(".png").with_stem(fp.stem + "-emcee"))
        hdi = samples.quantile([0.025, 0.975])["K"].to_list()
        print(f"Quantiles for K: {[f'{q:.3g}' for q in hdi]}")
        hdi = samples.quantile([0.03, 0.97])["K"].to_list()
        print(f"HDI (94%): {[f'{q:.3g}' for q in hdi]}")
        # R := S0 / S1
        # function := S1 * (R + (1 - R) * 10 ** (K - x) / (1 + 10 ** (K - x)))
        if is_ph:  # ratio between protonated un-protonated states
            ratios = {lbl: samples[f"S0_{lbl}"] / samples[f"S1_{lbl}"] for lbl in ds}
            # Combine ratio and K samples into a DataFrame for corner plot
            samples_ratios = pd.DataFrame({**ratios, "K": samples["K"]})
            fig_ratio = fitting.plotting.plot_emcee(samples_ratios)
            fig_ratio.savefig(fp.with_suffix(".png").with_stem(fp.stem + "-emc-ratios"))
            for lbl in ds:
                hdi = samples_ratios.quantile([0.025, 0.5, 0.975])[lbl].to_list()
                formatted_hdi = [f"{q:.3g}" for q in hdi]
                print(f"HDI (94%) for plateau ratio in dataset {lbl}: {formatted_hdi}")


########################################
# note_to_csv                          #
########################################
@click.command()
@click.argument("note", type=click.Path(exists=True))
@click.option("-o", "--output", default=None, help="Output CSV file.")
@click.option("-l", "--labels", default="A B", help="Labels to be appended.")
@click.option("-t", "--temp", default="37.0", help="Temperature to be appended.")
def note2csv(note: str, output: str, labels: str, temp: str) -> None:
    """Convert a tab-separated data file into a CSV file."""
    headers = ["Well", "pH", "Cl", "Name", "Temp", "Labels"]

    input_path = Path(note)
    output_path = Path(output) if output else input_path.with_suffix(".csv")

    if not output_path.exists():
        output_path.write_text(",".join(headers) + "\n", encoding="utf-8")

    # read data from note file, append to output file
    with input_path.open("r", encoding="utf-8") as datafile:
        reader = csv.reader(datafile, delimiter="\t")  # assuming tab-separated values
        next(reader)  # skip the header row

        with output_path.open("a", encoding="utf-8") as f:
            writer = csv.writer(f, lineterminator="\n")
            for row in reader:
                new_row = [*row[:4], temp, labels]
                writer.writerow(new_row)
