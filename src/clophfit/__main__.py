"""Command-line interface."""

from __future__ import annotations

import csv
import pprint
from collections import namedtuple
from pathlib import Path
from typing import Any

import click
import lmfit  # type: ignore[import-untyped]
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from click import Context, Path as cPath

from clophfit import __enspire_out_dir__, __tecan_out_dir__, binding, prenspire, prtecan
from clophfit.prenspire import EnspireFile
from clophfit.prtecan import TecanConfig, Titration


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
    click.echo(binding.fitting.kd(kd1=kd1, pka=pka, ph=ph))


@click.group()
@click.pass_context
@click.version_option(message="%(version)s")
@click.option("--verbose", "-v", count=True, help="Verbosity of messages.")
@click.option("--out", "-o", type=cPath(), help="Output folder.")
def ppr(ctx: Context, verbose: int, out: str) -> None:  # pragma: no cover
    """Parse Plate Reader `ppr` group command."""
    ctx.ensure_object(dict)
    ctx.obj["VERBOSE"] = verbose
    if out:
        ctx.obj["OUT"] = out


######################################
# pr.tecan                           #
######################################
@ppr.command()
@click.pass_context
@click.argument("list_file", type=cPath(exists=True))
@click.option("--cl", type=float, help="Cl titration: [Cl] (mM) of added aliquots.")
@click.option("--bg", is_flag=True, help="Subtract buffer (from scheme.txt).")
@click.option("--bg-adj", is_flag=True, help="Adjust bg to avoid negative value.")
@click.option("--dil", is_flag=True, help="Apply dilution correction.")
@click.option("--nrm", is_flag=True, help="Normalize using metadata.")
@click.option("--bg-mth", default="mean", show_default=True, help="Method for bg.")
@click.option("--sch", type=cPath(exists=True), help="Plate scheme (buffers CTRs).")
@click.option("--add", type=cPath(exists=True), help="Initial volume and additions.")
@click.option("--all", "comb", is_flag=True, help="Export (fit) all data combinations.")
@click.option("--lim", type=(float, float), help="Range MIN, MAX of plot_K.")
@click.option("--title", "-t", type=str, default="", help="Title for plots.")
@click.option("--fit/--no-fit", default=True, show_default=True, help="Perform also fit.")  # fmt: skip
@click.option("--png/--no-png", default=True, show_default=True, help="Export png files.")  # fmt: skip
def tecan(  # noqa: PLR0913
    ctx: Context,
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
) -> None:
    """Convert a list of Tecan-exported excel files into titrations.

    LIST_FILE : List of Tecan files and concentration values.

    Saves titrations as .dat files and fits all wells using 2 labels. The
    function produces:

    - K plot

    - csv tables for all labelblocks and global fittings.

    Note: Buffer is always subtracted if scheme indicates buffer well positions.
    """
    verbose: int = ctx.obj.get("VERBOSE", 0)
    out = ctx.obj.get("OUT", __tecan_out_dir__)
    out_fp = Path(out) / "Cl" if cl else Path(out) / "pH"
    out_fp.mkdir(parents=True, exist_ok=True)
    # Options validation.
    if cl and not add:
        msg = "--cl requires --add to be specified."
        raise click.UsageError(msg)
    if dil and not add:
        msg = "--dil requires --add to be specified."
        raise click.UsageError(msg)
    if bg and not sch:
        # Also --sch must contain valid buffers!
        msg = "Scheme is needed to compute buffer bg i.e. --bg requires --sch!"
        raise click.UsageError(msg)
    if comb and not (bg and sch and dil):
        msg = "All combinations requires --bg and --dil to be specified."
        raise click.UsageError(msg)
    # Config
    tecan_config = TecanConfig(out_fp, verbose, comb, lim, title, fit, png)
    # Load titration
    list_fp = Path(list_file)
    click.secho(f"** File: {list_fp.resolve()}", fg="green")
    click.echo(tecan_config)
    tit = Titration.fromlistfile(list_fp, not cl)
    tit.params.bg = bg
    tit.params.bg_adj = bg_adj
    tit.params.dil = dil
    tit.params.nrm = nrm
    tit.params.bg_mth = bg_mth
    click.echo(tit.params)
    if add:
        tit.load_additions(Path(add))
    if cl and tit.additions:
        tit.conc = prtecan.calculate_conc(tit.additions, cl)
    if sch:
        tit.load_scheme(Path(sch))
        f = tit.buffer.plot(title=title)
        f.savefig(out_fp / "buffer.png")
        f = tit.buffer.plot(nrm=True, title=title)
        f.savefig(out_fp / "buffer_norm.png")
    with (out_fp / "metadata-labels.txt").open("w", encoding="utf-8") as fp:
        for lbg in tit.labelblocksgroups:
            pprint.pprint(lbg.metadata, stream=fp)
    f = tit.plot_temperature(title=title)
    f.savefig(out_fp / "temperatures.png")
    # Output and export
    tit.export_data_fit(tecan_config)


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
    note = prenspire.prenspire.Note(note_fp, verbose=verbose)
    note.build_titrations(ef)
    dbands = {label: (ini, fin) for label, ini, fin in bands} if bands else {}
    x_combined = {}
    y_combined = {}
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
                    fit_result = binding.fitting.analyze_spectra(data, is_ph, band)
                    if fit_result.is_valid() and fit_result.mini:
                        x_combined[label] = fit_result.mini.userargs[0]["default"].x
                        y_combined[label] = fit_result.mini.userargs[0]["default"].y
                        pdf_file = out_dir / f"{name}_{temp}_{label}_{tit}_{band}.pdf"
                        if fit_result.figure:
                            fit_result.figure.savefig(pdf_file)
                    _print_result(fit_result, pdf_file, str(band))
                # Global spectra analysis with more than 1 label.
                if (
                    len(d_tit.keys() - dbands.keys()) > 1  # svd > 1
                    or len(dbands.keys() & d_tit.keys()) > 1  # bands > 1
                ):
                    ds = binding.fitting.Dataset(x_combined, y_combined, is_ph)
                    spectra_gres = binding.fitting.analyze_spectra_glob(
                        d_tit, ds, dbands
                    )
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


def _print_result(
    fit_result: binding.fitting.FitResult,
    pdf_file: Path,
    band_str: str,
) -> None:
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
    Notes = namedtuple("Notes", ["wells", "conc"])
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
    fit_result = binding.fitting.analyze_spectra(spectra, is_ph, band)
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
    yc = {lbl: file_df[lbl].to_numpy() for lbl in file_df.columns[1:]}
    ds = binding.fitting.Dataset(file_df["x"].to_numpy(), yc, is_ph)
    if weight:
        binding.fitting.weight_multi_ds_titration(ds)
    f_res = binding.fitting.fit_binding_glob(ds)
    # Figure
    figure, ax = plt.subplots()
    binding.fitting.plot_fit(
        ax, ds, f_res.result, nboot=30, pp=binding.plotting.PlotParameters(is_ph)
    )
    lmfit.printfuncs.report_fit(f_res.result, min_correl=min_correl_to_print)
    figure.savefig(Path(file).with_suffix(".png"))
    if boot and f_res.mini:
        # Emcee
        samples = f_res.mini.emcee(burn=burn, steps=boot).flatchain
        fig = binding.plotting.plot_emcee(samples)
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
            samples_ratios = pd.DataFrame({**ratios, **{"K": samples["K"]}})
            fig_ratio = binding.plotting.plot_emcee(samples_ratios)
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
                new_row = row[:4] + [temp, labels]
                writer.writerow(new_row)
