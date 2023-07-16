"""Command-line interface."""
from __future__ import annotations

import csv
import pprint
import warnings
from collections import namedtuple
from pathlib import Path
from typing import Any

import arviz as az
import click
import corner  # type: ignore
import lmfit  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import pandas as pd
from click import Context, Path as cPath

from clophfit import __enspire_out_dir__, __tecan_out_dir__, binding, prenspire, prtecan
from clophfit.prenspire import EnspireFile
from clophfit.prtecan import TitrationAnalysis


@click.group()
@click.version_option(message="%(version)s")
def clop() -> None:  # pragma: no cover
    """Group command."""


@clop.command()
@click.argument("kd1", type=float)
@click.argument("pka", type=float)
@click.argument("ph", type=float)
def eq1(kd1: float, pka: float, ph: float) -> None:
    """pH-deps for Kd."""
    click.echo(binding.kd(kd1=kd1, pka=pka, ph=ph))


@click.group()
@click.pass_context
@click.version_option(message="%(version)s")
@click.option("--verbose", "-v", count=True, help="Verbosity of messages.")
@click.option("--out", "-o", type=cPath(), help="Output folder.")
def ppr(ctx: Context, verbose: int, out: str) -> None:  # pragma: no cover
    """Parse Plate Reader `ppr` group command."""
    ctx.ensure_object(dict)
    ctx.obj["VERBOSE"] = verbose
    ctx.obj["OUT"] = out


######################################
# pr.tecan                           #
######################################
@ppr.command()
@click.pass_context
@click.argument("list_file", type=cPath(exists=True))
@click.option("--scheme", type=cPath(exists=True), help="Plate scheme (buffers CTRs).")
@click.option("--dil", type=cPath(exists=True), help="Initial volume and additions.")
@click.option("--bg", is_flag=True, help="Subtract buffer (scheme wells=='buffer').")
@click.option("--norm", is_flag=True, help="Normalize using metadata (gain, flashes).")
@click.option(
    "--weight/--no-weight", default=True, show_default=True, help="Use residue weights."
)
@click.option(
    "--is-ph/--no-is-ph", default=True, show_default=True, help="Concentrations are pH."
)
@click.option(
    "--fit/--no-fit", default=True, show_default=True, help="Perform also fit."
)
@click.option("--fit-all", is_flag=True, help="Fit all exported data.")
@click.option(
    "--png/--no-png", default=True, show_default=True, help="Export png files."
)
@click.option("--pdf", is_flag=True, help="Full report in pdf file.")
@click.option("--title", "-t", default="", help="Title for some plots.")
@click.option(
    "--Klim", default=None, type=(float, float), help="Range MIN, MAX of plot_K."
)
@click.option(
    "--sel", default=None, type=(float, float), help="Select from K_MIN S1_MIN."
)
def tecan(  # noqa: PLR0913
    ctx: Context,
    list_file: str,
    scheme: str | None,
    dil: str | None,
    bg: bool,
    norm: bool,
    weight: bool,
    is_ph: bool,
    fit: bool,
    fit_all: bool,
    png: bool,
    pdf: bool,
    title: str,
    klim: tuple[float, float] | None,
    sel: tuple[float, float] | None,
) -> None:
    """Convert a list of plate reader acquisitions into titrations.

    LIST_FILE : List of Tecan files and concentration values.

    Saves titrations as .dat files and fits all wells using 2 labels. The function produces:

    - K plot

    - ebar and (for selection) ebarZ plot

    - all_wells pdf

    - csv tables for all labelblocks and global fittings.

    Note: Buffer is always subtracted if scheme indicates buffer well positions.
    """
    verbose = ctx.obj.get("VERBOSE", 0)
    out = ctx.obj.get("OUT", __tecan_out_dir__)
    out_fp = Path(out) / "pH" if is_ph else Path(out) / "Cl"
    list_fp = Path(list_file)
    titan = TitrationAnalysis.fromlistfile(list_fp, is_ph)

    if scheme:
        titan.load_scheme(Path(scheme))
        if dil:
            titan.load_additions(Path(dil))
            bg = True  # should not be needed but influence decimals of
            # exported values; however ``dil imply bg```
            if not is_ph:  # XXX cl conc must be elsewhere
                titan.conc = prtecan.calculate_conc(titan.additions, 1000.0)  # type: ignore
    titan.export_data(out_fp)

    with (out_fp / "metadata-labels.txt").open("w", encoding="utf-8") as fp:
        for lbg in titan.labelblocksgroups:
            pprint.pprint(lbg.metadata, stream=fp)
    if scheme:
        f = titan.plot_buffer(title=title)
        f.savefig(out_fp / "buffer.png")

    if fit:
        if bg and not scheme:
            # ``as bg requires scheme even though scheme does not imply bg```
            warnings.warn("Scheme is needed to compute buffer bg!", stacklevel=2)
        if fit_all:
            for n, b, d, out2 in [
                (0, 0, 0, "dat"),
                (1, 0, 0, "dat_nrm"),
                (0, 1, 0, "dat_bg"),
                (1, 1, 0, "dat_bg_nrm"),
                (0, 1, 1, "dat_bg_dil"),
                (1, 1, 1, "dat_bg_dil_nrm"),
            ]:
                out_fit = out_fp / out2 / "0fit"
                out_fit.mkdir(parents=True, exist_ok=True)
                fit_tecan(
                    titan,
                    weight,
                    bool(n),
                    bool(b),
                    bool(d),
                    verbose,
                    out_fit,
                    klim,
                    title,
                    sel,
                    png,
                    pdf,
                )
        else:
            fit_tecan(
                titan,
                weight,
                norm,
                bg,
                bool(dil),
                verbose,
                out_fp,
                klim,
                title,
                sel,
                png,
                pdf,
            )


def fit_tecan(  # noqa: PLR0913
    titan: TitrationAnalysis,
    weight: bool,
    norm: bool,
    bg: bool,
    dil: bool,
    verbose: int,
    out: Path,
    klim: tuple[float, float] | None,
    title: str | None,
    sel: tuple[float, float] | None,
    png: bool,
    pdf: bool,
) -> None:
    """Help main."""
    titan.fitdata_params = {"bg": bg, "nrm": norm, "dil": dil}
    titan.fitkws = TitrationAnalysis.FitKwargs(weight=weight)
    # lb = 0, 1, 2(for glob)
    for i, fit in enumerate(titan.result_dfs):
        if verbose:
            try:
                print(fit)
                print(klim)
                meta = titan.labelblocksgroups[i].metadata
                print("-" * 79)
                print(f"\nlabel{i:d}")
                pprint.pprint(meta)
            except IndexError:
                print("-" * 79)
                print("\nGlobal on both labels")
            titan.print_fitting(i)
        # CSV tables
        fit.sort_index().to_csv(out / Path("ffit" + str(i) + ".csv"))
        if "S1_y1" in fit.columns:
            order = ["ctrl", "K", "sK", "S0_y0", "sS0_y0", "S1_y0", "sS1_y0"]
            order.extend(["S0_y1", "sS0_y1", "S1_y1", "sS1_y1"])
            ebar_y, ebar_yerr = "S1_y1", "sS1_y1"
        else:
            order = ["ctrl", "K", "sK", "S0_default", "sS0_default"]
            order.extend(["S1_default", "sS1_default"])
            ebar_y, ebar_yerr = "S1_default", "sS1_default"
        out_df = fit.reindex(order, axis=1).sort_index()
        out_df.to_csv(out / f"fit{i}.csv", float_format="%.3g")
        # Plots
        f = titan.plot_k(i, hue_column=ebar_y, xlim=klim, title=title)
        f.savefig(out / f"K{i}.png")
        f = titan.plot_ebar(i, ebar_y, ebar_yerr, title=title)
        f.savefig(out / f"ebar{i}.png")
        if sel:
            xm, ym = sel
            xmin = xm if titan.is_ph else None
            xmax = xm if not titan.is_ph else None
            f = titan.plot_ebar(
                i, ebar_y, ebar_yerr, xmin=xmin, xmax=xmax, ymin=ym, title=title
            )
            f.savefig(out / f"ebar{i}_sel{xm},{ym}.png")
        if png:
            titan.export_png(i, out)
    if pdf:
        # FIXME: export pdf
        titan.plot_all_wells(2, out / "all_wells.pdf")


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


def fit_enspire(
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
                    if fit_result.is_valid():
                        x_combined[label] = fit_result.mini.userargs[0]["default"].x
                        y_combined[label] = fit_result.mini.userargs[0]["default"].y
                        pdf_file = out_dir / f"{name}_{temp}_{label}_{tit}_{band}.pdf"
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
                    if spectra_gres.svd:
                        pdf_file = out_dir / f"{name}_{temp}_all_{tit}_SVD.pdf"
                        spectra_gres.svd.figure.savefig(pdf_file)
                        _print_result(spectra_gres.svd, pdf_file, "")
                    if spectra_gres.gsvd:
                        pdf_file = out_dir / f"{name}_{temp}_g_{tit}_SVD.pdf"
                        spectra_gres.gsvd.figure.savefig(pdf_file)
                        _print_result(spectra_gres.gsvd, pdf_file, "")
                    if spectra_gres.bands:
                        keys = dbands.keys() & d_tit.keys()
                        lname = [f"{k}({dbands[k][0]},{dbands[k][1]})" for k in keys]
                        bands_str = "".join(lname)
                        pdf_file = out_dir / f"{name}_{temp}_all_{tit}_{bands_str}.pdf"
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


@clop.command("fit_titration")
@click.argument("csv_f", type=click.Path(exists=True))
@click.argument("note_f", type=click.Path(exists=True))
@click.option("-d", "--out", default=Path("."), type=Path, help="destination directory")
@click.option(
    "-t",
    "titration_type",
    default="pH",
    type=click.Choice(["pH", "Cl"], case_sensitive=False),
    help="titration type (default: pH)",
)
@click.option(
    "-b", "--band", nargs=2, type=int, help="Integration interval from <1> to <2>"
)
@click.option("-v", "--verbose", is_flag=True, help="increase output verbosity")
def fit_titration(  # noqa: PLR0913
    csv_f: str,
    note_f: str,
    out: Path,
    titration_type: str,
    band: tuple[int, int] | None,
    verbose: bool,
) -> None:
    """Update old svd or band fit of titration spectra."""
    note_df = pd.read_csv(note_f, sep="\t")
    csv = pd.read_csv(csv_f)
    # Ignore buffer wells! SVD will use differences between spectra.
    note_df = note_df[note_df["mutant"] != "buffer"]
    Notes = namedtuple("Notes", ["wells", "conc"])
    note = Notes(list(note_df["well"]), list(note_df[titration_type]))
    spectra = csv[note.wells]
    spectra.index = csv["lambda"]
    spectra.columns = np.array(note.conc)
    if verbose:
        print(csv)
        click.echo(note_f)
        print(note)
        print("DataFrame\n", spectra)
    is_ph = titration_type == "pH"
    fit_result = binding.fitting.analyze_spectra(spectra, is_ph, band)
    # output
    out.mkdir(parents=True, exist_ok=True)
    pdf_file = out / f"{Path(csv_f).stem}_{band}_{Path(note_f).stem}.pdf"
    if fit_result.figure:
        fit_result.figure.savefig(pdf_file)
    _print_result(fit_result, pdf_file, str(band))


@clop.command("fit_titration_global")
@click.argument("file", type=click.Path(exists=True))
@click.option("-d", "--out", default=Path("."), type=Path, help="destination directory")
@click.option(
    "-t",
    "titration_type",
    default="pH",
    type=click.Choice(["pH", "Cl"], case_sensitive=False),
    help="titration type (default: pH)",
)
@click.option("-b", "--boot", type=int, help="Number of booting iterations")
@click.option("-v", "--verbose", is_flag=True, help="increase output verbosity")
def fit_titration_global(file, out, titration_type, boot, verbose):  # type: ignore
    """Update old svd or band fit of titration spectra."""
    file_df = pd.read_csv(file)
    if verbose:
        click.echo(file_df)
        print(out)
        print(titration_type)
        print(boot)
    xc = {}
    yc = {}
    for label in file_df.columns[1:]:
        xc[label] = file_df["x"].to_numpy()
        yc[label] = file_df[label].to_numpy()
    ds = binding.fitting.Dataset(xc, yc, titration_type == "pH")
    f_res = binding.fitting.fit_binding_glob(ds, True)
    figure, ax = plt.subplots()
    binding.fitting.plot_fit(ax, ds, f_res.result)
    lmfit.printfuncs.report_fit(f_res.result, min_correl=0.65)
    figure.savefig(Path(file).with_suffix(".png"))
    result_emcee = f_res.mini.emcee(burn=150, steps=1800)
    samples = result_emcee.flatchain
    # Convert the dictionary of flatchains to an ArviZ InferenceData object
    samples_dict = {key: np.array(val) for key, val in samples.items()}
    idata = az.from_dict(posterior=samples_dict)
    quantiles = np.percentile(idata.posterior["K"], [2.275, 15.865, 50, 84.135, 97.275])
    emcee_plot = corner.corner(samples)
    emcee_plot.savefig("e.png")
    print(quantiles)
    for lbl in ds:
        ratio = (
            f_res.result.params[f"S0_{lbl}"].value
            / f_res.result.params[f"S1_{lbl}"].value
        )
        print(f"Ratio of {lbl}: {ratio}")


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
        output_path.write_text(",".join(headers) + "\n")

    # read data from note file, append to output file
    with input_path.open("r") as datafile:
        reader = csv.reader(datafile, delimiter="\t")  # assuming tab-separated values
        next(reader)  # skip the header row

        with output_path.open("a") as f:
            writer = csv.writer(f)
            for row in reader:
                new_row = row[:4] + [temp, labels]
                writer.writerow(new_row)
