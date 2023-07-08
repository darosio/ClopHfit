"""Command-line interface."""
from __future__ import annotations

import csv
import pprint
import warnings
from collections import namedtuple
from pathlib import Path

import arviz as az
import click
import corner  # type: ignore
import lmfit  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import pandas as pd

from clophfit import __default_enspire_out_dir__, binding, prenspire, prtecan
from clophfit.prenspire import EnspireFile


def fit_routine(  # noqa: PLR0913
    titan: prtecan.TitrationAnalysis,
    kind: str,
    weight: bool,
    confint: float,
    norm: bool,
    bg: bool,
    dil: bool,
    verbose: int,
    out: Path,
    klim: tuple[float, float] | None,
    title: str | None,
    sel: tuple[float, float] | None,
    pdf: bool,
) -> None:
    """Help main."""
    titan.datafit_params = {"bg": bg, "nrm": norm, "dil": dil}
    titan.fit_args = {"kind": kind, "no_weight": (not weight)}
    for i, fit in enumerate(titan.fitresults):
        # Printout
        if verbose:
            try:
                print(fit)
                print(klim)
                print(confint)
                meta = titan.labelblocksgroups[i].metadata
                print("-" * 79)
                print(f"\nlabel{i:d}")
                pprint.pprint(meta)
            except IndexError:
                print("-" * 79)
                print("\nGlobal on both labels")
            titan.print_fitting(i)
        """# XXX: plots
        # Csv tables
        fit.sort_index().to_csv(out / Path("ffit" + str(i) + ".csv"))
        if "SA2" in fit:
            out_cols = ["K", "sK", "SA", "sSA", "SB", "sSB"]
            out_cols.extend(["SA2", "sSA2", "SB2", "sSB2"])
        else:
            out_cols = ["K", "sK", "SA", "sSA", "SB", "sSB"]
        fit[out_cols].sort_index().to_csv(
            out / Path("fit" + str(i) + ".csv"), float_format="%5.1f"
        )
        Plots
        f = titan.plot_k(i, xlim=klim, title=title)
        f.savefig(out / Path("K" + str(i) + ".png"))
        f = titan.plot_ebar(i, title=title)
        f.savefig(out / Path("ebar" + str(i) + ".png"))
        """
        if sel:
            if kind.lower() == "ph":
                xmin, ymin = sel
                f = titan.plot_ebar(i, xmin=xmin, ymin=ymin, title=title)
            if kind.lower() == "cl":
                xmax, ymin = sel
                f = titan.plot_ebar(i, xmax=xmax, ymin=ymin, title=title)
            f.savefig(out / Path("ebarZ" + str(i) + ".png"))
    if pdf:
        titan.plot_all_wells(2, out / "all_wells.pdf")


@click.group()
@click.version_option(message="%(version)s")
def clop() -> None:  # pragma: no cover
    """Group command."""


@clop.command("eq1")
@click.argument("kd1", type=float)
@click.argument("pka", type=float)
@click.argument("ph", type=float)
def eq1(  # type: ignore
    kd1,
    pka,
    ph,
):
    """pH-deps for Kd."""
    print(binding.kd(kd1=kd1, pka=pka, ph=ph))


@clop.command("pr.tecan")
@click.argument("list_file", type=click.Path(path_type=Path))
@click.option(
    "--dil",
    type=click.Path(path_type=Path),
    is_flag=False,
    flag_value="additions.pH",
    show_default="additions.pH",
    help="Initial volume and additions",
)
@click.option("--norm", is_flag=True, help="Normalize using metadata (gain, flashes).")
@click.option(
    "--scheme",
    type=click.Path(path_type=Path),
    is_flag=False,
    flag_value="scheme.txt",
    show_default="scheme.txt",
    help="Positions of buffer and controls wells.",
)
@click.option("--bg", is_flag=True, help="Subtract buffer (scheme wells=='buffer').")
@click.option(
    "--kind",
    "-k",
    type=click.Choice(["pH", "Cl"], case_sensitive=False),
    default="pH",
    help="Kind of titration.",
    show_default=True,
)
@click.option(
    "--weight/--no-weight",
    default=False,
    show_default=True,
    help="Global fitting without relative residues weights.",
)
@click.option(
    "--fit/--no-fit",
    default=True,
    show_default=True,
    help="Perform also fit.",
)
@click.option("--fit-all", is_flag=True, help="Fit all exported data.")
@click.option(
    "--confint",
    default=0.95,
    type=click.FloatRange(0, 1, clamp=True),
    show_default=True,
    help="Confidence value for the calculation of parameter errors.",
)
@click.option(
    "--out",
    type=click.Path(path_type=Path),
    default="out2",
    help="Path to output results.",
    show_default=True,
)
@click.option("--pdf", is_flag=True, help="Full report in pdf file.")
@click.option("--title", "-t", default=None, help="Title for some plots.")
@click.option(
    "--Klim",
    default=None,
    type=(float, float),
    help="Range MIN, MAX (xlim) for plot_K.",
)
@click.option(
    "--sel",
    default=None,
    type=(float, float),
    help="Errorbar plot for selection with K_min SA_min.",
)
@click.option("--verbose", "-v", count=True, help="Verbosity of messages.")
def tecan(  # type: ignore  # noqa: PLR0913
    list_file,
    scheme,
    dil,
    verbose,
    bg,
    kind,
    norm,
    out,
    weight,
    fit,
    fit_all,
    confint,
    klim,
    title,
    sel,
    pdf,
):
    """Convert a list of plate reader acquisitions into titrations.

    LIST_FILE : List of Tecan files and concentration values.

    Save titrations as .dat files.

    Fits all wells using 2 labels and produces:

    - K plot

    - ebar and (for selection) ebarZ plot

    - all_wells pdf

    - csv tables for all labelblocks and global fittings.

    Notes: Buffer is always subtracted if scheme indicates buffer well positions.

    """
    titan = prtecan.TitrationAnalysis.fromlistfile(list_file)
    if scheme:
        titan.load_scheme(scheme)
        if dil:
            titan.load_additions(dil)
            bg = True  # should not be needed but influence decimals of
            # exported values; however ``dil imply bg```
            if kind.lower() == "cl":  # XXX cl conc must be elsewhere
                titan.conc = list(prtecan.calculate_conc(titan.additions, 1000.0))  # type: ignore
    titan.export_data(out)
    with (out / "metadata-labels.txt").open("w", encoding="utf-8") as fp:
        for lbg in titan.labelblocksgroups:
            pprint.pprint(lbg.metadata, stream=fp)
    if scheme:
        f = titan.plot_buffer(title=title)
        f.savefig(out / Path("buffer.png"))

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
                out_fit = out / out2 / "0fit"
                out_fit.mkdir(parents=True, exist_ok=True)
                fit_routine(
                    titan,
                    kind,
                    weight,
                    confint,
                    bool(n),
                    bool(b),
                    bool(d),
                    verbose,
                    out_fit,
                    klim,
                    title,
                    sel,
                    pdf,
                )
        else:
            fit_routine(
                titan,
                kind,
                weight,
                confint,
                norm,
                bg,
                bool(dil),
                verbose,
                out,
                klim,
                title,
                sel,
                pdf,
            )


@clop.command("pr.enspire")
@click.argument("csv", type=click.Path(exists=True, path_type=Path))
@click.argument("note_fp", type=click.Path(exists=True, path_type=Path), required=False)
@click.option(
    "-b",
    "--band-intervals",
    "bands",
    multiple=True,
    nargs=3,
    type=(str, int, int),
    help="Label and band interval (format: LABEL LOWER UPPER)",
)
@click.option(
    "--out",
    "-d",
    type=click.Path(path_type=Path),
    default=__default_enspire_out_dir__,
    help="Path to output results.",
    show_default=True,
)
@click.option("--verbose", "-v", count=True, help="Verbosity of messages.")
def enspire(csv, note_fp, out, bands, verbose):  # type: ignore
    """Save spectra as csv tables from EnSpire xls file."""
    ef = EnspireFile(csv, verbose=verbose)
    ef.export_measurements(out)
    if note_fp is not None:
        fit_enspire(ef, note_fp, out, bands, verbose)


def fit_enspire(
    ef: EnspireFile,
    note_fp: Path,
    out_dir: Path,
    bands: list[tuple[str, int, int]] | None,
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
@click.argument("csv_fp", type=click.Path(exists=True, path_type=Path))
@click.argument("note_fp", type=click.Path(exists=True, path_type=Path))
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
    csv_fp: Path,
    note_fp: Path,
    out: Path,
    titration_type: str,
    band: tuple[int, int] | None,
    verbose: bool,
) -> None:
    """Update old svd or band fit of titration spectra."""
    note_df = pd.read_csv(note_fp, sep="\t")
    csv = pd.read_csv(csv_fp)
    out_fp = Path(out)
    # Ignore buffer wells! SVD will use differences between spectra.
    note_df = note_df[note_df["mutant"] != "buffer"]
    Notes = namedtuple("Notes", ["wells", "conc"])
    note = Notes(list(note_df["well"]), list(note_df[titration_type]))
    spectra = csv[note.wells]
    spectra.index = csv["lambda"]
    spectra.columns = np.array(note.conc)
    if verbose:
        print(csv)
        click.echo(note_fp)
        print(note)
        print("DataFrame\n", spectra)
    is_ph = titration_type == "pH"
    fit_result = binding.fitting.analyze_spectra(spectra, is_ph, band)
    # output
    out_fp.mkdir(parents=True, exist_ok=True)
    pdf_file = out_fp / f"{csv_fp.stem}_{band}_{note_fp.stem}.pdf"
    if fit_result.figure:
        fit_result.figure.savefig(pdf_file)
    _print_result(fit_result, pdf_file, str(band))


@clop.command("fit_titration_global")
@click.argument("file", type=click.Path(exists=True, path_type=Path))
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


@click.command()
@click.argument("note", type=click.Path(exists=True, path_type=Path))
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
