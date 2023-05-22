"""Command-line interface."""
from __future__ import annotations

import pprint
import warnings
from collections import namedtuple
from pathlib import Path

import click
import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import pandas as pd
import seaborn  # type: ignore # noqa: ICN001
from numpy.typing import NDArray

from clophfit import binding
from clophfit import prtecan
from clophfit.binding import fit_titration_spectra
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
    titan.fit(
        kind,
        no_weight=(not weight),
        tval=float(confint),
        nrm=norm,
        bg=bg,
        dil=dil,
    )
    for i, fit in enumerate(titan.fittings):
        # Printout
        if verbose:
            try:
                meta = titan.labelblocksgroups[i].metadata
                print("-" * 79)
                print(f"\nlabel{i:d}")
                pprint.pprint(meta)
            except IndexError:
                print("-" * 79)
                print("\nGlobal on both labels")
            titan.print_fitting(i)
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
        # Plots
        f = titan.plot_k(i, xlim=klim, title=title)
        f.savefig(out / Path("K" + str(i) + ".png"))
        f = titan.plot_ebar(i, title=title)
        f.savefig(out / Path("ebar" + str(i) + ".png"))
        if sel:
            if kind.lower() == "ph":
                xmin, ymin = sel
                f = titan.plot_ebar(i, xmin=xmin, ymin=ymin, title=title)
            if kind.lower() == "cl":
                xmax, ymin = sel
                f = titan.plot_ebar(i, xmax=xmax, ymin=ymin, title=title)
            f.savefig(out / Path("ebarZ" + str(i) + ".png"))
    if pdf:
        titan.plot_all_wells(out / "all_wells.pdf")


@click.group()
@click.version_option(message="%(version)s")
def clop() -> None:  # pragma: no cover
    """Group command."""


@clop.command()
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


@clop.command("prtecan")
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


@clop.command("prenspire")
@click.argument("csv", type=click.Path(path_type=Path))
@click.option(
    "--out",
    type=click.Path(path_type=Path),
    default="Meas",
    help="Path to output results.",
    show_default=True,
)
@click.option("--verbose", "-v", count=True, help="Verbosity of messages.")
def enspire(csv, out, verbose):  # type: ignore
    """Save spectra as csv tables from EnSpire xls file."""
    ef = EnspireFile(csv, verbose=verbose)
    ef.export_measurements(out)


@clop.command()
@click.argument("csvtable")
@click.argument("note_fp")
@click.option(
    "-d", "--out", "out_dir", default=Path("."), type=Path, help="destination directory"
)
@click.option(
    "-m",
    "--method-of-analysis",
    "analysis_method",
    default="svd",
    type=click.Choice(["svd", "band"], case_sensitive=False),
    help="analysis method (default: svd)",
)
@click.option(
    "-t",
    "--titration-of",
    "titration_type",
    default="pH",
    type=click.Choice(["pH", "cl"], case_sensitive=False),
    help="titration type (default: pH)",
)
@click.option(
    "-b",
    "--band-interval",
    "band",
    nargs=2,
    type=int,
    help="Integration interval from <1> to <2>",
)
@click.option("-v", "--verbose", is_flag=True, help="increase output verbosity")
def fit(csvtable, note_fp, out_dir, analysis_method, titration_type, band, verbose):  # type: ignore # noqa: PLR0913,PLR0915
    """Old fit titration."""
    # input of spectra (csv) and titration (note) data
    csv = pd.read_csv(csvtable)
    note_file = pd.read_table(note_fp)  # noqa: PD012
    note_file = note_file[note_file["mutant"] != "buffer"]
    # TODO aggregation logic for some pH or cloride
    nnote = namedtuple("nnote", "wells conc")

    # pH or Cl titration
    def fz_kd(kd: float, p: list[float], x: NDArray[np.float_]) -> NDArray[np.float_]:
        return (p[0] + p[1] * x / kd) / (1 + x / kd)

    def fz_pk(pk: float, p: list[float], x: NDArray[np.float_]) -> NDArray[np.float_]:
        return (p[1] + p[0] * 10 ** (pk - x)) / (1 + 10 ** (pk - x))

    if titration_type == "cl":
        note = nnote(list(note_file.well), list(note_file.Cl))
        fz = fz_kd
    elif titration_type == "pH":
        note = nnote(list(note_file.well), list(note_file.pH))
        fz = fz_pk  # type: ignore

    df = csv[note.wells]  # noqa: PD901
    df.index = csv["lambda"]
    conc = np.array(note.conc)
    # sideeffect print input data
    if verbose:
        print(csv)
        print(note)
        print("conc vector\n", conc)
        print("DataFrame\n", df)
    if analysis_method == "svd":
        # svd on difference spectra
        ddf = df.sub(df.iloc[:, 0], axis=0)
        u, s, v = np.linalg.svd(ddf)
        # fitting
        result = fit_titration_spectra(fz, conc, v[0, :])
        # plotting
        seaborn.set_style("ticks")
        fig1 = plt.figure(figsize=(12, 8))
        ax1 = fig1.add_axes([0.05, 0.65, 0.32, 0.31])
        plt.grid(True)
        ax2 = fig1.add_axes([0.42, 0.65, 0.32, 0.31])
        plt.grid(True)
        ax1.plot(df)
        ax2.plot(ddf.index, u[:, 0], "k-", lw=3)
        ax2.plot(ddf.index, u[:, 1], "b--")
        ax3 = fig1.add_axes(
            [0.80, 0.65, 0.18, 0.31],
            yscale="log",
            xticks=[1, 2, 3, 4],
            title="autovalues",
        )
        ax3.bar(
            [1, 2, 3, 4],
            (s**2 / sum(s**2))[:4],
            align="center",
            alpha=0.7,
            width=0.66,
        )
        ax4 = fig1.add_axes([0.05, 0.08, 0.50, 0.50], title="fitting")
        ax5 = fig1.add_axes(
            [0.63, 0.08, 0.35, 0.50],
            title="SVD coefficients",
            xlabel="1$^{st}$ autovector",
            ylabel="2$^{nd}$ autovector",
        )
        ax4.scatter(conc, v[0, :])
        xmin = conc.min()
        xmax = conc.max()
        xmax += (xmax - xmin) / 7
        xlin = np.linspace(xmin, xmax, 100)
        ax4.plot(xlin, fz(result.K, [result.SA, result.SB], xlin))
        title = str(round(result.K, 2)) + " \u00B1 " + str(round(result.sK, 2))
        plt.figtext(0.26, 0.54, title, size=20)
        ax5.plot(v[:, 1], v[:, 2], lw=0.8)
        for x, y, w in zip(v[:, 1], v[:, 2], note.wells):
            ax5.text(x, y, w)
        kd = []
        sa = []
        sb = []
        for _ in range(100):
            boot_idxs = np.random.randint(0, len(ddf.columns) - 1, len(ddf.columns))
            ddf2 = df.iloc[:, boot_idxs]
            conc2 = conc[boot_idxs]
            u, s, v = np.linalg.svd(ddf2)
            result2 = fit_titration_spectra(fz, conc2, v[0, :])
            kd.append(result2.K)
            sa.append(result2.SA)
            sb.append(result2.SB)
        bs = pd.DataFrame({"kd": kd, "SA": sa, "SB": sb})
        bs.to_csv("bs.txt")
    elif analysis_method == "band":
        # fitting
        try:
            ini = band[0]
            fin = band[1]
            y = []
            for c in df.columns:
                y.append(df[c].loc[ini:fin].sum())
            result = fit_titration_spectra(fz, conc, y)
        except:
            print(
                f"""bands [{ini}, {fin}] not in index.
                  Try other values"""
            )
            raise
        # plotting
        fig1 = plt.figure(figsize=(12, 8))
        ax4 = fig1.add_axes([0.05, 0.08, 0.50, 0.50], title="fitting")
        ax4.scatter(conc, y)
        xmin = conc.min()
        xmax = conc.max()
        xmax += (xmax - xmin) / 7
        xlin = np.linspace(xmin, xmax, 100)
        ax4.plot(xlin, fz(result.K, [result.SA, result.SB], xlin))
        title = (
            str(round(result.K, 2))
            + " \u00B1 "
            + str(round(result.sK, 2))
            + "["
            + str(ini)
            + ":"
            + str(fin)
            + "]"
        )
        plt.figtext(0.26, 0.54, title, size=20)
        for c, yy, well in zip(conc, y, note.wells):
            ax4.text(c, yy, well)

    # output
    f_csv_shortname = Path(csvtable).stem
    f_note_shortname = Path(note_fp).stem
    f_out = "_".join([f_csv_shortname, analysis_method, f_note_shortname])
    print("f-out: ", f_out)
    if not out_dir.is_dir():
        out_dir.mkdir(parents=True)
    f_out = out_dir / Path(f_out).with_suffix(".pdf")
    fig1.savefig(f_out)

    print("best-fitting using: ", analysis_method)
    print("spectra csv file: ", f_csv_shortname)
    print("note file: ", f_note_shortname)
    print("K = ", round(result.K, 3))
    print("sK = ", round(result.sK, 3))
    print("SA = ", round(result.SA, 3))
    print("sSA = ", round(result.sSA, 3))
    print("SB = ", round(result.SB, 3))
    print("sSB = ", round(result.sSB, 3))
