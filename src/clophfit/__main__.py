"""Command-line interface."""
from __future__ import annotations

import pprint
import warnings
from pathlib import Path

import click

from clophfit import binding
from clophfit import prtecan


def fit_routine(
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
def tecan(  # type: ignore
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
    with open(out / "metadata-labels.txt", "w", encoding="utf-8") as fp:
        for lbg in titan.labelblocksgroups:
            pprint.pprint(lbg.metadata, stream=fp)
    if scheme:
        f = titan.plot_buffer(title=title)
        f.savefig(out / Path("buffer.png"))

    if fit:
        if bg and not scheme:
            # ``as bg requires scheme even though scheme does not imply bg```
            warnings.warn("Scheme is needed to compute buffer bg!")
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
