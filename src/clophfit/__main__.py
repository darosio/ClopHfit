"""Command-line interface."""
from __future__ import annotations

import pprint
import sys
from pathlib import Path

import click

from clophfit import binding
from clophfit import prtecan


@click.group()
@click.version_option()
def clop() -> None:
    """Group command."""
    pass


@clop.command()
@click.option("--some", flag_value="bello", is_flag=False, show_default="LLL")
@click.argument("kd1", type=float)
@click.argument("pka", type=float)
@click.argument("ph", type=float)
def eq1(  # type: ignore
    kd1,
    pka,
    ph,
    some,
):
    """pH-deps for Kd."""
    if some:
        print(some)
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
@click.option("--bg", is_flag=True, help="Substract buffer (scheme wells=='buffer').")
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
@click.option(
    "--dat",
    type=click.Path(path_type=Path),
    default="./dat",
    help="Path to output dat files.",
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
    dat,
    weight,
    fit,
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

    """
    titan = prtecan.TitrationAnalysis.fromlistfile(list_file)
    if scheme:
        titan.load_scheme(scheme)
        if bg:
            pass  # titan.subtract_bg()
        if dil:
            titan.load_additions(dil)
            if kind.lower() == "cl":  # XXX cl conc must be elsewhere
                titan.conc = list(prtecan.calculate_conc(titan.additions, 1000.0))  # type: ignore
        if norm:
            pass  # titan.metadata_normalization()
    titan.export_dat(out / dat)
    # Fit
    if not fit:
        sys.exit(0)
    titan.fit(kind, no_weight=(not weight), tval=float(confint))
    # metadata-labels.txt
    fp = open(out / "metadata-labels.txt", "w")
    for lbg in titan.labelblocksgroups:
        pprint.pprint(lbg.metadata, stream=fp)
    fp.close()
    # Loop over fittings[]
    for i, fit in enumerate(titan.fittings):
        # Printout
        if verbose:
            try:
                meta = titan.labelblocksgroups[i].metadata
                print("{:s}".format("-" * 79))
                print(f"\nlabel{i:d}")
                pprint.pprint(meta)
            except IndexError:
                print("{:s}".format("-" * 79))
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
            if kind.lower() == "ph":  # FIXME **kw?
                xmin, ymin = sel
                f = titan.plot_ebar(i, xmin=xmin, ymin=ymin, title=title)
            if kind.lower() == "cl":
                xmax, ymin = sel
                f = titan.plot_ebar(i, xmax=xmax, ymin=ymin, title=title)
            f.savefig(out / Path("ebarZ" + str(i) + ".png"))
    # ---------- ebar ---------------------------
    f = titan.plot_buffer(title=title)
    f.savefig(out / Path("buffer.png"))
    if pdf:
        titan.plot_all_wells(out / Path("all_wells.pdf"))
