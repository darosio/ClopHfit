"""Command-line interface."""
import functools
import os
import pprint
from enum import Enum
from pathlib import Path
from typing import Tuple

import click

from clophfit import binding, prtecan


@click.group()
@click.version_option()
def clop() -> None:
    """Group command."""
    pass


@clop.command()
@click.option("--some", flag_value="bello", is_flag=False, show_default="LLL")
@click.argument("kd1", type=float, default=13.2)
@click.argument("pka", type=float)
@click.argument("ph", type=float)
def eq1(  # type: ignore
    kd1,
    pka,
    ph,
    some,
):  # type: ignore
    """pH-deps for Kd."""
    if some:
        print(some)
    print(binding.kd(Kd1=kd1, pKa=pka, pH=ph))


class FitKind(str, Enum):
    """Enum for kind of fit: pH or Cl."""

    pH = "pH"
    Cl = "Cl"


# @click.argument("list_file", type=click.Path(path_type=Path))


@clop.command("prtecan")
@click.argument("list_file", type=str)
@click.option(
    "--scheme",
    type=click.Path(path_type=Path),
    is_flag=False,
    flag_value="scheme.txt",
    show_default="scheme.txt",
    help="Positions of buffer and controls wells.",
)
@click.option(
    "--dil",
    type=click.Path(path_type=Path),
    is_flag=False,
    flag_value="additions.pH",
    show_default="additions.pH",
    help="Initial volume and additions",
)
@click.option(
    "--kind",
    "-k",
    type=click.Choice(['pH', 'Cl'], case_sensitive=False),
    default="pH",
    help="Titration type.",
    show_default=True,
)
@click.option(
    "--out", type=str, default="out2", help="Path to output results.", show_default=True
)
@click.option(
    "--dat",
    type=str,
    default="./dat",
    help="Path to output dat files.",
    show_default=True,
)
@click.option("--pdf", is_flag=True, help="Full report in pdf file.")
@click.option(
    "--Klim",
    default=None,
    type=Tuple[float, float],
    help="Range MIN, MAX (xlim) for plot_K.",
)
@click.option("--title", "-t", default=None, help="Title for some plots.")
@click.option(
    "--sel",
    default=None,
    type=Tuple[float, float],
    help="Errorbar plot for selection with K_min SA_min.",
)
@click.option(
    "--weigth/--no-weight",
    default=True,
    show_default=True,
    help="Global fitting without relative residues weights.",
)
@click.option(
    "--confint",
    default=0.95,
    type=click.FloatRange(0, 1, clamp=True),
    show_default=True,
    help="Confidence value for the calculation of parameter errors.",
)
@click.option("--norm", is_flag=True, help="Normalize using metadata (gain, flashes).")
@click.option("--bg", is_flag=True, help="Substract buffer (scheme wells=='buffer').")
@click.option("--verbose", "-v", count=True)
def tecan(  # type: ignore
    list_file,
    scheme,
    dil,
    verbose,
    version,
    bg,
    kind,
    norm,
    out,
    dat,
    no_weight,
    confint,
    Klim,
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
    tit = prtecan.Titration(list_file)
    # TitrationAnalysis
    if scheme:
        tit = prtecan.TitrationAnalysis(tit, scheme)
        if bg:
            tit.subtract_bg()
        if dil:
            tit.dilution_correction(dil)
            if kind == 'cl':  # XXX cl conc must be elsewhere
                tit.conc = tit.calculate_conc(tit.additions, 1000)
        if norm:
            tit.metadata_normalization()
    else:
        tit = prtecan.TitrationAnalysis(tit)
    # Output dir
    if out:
        if not os.path.isdir(out):
            os.makedirs(out)
        ttff = functools.partial(os.path.join, out)
    else:
        ttff = functools.partial(os.path.join, '')
    # Export .dat
    tit.export_dat(ttff(dat))
    # Fit
    tit.fit(kind.value, no_weight=no_weight, tval_conf=float(confint))
    # metadata-labels.txt
    fp = open(ttff('metadata-labels.txt'), 'w')
    for lbg in tit.labelblocksgroups:
        pprint.pprint(lbg.metadata, stream=fp)
    fp.close()
    # Loop over fittings[]
    for i, fit in enumerate(tit.fittings):
        # Printout
        if verbose:
            try:
                meta = tit.labelblocksgroups[i].metadata
                print('{:s}'.format('-' * 79))
                print('\nlabel{:d}'.format(i))
                pprint.pprint(meta)
            except IndexError:
                print('{:s}'.format('-' * 79))
                print('\nGlobal on both labels')
            tit.print_fitting(i)
        # Csv tables
        fit.sort_index().to_csv(ttff('ffit' + str(i) + '.csv'))
        if 'SA2' in fit:
            out_cols = [
                'K',
                'sK',
                'SA',
                'sSA',
                'SB',
                'sSB',
                'SA2',
                'sSA2',
                'SB2',
                'sSB2',
            ]
        else:
            out_cols = ['K', 'sK', 'SA', 'sSA', 'SB', 'sSB']
        fit[out_cols].sort_index().to_csv(
            ttff('fit' + str(i) + '.csv'), float_format='%5.1f'
        )
        # Plots
        f = tit.plot_K(i, xlim=Klim, title=title)
        f.savefig(ttff('K' + str(i) + '.png'))
        f = tit.plot_ebar(i, title=title)
        f.savefig(ttff('ebar' + str(i) + '.png'))
        if sel:
            if kind.value.lower() == 'ph':  # FIXME **kw?
                xmin, ymin = sel
                f = tit.plot_ebar(i, xmin=xmin, ymin=ymin, title=title)
            if kind.value.lower() == 'cl':
                xmax, ymin = sel
                f = tit.plot_ebar(i, xmax=xmax, ymin=ymin, title=title)
            f.savefig(ttff('ebarZ' + str(i) + '.png'))
    # ---------- ebar ---------------------------
    if hasattr(tit.labelblocksgroups[0], 'buffer'):
        f = tit.plot_buffer(title=title)
        f.savefig(ttff('buffer.png'))
    if pdf:
        tit.plot_all_wells(ttff('all_wells.pdf'))
