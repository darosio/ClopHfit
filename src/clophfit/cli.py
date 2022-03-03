"""Command-line interface."""
import functools
import os
import pprint
from enum import Enum
from typing import Optional, Tuple

import typer

from clophfit import binding, prtecan

from . import __version__

app = typer.Typer()


# @typer.version_option(version=__version__)
def version_callback(value: bool) -> None:
    """Version callback function."""
    if value:
        typer.echo(f"{__version__}")
        raise typer.Exit()


version = typer.Option(None, "--version", callback=version_callback, is_eager=True)


@app.command("eq1")
def eq1(
    kd1: float,  # Must be lowercase.
    pka: float,
    ph: float,
    version: bool = version,
) -> None:
    """pH-deps for Kd."""
    print(binding.kd(Kd1=kd1, pKa=pka, pH=ph))


class FitKind(str, Enum):
    """Enum for kind of fit: pH or Cl."""

    pH = "pH"
    Cl = "Cl"


# -Arguments # TODO maybe use pathlib.Path
list_file = typer.Argument(..., help="List of Tecan files and concentration values.")
scheme = typer.Argument("", help="Describe content of wells.", show_default=False)
dil = typer.Argument("", help="Describe volume addictions.", show_default=False)
# -Options
verbose = typer.Option(0, "--verbose", "-v", count=True)
version = typer.Option(None, "--version", callback=version_callback, is_eager=True)
bg = typer.Option(True, help="Substract buffer (scheme wells=='buffer').")
kind = typer.Option(
    FitKind.pH, "--kind", "-k", case_sensitive=False, help="Titration type."
)
norm = typer.Option(False, help="Normalize using metadata (gain, flashes).")
out = typer.Option("out2", help="Path to output results.")
dat = typer.Option("./dat", help="Path to output dat files.")
no_weight = typer.Option(
    False, "--no-weight/ ", help="Global fitting without relative residues weights."
)
confint = typer.Option(
    0.95,
    min=0,
    max=1,
    help="Confidence value for the calculation of parameter errors.",
)
Klim = typer.Option(None, "--Klim", help="Range MIN, MAX (xlim) for plot_K.")
title = typer.Option(None, "-t", help="Title for some plots.")
sel = typer.Option(None, help="Errorbar plot for selection with K_min SA_min.")
pdf = typer.Option(False, help="Full report in pdf file.")


"""
    Positions of buffer and control samples goes into SCHEME [scheme.txt].
    Initial volume and titration additions go into DIL [addition.cl|pH].

    Save titrations as .dat files.
    Fits all wells using 2 labels and produces:
    - K plot
    - ebar and (for selection) ebarZ plot
    - all_wells pdf
    - csv tables for all labelblocks and global fittings.

"""


@app.command("prtecan")
def tecan(
    list_file: str = list_file,
    scheme: Optional[str] = scheme,
    verbose: int = verbose,
    version: bool = version,
    bg: bool = bg,
    dil: Optional[str] = dil,
    kind: FitKind = kind,
    norm: bool = norm,
    out: str = out,
    dat: str = dat,
    no_weight: bool = no_weight,
    confint: float = confint,
    Klim: Optional[Tuple[float, float]] = Klim,
    title: str = title,
    sel: Tuple[float, float] = sel,
    pdf: bool = pdf,
) -> None:
    """Convert a list of plate reader acquisitions into titrations."""
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


def run() -> None:
    """Run commands."""
    app()
