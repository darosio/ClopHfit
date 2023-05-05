"""Cli: pr.enspire."""
from typing import Optional

import typer

import clophfit.prenspire
from clophfit.prenspire.prenspire import EnspireFile


def version(value: bool):
    """Printout version."""
    if value:
        typer.echo(f"{prenspire.__version__}")
        raise typer.Exit()


app = typer.Typer(help="Export EnSpire measurements to csv tables in a directory.")
ver = typer.Option(None, "--version", callback=version)


@app.command()
def enspire(
    csv: str,
    out: Optional[str] = "Meas",
    verbose: bool = False,
    version: Optional[bool] = ver,
) -> None:
    """Save spectra as csv tables from EnSpire xls file."""
    ef = EnspireFile(csv, verbose=verbose)
    ef.extract_measurements(verbose=verbose)
    ef.export_measurements(out)
