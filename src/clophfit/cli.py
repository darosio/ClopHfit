"""Command-line interface."""

import typer

from clophfit import binding

from . import __version__

app = typer.Typer()


# @typer.version_option(version=__version__)
def version_callback(value: bool) -> None:
    """Version callback function."""
    if value:
        typer.echo(f"{__version__}")
        raise typer.Exit()


version = typer.Option(None, "--version", callback=version_callback, is_eager=True)


@app.command("ci")
def ciao() -> None:
    """Temporary placeholder."""
    print("ciao")


@app.command("eq1")
def eq1(
    kd1: float,  # Must be lowercase.
    pka: float,
    ph: float,
    version: bool = version,
) -> None:
    """pH-deps for Kd."""
    print(binding.kd(Kd1=kd1, pKa=pka, pH=ph))


def run() -> None:
    """Run commands."""
    app()
