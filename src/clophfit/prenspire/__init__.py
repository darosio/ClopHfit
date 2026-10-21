"""Parse EnSpire files and optionally build titrations using a note file."""

from clophfit.prenspire.mcr import McrFit, mcr_als, mcr_titration
from clophfit.prenspire.prenspire import (
    CsvLineError,
    EnspireFile,
    ManyLinesFoundError,
    Note,
    verbose_print,
)
from clophfit.prenspire.thermo import (
    VanthoffFit,
    VanthoffModel,
    fit_vanthoff_model,
    titrant_kind,
    vanthoff_titration,
)

__all__ = [
    "CsvLineError",
    "EnspireFile",
    "ManyLinesFoundError",
    "McrFit",
    "Note",
    "VanthoffFit",
    "VanthoffModel",
    "fit_vanthoff_model",
    "mcr_als",
    "mcr_titration",
    "titrant_kind",
    "vanthoff_titration",
    "verbose_print",
]
