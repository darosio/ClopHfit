"""Parse EnSpire files and optionally build titrations using a note file."""

from clophfit.prenspire.mcr import McrFit, mcr_als, mcr_titration
from clophfit.prenspire.prenspire import (
    CsvLineError,
    EnspireFile,
    ManyLinesFoundError,
    Note,
    merge_enspire_files,
    verbose_print,
)
from clophfit.prenspire.thermo import (
    VanthoffCurves,
    VanthoffFit,
    VanthoffModel,
    fit_vanthoff_model,
    fit_vanthoff_model_bayes,
    titrant_kind,
    vanthoff_curves,
    vanthoff_titration,
)

__all__ = [
    "CsvLineError",
    "EnspireFile",
    "ManyLinesFoundError",
    "McrFit",
    "Note",
    "VanthoffCurves",
    "VanthoffFit",
    "VanthoffModel",
    "fit_vanthoff_model",
    "fit_vanthoff_model_bayes",
    "mcr_als",
    "mcr_titration",
    "merge_enspire_files",
    "titrant_kind",
    "vanthoff_curves",
    "vanthoff_titration",
    "verbose_print",
]
