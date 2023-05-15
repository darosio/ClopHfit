"""
Parse Tecan files and fit files that are grouped into titrations.

A 'titration' in this context is defined within a list.pH or list.cl file. These files describe the properties and parameters of the titration experiment.

The command-line interface 'prtecan' is used to construct 96 distinct titrations and export the results to .txt files. This allows for further analysis or visualization of the titration data.

In cases where two label blocks exist in the Tecan files, the module performs a 'global fitting' operation. Global fitting is a method used in data analysis to best fit a model to multiple data sets simultaneously, under the assumption that some underlying parameters are shared between the data sets.

Please consult the documentation for each function in this module for more detailed information on their operation and use.

"""
from clophfit.prtecan.prtecan import FitFirstError
from clophfit.prtecan.prtecan import Labelblock
from clophfit.prtecan.prtecan import LabelblocksGroup
from clophfit.prtecan.prtecan import Metadata
from clophfit.prtecan.prtecan import PlateScheme
from clophfit.prtecan.prtecan import Tecanfile
from clophfit.prtecan.prtecan import TecanfilesGroup
from clophfit.prtecan.prtecan import Titration
from clophfit.prtecan.prtecan import TitrationAnalysis
from clophfit.prtecan.prtecan import calculate_conc
from clophfit.prtecan.prtecan import dilution_correction
from clophfit.prtecan.prtecan import extract_metadata
from clophfit.prtecan.prtecan import lookup_listoflines
from clophfit.prtecan.prtecan import merge_md
from clophfit.prtecan.prtecan import read_xls
from clophfit.prtecan.prtecan import strip_lines

__all__ = [
    "FitFirstError",
    "Labelblock",
    "LabelblocksGroup",
    "Metadata",
    "PlateScheme",
    "Tecanfile",
    "TecanfilesGroup",
    "Titration",
    "TitrationAnalysis",
    "calculate_conc",
    "dilution_correction",
    "extract_metadata",
    "lookup_listoflines",
    "merge_md",
    "read_xls",
    "strip_lines",
]
