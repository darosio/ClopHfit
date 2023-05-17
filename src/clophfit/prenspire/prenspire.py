"""Parses EnSpire data files."""
from __future__ import annotations

import csv
import warnings
from collections import Counter
from collections import namedtuple
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Sequence

import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import pandas as pd
import pyparsing

# TODO: kd1.csv kd2.csv kd3.csv kd1-nota kd2-nota kd3-nota --> Titration
# TODO: Titration.data ['A' 'B' 'C'] -- global fit


def verbose_print(verbose: int) -> None | Callable[..., Any]:
    """Print when verbose output is True."""
    return print if verbose else lambda *_, **__: None


class ManyLinesFoundError(Exception):
    """Raised when there are multiple lines containing a specified search string."""


def line_index(lines: list[list[str]], search: list[str]) -> int:
    """Index function checking the existence of a unique match.

    It behaves like list.index() but raise an Exception if the search string
    occurs multiple times. Returns 0 otherwise.

    Parameters
    ----------
    lines : list[list[str]]
        List of lines.
    search : list[str]
        The search term.

    Returns
    -------
    int
        The position index of search within lines.

    Raises
    ------
    ManyLinesFoundError
        If search term is present 2 or more times.

    """
    count: int = lines.count(search)
    if count <= 1:
        return lines.index(search)  # [].index ValueError if count == 0
    else:
        raise ManyLinesFoundError("Many " + str(search) + " lines.")


class CsvLineError(Exception):
    """Exception raised when the lines list has issues."""


@dataclass
class EnspireFile:
    """Read an EnSpire-generated csv file.

    Read the files and check the formats.
    extract_measurements(): create the measurements dictionary structure storing
    all data and metadata (description of measurement).

    Parameters
    ----------
    file : Path
        Path to the EnSpire csv file
    verbose : int
        Level of verbosity; 0 is silent, higher values are more verbose (default=0).

    Raises
    ------
    Exception
        If unexpected format is found.

    Examples
    --------
    >>> from clophfit.prenspire import EnspireFile
    >>> ef = EnspireFile("tests/EnSpire/h148g-spettroC.csv", verbose=0)
    >>> ef.extract_measurements()
    >>> ef.measurements['A']['lambda'][2]
    274.0
    """

    file: Path
    verbose: int = 0
    #: General metadata.
    metadata: dict[str, str | list[str]] = field(default_factory=dict, init=False)
    #: Spectra and metadata for each Meas.
    measurements: dict[str, Any] = field(default_factory=dict, init=False)
    # #: List of wells.
    #: wells: list[str] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        """Complete initialization."""
        verboseprint = verbose_print(self.verbose)
        csvl = self._read_csv_file(Path(self.file), self.verbose)
        self._ini, self._fin = self._find_data_indices(csvl)
        verboseprint("ini =", self._ini)  # type: ignore
        verboseprint("fin =", self._fin)  # type: ignore
        self._check_csvl_ini_fin(csvl, self._ini, self._fin)
        verboseprint("checked csv format around ini and fin")  # type: ignore
        pre = csvl[0 : self._ini - 2]  # -3
        verboseprint("saved metadata_pre attribute")  # type: ignore
        self._data_list = csvl[self._ini - 1 : self._fin]
        verboseprint("saved _data_list attribute")  # type: ignore
        self._metadata_post = csvl[self._fin + 1 :]
        verboseprint("saved metadata_post attribute")  # type: ignore
        self._well_list_platemap, self._platemap = self._extract_platemap(
            self._metadata_post
        )
        verboseprint("saved _well_list_platemap attribute")  # type: ignore
        self.metadata = self._create_metadata(pre, self._metadata_post)

    def _read_csv_file(self, file: Path, verbose: int) -> list[list[str]]:
        verboseprint = verbose_print(verbose)
        csvl = list(csv.reader(file.open(encoding="iso-8859-1"), dialect="excel-tab"))
        verboseprint("read file csv")  # type: ignore
        return csvl

    def _find_data_indices(self, csvl: list[list[str]]) -> tuple[int, int]:
        def get_data_ini(lines: list[list[str]]) -> int:
            """Find the line index containing Well and Sample headers in a list of lines.

            Get index of the line ['Well', 'Sample', ...].
            Check for the presence of a unique ['well', 'sample', ...] line.

            Parameters
            ----------
            lines : list
                List of lines.

            Raises
            ------
            CsvLineError
                If not unique or absent.

            Returns
            -------
            int

            """
            min_line_length = 2  # for key: value
            count = 0
            for i, line in enumerate(lines):
                if len(line) >= min_line_length and line[:1] == ["Well"]:
                    count += 1
                    idx = i
            if count == 0:
                msg = f"No line starting with ['Well', ...] found in {lines[:9]}"
                raise CsvLineError(msg)
            elif count == 1:
                return idx
            else:  # count > 1
                msg = f"Multiple lines starting with ['Well', ...] in {lines[:9]}"
                raise CsvLineError(msg)

        # TODO: try prparser.line_index()
        ini = 1 + get_data_ini(csvl)
        fin = -1 + line_index(csvl, ["Basic assay information "])
        return ini, fin

    def _check_csvl_ini_fin(self, csvl: list[list[str]], ini: int, fin: int) -> None:
        """Check csv format around ini and fin."""
        if not (csvl[ini - 3] == csvl[ini - 2] == []):
            msg = "Expecting two empty lines before _ini"
            raise CsvLineError(msg)
        if csvl[fin] != []:
            msg = "Expecting an empty line after _fin"
            raise CsvLineError(msg)

    def _extract_platemap(
        self, post: list[list[str]]
    ) -> tuple[list[str], list[list[str]]]:
        """Extract well list and Platemap from _metadata_post.

        Parameters
        ----------
        lines: list[list[str]]
            List of lines of metadata post containing plate map.

        Returns
        -------
        tuple[list[str], list[list[str]]]
            A tuple containing:
            - A list of well IDs in the format "A01", "A02", etc.
            - A list of lists representing the Platemap.

        Raises
        ------
        CsvLineError
            If the column '01' is not present 3 lines below ['Platemap:'].

        """
        idx: int = line_index(post, ["Platemap:"])
        if "01" not in post[idx + 3]:
            msg = "stop: Platemap format unexpected"
            raise CsvLineError(msg)
        plate: list[list[str]] = []
        for i in range(idx + 4, len(post)):
            if not post[i]:
                break
            plate.append(post[i])
        well_list: list[str] = [
            f"{r[0]}{c:02}" for r in plate for c in range(1, len(r)) if r[c].strip()
        ]
        return well_list, plate

    def _create_metadata(
        self, pre: list[list[str]], post: list[list[str]]
    ) -> dict[str, str | list[str]]:
        """Create metadata dictionary."""
        metadata: dict[str, str | list[str]] = {}
        metadata[pre[1][3]] = pre[2][3]
        metadata[pre[1][4]] = pre[2][4]
        metadata[pre[1][5]] = pre[2][5]
        metadata[pre[1][6]] = pre[2][6]
        metadata[pre[1][7]] = pre[2][7]
        metadata["Protocol name"] = post[7][4]
        metadata["Exported data"] = [
            ll[4]
            for ll in [line for line in post if line[:-1]]
            if ll[0] == "Exported data"
        ][0]
        metadata["warnings"] = [
            line[0] for line in post if len(line) == 1 and "WARNING:" in line[0]
        ]
        return metadata

    def extract_measurements(self, verbose: int = 0) -> None:  # noqa: PLR0915
        """Extract the measurements dictionary.

        Add 3 attributes: wells, samples, measurements (as list, list, dict)

        Parameters
        ----------
        verbose : int
            It passes extra parameters.

        Raises
        ------
        CsvLineError
            When something went wrong.
        """
        verboseprint = verbose_print(verbose)
        pyparsing.ParserElement.setDefaultWhitespaceChars(" \t")

        def line(keyword: str) -> pyparsing.ParserElement:
            EOL = pyparsing.LineEnd().suppress()  # type: ignore # noqa: N806
            w = pyparsing.Word(pyparsing.alphanums + ".\u00B0%")  # . | deg | %
            return (
                pyparsing.ZeroOrMore(pyparsing.White(" \t")).suppress()
                + pyparsing.Keyword(keyword)("name")
                + pyparsing.ZeroOrMore(pyparsing.White(" \t")).suppress()
                + w("value")
                + pyparsing.Optional(w)
                + EOL
            )

        meas: dict[str, Any] = {}
        temp = [0]
        meas_key = ["zz"]

        def aa(tokens: pyparsing.ParseResults) -> None:
            name = tokens[0]
            value = tokens[1]
            verboseprint(name, "=", value)  # type: ignore
            if name == "Measurement chamber temperature":
                temp[0] = value
                return
            if name == "Meas":
                meas_key[0] = value
                if value not in meas.keys():
                    # Initialize new "Measurement"
                    meas[meas_key[0]] = {}
                    meas[value]["metadata"] = {}
                    meas[value]["metadata"]["temp"] = temp[0]
                return
            meas[meas_key[0]]["metadata"][name] = value

        block_lines = (
            line("Measurement chamber temperature")
            | line("Meas")
            | line("Monochromator")
            | line("Min wavelength")
            | line("Max wavelength")
            | line("Wavelength")
            | line("Using of excitation filter")
            | line("Measurement height")
            | line("Number of flashes integrated")
            | line("Number of flashes")
            | line("Flash power")
        )
        pr = block_lines.setParseAction(aa)

        ps1 = ["\t".join(line) for line in self._metadata_post]
        verboseprint("metadata_post ps1 conversion... done")  # type: ignore
        ps2 = "\n".join(ps1)
        verboseprint("metadata_post ps2 conversion... done")  # type: ignore
        pr.searchString(ps2)
        verboseprint("metadata_post pyparsing... done")  # type: ignore
        self.measurements = meas

        def headerdata_measurementskeys_check() -> bool:
            """Check header and measurements.keys()."""
            counter_constant = 3  # Not sure, maybe for md with units.
            meas = [line.split(":")[0].replace("Meas", "") for line in headerdata]
            b = {k for k, v in Counter(meas).items() if v == counter_constant}
            a = set(self.measurements.keys())
            verboseprint("check header and measurements.keys()", a == b, a, b)  # type: ignore
            return a == b

        headerdata = self._data_list[0]
        if not headerdata_measurementskeys_check():
            msg = "check header and measurements.keys() FAILED."
            raise CsvLineError(msg)

        def check_lists() -> bool:
            """Check that lists derived from .csv data and Platemap metadata are identical.

            if not raises an *Exception*.
            Will Raise *Exception* if well_list from csv (data_list) and note
            disagree. Raise *Warning* if well_list from csv (data_list) and
            platemap disagree.

            Returns
            -------
            bool
                Ckeck correctness.

            """
            if self.wells != self._well_list_platemap:
                warnings.warn(
                    "well_list from data_list and platemap differ. It might be you did not exported data for all acquired wells",
                    stacklevel=2,
                )
            return True

        columns = [r.replace(":", "") for r in headerdata]
        dfdata = pd.DataFrame(self._data_list[1:], columns=columns)
        w = dfdata.drop_duplicates(["Well"])
        self.wells = w.Well.tolist()
        check_lists()
        # Monochromator is expected to be either Exc or Ems
        for k, v in self.measurements.items():
            label = f"Meas{k}"
            heading = namedtuple("heading", "ex em res")
            head = heading(
                f"{label}WavelengthExc", f"{label}WavelengthEms", f"{label}Result"
            )
            # excitation spectra must have only one emission wavelength
            if v["metadata"]["Monochromator"] == "Excitation":
                x = [r for r in dfdata[head.em] if r]
                c = Counter(x)
                if len(c) != 1 or list(c.keys())[0] != v["metadata"]["Wavelength"]:
                    msg = f"Excitation spectra with unexpected emission in {label}"
                    raise CsvLineError(msg)
                v["lambda"] = [
                    float(r) for r in dfdata[head.ex][dfdata.Well == self.wells[0]] if r
                ]
            # emission spectra must have only one excitation wavelength
            elif v["metadata"]["Monochromator"] == "Emission":
                x = [r for r in dfdata[head.ex] if r]
                c = Counter(x)
                if len(c) != 1 or list(c.keys())[0] != v["metadata"]["Wavelength"]:
                    msg = f"Emission spectra with unexpected excitation in {label}"
                    raise CsvLineError(msg)
                v["lambda"] = [
                    float(r) for r in dfdata[head.em][dfdata.Well == self.wells[0]] if r
                ]
            else:
                msg = f'Unknown "Monochromator": {v["metadata"]["Monochromator"]} in {label}'
                raise CsvLineError(msg)
            for w in self.wells:
                v[w] = [float(r) for r in dfdata[head.res][dfdata.Well == w] if r]

    def export_measurements(self, output_dir: Path = Path("Meas")) -> None:
        """Create table as DataFrame and plot; save into Meas folder."""
        output_dir.mkdir(parents=True, exist_ok=True)
        for m in self.measurements:
            data_dict = {"lambda": list(map(float, self.measurements[m]["lambda"]))}
            data_dict.update(
                {w: list(map(float, self.measurements[m][w])) for w in self.wells}
            )
            dfdata = pd.DataFrame(data=data_dict).set_index("lambda")
            # Create plot
            dfdata.plot(title=m, legend=False)
            # Save files
            file = output_dir / (self.file.stem + "_" + m + ".csv")
            while file.exists():
                # because with_stem was introduced in py3.9
                file = file.with_name(file.stem + "-b" + file.suffix)
            dfdata.to_csv(str(file))
            plt.savefig(str(file.with_suffix(".png")))


@dataclass
class ExpNote:
    """Read and processes an Experimental Note file.

    This class is responsible for handling Experimental Note files
    that describe an EnSpire experiment collecting spectrum. It reads the files and
    stores the extracted information as a list of lines: note_list and list of wells.

    Parameters
    ----------
    note_file : Path
        The path to the Experimental Note file to be processed.
    verbose : int
        Level of verbosity; 0 is silent, higher values are more verbose (default=0).

    Example
    -------
    >>> from clophfit.prenspire import ExpNote
    >>> en = ExpNote("tests/EnSpire/h148g-spettroC-nota")
    >>> en.wells[2]
    'A03'
    """

    note_file: Path
    verbose: int = 0
    #: A list of wells generated from the note file.
    wells: list[str] = field(init=False, default_factory=list)
    # A list of lines extracted from the note file.
    _note_list: list[list[str]] = field(init=False, default_factory=list)
    #: A list of titrations extracted from the note file.
    titrations: list[Titration] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        """Complete the initialization generating wells and _note_list."""
        verboseprint = verbose_print(self.verbose)
        with Path(self.note_file).open(encoding="iso-8859-1") as f:
            # Differ from pandas because all fields/cells are strings.
            self._note_list = list(csv.reader(f, dialect="excel-tab"))
        verboseprint("Read (experimental) note file")  # type: ignore
        self.wells: list[str] = np.array(self._note_list)[1:, 0].tolist()
        verboseprint("Wells generated")  # type: ignore

    def build_titrations(self, ef: EnspireFile) -> None:
        """Extract titrations from the given ef (_note file like: <well, pH, Cl>)."""
        # 1 pH titration
        conc_well = [(line[1], line[0]) for line in self._note_list if line[2] == "0"]
        conc = [float(tpl[0]) for tpl in conc_well]
        well = [tpl[1] for tpl in conc_well]
        data = {}
        for m, measurement in ef.measurements.items():
            data[m] = pd.DataFrame(
                data=np.transpose([list(map(float, measurement[w])) for w in well]),
                columns=[conc, well],
                index=pd.Index(
                    data=list(map(float, measurement["lambda"])), name="lambda"
                ),
            )
        self.titrations = [Titration(conc, data, cl="0")]
        # n cl titrations
        self.pH_values = sorted(
            {
                line[1]
                for line in self._note_list
                if line[2].replace(".", "").isnumeric()
            }
        )
        for ph in self.pH_values:
            conc_well = [
                (line[2], line[0])
                for line in self._note_list
                if line[1] == ph and line[2].replace(".", "").isnumeric()
            ]
            conc = [float(tpl[0]) for tpl in conc_well]
            well = [tpl[1] for tpl in conc_well]
            data = {}
            for m, measurement in ef.measurements.items():
                data[m] = pd.DataFrame(
                    data=np.transpose([list(map(float, measurement[w])) for w in well]),
                    columns=[conc, well],
                    index=pd.Index(
                        data=list(map(float, measurement["lambda"])),
                        name="lambda",
                    ),
                )
            self.titrations.append(Titration(conc, data, ph=ph))

        # TODO: BUFFER


# TODO: +titrations attribute even though created by build_titrations()
# TODO: PlateScheme?
# TODO: Metadata
# TODO: _get_init


class Titration:
    """Store titration data and fit results."""

    def __init__(
        self,
        conc: Sequence[float],
        data: dict[str, pd.DataFrame],
        cl: str | None = None,
        ph: str | None = None,
    ) -> None:
        self.conc = conc
        self.data = data
        if ph:
            self.ph = ph
        if cl:
            self.cl = cl

    def plot(self) -> None:
        """Plot the titration spectra."""
        for m in self.data:
            self.data[m].plot()
