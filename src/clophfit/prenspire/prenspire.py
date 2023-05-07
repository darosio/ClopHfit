"""Parses EnSpire data files."""
from __future__ import annotations

import csv
import warnings
from collections import Counter
from collections import namedtuple
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Sequence

import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import pandas as pd
import pyparsing  # TODO: indentBlock, ParseResults

# TODO: kd1.csv kd2.csv kd3.csv kd1-nota kd2-nota kd3-nota --> Titration
# TODO: Titration.data ['A' 'B' 'C'] -- global fit


def verbose_print(verbose: int) -> None | Callable[..., Any]:
    """Print when verbose output is True."""
    return print if verbose else lambda *_, **__: None


class ManyLinesFoundError(Exception):
    """Raised when there are multiple lines containing a specified search string."""

    pass


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


class EnspireFile:
    """Read an EnSpire-generated csv file.

    Read the files and check the formats.
    extract_measurements(): create the measurements dictionary structure storing
    all data and metadata (description of measurement).

    Parameters
    ----------
    file : str
        filename
    **kwargs : dict
        keywords

    Raises
    ------
    Exception
        If unexpected format is found.

    Examples
    --------
    >>> from prenspire.prenspire import EnspireFile
    >>> ef = EnspireFile("tests/EnSpire/h148g-spettroC.csv", verbose=0)
    >>> ef.extract_measurements()
    >>> ef.measurements['A']['lambda'][2]
    274.0

    """

    def __init__(self, file: Path, verbose: int = 0) -> None:
        def get_data_ini(lines: list[list[str]]) -> int:
            """Get index of the line ['Well', 'Sample', ...].

            Check for the presence of a unique ['well', 'sample', ...] line.

            Parameters
            ----------
            lines : list
                List of lines.

            Raises
            ------
            Exception
                If not unique or absent.

            Returns
            -------
            int

            """
            count = 0
            for i, line in enumerate(lines):
                if len(line) >= 2:
                    if line[0:2] == ["Well", "Sample"]:
                        count += 1
                        idx = i
            if count == 0:
                raise Exception("No line starting with ['Well', 'Sample',...]")
            elif count == 1:
                return idx
            else:  # count > 1
                raise Exception("2 lines starting with ['Well', 'Sample',...]")

        def get_list_from_platemap() -> tuple[list[str], list[list[str]]]:
            """Get well_list from Platemap contained in metadata_post.

            Returns
            -------
                also Platemap, at this time.

            Raises
            ------
            Exception
                If the number of plate columns is not equal to 12.

            """
            lines = self._metadata_post
            idx = line_index(lines, ["Platemap:"])
            if lines[idx + 3] != [
                "",
                "01",
                "02",
                "03",
                "04",
                "05",
                "06",
                "07",
                "08",
                "09",
                "10",
                "11",
                "12",
                "",
            ]:
                raise Exception("stop: Platemap format unexpected")
            plate = lines[idx + 4 : idx + 12]
            p = []
            for r in plate:
                letter = r[0]
                for c in range(1, len(r)):
                    # strip out white spaces;fix "no background info available"
                    if r[c].strip():
                        p.append(f"{letter}{c:02}")
            return p, plate

        def create_metadata() -> None:
            """Create metadata dictionary."""
            self.metadata: dict[str, str | list[str]] = {}
            self.metadata[pre[1][3]] = pre[2][3]
            self.metadata[pre[1][4]] = pre[2][4]
            self.metadata[pre[1][5]] = pre[2][5]
            self.metadata[pre[1][6]] = pre[2][6]
            self.metadata[pre[1][7]] = pre[2][7]
            self.metadata["Protocol name"] = self._metadata_post[7][4]
            self.metadata["Exported data"] = [
                ll[4]
                for ll in [line for line in self._metadata_post if line[:-1]]
                if ll[0] == "Exported data"
            ][0]
            self.metadata["warnings"] = [
                line[0]
                for line in self._metadata_post
                if len(line) == 1 and "WARNING:" in line[0]
            ]

        verboseprint = verbose_print(verbose)
        csvl = list(csv.reader(file.open(encoding="iso-8859-1"), dialect="excel-tab"))
        verboseprint("read file csv")  # type: ignore
        # TODO: try prparser.line_index()
        self._ini = 1 + get_data_ini(csvl)
        verboseprint("ini =", self._ini)  # type: ignore
        self._fin = -1 + line_index(csvl, ["Basic assay information "])
        verboseprint("fin =", self._fin)  # type: ignore
        # check csv format around ini and fin
        if not (csvl[self._ini - 3] == csvl[self._ini - 2] == []):
            raise Exception("Expecting two empty lines before _ini")
        if not (csvl[self._fin] == []):
            raise Exception("Expecting an empty line after _fin")
        verboseprint("checked csv format around ini and fin")  # type: ignore
        pre = csvl[0 : self._ini - 2]  # -3
        verboseprint("saved metadata_pre attribute")  # type: ignore
        self._data_list = csvl[self._ini - 1 : self._fin]
        verboseprint("saved _data_list attribute")  # type: ignore
        self._metadata_post = csvl[self._fin + 1 :]
        verboseprint("saved metadata_post attribute")  # type: ignore
        self._well_list_platemap, self._platemap = get_list_from_platemap()
        verboseprint("saved _well_list_platemap attribute")  # type: ignore
        create_metadata()
        self._filename = str(file)

    def extract_measurements(self, verbose: int = 0) -> None:
        """Extract the measurements dictionary.

        Add 3 attributes: wells, samples, measurements (as list, list, dict)

        Parameters
        ----------
        verbose : int
            It passes extra parameters.

        Raises
        ------
        Exception
            When something went wrong.
        """
        verboseprint = verbose_print(verbose)
        pyparsing.ParserElement.setDefaultWhitespaceChars(" \t")

        def line(keyword: str) -> Any:
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
            meas = [line.split(":")[0].replace("Meas", "") for line in headerdata]
            b = {k for k, v in Counter(meas).items() if v == 3}
            a = set(self.measurements.keys())
            verboseprint("check header and measurements.keys()", a == b, a, b)  # type: ignore
            return a == b

        headerdata = self._data_list[0]
        if not headerdata_measurementskeys_check():
            raise Exception("check header and measurements.keys() FAILED.")

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
            if not self.wells == self._well_list_platemap:
                warnings.warn(
                    "well_list from data_list and platemap differ. \
                    It might be you did not exported data for all acquired wells"
                )
            return True

        columns = [r.replace(":", "") for r in headerdata]
        df = pd.DataFrame(self._data_list[1:], columns=columns)
        w = df.drop_duplicates(["Well", "Sample"])
        self.wells = w.Well.tolist()
        self.samples = w.Sample.tolist()
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
                x = [r for r in df[head.em] if not r == ""]
                c = Counter(x)
                if (
                    not len(c) == 1
                    or not list(c.keys())[0] == v["metadata"]["Wavelength"]
                ):
                    raise Exception(
                        "Excitation spectra with unexpected emission in " + label
                    )
                v["lambda"] = [
                    float(r)
                    for r in df[head.ex][df.Well == self.wells[0]]
                    if not r == ""
                ]
            # emission spectra must have only one excitation wavelength
            elif v["metadata"]["Monochromator"] == "Emission":
                x = [r for r in df[head.ex] if not r == ""]
                c = Counter(x)
                if (
                    not len(c) == 1
                    or not list(c.keys())[0] == v["metadata"]["Wavelength"]
                ):
                    raise Exception(
                        "Emission spectra with unexpected excitation in " + label
                    )
                v["lambda"] = [
                    float(r)
                    for r in df[head.em][df.Well == self.wells[0]]
                    if not r == ""
                ]
            else:
                raise Exception(
                    'Unknown "Monochromator": '
                    + v["metadata"]["Monochromator"]
                    + " in "
                    + label
                )
            for w in self.wells:
                v[w] = [float(r) for r in df[head.res][df.Well == w] if not r == ""]

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
            file = output_dir / (Path(self._filename).stem + "_" + m + ".csv")
            while file.exists():
                file = file.with_stem(file.stem + "-b")
            dfdata.to_csv(str(file))
            plt.savefig(str(file.with_suffix(".png")))


class ExpNote:
    """Read an Experimental Note file.

    For describing an EnSpire experiment collecting spectrum. Store info as list
    of lines: note_list and well_list.

    Example
    -------
    >>> from prenspire.prenspire import ExpNote
    >>> en = ExpNote("tests/EnSpire/h148g-spettroC-nota")
    >>> en.wells[2]
    'A03'

    """

    def __init__(self, note_file: Path, verbose: int = 0) -> None:
        """Initialize an object."""
        verboseprint = verbose_print(verbose)
        with note_file.open(encoding="iso-8859-1") as f:
            # Differ from pandas because all fields/cells are strings.
            self.note_list = list(csv.reader(f, dialect="excel-tab"))
        verboseprint("read (experimental) note file")  # type: ignore
        self.wells: list[str] = np.array(self.note_list)[1:, 0].tolist()
        verboseprint("wells generated")  # type: ignore

    def check_wells(self, ef: EnspireFile) -> bool:
        """Is (EnspireFile) ef.wells == ExpNote.wells? Return False-or-True."""
        return self.wells == ef.wells

    def build_titrations(self, ef: EnspireFile) -> None:
        """Extract titrations from the given ef (_note file like: <well, pH, Cl>)."""
        # 1 pH titration
        conc_well = [(line[1], line[0]) for line in self.note_list if line[2] == "0"]
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
        self.titrations = [Titration(conc, data, cl=0)]
        # n cl titrations
        self.pH_values = sorted(
            {line[1] for line in self.note_list if line[2].replace(".", "").isnumeric()}
        )
        for ph in self.pH_values:
            conc_well = [
                (line[2], line[0])
                for line in self.note_list
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


class Titration:
    """Store titration data and fit results."""

    def __init__(
        self, conc: Sequence[float], data: dict[str, pd.DataFrame], **kwargs: Any
    ) -> None:
        self.conc = conc
        self.data = data
        if "ph" in kwargs:
            self.ph = kwargs["ph"]
        if "cl" in kwargs:
            self.cl = kwargs["cl"]
        if "func" in kwargs:
            self.func = kwargs["func"]

    def plot(self) -> None:
        """Plot the titration spectra."""
        for m in self.data:
            self.data[m].plot()
