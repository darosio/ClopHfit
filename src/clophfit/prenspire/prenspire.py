"""Parses EnSpire data files."""

from __future__ import annotations

import collections
import csv
import typing
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyparsing

from clophfit import __enspire_out_dir__
from clophfit.prtecan import lookup_listoflines

# MAYBE: kd1.csv kd2.csv kd3.csv kd1-nota kd2-nota kd3-nota --> Titration


def verbose_print(verbose: int) -> typing.Callable[..., typing.Any]:
    """Return print function when verbose output is True."""
    if verbose:
        return print
    else:
        return lambda *_, **__: None


class ManyLinesFoundError(Exception):
    """Raised when there are multiple lines containing a specified search string."""


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
    >>> from pathlib import Path
    >>> ef = EnspireFile(Path("tests/EnSpire/h148g-spettroC.csv"), verbose=0)
    >>> ef.measurements["A"]["lambda"][2]
    274.0
    """

    file: Path
    verbose: int = 0
    #: General metadata.
    metadata: dict[str, str | list[str]] = field(default_factory=dict, init=False)
    #: Spectra and metadata for each label, such as "MeasB".
    measurements: dict[str, typing.Any] = field(default_factory=dict, init=False)
    #: List of exported wells.
    wells: list[str] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        """Complete initialization."""
        verboseprint = verbose_print(self.verbose)
        csvl = self._read_csv_file(self.file, verboseprint)
        ini, fin = self._find_data_indices(csvl, verboseprint)
        self._ini, self._fin = ini, fin
        self._check_csvl_format(csvl, ini, fin, verboseprint)
        self._wells_platemap, self._platemap = self._extract_platemap(
            csvl[fin + 1 :], verboseprint
        )
        self.metadata = self._create_metadata(csvl[0 : ini - 2], csvl[fin + 1 :])
        self.wells, self.measurements = self._extract_measurements(
            csvl[ini - 1 : fin], csvl[fin + 1 :], verboseprint
        )

    def export_measurements(self, out_dir: Path = Path(__enspire_out_dir__)) -> None:
        """Save measurements, metadata and plots into out_dir."""
        out_dir.mkdir(parents=True, exist_ok=True)
        for m in self.measurements:
            # Prepare data dictionary and construct DataFrame
            data_dict = {"lambda": self.measurements[m]["lambda"]}
            data_dict.update({w: self.measurements[m][w] for w in self.wells})
            dfdata = pd.DataFrame(data=data_dict).set_index("lambda")
            basename = f"{self.file.stem}_{m}"
            if (out_dir / f"{basename}.csv").exists():
                timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")  # noqa: UP017 [for py3.10]
                basename = f"{self.file.stem}_{m}_{timestamp}"
            csv_file = out_dir / f"{basename}.csv"
            md_file = out_dir / f"{basename}.json"
            png_file = out_dir / f"{basename}.png"
            dfdata.to_csv(csv_file)
            pd.DataFrame([self.measurements[m]["metadata"]]).to_json(md_file, indent=4)
            # Create and save plot
            fig, ax = plt.subplots()
            dfdata.plot(title=m, legend=False, ax=ax)
            plt.savefig(png_file)
            plt.close(fig)

    # Helpers
    def _read_csv_file(
        self, file: Path, verboseprint: typing.Callable[..., typing.Any]
    ) -> list[list[str]]:
        """Read EnSpire exported file into csvl."""
        csvl = list(csv.reader(file.open(encoding="iso-8859-1"), dialect="excel-tab"))
        verboseprint("read file csv")
        return csvl

    def _find_data_indices(
        self, csvl: list[list[str]], verboseprint: typing.Callable[..., typing.Any]
    ) -> tuple[int, int]:
        """Find the indices of the data blocks in the input file."""
        inil = lookup_listoflines(csvl, pattern="Well", col=0)
        if len(inil) == 0:
            msg = f"No line starting with ['Well', ...] found in {csvl[:9]}"
            raise CsvLineError(msg)
        elif len(inil) == 1:
            ini = 1 + inil[0]
        else:
            msg = f"Multiple lines starting with ['Well', ...] in {csvl[:9]}"
            raise CsvLineError(msg)
        verboseprint("ini =", ini)
        fin = -1
        fin += lookup_listoflines(csvl, pattern="Basic assay information ", col=0)[0]
        verboseprint("fin =", fin)
        return ini, fin

    def _check_csvl_format(
        self,
        csvl: list[list[str]],
        ini: int,
        fin: int,
        verboseprint: typing.Callable[..., typing.Any],
    ) -> None:
        """Check csv format around ini and fin."""
        if not (csvl[ini - 3] == csvl[ini - 2] == []):
            msg = "Expecting two empty lines before _ini"
            raise CsvLineError(msg)
        if csvl[fin] != []:
            msg = "Expecting an empty line after _fin"
            raise CsvLineError(msg)
        verboseprint("checked csv format around ini and fin")

    def _extract_platemap(
        self, post: list[list[str]], verboseprint: typing.Callable[..., typing.Any]
    ) -> tuple[list[str], list[list[str]]]:
        """Extract well list and Platemap from _metadata_post.

        Parameters
        ----------
        post: list[list[str]]
            List of lines of metadata post containing plate map.
        verboseprint : typing.Callable[..., typing.Any]
            Function to print verbose information.

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
        idx = lookup_listoflines(post, pattern="Platemap:", col=0)[0]
        if "01" not in post[idx + 3]:
            msg = "stop: Platemap format unexpected"
            raise CsvLineError(msg)
        platemap: list[list[str]] = []
        for i in range(idx + 4, len(post)):
            if not post[i]:
                break
            platemap.append(post[i])
        wells: list[str] = [
            f"{r[0]}{c:02}" for r in platemap for c in range(1, len(r)) if r[c].strip()
        ]
        verboseprint("Created attributes _wells_platemap and _platemap.")
        return wells, platemap

    def _create_metadata(
        self, pre: list[list[str]], post: list[list[str]]
    ) -> dict[str, str | list[str]]:
        """Create metadata dictionary."""
        pre_md_start_line = 3
        pre_md_end_line = 8
        metadata: dict[str, str | list[str]] = {}
        for i in range(pre_md_start_line, pre_md_end_line):
            metadata[pre[1][i]] = pre[2][i]
        metadata["Protocol name"] = post[7][4]
        metadata["Exported data"] = next(
            ll[4]
            for ll in [line for line in post if line[:-1]]
            if ll[0] == "Exported data"
        )
        metadata["warnings"] = [
            line[0] for line in post if len(line) == 1 and "WARNING:" in line[0]
        ]
        return metadata

    def _extract_measurements(
        self,
        csvl_data: list[list[str]],
        csvl_post: list[list[str]],
        verboseprint: typing.Callable[..., typing.Any],
    ) -> tuple[list[str], dict[str, typing.Any]]:
        """Extract the measurements dictionary.

        For each measurement label extracts metadata, lambda and for each well the spectrum.

        Parameters
        ----------
        csvl_data : list[list[str]]
            Lines of csvl containing data.
        csvl_post : list[list[str]]
            Lines of csvl containing metadata after data.
        verboseprint : typing.Callable[..., typing.Any]
            Function to print verbose information.

        Returns
        -------
        tuple[list[str], dict[str, typing.Any]]
            A tuple containing a list of wells and the measurements dictionary.

        Raises
        ------
        CsvLineError
            When an error occurs during the extraction process.
        """
        measurements = self._parse_measurements_metadata(csvl_post, verboseprint)
        header = csvl_data[0]
        if not self._check_header_measurements_keys(header, measurements, verboseprint):
            msg = "check header and measurements.keys() FAILED."
            raise CsvLineError(msg)
        columns = [r.replace(":", "") for r in header]
        dfdata = pd.DataFrame(csvl_data[1:], columns=columns)
        w = dfdata.drop_duplicates(["Well"])
        wells = w.Well.tolist()
        if wells != self._wells_platemap:
            msg = "well_list from data_list and platemap differ. It might be that you did not export data for all acquired wells"
            warnings.warn(msg, stacklevel=2)

        # Monochromator is expected to be either Exc or Ems
        for k, measurement in measurements.items():
            label = f"Meas{k}"
            heading = collections.namedtuple("heading", "ex em res")
            head = heading(
                f"{label}WavelengthExc", f"{label}WavelengthEms", f"{label}Result"
            )
            # excitation spectra must have only one emission wavelength
            if measurement["metadata"]["Monochromator"] == "Excitation":
                x = [r for r in dfdata[head.em] if r]
                c = collections.Counter(x)
                if (
                    len(c) != 1
                    or next(iter(c.keys())) != measurement["metadata"]["Wavelength"]
                ):
                    msg = f"Excitation spectra with unexpected emission in {label}"
                    raise CsvLineError(msg)
                measurement["lambda"] = [
                    float(r) for r in dfdata[head.ex][dfdata.Well == wells[0]] if r
                ]
            # emission spectra must have only one excitation wavelength
            elif measurement["metadata"]["Monochromator"] == "Emission":
                x = [r for r in dfdata[head.ex] if r]
                c = collections.Counter(x)
                if (
                    len(c) != 1
                    or next(iter(c.keys())) != measurement["metadata"]["Wavelength"]
                ):
                    msg = f"Emission spectra with unexpected excitation in {label}"
                    raise CsvLineError(msg)
                measurement["lambda"] = [
                    float(r) for r in dfdata[head.em][dfdata.Well == wells[0]] if r
                ]
            else:
                msg = f'Unknown "Monochromator": {measurement["metadata"]["Monochromator"]} in {label}'
                raise CsvLineError(msg)
            for w in wells:
                measurement[w] = [
                    float(r) for r in dfdata[head.res][dfdata.Well == w] if r
                ]
        return wells, measurements

    def _parse_measurements_metadata(
        self, csvl_post: list[list[str]], verboseprint: typing.Callable[..., typing.Any]
    ) -> dict[str, typing.Any]:
        """Initialize measurements with metadata for each label."""
        pyparsing.ParserElement.setDefaultWhitespaceChars(" \t")

        def line(keyword: str) -> pyparsing.ParserElement:
            EOL = pyparsing.LineEnd().suppress()  # type: ignore # noqa: N806
            w = pyparsing.Word(pyparsing.alphanums + ".\u00b0%")  # . | deg | %
            return (
                pyparsing.ZeroOrMore(pyparsing.White(" \t")).suppress()
                + pyparsing.Keyword(keyword)("name")
                + pyparsing.ZeroOrMore(pyparsing.White(" \t")).suppress()
                + w("value")
                + pyparsing.Optional(w)
                + EOL
            )

        measurements: dict[str, typing.Any] = {}
        temp = [0.0]
        meas_key = [""]

        def aa(tokens: pyparsing.ParseResults) -> None:
            name = tokens[0]
            value = tokens[1]
            verboseprint(name, "=", value)
            if name == "Measurement chamber temperature":
                temp[0] = value
                return
            if name == "Meas":
                meas_key[0] = value
                if value not in measurements:
                    measurements[meas_key[0]] = {}
                    measurements[value]["metadata"] = {}
                    measurements[value]["metadata"]["temp"] = temp[0]
                return
            measurements[meas_key[0]]["metadata"][name] = value

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
        ps1 = ["\t".join(line) for line in csvl_post]
        ps2 = "\n".join(ps1)
        pr.searchString(ps2)
        return measurements

    def _check_header_measurements_keys(
        self,
        headerdata: list[str],
        measurements: dict[str, typing.Any],
        verboseprint: typing.Callable[..., typing.Any],
    ) -> bool:
        """Check header and measurements.keys()."""
        counter_constant = 3  # Not sure, maybe for md with units. <Exc, Ems, F>
        meas = [line.split(":")[0].replace("Meas", "") for line in headerdata]
        b = {k for k, v in collections.Counter(meas).items() if v == counter_constant}
        a = set(measurements.keys())
        verboseprint("check header and measurements.keys()", a == b, a, b)
        return a == b


@dataclass
class Note:
    """Read and processes a Note csv file.

    This class is responsible for handling Experimental Note files
    that describe an EnSpire experiment collecting spectrum.

    Parameters
    ----------
    fpath : Path
        The path to the Experimental Note file to be processed.
    verbose : int
        Level of verbosity; 0 is silent, higher values are more verbose (default=0).

    Example
    -------
    >>> from clophfit.prenspire import Note
    >>> n = Note("tests/EnSpire/h148g-spettroC-nota.csv")
    >>> n.wells[2]
    'A03'
    """

    fpath: Path
    verbose: int = 0
    #: A list of wells generated from the note file.
    wells: list[str] = field(init=False, default_factory=list)
    # A list of lines extracted from the note file.
    _note: pd.DataFrame = field(init=False, default_factory=pd.DataFrame)
    #: A list of titrations extracted from the note file.
    titrations: dict[typing.Any, typing.Any] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        """Complete the initialization generating wells and _note_list."""
        verboseprint = verbose_print(self.verbose)
        with Path(self.fpath).open("r", newline="") as file:
            sample_data = file.read(1024)  # Read a sample of the CSV data
        dialect = typing.cast(csv.Dialect, csv.Sniffer().sniff(sample_data))
        self._note = pd.read_csv(self.fpath, dialect=dialect)
        self.wells: list[str] = np.array(self._note)[:, 0].tolist()
        verboseprint(f"Wells {self.wells[:2]}...{self.wells[-2:]} generated.")

    # MAYBE: Add buffer subtraction logic to prenspire Note.

    def build_titrations(self, ef: EnspireFile) -> None:
        """Extract titrations from the given ef (_note file like: <well, pH, Cl>)."""
        df_no_buffer = self._note.query('Name != "buffer"')
        threshold = 3
        grouped0 = df_no_buffer.groupby("Name")
        titrations: dict[typing.Any, typing.Any] = {}
        for name0, group0 in grouped0:
            grouped1 = group0.groupby("Temp")
            titrations[name0] = {}
            for name1, group1 in grouped1:
                titrations[name0][name1] = {}
                # Group by 'pH' and 'Cl' and keep only groups with more than 'threshold' rows.
                for grouping in ["pH", "Cl"]:
                    grouped2 = group1.groupby(grouping)
                    for name2, group2 in grouped2:
                        if len(group2) > threshold:
                            grouped3 = group2.groupby("Labels")
                            for name3, group3 in grouped3:
                                wells = group3["Well"].to_list()
                                meas_dict = {}
                                for label in str(name3).split():
                                    d = {w: ef.measurements[label][w] for w in wells}
                                    value_df = pd.DataFrame(
                                        d,
                                        index=ef.measurements[label]["lambda"],
                                        columns=wells,
                                    )
                                    if group3["pH"].nunique() > group3["Cl"].nunique():
                                        value_df.columns = pd.Index(
                                            group3["pH"].to_list()
                                        )
                                    else:
                                        value_df.columns = pd.Index(
                                            group3["Cl"].to_list()
                                        )
                                    meas_dict[label] = value_df
                                titrations[name0][name1][f"{grouping}_{name2}"] = (
                                    meas_dict
                                )
        self.titrations = titrations
