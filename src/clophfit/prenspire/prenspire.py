"""Parses EnSpire data files."""

from __future__ import annotations

import collections
import copy
import csv
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyparsing

from clophfit import __enspire_out_dir__
from clophfit.utils import lookup_listoflines

# MAYBE: kd1.csv kd2.csv kd3.csv kd1-nota kd2-nota kd3-nota --> Titration

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from types import BuiltinFunctionType


def verbose_print(verbose: int) -> BuiltinFunctionType | Callable[..., None]:
    """Return print function when verbose output is True."""
    return print if verbose else lambda *_, **__: None


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
    verbose :

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
    """Level of verbosity; 0 is silent."""
    metadata: dict[str, str | list[str]] = field(default_factory=dict, init=False)
    """General metadata."""
    measurements: dict[str, Any] = field(default_factory=dict, init=False)
    """Spectra and metadata for each label, such as "MeasB"."""
    wells: list[str] = field(default_factory=list, init=False)
    """List of exported wells."""

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

    def export_measurements(
        self,
        out_dir: Path = Path(__enspire_out_dir__),
        *,
        mode: str = "auto",
    ) -> list[str]:
        """Write each measurement's table, metadata and figure into ``out_dir``.

        The file names are canonical - ``<export stem>_<label>.csv/json/png`` - so
        a second run over the same export lands on the same files instead of
        accumulating timestamped copies. Running several fitting methods on one
        plate used to leave five copies of every table and figure in the output
        directory, because an existing file made the writer append a timestamp.

        Parameters
        ----------
        out_dir : Path
            Destination directory; created when missing.
        mode : str
            ``"auto"`` (default) rewrites a measurement only when the export is
            newer than what was written (or the files are missing), ``"always"``
            rewrites every time, ``"never"`` writes nothing.

        Returns
        -------
        list[str]
            Base names written, in measurement order; empty when everything was
            already up to date.

        Raises
        ------
        ValueError
            If ``mode`` is not one of the three accepted values.
        """
        if mode not in {"auto", "always", "never"}:
            msg = f"export mode must be auto, always or never, not {mode!r}"
            raise ValueError(msg)
        if mode == "never":
            return []
        out_dir.mkdir(parents=True, exist_ok=True)
        written: list[str] = []
        source_mtime = self.file.stat().st_mtime
        for measurement_name, measurement_data in self.measurements.items():
            basename = f"{self.file.stem}_{measurement_name}"
            csv_file = out_dir / f"{basename}.csv"
            md_file = out_dir / f"{basename}.json"
            png_file = out_dir / f"{basename}.png"
            targets = (csv_file, md_file, png_file)
            if mode == "auto" and all(
                path.exists() and path.stat().st_mtime >= source_mtime
                for path in targets
            ):
                continue
            data_dict: dict[str, Any] = {"lambda": measurement_data["lambda"]}
            data_dict.update({well: measurement_data[well] for well in self.wells})
            dfdata = pd.DataFrame(data=data_dict).set_index("lambda")
            dfdata.to_csv(csv_file)
            pd.DataFrame([measurement_data["metadata"]]).to_json(md_file, indent=4)
            fig, ax = plt.subplots()
            dfdata.plot(title=measurement_name, legend=False, ax=ax)
            fig.savefig(png_file)
            plt.close(fig)
            written.append(basename)
        return written

    # Helpers
    @staticmethod
    def _read_csv_file(file: Path, verboseprint: Callable[..., Any]) -> list[list[str]]:
        """Read EnSpire exported file into csvl."""
        with file.open(encoding="iso-8859-1") as f:
            csvl = list(csv.reader(f, dialect="excel-tab"))
        verboseprint("read file csv")
        return csvl

    @staticmethod
    def _find_data_indices(
        csvl: list[list[str]], verboseprint: Callable[..., Any]
    ) -> tuple[int, int]:
        """Find the indices of the data blocks in the input file."""
        inil = lookup_listoflines(csvl, pattern="Well", col=0)
        if len(inil) == 0:
            msg = f"No line starting with ['Well', ...] found in {csvl[:9]}"
            raise CsvLineError(msg)
        if len(inil) == 1:
            ini = 1 + inil[0]
        else:
            msg = f"Multiple lines starting with ['Well', ...] in {csvl[:9]}"
            raise CsvLineError(msg)
        verboseprint("ini =", ini)
        fin = -1
        fin += lookup_listoflines(csvl, pattern="Basic assay information ", col=0)[0]
        verboseprint("fin =", fin)
        return ini, fin

    @staticmethod
    def _check_csvl_format(
        csvl: list[list[str]], ini: int, fin: int, verboseprint: Callable[..., Any]
    ) -> None:
        """Check csv format around ini and fin."""
        if not csvl[ini - 3] == csvl[ini - 2] == []:
            msg = "Expecting two empty lines before _ini"
            raise CsvLineError(msg)
        if csvl[fin] != []:
            msg = "Expecting an empty line after _fin"
            raise CsvLineError(msg)
        verboseprint("checked csv format around ini and fin")

    @staticmethod
    def _extract_platemap(
        post: list[list[str]], verboseprint: Callable[..., Any]
    ) -> tuple[list[str], list[list[str]]]:
        """Extract well list and Platemap from _metadata_post.

        Parameters
        ----------
        post: list[list[str]]
            List of lines of metadata post containing plate map.
        verboseprint : Callable[..., Any]
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

    @staticmethod
    def _create_metadata(
        pre: list[list[str]], post: list[list[str]]
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
        verboseprint: Callable[..., Any],
    ) -> tuple[list[str], dict[str, Any]]:
        """Extract the measurements dictionary.

        For each measurement label extracts metadata, lambda and for each well
        the spectrum.

        Parameters
        ----------
        csvl_data : list[list[str]]
            Lines of csvl containing data.
        csvl_post : list[list[str]]
            Lines of csvl containing metadata after data.
        verboseprint : Callable[..., Any]
            Function to print verbose information.

        Returns
        -------
        tuple[list[str], dict[str, Any]]
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
            warnings.warn(
                (
                    "well_list from data_list and platemap differ. It might be that "
                    "you did not export data for all acquired wells"
                ),
                stacklevel=2,
            )
        # Monochromator is expected to be either Exc or Ems
        for k, measurement in measurements.items():
            label = f"Meas{k}"

            class Heading(NamedTuple):
                ex: str
                em: str
                res: str

            heading = Heading
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
                monochromator = measurement["metadata"]["Monochromator"]
                msg = f'Unknown "Monochromator": {monochromator} in {label}'
                raise CsvLineError(msg)
            for w in wells:
                measurement[w] = [
                    float(r) for r in dfdata[head.res][dfdata.Well == w] if r
                ]
        return wells, measurements

    @staticmethod
    def _parse_measurements_metadata(
        csvl_post: list[list[str]], verboseprint: Callable[..., Any]
    ) -> dict[str, Any]:
        """Initialize measurements with metadata for each label."""
        pe = pyparsing.ParserElement
        if hasattr(pe, "set_default_whitespace_chars"):
            pe.set_default_whitespace_chars(" \t")
        else:  # pragma: no cover
            pe.setDefaultWhitespaceChars(" \t")

        def line(keyword: str) -> pyparsing.ParserElement:
            EOL = pyparsing.LineEnd().suppress()  # ruff: ignore[non-lowercase-variable-in-function]
            w = pyparsing.Word(pyparsing.alphanums + ".\u00b0%")  # . | deg | %
            return (
                pyparsing.ZeroOrMore(pyparsing.White(" \t")).suppress()
                + pyparsing.Keyword(keyword)("name")
                + pyparsing.ZeroOrMore(pyparsing.White(" \t")).suppress()
                + w("value")
                + pyparsing.Optional(w)
                + EOL
            )

        measurements: dict[str, Any] = {}
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
        if hasattr(block_lines, "set_parse_action"):
            pr = block_lines.set_parse_action(aa)
        else:  # pragma: no cover
            pr = block_lines.setParseAction(aa)
        ps1 = ["\t".join(line) for line in csvl_post]
        ps2 = "\n".join(ps1)
        if hasattr(pr, "search_string"):
            pr.search_string(ps2)
        else:  # pragma: no cover
            pr.searchString(ps2)
        return measurements

    @staticmethod
    def _check_header_measurements_keys(
        headerdata: list[str],
        measurements: dict[str, Any],
        verboseprint: Callable[..., Any],
    ) -> bool:
        """Check header and measurements.keys()."""
        counter_constant = 3  # Not sure, maybe for md with units. <Exc, Ems, F>
        meas = [line.split(":")[0].replace("Meas", "") for line in headerdata]
        b = {k for k, v in collections.Counter(meas).items() if v == counter_constant}
        a = set(measurements.keys())
        verboseprint("check header and measurements.keys()", a == b, a, b)
        return a == b


_COLUMN_ALIASES = {
    "well": "Well",
    "mutant": "Name",
    "name": "Name",
    "sample": "Name",
    "ph": "pH",
    "cl": "Cl",
    "temp": "Temp",
    "labels": "Labels",
}
_TEMP_IN_NAME = re.compile(r"[_-](\d{2})[_-]?note", re.IGNORECASE)


def _normalize_columns(note: pd.DataFrame, fpath: Path) -> pd.DataFrame:
    """Give a note the column names the rest of the code expects.

    Two dialects are in use: ``Well,pH,Cl,Name,Temp,Labels`` and
    ``well,pH,Cl,Mutant`` with the temperature in the file name. Aliases are
    resolved case-insensitively, and ``Name``/``Temp`` are taken from the file
    name when the note has no column for them.

    Parameters
    ----------
    note : pd.DataFrame
        The note as read.
    fpath : Path
        Where it came from; its name carries the temperature in one dialect.

    Returns
    -------
    pd.DataFrame
        The note with normalised columns.
    """
    note.columns = pd.Index([str(c).strip() for c in note.columns])
    note = note.rename(
        columns={c: _COLUMN_ALIASES.get(str(c).lower(), str(c)) for c in note.columns}
    )
    if "Temp" not in note.columns:
        found = _TEMP_IN_NAME.search(fpath.name)
        note["Temp"] = float(found.group(1)) if found else np.nan
    if "Name" not in note.columns:
        note["Name"] = fpath.name.removesuffix("_note.csv")
    for col in ("pH", "Cl"):
        if col not in note.columns:
            note[col] = 0.0
    return note


@dataclass
class Note:
    """Read and processes a Note csv file.

    This class is responsible for handling Experimental Note files
    that describe an EnSpire experiment collecting spectrum.

    Parameters
    ----------
    fpath : Path
        The path to the Experimental Note file to be processed.
    verbose :

    Example
    -------
    >>> from clophfit.prenspire import Note
    >>> n = Note("tests/EnSpire/h148g-spettroC-nota.csv")
    >>> n.wells[2]
    'A03'
    """

    fpath: Path
    verbose: int = 0
    """Level of verbosity; 0 is silent."""
    wells: list[str] = field(init=False, default_factory=list)
    """A list of wells generated from the note file."""
    # A list of lines extracted from the note file.
    _note: pd.DataFrame = field(init=False, default_factory=pd.DataFrame)
    titrations: dict[Any, Any] = field(init=False, default_factory=dict)
    """A list of titrations extracted from the note file."""

    def __post_init__(self) -> None:
        """Complete the initialization generating wells and _note_list."""
        verboseprint = verbose_print(self.verbose)
        with Path(self.fpath).open("r", newline="", encoding="utf-8") as file:
            sample_data = file.read(1024)  # Read a sample of the CSV data
        dialect = cast("csv.Dialect", csv.Sniffer().sniff(sample_data))
        self._note = _normalize_columns(
            pd.read_csv(self.fpath, dialect=dialect), Path(self.fpath)
        )
        self.wells = self._note.iloc[:, 0].tolist()
        verboseprint(f"Wells {self.wells[:2]}...{self.wells[-2:]} generated.")

    @property
    def note(self) -> pd.DataFrame:
        """The note as a frame, with ``Well``, ``pH``, ``Cl``, ``Name`` and ``Temp``."""
        return self._note

    # MAYBE: Add buffer subtraction logic to prenspire Note.

    def build_titrations(  # ruff: ignore[complex-structure, too-many-branches]
        self,
        ef: EnspireFile,
        *,
        isosbestic: float | str | None = None,
        drop_wells: dict[tuple[str, str], list[str]] | None = None,
    ) -> None:
        """Extract titrations from the given ef (_note file like: <well, pH, Cl>).

        Parameters
        ----------
        ef : EnspireFile
            The parsed EnSpire export.
        isosbestic : float | str | None
            Divide every well by its signal at the isosbestic point before the
            titration is assembled, so the legacy projection path is normalised
            the same way the band and whole-spectrum paths are: a wavelength
            [nm], ``"auto"`` to locate it per group, ``"auto-scale"`` to locate
            every group on the plate at once, or ``None``. The
            scaling is applied to every label of the group from the excitation
            scan's crossing, because the crossing belongs to the sample, not to
            the detector window.
        drop_wells : dict[tuple[str, str], list[str]] | None
            Wells to leave out, keyed by ``(sample, temperature)`` as strings. The
            legacy projection path has no screen of its own; this is where a list
            produced elsewhere (the whole-spectrum screen) enters it.
        """
        if "Labels" not in self._note.columns:
            # the second note dialect names no labels: take the ones the file
            # measured at each row's temperature
            at_temp: dict[str, str] = {}
            for label, meas in ef.measurements.items():
                temp = str(meas["metadata"].get("temp")).split(".")[0]
                at_temp[temp] = f"{at_temp.get(temp, '')} {label}".strip()
            self._note["Labels"] = [
                at_temp.get(str(t).split(".")[0], " ".join(ef.measurements))
                for t in self._note["Temp"]
            ]
        if isinstance(isosbestic, str) and isosbestic == "auto-scale":
            from clophfit.prenspire.spectral import isosbestic_by_group  # ruff: ignore[import-outside-top-level]

            lam_map = isosbestic_by_group(ef, self._note)
        else:
            lam_map = {}
        df_no_buffer = self._note.query('Name != "buffer"')
        threshold = 3
        grouped0 = df_no_buffer.groupby("Name")
        titrations: dict[Any, Any] = {}
        for name0, group0 in grouped0:
            grouped1 = group0.groupby("Temp")
            titrations[name0] = {}
            for name1, group1 in grouped1:
                titrations[name0][name1] = {}
                # Group by 'pH' and 'Cl' and keep only groups with n rows > 'threshold'.
                for grouping in ["pH", "Cl"]:
                    grouped2 = group1.groupby(grouping)
                    for name2, group2 in grouped2:
                        if len(group2) > threshold:
                            grouped3 = group2.groupby("Labels")
                            for name3, group3 in grouped3:
                                all_wells = group3["Well"].to_list()
                                skip = set(
                                    (drop_wells or {}).get((str(name0), str(name1)), ())
                                )
                                keep = [
                                    i for i, w in enumerate(all_wells) if w not in skip
                                ]
                                wells = [all_wells[i] for i in keep]
                                if group3["pH"].nunique() > group3["Cl"].nunique():
                                    axis = [group3["pH"].to_list()[i] for i in keep]
                                else:
                                    axis = [group3["Cl"].to_list()[i] for i in keep]
                                meas_dict = {}
                                for label in str(name3).split():
                                    d = {w: ef.measurements[label][w] for w in wells}
                                    value_df = pd.DataFrame(
                                        d,
                                        index=ef.measurements[label]["lambda"],
                                        columns=wells,
                                    )
                                    value_df.columns = pd.Index(axis)
                                    meas_dict[label] = value_df
                                lam = (
                                    lam_map.get((str(name0), float(str(name1))))
                                    if lam_map
                                    else isosbestic
                                )
                                if lam is not None:
                                    meas_dict = _normalise_titration_isosbestic(
                                        meas_dict, ef, lam
                                    )
                                titrations[name0][name1][f"{grouping}_{name2}"] = (
                                    meas_dict
                                )
        self.titrations = titrations


def _normalise_titration_isosbestic(
    meas_dict: dict[str, pd.DataFrame],
    ef: EnspireFile,
    isosbestic: float | str,
) -> dict[str, pd.DataFrame]:
    """Scale one titration group's spectra by its isosbestic signal.

    ``meas_dict`` holds the group's labels as wavelength-indexed frames whose
    columns are the titrant values, in well order. The crossing is located on
    the excitation scan (the only scan that carries the two-state crossover) and
    its per-well value is used as the brightness reference for every label of
    the group, which is what makes the group's readouts comparable.

    Parameters
    ----------
    meas_dict : dict[str, pd.DataFrame]
        Label to its spectra; wavelengths on the index, titrant per column.
    ef : EnspireFile
        The parsed export, for each label's metadata.
    isosbestic : float | str
        A wavelength [nm] or ``"auto"``.

    Returns
    -------
    dict[str, pd.DataFrame]
        The same frames, normalised. Unchanged when ``"auto"`` finds no crossing,
        which is what a chloride titration returns.
    """
    from clophfit.prenspire.spectral import (  # ruff: ignore[import-outside-top-level]
        EXCITATION_SCAN_NAMES,
        isosbestic_scale,
        plate_isosbestic,
    )

    excitation = next(
        (
            label
            for label in meas_dict
            if ef.measurements[label]["metadata"].get("Monochromator") == "Excitation"
        ),
        None,
    )
    if excitation is None:
        return meas_dict
    frame = meas_dict[excitation]
    spectra = {
        EXCITATION_SCAN_NAMES[0]: (
            frame.index.to_numpy(dtype=float),
            frame.to_numpy(dtype=float),
        )
    }
    x = frame.columns.to_numpy(dtype=float)
    lam_iso = (
        plate_isosbestic(spectra, x)
        if isinstance(isosbestic, str)
        else float(isosbestic)
    )
    if lam_iso is None:
        return meas_dict
    scale, _usable = isosbestic_scale(spectra, lam_iso)
    out: dict[str, pd.DataFrame] = {}
    for label, one in meas_dict.items():
        scaled = one.copy()
        scaled.iloc[:, :] = one.to_numpy(dtype=float) / scale
        out[label] = scaled
    return out


def merge_enspire_files(files: Sequence[EnspireFile]) -> EnspireFile:
    """Combine the measurements of several exports of the same plate.

    A construct can be titrated at two temperatures in two separate EnSpire
    exports, with its own note each time; a van't Hoff fit needs both in one
    place. The exports carry disjoint measurement labels (the instrument numbers
    them E/B/F at 37 C and C/D/A at 20 C on the same plate), so the merge is a
    union - and a *loud* failure if two files ever share a label, because then
    they are not two temperatures of one experiment but two experiments.

    Parameters
    ----------
    files : Sequence[EnspireFile]
        The exports to combine, first one first.

    Returns
    -------
    EnspireFile
        An export carrying every file's measurements and the union of their wells.
        Its ``file`` is the first input, which is what the export names use.

    Raises
    ------
    ValueError
        If the list is empty or two files share a measurement label.
    """
    if not files:
        msg = "merge_enspire_files needs at least one export"
        raise ValueError(msg)
    # start from the first export *as given* (no re-read: a caller may have
    # filtered its measurements), with its own containers
    merged = copy.copy(files[0])
    merged.measurements = dict(files[0].measurements)
    merged.wells = list(files[0].wells)
    for other in files[1:]:
        clash = set(merged.measurements) & set(other.measurements)
        if clash:
            msg = (
                f"{other.file.name} repeats measurement label(s) "
                f"{sorted(clash)}; two exports of one experiment must carry "
                "disjoint labels (different temperatures are labelled E/B/F and "
                "C/D/A), so these are not two temperatures of one titration"
            )
            raise ValueError(msg)
        merged.measurements.update(other.measurements)
        merged.wells = sorted(set(merged.wells) | set(other.wells))
    return merged
