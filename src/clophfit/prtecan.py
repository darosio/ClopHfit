"""Parse Tecan files, group lists and fit titrations.

- Titration is described in list.pH or list.cl file.
- Builds 96 titrations and export them in txt files.
- In the case of 2 labelblocks performs a global fit saving a png and printing the fitting results.

.. include:: ../../docs/api/prtecan.uml.rst

"""
from __future__ import annotations

import copy
import hashlib
import itertools
import warnings
from collections import defaultdict
from dataclasses import InitVar
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any  # , overload
from typing import Sequence

import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import pandas as pd
import scipy  # type: ignore
import scipy.stats  # type: ignore
import seaborn as sb  # type: ignore
from matplotlib.backends.backend_pdf import PdfPages  # type: ignore
from numpy.typing import NDArray


# list_of_lines
# after set([type(x) for l in csvl for x in l]) = float | int | str


def read_xls(path: Path) -> list[list[str | int | float]]:
    """Read first sheet of an xls file.

    Parameters
    ----------
    path : Path
        Path to `.xls` file.

    Returns
    -------
    list_of_lines
        Lines.

    """
    df = pd.read_excel(path)
    n0 = pd.DataFrame([[np.nan] * len(df.columns)], columns=df.columns)
    df = pd.concat([n0, df], ignore_index=True)
    df.fillna("", inplace=True)
    return list(df.values.tolist())


def lookup_listoflines(
    csvl: list[list[str | int | float]], pattern: str = "Label: Label", col: int = 0
) -> list[int]:
    """Lookup line numbers (row index) where given pattern occurs.

    Parameters
    ----------
    csvl : list_of_lines
        Lines of a csv/xls file.
    pattern : str
        Pattern to be searched (default="Label: Label").
    col : int
        Column to search (default=0).

    Returns
    -------
    list[int]
        Row/line index for all occurrences of pattern. Empty list for no occurrences.

    """
    return [
        tuple_i_line[0]
        for tuple_i_line in list(
            filter(
                lambda x: pattern in x[1][0] if isinstance(x[1][0], str) else None,
                enumerate(csvl),
            )
        )
    ]


def strip_lines(lines: list[list[str | int | float]]) -> list[list[str | int | float]]:
    """Remove empty fields/cells from lines read from a csv file.

    Parameters
    ----------
    lines : list_of_lines
        Lines that are a list of fields, typically from a csv/xls file.

    Returns
    -------
    list_of_lines
        Lines removed from blank cells.

    Examples
    --------
    >>> lines = [['Shaking (Linear) Amplitude:', '', '', '', 2, 'mm', '', '', '', '', '']]
    >>> strip_lines(lines)
    [['Shaking (Linear) Amplitude:', 2, 'mm']]

    """
    return [[e for e in line if e != ""] for line in lines]


# TODO with a filter ectract_metadata with a map


@dataclass(frozen=False)
class Metadata:
    """Value type of a metadata dictionary."""

    value: int | str | float | None  #: Value for the dictionary key.
    unit: Sequence[str | float | int] | None = None
    """First element is the unit, the following are somewhat unexpected."""


def extract_metadata(
    lines: list[list[str | int | float]],
) -> dict[str, Metadata]:
    """Extract metadata into both Tecanfile and Labelblock.

    From a list of stripped lines takes the first field as the **key** of the
    metadata dictionary, remaining fields goes into a list of values with the
    exception of Label ([str]) and Temperature ([float]).

    Parameters
    ----------
    lines : list_of_lines
        Lines that are a list of fields, typically from a csv/xls file.

    Returns
    -------
    dict[str, str | list[str | int | float]]
        Metadata for Tecanfile or Labelblock.

    Examples
    --------
    >>> lines = [['Shaking (Linear) Amplitude:', '', '', '', 2, 'mm', '', '', '', '', '']]
    >>> extract_metadata(lines)
    {'Shaking (Linear) Amplitude:': Metadata(value=2, unit=['mm'])}

    >>> lines = [['', 'Temperature: 26 °C', '', '', '', '', '', '', '', '', '']]
    >>> extract_metadata(lines)
    {'Temperature': Metadata(value=26.0, unit=['°C'])}

    >>> lines = [['Excitation Wavelength', '', '', '', 400, 'nm', '', '', '', '', '']]
    >>> extract_metadata(lines)
    {'Excitation Wavelength': Metadata(value=400, unit=['nm'])}

    >>> lines = [['Label: Label1', '', '', '', '', '', '', '', '', '', '', '', '']]
    >>> extract_metadata(lines)
    {'Label': Metadata(value='Label1', unit=None)}

    >>> lines = [['Mode', '', '', '', 'Fluorescence Top Reading', '', '', '', '', '']]
    >>> extract_metadata(lines)['Mode'].value
    'Fluorescence Top Reading'

    """
    md: dict[str, Metadata] = {}

    for line in strip_lines(lines):
        if len(line) > 2:
            md.update({str(line[0]): Metadata(line[1], line[2:])})
        elif len(line) == 2:
            md.update({str(line[0]): Metadata(line[1])})
        elif len(line) == 1 and isinstance(line[0], str) and ":" in line[0]:
            k, v = line[0].split(":")
            vals: list[str] = v.split()
            val: float | str
            try:
                val = float(vals[0])
            except ValueError:
                val = vals[0]
            if len(vals) == 1:
                md.update({k: Metadata(val)})
            else:
                md.update({k: Metadata(val, vals[1:])})
        elif line:
            md.update({str(line[0]): Metadata(line[0])})

    return md


def _merge_md(mds: list[dict[str, Metadata]]) -> dict[str, Metadata]:
    """Merge a list of metadata dict if the key value is the same in the list."""
    mmd = {k: v for k, v in mds[0].items() if all(v == md[k] for md in mds[1:])}
    # To account for the case 93"Optimal" and 93"Manual" in lb metadata
    if mmd.get("Gain") is None and mds[0].get("Gain") is not None:
        if all(mds[0]["Gain"].value == md["Gain"].value for md in mds[1:]):
            mmd["Gain"] = Metadata(mds[0]["Gain"].value)
    return mmd


def fz_kd_singlesite(
    k: float, p: NDArray[np.float_] | Sequence[float], x: NDArray[np.float_]
) -> NDArray[np.float_]:
    """Fit function for Cl titration."""
    return (float(p[0]) + float(p[1]) * x / k) / (1 + x / k)


def fz_pk_singlesite(
    k: float, p: NDArray[np.float_] | Sequence[float], x: NDArray[np.float_]
) -> NDArray[np.float_]:
    """Fit function for pH titration."""
    return (float(p[1]) + float(p[0]) * 10 ** (k - x)) / (1 + 10 ** (k - x))


def fit_titration(
    kind: str,
    x: Sequence[float],
    y: NDArray[np.float_],
    y2: NDArray[np.float_] | None = None,
    residue: NDArray[np.float_] | None = None,
    residue2: NDArray[np.float_] | None = None,
    tval_conf: float = 0.95,
) -> pd.DataFrame:
    """Fit pH or Cl titration using a single-site binding model.

    Returns confidence interval (default=0.95) for fitting params (cov*tval), rather than
    standard error of the fit. Use scipy leastsq. Determine 3 fitting parameters:
    - binding constant *K*
    - and 2 plateau *SA* and *SB*.

    Parameters
    ----------
    kind : str
        Titration type {'pH'|'Cl'}
    x : Sequence[float]
        Dataset x-values.
    y : NDArray[np.float]
        Dataset y-values.
    y2 : NDArray[np.float], optional
        Optional second dataset y-values (share x with main dataset).
    residue : NDArray[np.float], optional
        Residues for main dataset.
    residue2 : NDArray[np.float], optional
        Residues for second dataset.
    tval_conf : float
        Confidence level (default 0.95) for parameter estimations.

    Returns
    -------
    pd.DataFrame
        Fitting results.

    Raises
    ------
    NameError
        When kind is different than "pH" or "Cl".

    Examples
    --------
    >>> import numpy as np
    >>> fit_titration("Cl", np.array([1.0, 10, 30, 100, 200]), \
          np.array([10, 8, 5, 1, 0.1]))[["K", "sK"]]
               K         sK
    0  38.955406  30.201929

    """
    if kind == "pH":
        fz = fz_pk_singlesite
    elif kind == "Cl":
        fz = fz_kd_singlesite
    else:
        raise NameError("kind= pH or Cl")

    def compute_p0(x: Sequence[float], y: NDArray[np.float_]) -> NDArray[np.float_]:
        df = pd.DataFrame({"x": x, "y": y})
        p0sa = df.y[df.x == min(df.x)].values[0]
        p0sb = df.y[df.x == max(df.x)].values[0]
        p0k = np.average([max(y), min(y)])
        try:
            x1, y1 = df[df["y"] >= p0k].values[0]
        except IndexError:
            x1 = np.nan
            y1 = np.nan
        try:
            x2, y2 = df[df["y"] <= p0k].values[0]
        except IndexError:
            x2 = np.nan
            y2 = np.nan
        p0k = (x2 - x1) / (y2 - y1) * (p0k - y1) + x1
        return np.array(np.r_[p0k, p0sa, p0sb])

    if y2 is None:

        def ssq1(
            p: NDArray[np.float_], x: NDArray[np.float_], y1: NDArray[np.float_]
        ) -> NDArray[np.float_]:
            return np.array(np.r_[y1 - fz(p[0], p[1:3], x)])

        p0 = compute_p0(x, y)
        p, cov, info, msg, success = scipy.optimize.leastsq(
            ssq1, p0, args=(np.array(x), y), full_output=True, xtol=1e-11
        )
    else:

        def ssq2(
            p: NDArray[np.float_],
            x: NDArray[np.float_],
            y1: NDArray[np.float_],
            y2: NDArray[np.float_],
            rd1: NDArray[np.float_],
            rd2: NDArray[np.float_],
        ) -> NDArray[np.float_]:
            return np.array(
                np.r_[
                    (y1 - fz(p[0], p[1:3], x)) / rd1**2,
                    (y2 - fz(p[0], p[3:5], x)) / rd2**2,
                ]
            )

        p1 = compute_p0(x, y)
        p2 = compute_p0(x, y2)
        ave = np.average([p1[0], p2[0]])
        p0 = np.r_[ave, p1[1], p1[2], p2[1], p2[2]]
        tmp = scipy.optimize.leastsq(
            ssq2,
            p0,
            full_output=True,
            xtol=1e-11,
            args=(np.array(x), y, y2, residue, residue2),
        )
        p, cov, info, msg, success = tmp
    res = pd.DataFrame({"ss": [success]})
    res["msg"] = msg
    if 1 <= success <= 4:
        try:
            tval = (tval_conf + 1) / 2
            chisq = sum(info["fvec"] * info["fvec"])
            res["df"] = len(y) - len(p)
            res["tval"] = scipy.stats.distributions.t.ppf(tval, res.df)
            res["chisqr"] = chisq / res.df
            res["K"] = p[0]
            res["SA"] = p[1]
            res["SB"] = p[2]
            if y2 is not None:
                res["df"] += len(y2)
                res["tval"] = scipy.stats.distributions.t.ppf(tval, res.df)
                res["chisqr"] = chisq / res.df
                res["SA2"] = p[3]
                res["SB2"] = p[4]
                res["sSA2"] = np.sqrt(cov[3][3] * res.chisqr) * res.tval
                res["sSB2"] = np.sqrt(cov[4][4] * res.chisqr) * res.tval
            res["sK"] = np.sqrt(cov[0][0] * res.chisqr) * res.tval
            res["sSA"] = np.sqrt(cov[1][1] * res.chisqr) * res.tval
            res["sSB"] = np.sqrt(cov[2][2] * res.chisqr) * res.tval
        except TypeError:
            pass  # if some params are not successfully determined.
    return res


@dataclass
class Labelblock:
    """Parse a label block within a Tecan file.

    Parameters
    ----------
    lines : list[list[str | int | float]]
        Lines to create this Labelblock.
    path : Path, optional
        File path to the tecanfile that contains this labelblock.

    Raises
    ------
    Exception
        When data do not correspond to a complete 96-well plate.

    Warns
    -----
    Warning
        When it replaces "OVER" with ``np.nan`` for saturated values.

    """

    lines: InitVar[list[list[str | int | float]]]
    path: Path | None = None
    metadata: dict[str, Metadata] = field(init=False, repr=True)
    """Metadata specific for this Labelblock."""
    data: dict[str, float] = field(init=False, repr=True)
    """The 96 data values as {'well_name', value}."""
    _buffer: float = field(init=False, repr=False)
    _buffer_norm: float = field(init=False, repr=False)
    _sd_buffer: float = field(init=False, repr=False)
    _sd_buffer_norm: float = field(init=False, repr=False)
    _data_normalized: dict[str, float] | None = field(init=False, repr=False)
    _data_buffersubtracted: dict[str, float] | None = field(init=False, repr=False)
    _data_buffersubtracted_norm: dict[str, float] | None = field(init=False, repr=False)
    _buffer_wells: list[str] = field(init=False, repr=False)

    def __post_init__(self, lines: list[list[str | int | float]]) -> None:
        """Generate metadata and data for this labelblock."""
        if lines[14][0] == "<>" and lines[23] == lines[24] == [""] * 13:
            stripped = strip_lines(lines)
            stripped[14:23] = []
            self.metadata = extract_metadata(stripped)
            self.data = self._extract_data(lines[15:23])
            self._data_normalized = None
        else:
            raise ValueError("Cannot build Labelblock: not 96 wells?")

    def _extract_data(self, lines: list[list[str | int | float]]) -> dict[str, float]:
        """Convert data into a dictionary.

        {'A01' : value}
        :
        {'H12' : value}

        Parameters
        ----------
        lines : list_of_lines
            xls file read into lines.

        Returns
        -------
        dict[str, float]
            Data from a label block.

        Raises
        ------
        ValueError
            When something went wrong. Possibly because not 96-well.

        Warns
        -----
            When a cell contains saturated signal (converted into np.nan).

        """
        rownames = tuple("ABCDEFGH")
        data = {}
        try:
            assert len(lines) == 8
            for i, row in enumerate(rownames):
                assert lines[i][0] == row  # e.g. "A" == "A"
                for col in range(1, 13):
                    try:
                        data[row + f"{col:0>2}"] = float(lines[i][col])
                    except ValueError:
                        data[row + f"{col:0>2}"] = np.nan
                        warnings.warn(
                            f"OVER\n Overvalue in {self.metadata['Label'].value}:"
                            f"{row}{col:0>2} of tecanfile {self.path}"
                        )
        except AssertionError:
            raise ValueError("Cannot extract data in Labelblock: not 96 wells?")
        return data

    _KEYS = [
        "Emission Bandwidth",
        "Emission Wavelength",
        "Excitation Bandwidth",
        "Excitation Wavelength",
        "Mode",
        "Integration Time",
        "Number of Flashes",
    ]

    @property
    def data_normalized(self) -> dict[str, float]:
        """Normalize data by number of flashes, integration time and gain."""
        if self._data_normalized is None:
            if (
                isinstance(self.metadata["Gain"].value, (float, int))
                and isinstance(self.metadata["Number of Flashes"].value, (float, int))
                and isinstance(self.metadata["Integration Time"].value, (float, int))
            ):
                norm = (
                    1000.0
                    / self.metadata["Gain"].value
                    / self.metadata["Number of Flashes"].value
                    / self.metadata["Integration Time"].value
                )
            else:
                warnings.warn(
                    "Could not normalize for non numerical Gain, Number of Flashes or Integration time."
                )  # pragma: no cover
            self._data_normalized = {k: v * norm for k, v in self.data.items()}
        return self._data_normalized

    @property
    def buffer_wells(self) -> list[str]:
        """List of buffer wells."""
        return self._buffer_wells

    @buffer_wells.setter
    def buffer_wells(self, buffer_wells: list[str]) -> None:
        self._buffer_wells = buffer_wells
        self._buffer = float(np.average([self.data[k] for k in self.buffer_wells]))
        self._sd_buffer = float(np.std([self.data[k] for k in self.buffer_wells]))
        self._buffer_norm = float(
            np.average([self.data_normalized[k] for k in self.buffer_wells])
        )
        self._sd_buffer_norm = float(
            np.std([self.data_normalized[k] for k in self.buffer_wells])
        )
        self._data_buffersubtracted = None
        self._data_buffersubtracted_norm = None

    @property
    def data_buffersubtracted(self) -> dict[str, float]:
        """Buffer subtracted data."""
        if isinstance(self.buffer, (float, int)):
            if self._data_buffersubtracted is None:
                self._data_buffersubtracted = {
                    k: v - self.buffer for k, v in self.data.items()
                }
            return self._data_buffersubtracted
        else:
            raise AttributeError("Provide a buffer value first.")

    @property
    def data_buffersubtracted_norm(self) -> dict[str, float]:
        """Normalize buffer-subtracted data."""
        if isinstance(self.buffer_norm, (float, int)):
            if self._data_buffersubtracted_norm is None:
                self._data_buffersubtracted_norm = {
                    k: v - self.buffer_norm for k, v in self.data_normalized.items()
                }
            return self._data_buffersubtracted_norm
        else:
            raise AttributeError("Provide a buffer value first.")

    @property
    def buffer(self) -> float | None:
        """Background value to be subtracted before dilution correction."""
        return self._buffer

    @buffer.setter
    def buffer(self, value: float) -> None:
        if hasattr(self, "_buffer") and self._buffer == value:
            return None
        self._data_buffersubtracted = None
        self._buffer = value

    @property
    def sd_buffer(self) -> float | None:
        """Get standard deviation of buffer_wells values."""
        return self._sd_buffer

    @property
    def buffer_norm(self) -> float | None:
        """Background value to be subtracted before dilution correction."""
        return self._buffer_norm

    @buffer_norm.setter
    def buffer_norm(self, value: float) -> None:
        if hasattr(self, "_buffer_norm") and self._buffer_norm == value:
            return None
        self._data_buffersubtracted_norm = None
        self._buffer_norm = value

    @property
    def sd_buffer_norm(self) -> float | None:
        """Get standard deviation of normalized buffer_wells values."""
        return self._sd_buffer_norm

    def __eq__(self, other: object) -> bool:
        """Two labelblocks are equal when metadata KEYS are identical."""
        if not isinstance(other, Labelblock):
            return NotImplemented
        eq: bool = True
        for k in Labelblock._KEYS:
            eq &= self.metadata[k] == other.metadata[k]
        # 'Gain': [81.0, 'Manual'] = 'Gain': [81.0, 'Optimal'] They are equal
        eq &= self.metadata["Gain"].value == other.metadata["Gain"].value
        return eq

    def __almost_eq__(self, other: Labelblock) -> bool:
        """Two labelblocks are almost equal when they could be merged after normalization."""
        eq: bool = True
        # Integration Time, Number of Flashes and Gain can differ.
        for k in Labelblock._KEYS[:5]:
            eq &= self.metadata[k] == other.metadata[k]
        return eq


@dataclass
class Tecanfile:
    """Parse a Tecan .xls file.

    Parameters
    ----------
    path: Path
        Path to `.xls` file.

    Raises
    ------
    FileNotFoundError
        When path does not exist.
    Exception
        When no Labelblock is found.

    """

    path: Path
    metadata: dict[str, Metadata] = field(init=False, repr=True)
    """General metadata for Tecanfile, like `Date` and `Shaking Duration`."""
    labelblocks: list[Labelblock] = field(init=False, repr=True)
    """All labelblocks contained in this file."""

    def __post_init__(self) -> None:
        """Initialize."""
        csvl = read_xls(self.path)
        idxs = lookup_listoflines(csvl, pattern="Label: Label", col=0)
        if len(idxs) == 0:
            raise ValueError("No Labelblock found.")
        self.metadata = extract_metadata(csvl[: idxs[0]])
        labelblocks = []
        n_labelblocks = len(idxs)
        idxs.append(len(csvl))
        for i in range(n_labelblocks):
            labelblocks.append(Labelblock(csvl[idxs[i] : idxs[i + 1]], self.path))
        if any(
            labelblocks[i] == labelblocks[j]
            for i, j in itertools.combinations(range(n_labelblocks), 2)
        ):
            warnings.warn("Repeated labelblocks")
        self.labelblocks = labelblocks


@dataclass
class LabelblocksGroup:
    """Group labelblocks with compatible metadata.

    Parameters
    ----------
    labelblocks: list[Labelblock]
        Labelblocks to be grouped.
    allequal: bool
        True if labelblocks already tested to be equal.

    Raises
    ------
    Exception
        When labelblocks are neither equal nor almost equal.

    """

    labelblocks: list[Labelblock]
    allequal: bool = False
    #: TODO: remove because already in labelblock?
    buffer: dict[str, list[float]] | None = None
    #: The common metadata.
    metadata: dict[str, Metadata] = field(init=False, repr=True)
    #: Dict for data, like  Labelblock with well name as key and list of values as value.
    data: dict[str, list[float]] = field(init=False, repr=True)

    def __post_init__(self) -> None:
        """Create common metadata and data."""
        if not self.allequal:
            self.allequal = all(
                self.labelblocks[0] == lb for lb in self.labelblocks[1:]
            )
        datagrp = defaultdict(list)
        if self.allequal:
            for key in self.labelblocks[0].data.keys():
                for lb in self.labelblocks:
                    datagrp[key].append(lb.data[key])
        # labelblocks that can be merged only after normalization
        elif all(self.labelblocks[0].__almost_eq__(lb) for lb in self.labelblocks[1:]):
            for key in self.labelblocks[0].data.keys():
                for lb in self.labelblocks:
                    datagrp[key].append(lb.data_normalized[key])
        else:
            raise ValueError("Creation of labelblock group failed.")
        self.data = datagrp
        self.metadata = _merge_md([lb.metadata for lb in self.labelblocks])


@dataclass
class TecanfilesGroup:
    """Group of Tecanfiles containing at least one common Labelblock.

    Parameters
    ----------
    tecanfiles: list[Tecanfile]
        List of Tecanfiles.

    Raises
    ------
    Exception
        When all Labelblocks are not at least almost equal.

    Warns
    -----
    Warning
        The Tecanfiles listed in *filenames* are suppossed to contain the
        "same" list (of length N) of Labelblocks. So, N labelblocksgroups
        will be normally created. A warn will raise if not all Tecanfiles
        contains the same number of Labelblocks ('equal' mergeable) in the
        same order, but a number M < N of groups can be built.

    """

    tecanfiles: list[Tecanfile]
    #: Each group contains its own data like a titration. ??
    labelblocksgroups: list[LabelblocksGroup] = field(init=False, default_factory=list)
    #: FIXME: Metadata of the first Tecanfile.
    metadata: dict[str, Metadata] = field(init=False, repr=True)

    def __post_init__(self) -> None:
        """Create metadata and labelblocksgroups."""
        n_labelblocks = [len(tf.labelblocks) for tf in self.tecanfiles]
        tf0 = self.tecanfiles[0]
        if all([tf0.labelblocks == tf.labelblocks for tf in self.tecanfiles[1:]]):
            # Same number and order of labelblocks
            for i, _lb in enumerate(tf0.labelblocks):
                self.labelblocksgroups.append(
                    LabelblocksGroup(
                        [tf.labelblocks[i] for tf in self.tecanfiles], allequal=True
                    )
                )
        else:
            # Create as many as possible groups of labelblocks
            rngs = tuple([range(n) for n in n_labelblocks])
            for idx in itertools.product(*rngs):
                try:
                    gr = LabelblocksGroup(
                        [tf.labelblocks[idx[i]] for i, tf in enumerate(self.tecanfiles)]
                    )
                except ValueError:
                    continue
                # if labelblocks are all 'equal'
                else:
                    self.labelblocksgroups.append(gr)
            files = [tf.path.name for tf in self.tecanfiles]
            if len(self.labelblocksgroups) == 0:  # == []
                raise ValueError(f"No common labelblock in filenames: {files}.")
            else:
                warnings.warn(f"Different LabelblocksGroup among filenames: {files}.")
        self.metadata = _merge_md([tf.metadata for tf in self.tecanfiles])


@dataclass(init=False)
class Titration(TecanfilesGroup):
    """Group tecanfiles into a Titration as indicated by a listfile.

    The script will work from any directory: list.pH list filenames relative to
    its position.

    Parameters
    ----------
    listfile
        File path to the listfile ([tecan_file_path conc]).

    Raises
    ------
    FileNotFoundError
        When cannot access `listfile`.
    ValueError
        For unexpected file format, e.g. length of filename column differs from length of conc values.

    """

    #: Concentration values common to all 96 titrations.
    conc: Sequence[float] = field(init=False, repr=True)

    def __init__(self, listfile: Path | str) -> None:
        if isinstance(listfile, str):
            listfile = Path(listfile)
        try:
            df = pd.read_table(listfile, names=["filenames", "conc"])
        except FileNotFoundError:
            raise FileNotFoundError(f"Cannot find: {listfile}")
        if df["filenames"].count() != df["conc"].count():
            raise ValueError(f"Check format [filenames conc] for listfile: {listfile}")
        self.conc = df["conc"].tolist()
        tecanfiles = [Tecanfile(listfile.parent / f) for f in df["filenames"]]
        super().__init__(tecanfiles)

    def export_dat(self, path: Path) -> None:
        """Export dat files [x,y1,..,yN] from labelblocksgroups.

        Parameters
        ----------
        path : Path
            Path to output folder.

        """
        path.mkdir(parents=True, exist_ok=True)
        for key, dy1 in self.labelblocksgroups[0].data.items():
            df = pd.DataFrame({"x": self.conc, "y1": dy1})
            for n, lb in enumerate(self.labelblocksgroups[1:], start=2):
                dy = lb.data[key]
                df["y" + str(n)] = dy
            df.to_csv(path / Path(key).with_suffix(".dat"), index=False)


@dataclass
class TitrationAnalysis:
    """Perform analysis of a titration.

    Parameters
    ----------
    titration : Titration
        Titration object.
    schemefile : str | None
        File path to the schemefile (e.g. {"C01: 'V224Q'"}).

    Raises
    ------
    ValueError
        For unexpected file format, e.g. header `names`.

    """

    titration: Titration
    schemefile: str | None = None
    #: Scheme for known samples e.g. {'buffer', ['H12', 'H01']}
    scheme: pd.Series[Any] = field(init=False, repr=True)
    #: Concentration values common to all 96 titrations.
    conc: Sequence[float] = field(init=False, repr=True)
    #: Deepcopy from titration.
    labelblocksgroups: list[LabelblocksGroup] = field(init=False, repr=True)
    additions: Sequence[float] = field(init=False, repr=True)

    def __post_init__(self) -> None:
        """Create attributes."""
        if self.schemefile is None:
            self.scheme = pd.Series({"well": []})
        else:
            df = pd.read_table(self.schemefile)
            if (
                df.columns.tolist() != ["well", "sample"]
                or df["well"].count() != df["sample"].count()
            ):
                raise ValueError(
                    f"Check format [well sample] for schemefile: {self.schemefile}"
                )
            self.scheme = df.groupby("sample")["well"].unique()
        self.conc = self.titration.conc
        self.labelblocksgroups = copy.deepcopy(self.titration.labelblocksgroups)

    @classmethod
    def calculate_conc(
        cls,
        additions: Sequence[float],
        conc_stock: float,
        conc_ini: float = 0.0,
    ) -> NDArray[np.float_]:
        """Calculate concentration values.

        additions[0]=vol_ini; Stock concentration is a parameter.

        Parameters
        ----------
        additions : Sequence[float]
            Initial volume and all subsequent additions.
        conc_stock : float
            Concentration of the stock used for additions.
        conc_ini : float
            Initial concentration (default=0).

        Returns
        -------
        np.ndarray
            Concentrations as vector.

        """
        vol_tot = np.cumsum(additions)
        concs = np.ones(len(additions))
        concs[0] = conc_ini
        for i, add in enumerate(additions[1:], start=1):
            concs[i] = (
                concs[i - 1] * vol_tot[i - 1] + conc_stock * float(add)
            ) / vol_tot[i]
        return concs  # , vol_tot

    def subtract_bg(self) -> None:
        """Subtract average buffer values for each titration point."""
        buffer_keys = self.scheme.pop("buffer")
        for lbg in self.labelblocksgroups:
            lbg.buffer = {}
            for k in buffer_keys:
                lbg.buffer[k] = lbg.data.pop(k)
            bgs = list(lbg.buffer.values())
            bg = np.mean(bgs, axis=0)
            bg_sd = np.std(bgs, axis=0)
            for k in lbg.data:
                lbg.data[k] -= bg
            lbg.buffer["bg"] = bg
            lbg.buffer["bg_sd"] = bg_sd

    def dilution_correction(self, additionsfile: str) -> None:
        """Apply dilution correction.

        Parameters
        ----------
        additionsfile: str
            File listing volume additions during titration.

        """
        if hasattr(self, "additions"):
            warnings.warn("Dilution correction was already applied.")
            return
        df = pd.read_table(additionsfile, names=["add"])
        self.additions = df["add"].tolist()
        volumes = np.cumsum(self.additions)
        corr = volumes / volumes[0]
        for lbg in self.labelblocksgroups:
            for k in lbg.data:
                lbg.data[k] *= corr

    def metadata_normalization(self) -> None:
        """Normalize signal using gain, flashes and integration time."""
        if hasattr(self, "normalized"):
            warnings.warn("Normalization using metadata was already applied.")
            return
        for lbg in self.labelblocksgroups:
            # new lbg metadata
            corr = 1000 / float(lbg.metadata["Gain"].value)  # type: ignore
            corr /= float(lbg.metadata["Integration Time"].value)  # type: ignore
            corr /= float(lbg.metadata["Number of Flashes"].value)  # type: ignore
            for k in lbg.data.keys():
                lbg.data[k] = [v * corr for v in lbg.data[k]]
        self.normalized = True

    def _get_keys(self) -> None:
        """Get plate positions of crtl and unk samples."""
        self.keys_ctrl = [k for ctr in self.scheme.tolist() for k in ctr]
        self.names_ctrl = list(self.scheme.to_dict())
        self.keys_unk = list(
            self.labelblocksgroups[0].data.keys() - set(self.keys_ctrl)
        )

    def fit(
        self,
        kind: str,
        ini: int = 0,
        fin: int | None = None,
        no_weight: bool = False,
        tval: float = 0.95,
    ) -> None:
        """Fit titrations.

        Here is less general. It is for 2 labelblocks.

        Parameters
        ----------
        kind : str
            Titration type {'pH'|'Cl'}
        ini : int
            Initial point (default: 0).
        fin : int, optional
            Final point (default: None).
        no_weight : bool
            Do not use residues from single Labelblock fit as weight for global fitting.
        tval : float
            Only for tval different from default=0.95 for the confint calculation.

        Notes
        -----
        Create (: list) 3 fitting tables into self.fittings.

        """
        if kind == "Cl":
            self.fz = fz_kd_singlesite
        elif kind == "pH":
            self.fz = fz_pk_singlesite
        x = np.array(self.conc)
        fittings = []
        for lbg in self.labelblocksgroups:
            fitting = pd.DataFrame()
            for k, y in lbg.data.items():
                res = fit_titration(
                    kind, self.conc[ini:fin], np.array(y[ini:fin]), tval_conf=tval
                )
                res.index = pd.Index([k])
                # fitting = fitting.append(res, sort=False) DDD
                fitting = pd.concat([fitting, res], sort=False)
                # TODO assert (fitting.columns == res.columns).all()
                # better to refactor this function

            fittings.append(fitting)
        # global weighted on relative residues of single fittings
        fitting = pd.DataFrame()
        for k, y in self.labelblocksgroups[0].data.items():
            y2 = np.array(self.labelblocksgroups[1].data[k])
            residue = y - self.fz(
                fittings[0]["K"].loc[k],
                [fittings[0]["SA"].loc[k], fittings[0]["SB"].loc[k]],
                x,
            )
            residue /= y  # TODO residue or
            # log(residue/y) https://www.tandfonline.com/doi/abs/10.1080/00031305.1985.10479385
            residue2 = y2 - self.fz(
                fittings[1]["K"].loc[k],
                [fittings[1]["SA"].loc[k], fittings[1]["SB"].loc[k]],
                x,
            )
            residue2 /= y2
            if no_weight:
                for i, _rr in enumerate(residue):
                    residue[i] = 1  # TODO use np.ones() but first find a way to test
                    residue2[i] = 1
            res = fit_titration(
                kind,
                self.conc[ini:fin],
                np.array(y[ini:fin]),
                y2=y2[ini:fin],
                residue=residue[ini:fin],
                residue2=residue2[ini:fin],
                tval_conf=tval,
            )
            res.index = pd.Index([k])
            # fitting = fitting.append(res, sort=False) DDD
            fitting = pd.concat([fitting, res], sort=False)
        fittings.append(fitting)
        # Write the name of the control e.g. S202N in the "ctrl" column
        for fitting in fittings:
            for ctrl, v in self.scheme.items():
                for k in v:
                    fitting.loc[k, "ctrl"] = ctrl  # type: ignore
        # self.fittings and self.fz
        self.fittings = fittings
        self._get_keys()

    def plot_k(
        self,
        lb: int,
        xlim: tuple[float, float] | None = None,
        title: str | None = None,
    ) -> plt.figure:
        """Plot K values as stripplot.

        Parameters
        ----------
        lb: int
            Labelblock index.
        xlim : tuple[float, float], optional
            Range.
        title : str, optional
            To name the plot.

        Returns
        -------
        plt.figure
            The figure.

        Raises
        ------
        Exception
            When no fitting results are available (in this object).

        """
        if not hasattr(self, "fittings"):
            raise Exception("run fit first")
        sb.set(style="whitegrid")
        f = plt.figure(figsize=(12, 16))
        # Ctrls
        ax1 = plt.subplot2grid((8, 1), loc=(0, 0))
        if len(self.keys_ctrl) > 0:
            res_ctrl = self.fittings[lb].loc[self.keys_ctrl]
            sb.stripplot(
                x=res_ctrl["K"],
                y=res_ctrl.index,
                size=8,
                orient="h",
                hue=res_ctrl.ctrl,
                ax=ax1,
            )
            plt.errorbar(
                res_ctrl.K,
                range(len(res_ctrl)),
                xerr=res_ctrl.sK,  # xerr=res_ctrl.sK*res_ctrl.tval,
                fmt=".",
                c="lightgray",
                lw=8,
            )
            plt.grid(1, axis="both")
        # Unks
        #  FIXME keys_unk is an attribute or a property
        res_unk = self.fittings[lb].loc[self.keys_unk]
        ax2 = plt.subplot2grid((8, 1), loc=(1, 0), rowspan=7)
        sb.stripplot(
            x=res_unk["K"].sort_index(),
            y=res_unk.index,
            size=12,
            orient="h",
            palette="Greys",
            hue=res_unk["SA"].sort_index(),
            ax=ax2,
        )
        plt.legend("")
        plt.errorbar(
            res_unk["K"].sort_index(),
            range(len(res_unk)),
            xerr=res_unk["sK"].sort_index(),
            fmt=".",
            c="gray",
            lw=2,
        )
        plt.yticks(range(len(res_unk)), res_unk.index.sort_values())
        plt.ylim(-1, len(res_unk))
        plt.grid(1, axis="both")
        if not xlim:
            xlim = (res_unk["K"].min(), res_unk["K"].max())
            if len(self.keys_ctrl) > 0:
                xlim = (
                    0.99 * min(res_ctrl["K"].min(), xlim[0]),
                    1.01 * max(res_ctrl["K"].max(), xlim[1]),
                )
            xlim = (0.99 * xlim[0], 1.01 * xlim[1])
        ax1.set_xlim(xlim)
        ax2.set_xlim(xlim)
        ax1.set_xticklabels([])
        ax1.set_xlabel("")
        title = title if title else ""
        title += "  label:" + str(lb)
        f.suptitle(title, fontsize=16)
        f.tight_layout(pad=1.2, w_pad=0.1, h_pad=0.5, rect=(0, 0, 1, 0.97))
        return f

    def plot_well(self, key: str) -> plt.figure:
        """Plot global fitting using 2 labelblocks.

        Here is less general. It is for 2 labelblocks.

        Parameters
        ----------
        key: str
            Well position as dictionary key like "A01".

        Returns
        -------
        plt.figure
            Pointer to mpl.figure.

        Raises
        ------
        Exception
            When fit is not yet run.

        """
        if not hasattr(self, "fittings"):
            raise Exception("run fit first")
        plt.style.use(["seaborn-ticks", "seaborn-whitegrid"])
        out = ["K", "sK", "SA", "sSA", "SB", "sSB"]
        out2 = ["K", "sK", "SA", "sSA", "SB", "sSB", "SA2", "sSA2", "SB2", "sSB2"]
        x = np.array(self.conc)
        xfit = np.linspace(min(x) * 0.98, max(x) * 1.02, 50)
        residues = []
        colors = []
        lines = []
        f = plt.figure(figsize=(10, 7))
        ax_data = plt.subplot2grid((3, 1), loc=(0, 0), rowspan=2)
        # labelblocks
        for i, (lbg, df) in enumerate(zip(self.labelblocksgroups, self.fittings)):
            y = lbg.data[key]
            # ## data
            colors.append(plt.cm.Set2((i + 2) * 10))
            ax_data.plot(
                x, y, "o", color=colors[i], markersize=12, label="label" + str(i)
            )
            ax_data.plot(
                xfit,
                self.fz(df.K.loc[key], [df.SA.loc[key], df.SB.loc[key]], xfit),
                "-",
                lw=2,
                color=colors[i],
                alpha=0.8,
            )
            ax_data.set_xticks(ax_data.get_xticks()[1:-1])
            # MAYBE ax_data.set_yscale('log')
            residues.append(
                y - self.fz(df.K.loc[key], [df.SA.loc[key], df.SB.loc[key]], x)
            )
            # Print out.
            line = ["%1.2f" % v for v in list(df[out].loc[key])]
            for _i in range(4):
                line.append("")
            lines.append(line)
        # ## residues
        ax1 = plt.subplot2grid((3, 1), loc=(2, 0))
        ax1.plot(
            x, residues[0], "o-", lw=2.5, color=colors[0], alpha=0.6, markersize=12
        )
        ax2 = plt.twinx(ax1)
        ax2.plot(
            x, residues[1], "o-", lw=2.5, color=colors[1], alpha=0.6, markersize=12
        )
        plt.subplots_adjust(hspace=0)
        ax1.set_xlim(ax_data.get_xlim())
        ax_data.legend()
        # global
        df = self.fittings[-1]
        lines.append(["%1.2f" % v for v in list(df[out2].loc[key])])
        ax_data.plot(
            xfit,
            self.fz(df.K.loc[key], [df.SA.loc[key], df.SB.loc[key]], xfit),
            "b--",
            lw=0.5,
        )
        ax_data.plot(
            xfit,
            self.fz(df.K.loc[key], [df.SA2.loc[key], df.SB2.loc[key]], xfit),
            "b--",
            lw=0.5,
        )
        ax_data.table(cellText=lines, colLabels=out2, loc="top")
        ax1.grid(0, axis="y")  # switch off horizontal
        ax2.grid(1, axis="both")
        # ## only residues
        y = self.labelblocksgroups[0].data[key]
        ax1.plot(
            x,
            (y - self.fz(df.K.loc[key], [df.SA.loc[key], df.SB.loc[key]], x)),
            "--",
            lw=1.5,
            color=colors[0],
        )
        y = self.labelblocksgroups[1].data[key]
        ax2.plot(
            x,
            (y - self.fz(df.K.loc[key], [df.SA2.loc[key], df.SB2.loc[key]], x)),
            "--",
            lw=1.5,
            color=colors[1],
        )
        if key in self.keys_ctrl:
            plt.title(
                "Ctrl: " + df["ctrl"].loc[key] + "  [" + key + "]", {"fontsize": 16}
            )
        else:
            plt.title(key, {"fontsize": 16})
        plt.close()
        return f

    def plot_all_wells(self, path: str) -> None:
        """Plot all wells into a pdf.

        Parameters
        ----------
        path : str
            Where the pdf file is saved.

        Raises
        ------
        Exception
            When fit is not yet run.

        """
        if not hasattr(self, "fittings"):
            raise Exception("run fit first")
        out = PdfPages(path)
        for k in self.fittings[0].loc[self.keys_ctrl].index:
            out.savefig(self.plot_well(k))
        for k in self.fittings[0].loc[self.keys_unk].sort_index().index:
            out.savefig(self.plot_well(k))
        out.close()

    def plot_ebar(
        self,
        lb: int,
        x: str = "K",
        y: str = "SA",
        xerr: str = "sK",
        yerr: str = "sSA",
        xmin: float | None = None,
        ymin: float | None = None,
        xmax: float | None = None,
        title: str | None = None,
    ) -> plt.figure:
        """Plot SA vs. K with errorbar for the whole plate."""
        if not hasattr(self, "fittings"):
            raise Exception("run fit first")
        df = self.fittings[lb]
        with plt.style.context("fivethirtyeight"):
            f = plt.figure(figsize=(10, 10))
            if xmin:
                df = df[df[x] > xmin]
            if xmax:
                df = df[df[x] < xmax]
            if ymin:
                df = df[df[y] > ymin]
            try:
                plt.errorbar(
                    df[x],
                    df[y],
                    xerr=df[xerr],
                    yerr=df[yerr],
                    fmt="o",
                    elinewidth=1,
                    markersize=10,
                    alpha=0.7,
                )
            except ValueError:
                pass
            if "ctrl" not in df:
                df["ctrl"] = 0
            df = df[~np.isnan(df[x])]
            df = df[~np.isnan(df[y])]
            for idx, xv, yv, l in zip(df.index, df[x], df[y], df["ctrl"]):
                # x or y do not exhist.# try:
                if type(l) == str:
                    color = "#" + hashlib.sha224(l.encode()).hexdigest()[2:8]
                    plt.text(xv, yv, l, fontsize=13, color=color)
                else:
                    plt.text(xv, yv, idx, fontsize=12)
                # x or y do not exhist.# except:
                # x or y do not exhist.# continue
            plt.yscale("log")
            # min(x) can be = NaN
            min_x = min(max([0.01, df[x].min()]), 14)
            min_y = min(max([0.01, df[y].min()]), 5000)
            plt.xlim(0.99 * min_x, 1.01 * df[x].max())
            plt.ylim(0.90 * min_y, 1.10 * df[y].max())
            plt.grid(1, axis="both")
            plt.ylabel(y)
            plt.xlabel(x)
            title = title if title else ""
            title += "  label:" + str(lb)
            plt.title(title, fontsize=15)
            return f

    def print_fitting(self, lb: int) -> None:
        """Print fitting parameters for the whole plate."""

        def df_print(df: pd.DataFrame) -> None:
            for i, r in df.iterrows():
                print(f"{i:s}", end=" ")
                for k in out[:2]:
                    print(f"{r[k]:7.2f}", end=" ")
                for k in out[2:]:
                    print(f"{r[k]:7.0f}", end=" ")
                print()

        df = self.fittings[lb]
        if "SA2" in df.keys():
            out = ["K", "sK", "SA", "sSA", "SB", "sSB", "SA2", "sSA2", "SB2", "sSB2"]
        else:
            out = ["K", "sK", "SA", "sSA", "SB", "sSB"]
        if len(self.keys_ctrl) > 0:
            res_ctrl = df.loc[self.keys_ctrl]
            gr = res_ctrl.groupby("ctrl")
            print("    " + " ".join([f"{x:>7s}" for x in out]))
            for g in gr:
                print(" ", g[0])
                df_print(g[1][out])
        res_unk = df.loc[self.keys_unk]
        print()
        print("    " + " ".join([f"{x:>7s}" for x in out]))
        print("  UNK")
        df_print(res_unk.sort_index())

    def plot_buffer(self, title: str | None = None) -> plt.figure:
        """Plot buffers (indicated in scheme) for all labelblocksgroups."""
        x = self.conc
        f, ax = plt.subplots(2, 1, figsize=(10, 10))
        for i, lbg in enumerate(self.labelblocksgroups):
            buf = copy.deepcopy(lbg.buffer)
            bg = buf.pop("bg")  # type: ignore
            bg_sd = buf.pop("bg_sd")  # type: ignore
            rowlabel = ["Temp"]
            lines = [
                [
                    f"{x:6.1f}"
                    for x in [
                        lb.metadata["Temperature"].value for lb in lbg.labelblocks
                    ]
                ]
            ]
            colors = plt.cm.Set3(np.linspace(0, 1, len(buf) + 1))  # type: ignore
            for j, (k, v) in enumerate(buf.items(), start=1):  # type: ignore
                rowlabel.append(k)
                lines.append([f"{x:6.1f}" for x in v])
                ax[i].plot(x, v, "o-", alpha=0.8, lw=2, markersize=3, color=colors[j])
            ax[i].errorbar(
                x,
                bg,
                yerr=bg_sd,
                fmt="o-.",
                markersize=15,
                lw=1,
                elinewidth=3,
                alpha=0.8,
                color="grey",
                label="label" + str(i),
            )
            plt.subplots_adjust(hspace=0.0)
            ax[i].legend(fontsize=22)
            if x[0] > x[-1]:  # reverse
                for line in lines:
                    line.reverse()
            ax[i].table(
                cellText=lines,
                rowLabels=rowlabel,
                loc="top",
                rowColours=colors,
                alpha=0.4,
            )
            ax[i].set_xlim(min(x) * 0.96, max(x) * 1.02)
            ax[i].set_yticks(ax[i].get_yticks()[:-1])
        ax[0].set_yticks(ax[0].get_yticks()[1:])
        ax[0].set_xticklabels("")
        if title:
            f.suptitle(title, fontsize=18)
        f.tight_layout(h_pad=5, rect=(0, 0, 1, 0.87))
        return f
