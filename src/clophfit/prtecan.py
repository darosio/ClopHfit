"""Parse Tecan files, group lists and fit titrations.

(Titrations are described in list.pH or list.cl file.

Builds 96 titrations and export them in txt files. In the case of 2 labelblocks
performs a global fit saving a png and printing the fitting results.)

:ref:`prtecan parse`:

* Labelblock
* Tecanfile

:ref:`prtecan group`:

* LabelblocksGroup
* TecanfilesGroup
* Titration
* TitrationAnalysis

Functions
---------
.. autofunction:: fit_titration
.. autofunction:: fz_Kd_singlesite
.. autofunction:: fz_pK_singlesite
.. autofunction:: extract_metadata
.. autofunction:: strip_lines

"""
from __future__ import annotations

import copy
import hashlib
import itertools
import os
import warnings
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any  # , overload
from typing import List
from typing import Sequence

import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import pandas as pd
import scipy  # type: ignore
import scipy.stats  # type: ignore
import seaborn as sb  # type: ignore
from matplotlib.backends.backend_pdf import PdfPages  # type: ignore
from numpy.typing import NDArray


# after set([type(ll[i][j]) for i in range(len(ll)) for j in range(13)])
list_of_lines = List[List[Any]]


def strip_lines(lines: list_of_lines) -> list_of_lines:
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
    stripped_lines = []
    for line in lines:
        sl = [line[i] for i in range(len(line)) if line[i] != ""]
        stripped_lines.append(sl)
    return stripped_lines


def extract_metadata(
    lines: list_of_lines,
) -> dict[str, str | list[str | int | float]]:
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
    {'Shaking (Linear) Amplitude:': [2, 'mm']}

    >>> lines = [['Excitation Wavelength', '', '', '', 400, 'nm', '', '', '', '', '']]
    >>> lines.append(['', 'Temperature: 26 °C', '', '', '', '', '', '', '', '', ''])
    >>> extract_metadata(lines)
    {'Excitation Wavelength': [400, 'nm'], 'Temperature': [26.0]}

    >>> lines = [['Label: Label1', '', '', '', '', '', '', '', '', '', '', '', '']]
    >>> extract_metadata(lines)
    {'Label': ['Label1']}

    >>> lines = [['Mode', '', '', '', 'Fluorescence Top Reading', '', '', '', '', '']]
    >>> extract_metadata(lines)
    {'Mode': ['Fluorescence Top Reading']}

    """
    stripped_lines = strip_lines(lines)
    temp: dict[str, list[float | int | str]] = {
        "Temperature": [float(line[0].split(":")[1].split("°C")[0])]
        for line in stripped_lines
        if len(line) == 1 and "Temperature" in line[0]
    }
    labl: dict[str, list[str | int | float]] = {
        "Label": [line[0].split(":")[1].strip()]
        for line in stripped_lines
        if len(line) == 1 and "Label" in line[0]
    }
    m1: dict[str, str | list[str | int | float]] = {
        line[0]: line[0]
        for line in stripped_lines
        if len(line) == 1 and "Label" not in line[0] and "Temperature" not in line[0]
    }
    m2: dict[str, str | list[str | int | float]] = {
        line[0]: line[1:] for line in stripped_lines if len(line) > 1
    }
    m2.update(m1)
    m2.update(temp)
    m2.update(labl)
    return m2


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
    tecanfile : Tecanfile | None
        Object containing (has-a) this Labelblock.
    lines : list_of_lines
        Lines for this Labelblock.

    Attributes
    ----------
    metadata : dict
        Metadata specific for this Labelblock.
    data : Dict[str, float]
        The 96 data values as {'well_name': value}.

    Raises
    ------
    Exception
        When data do not correspond to a complete 96-well plate.

    Warns
    -----
    Warning
        When it replaces "OVER" with ``np.nan`` for any saturated value.

    """

    tecanfile: Tecanfile | None
    lines: list_of_lines
    metadata: dict[str, str | list[str | int | float]] = field(init=False, repr=True)
    data: dict[str, float] = field(init=False, repr=True)

    def __post_init__(self) -> None:
        """Generate metadata and data for this labelblock."""
        if self.lines[14][0] == "<>" and self.lines[23] == self.lines[24] == [""] * 13:
            stripped = strip_lines(self.lines)
            stripped[14:23] = []
            self.metadata = extract_metadata(stripped)
            self.data = self._extract_data(self.lines[15:23])
        else:
            raise ValueError("Cannot build Labelblock: not 96 wells?")

    def _extract_data(self, lines: list_of_lines) -> dict[str, float]:
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
                        path = self.tecanfile.path if self.tecanfile else ""
                        warnings.warn(
                            "OVER value in {}{:0>2} well for {} of tecanfile: {}".format(
                                row, col, self.metadata["Label"], path
                            )
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

    def __eq__(self, other: object) -> bool:
        """Two labelblocks are equal when metadata KEYS are identical."""
        if not isinstance(other, Labelblock):
            return NotImplemented
        eq: bool = True
        for k in Labelblock._KEYS:
            eq &= self.metadata[k] == other.metadata[k]
        # 'Gain': [81.0, 'Manual'] = 'Gain': [81.0, 'Optimal'] They are equal
        eq &= self.metadata["Gain"][0] == other.metadata["Gain"][0]
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
    """Parse a .xls file as exported from Tecan.

    Parameters
    ----------
    path
        Name of the xls file.

    Attributes
    ----------
    path: str
        Tecan file path.
    metadata : dict[str, str | list[str | int | float]]
        General metadata for Tecanfile e.g. 'Date:' or 'Shaking Duration:'.
    labelblocks : List[Labelblock]
        All labelblocks contained in the file.

    Methods
    -------
    read_xls(path) :
        Read xls file at path.
    lookup_csv_lines(csvl, pattern='Label: Label', col=0) :
        Return row index for pattern found at col.

    Raises
    ------
    FileNotFoundError
        When path does not exist.
    Exception
        When no Labelblock is found.

    """

    path: str
    metadata: dict[str, str | list[str | int | float]] = field(init=False, repr=True)
    labelblocks: list[Labelblock] = field(init=False, repr=True)

    def __post_init__(self) -> None:
        """Initialize."""
        csvl = Tecanfile.read_xls(self.path)
        idxs = Tecanfile.lookup_csv_lines(csvl, pattern="Label: Label", col=0)
        if len(idxs) == 0:
            raise ValueError("No Labelblock found.")
        self.metadata = extract_metadata(csvl[: idxs[0]])
        labelblocks = []
        n_labelblocks = len(idxs)
        idxs.append(len(csvl))
        for i in range(n_labelblocks):
            labelblocks.append(Labelblock(self, csvl[idxs[i] : idxs[i + 1]]))
        self.labelblocks = labelblocks

    def __hash__(self) -> int:
        """Define hash (related to __eq__) using self.path."""
        return hash(self.path)

    @classmethod
    def read_xls(cls, path: str) -> list_of_lines:
        """Read first sheet of an xls file.

        Parameters
        ----------
        path : str
            Path to .xls file.

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

    @classmethod
    def lookup_csv_lines(
        cls,
        csvl: list_of_lines,
        pattern: str = "Label: Label",
        col: int = 0,
    ) -> list[int]:
        """Lookup the line number where given pattern occurs.

        If nothing found return empty list.

        Parameters
        ----------
        csvl : list_of_lines
            Lines of a csv/xls file.
        pattern : str
            Pattern to be searched for., default="Label: Label"
        col : int
            Column to search (line-by-line).

        Returns
        -------
        list[int]
            Row/line index for all occurrences of pattern.

        """
        idxs = []
        for i, line in enumerate(csvl):
            if pattern in line[col]:
                idxs.append(i)
        return idxs


@dataclass
class LabelblocksGroup:
    """Group of labelblocks with 'equal' metadata.

    Parameters
    ----------
    labelblocks
        List of labelblocks with 'equal' metadata.

    Attributes
    ----------
    metadata : dict
        The common metadata.
    temperatures : List[float]
        The temperatire value for each Labelblock.
    data : Dict[str, List[float]]
        The usual dict for data (see Labelblock) with well name as key but with
        list of values as value.

    Raises
    ------
    Exception
        If metadata are not all 'equal'.

    """

    labelblocks: list[Labelblock]
    metadata: dict[str, str | list[str | int | float]] = field(init=False, repr=True)
    temperature: Sequence[float] = field(init=False, repr=True)
    data: dict[str, list[float]] = field(init=False, repr=True)
    buffer: dict[str, list[float]] = field(init=False, repr=True)

    def __post_init__(self) -> None:
        """Create common metadata and list for data and temperatures."""
        if all(self.labelblocks[0] == lb for lb in self.labelblocks[1:]):
            # build common metadata only
            metadata = {}
            for k in Labelblock._KEYS:
                metadata[k] = self.labelblocks[0].metadata[k]
                # list with first element don't care about Manual/Optimal
            metadata["Gain"] = [self.labelblocks[0].metadata["Gain"][0]]
            self.metadata = metadata
            # temperatures
            temperatures = []
            for lb in self.labelblocks:
                temperatures.append(lb.metadata["Temperature"][0])
            self.temperatures = temperatures
            # data
            datagrp: dict[str, list[float]] = {}
            for key in self.labelblocks[0].data.keys():
                datagrp[key] = []
                for lb in self.labelblocks:
                    datagrp[key].append(lb.data[key])
            self.data = datagrp
        else:
            raise ValueError("Creation of labelblock group failed.")


@dataclass
class TecanfilesGroup:
    """Group of Tecanfiles containing at least one common Labelblock.

    Parameters
    ----------
    filenames
        List of xls (paths) filenames.

    Attributes
    ----------
    labelblocksgroups : List[LabelblocksGroup]
       Each group contains its own data like a titration.

    Raises
    ------
    Exception
        When is not possible to build any LabelblocksGroup because nothing
        in common between files (listed in filenames).

    Warns
    -----
    Warning
        The Tecanfiles listed in *filenames* are suppossed to contain the
        "same" list (of length N) of Labelblocks. So, N labelblocksgroups
        will be normally created. A warn will raise if not all Tecanfiles
        contains the same number of Labelblocks ('equal' mergeable) in the
        same order, but a number M < N of groups can be built.

    """

    filenames: list[str]
    metadata: dict[str, str | list[str | int | float]] = field(init=False, repr=True)
    labelblocksgroups: list[LabelblocksGroup] = field(init=False, repr=True)

    def __post_init__(self) -> None:
        """Create metadata and labelblocksgroups."""
        tecanfiles = []
        for f in self.filenames:
            tecanfiles.append(Tecanfile(f))
        tf0 = tecanfiles[0]
        grps = []
        if all([tf0.labelblocks == tf.labelblocks for tf in tecanfiles[1:]]):
            # expected behaviour
            for i, _lb in enumerate(tf0.labelblocks):
                gr = LabelblocksGroup([tf.labelblocks[i] for tf in tecanfiles])
                grps.append(gr)
        else:
            # Try to creates as many as possible groups of labelblocks
            # with length=len(tecanfiles).
            # Not for 'equal' labelblocks within the same tecanfile.
            n_tecanfiles = len(tecanfiles)
            nmax_labelblocks = max(len(tf.labelblocks) for tf in tecanfiles)
            for idx in itertools.product(range(nmax_labelblocks), repeat=n_tecanfiles):
                try:
                    for i, tf in enumerate(tecanfiles):
                        tf.labelblocks[idx[i]]
                except IndexError:
                    continue
                # if all labelblocks exhist
                else:
                    try:
                        gr = LabelblocksGroup(
                            [tf.labelblocks[idx[i]] for i, tf in enumerate(tecanfiles)]
                        )
                    except AssertionError:
                        continue
                    # if labelblocks are all 'equal'
                    else:
                        grps.append(gr)
            if len(grps) == 0:
                raise Exception(
                    "No common labelblock in filenames" + str(self.filenames)
                )
            else:
                warnings.warn(
                    "Different LabelblocksGroup among filenames." + str(self.filenames)
                )
        self.metadata = tecanfiles[0].metadata
        self.labelblocksgroups = grps


@dataclass(init=False)
class Titration(TecanfilesGroup):
    """Group tecanfiles into a Titration as indicated by a listfile.

    The script will work from any directory: list.pH list filenames relative to
    its position.

    Parameters
    ----------
    listfile
        File path to the listfile ([tecan_file_path conc]).

    Attributes
    ----------
    conc : List[float]
        Concentration values common to all 96 titrations.
    labelblocksgroups: List[LabelblocksGroup]
        List of labelblocksgroups.

    """

    conc: Sequence[float] = field(init=False, repr=True)

    def __init__(self, listfile: Path) -> None:
        try:
            df = pd.read_table(listfile, names=["filenames", "conc"])
        except FileNotFoundError:
            raise FileNotFoundError(f"Cannot find: {listfile}")
        if df["filenames"].count() != df["conc"].count():
            raise ValueError(f"Check format [filenames conc] for listfile: {listfile}")
        self.conc = df["conc"].tolist()
        dirname = os.path.dirname(listfile)
        filenames = [os.path.join(dirname, fn) for fn in df["filenames"]]
        super().__init__(filenames)

    def export_dat(self, path: str) -> None:
        """Export dat files [x,y1,..,yN] from labelblocksgroups.

        Parameters
        ----------
        path : str
            Path to output folder.

        """
        if not os.path.isdir(path):
            os.makedirs(path)
        for key, dy1 in self.labelblocksgroups[0].data.items():
            df = pd.DataFrame({"x": self.conc, "y1": dy1})
            for n, lb in enumerate(self.labelblocksgroups[1:], start=2):
                dy = lb.data[key]
                df["y" + str(n)] = dy
            df.to_csv(os.path.join(path, key + ".dat"), index=False)


@dataclass
class TitrationAnalysis:
    """Perform analysis of a titration.

    Parameters
    ----------
    titration
        Titration object.
    schemefile
        File path to the schemefile (e.g. {"C01: 'V224Q'"}).

    Attributes
    ----------
    scheme : pd.DataFrame or pd.Series FIXME
        e.g. {'buffer': ['H12']}
    conc : List[float]
        Concentration values common to all 96 titrations.
    labelblocksgroups : List[LabelblocksGroup]
        Deepcopy from titration.

    Methods
    -------
    subtract_bg

    dilution_correction

    metadata_normalization

    calculate_conc

    fit

    """

    titration: Titration
    schemefile: str | None = None
    scheme: pd.Series[Any] = field(init=False, repr=True)
    conc: Sequence[float] = field(init=False, repr=True)
    labelblocksgroups: list[LabelblocksGroup] = field(init=False, repr=True)
    additions: Sequence[float] = field(init=False, repr=True)

    def __post_init__(self) -> None:
        """Create attributes."""
        if self.schemefile is None:
            self.scheme = pd.Series({"well": []})
        else:
            df = pd.read_table(self.schemefile)
            try:
                assert df.columns.tolist() == ["well", "sample"]
                assert df["well"].count() == df["sample"].count()
            except AssertionError as err:
                msg = "Check format [well sample] for schemefile: "
                raise AssertionError(msg + self.schemefile) from err
            self.scheme = df.groupby("sample")["well"].unique()
        self.conc = self.titration.conc
        self.labelblocksgroups = copy.deepcopy(self.titration.labelblocksgroups)

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

    def metadata_normalization(self) -> None:
        """Normalize signal using gain, flashes and integration time."""
        if hasattr(self, "normalized"):
            warnings.warn("Normalization using metadata was already applied.")
            return
        for lbg in self.labelblocksgroups:
            corr = 1000 / float(lbg.metadata["Gain"][0])
            corr /= float(lbg.metadata["Integration Time"][0])
            corr /= float(lbg.metadata["Number of Flashes"][0])
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
            bg = buf.pop("bg")
            bg_sd = buf.pop("bg_sd")
            rowlabel = ["Temp"]
            lines = [[f"{x:6.1f}" for x in lbg.temperatures]]
            colors = plt.cm.Set3(np.linspace(0, 1, len(buf) + 1))
            for j, (k, v) in enumerate(buf.items(), start=1):
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
