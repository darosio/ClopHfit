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
import copy
import hashlib
import itertools
import os
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union  # , overload

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import scipy.stats
import seaborn as sb
from matplotlib.backends.backend_pdf import PdfPages

list_of_lines = List[List]
# bug xdoctest-3.7 #import numpy.typing as npt


def strip_lines(lines: list_of_lines) -> list_of_lines:
    """Remove empty fields/cells from lines read from a csv file.

    ([a,,b,,,]-->[a,b])

    Parameters
    ----------
    lines
        Lines that are a list of fields, typically from a csv/xls file.

    Returns
    -------
        Lines removed from blank cells.

    """
    stripped_lines = []
    for line in lines:
        sl = [line[i] for i in range(len(line)) if line[i] != '']
        stripped_lines.append(sl)
    return stripped_lines


# def extract_metadata(lines: list_of_lines) -> Dict[str, Union[str, float, List[Any]]]:
def extract_metadata(lines: list_of_lines) -> Dict[str, Any]:
    """Extract metadata from a list of stripped lines.

    First field is the *key*, remaining fields goes into a list of values::

      ['', 'key', '', '', 'value1', '', ..., 'valueN', ''] -->
                                        {key: [value1, ..., valueN]}

    *Label* and *Temperature* are two exceptions::

      ['Label: labelXX', '', '', '', '', '', '', '']
      ['', 'Temperature: XX °C', '', '', '', '', '', '']

    Parameters
    ----------
    lines
        Lines that are a list of fields, typically from a csv/xls file.

    Returns
    -------
        Metadata for Tecanfile or Labelblock.

    """
    stripped_lines = strip_lines(lines)
    temp = {
        'Temperature': float(line[0].split(':')[1].split('°C')[0])
        for line in stripped_lines
        if len(line) == 1 and 'Temperature' in line[0]
    }
    labl = {
        'Label': line[0].split(':')[1].strip()
        for line in stripped_lines
        if len(line) == 1 and 'Label' in line[0]
    }
    m1 = {
        line[0]: line[0]
        for line in stripped_lines
        if len(line) == 1 and 'Label' not in line[0] and 'Temperature' not in line[0]
    }
    m2: Dict[str, Union[str, float, List[str]]] = {
        line[0]: line[1:] for line in stripped_lines if len(line) > 1
    }
    m2.update(m1)
    m2.update(temp)
    m2.update(labl)
    return m2


def fz_Kd_singlesite(K: float, p: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Fit function for Cl titration."""
    return (p[0] + p[1] * x / K) / (1 + x / K)


def fz_pK_singlesite(K: float, p: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Fit function for pH titration."""
    return (p[1] + p[0] * 10 ** (K - x)) / (1 + 10 ** (K - x))


def fit_titration(
    kind: str,
    x: np.ndarray,
    y: np.ndarray,
    y2: Optional[np.ndarray] = None,
    residue: Optional[np.ndarray] = None,
    residue2: Optional[np.ndarray] = None,
    tval_conf: float = 0.95,
) -> pd.DataFrame:
    """Fit pH or Cl titration using a single-site binding model.

    Returns confidence interval (default=0.95) for fitting params (cov*tval), rather than
    standard error of the fit. Use scipy leastsq. Determine 3 fitting parameters:
    - binding constant *K*
    - and 2 plateau *SA* and *SB*.

    Parameters
    ----------
    kind
        Titration type {'pH'|'Cl'}
    x, y
        Main dataset.
    y2
        Second dataset (share x with main dataset).
    residue
        Residues for main dataset.
    residue2
        Residues for second dataset.
    tval_conf
        Confidence level (default 0.95) for parameter estimations.

    Returns
    -------
        Fitting results.

    Raises
    ------
    NameError
        When kind is different than "pH" or "Cl".

    Examples
    --------
    >>> fit_titration("Cl", [1, 10, 30, 100, 200], [10, 8, 5, 1, 0.1])[["K", "sK"]]
               K         sK
    0  38.955406  30.201929

    """
    if kind == 'pH':
        fz = fz_pK_singlesite
    elif kind == 'Cl':
        fz = fz_Kd_singlesite
    else:
        raise NameError('kind= pH or Cl')

    def compute_p0(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        df = pd.DataFrame({'x': x, 'y': y})
        SA = df.y[df.x == min(df.x)].values[0]
        SB = df.y[df.x == max(df.x)].values[0]
        K = np.average([max(y), min(y)])
        try:
            x1, y1 = df[df['y'] >= K].values[0]
        except IndexError:
            x1 = np.nan
            y1 = np.nan
        try:
            x2, y2 = df[df['y'] <= K].values[0]
        except IndexError:
            x2 = np.nan
            y2 = np.nan
        K = (x2 - x1) / (y2 - y1) * (K - y1) + x1
        return np.r_[K, SA, SB]

    x = np.array(x)
    y = np.array(y)

    if y2 is None:

        def ssq1(p: np.ndarray, x: np.ndarray, y1: np.ndarray) -> np.ndarray:
            return np.r_[y1 - fz(p[0], p[1:3], x)]

        p0 = compute_p0(x, y)
        p, cov, info, msg, success = scipy.optimize.leastsq(
            ssq1, p0, args=(x, y), full_output=True, xtol=1e-11
        )
    else:

        def ssq2(
            p: np.ndarray,
            x: np.ndarray,
            y1: np.ndarray,
            y2: np.ndarray,
            rd1: np.ndarray,
            rd2: np.ndarray,
        ) -> np.ndarray:
            return np.r_[
                (y1 - fz(p[0], p[1:3], x)) / rd1**2,
                (y2 - fz(p[0], p[3:5], x)) / rd2**2,
            ]

        p1 = compute_p0(x, y)
        p2 = compute_p0(x, y2)
        ave = np.average([p1[0], p2[0]])
        p0 = np.r_[ave, p1[1], p1[2], p2[1], p2[2]]
        tmp = scipy.optimize.leastsq(
            ssq2, p0, full_output=True, xtol=1e-11, args=(x, y, y2, residue, residue2)
        )
        p, cov, info, msg, success = tmp
    res = pd.DataFrame({'ss': [success]})
    res['msg'] = msg
    if 1 <= success <= 4:
        try:
            tval = (tval_conf + 1) / 2
            chisq = sum(info['fvec'] * info['fvec'])
            res['df'] = len(y) - len(p)
            res['tval'] = scipy.stats.distributions.t.ppf(tval, res.df)
            res['chisqr'] = chisq / res.df
            res['K'] = p[0]
            res['SA'] = p[1]
            res['SB'] = p[2]
            if y2 is not None:
                res['df'] += len(y2)
                res['tval'] = scipy.stats.distributions.t.ppf(tval, res.df)
                res['chisqr'] = chisq / res.df
                res['SA2'] = p[3]
                res['SB2'] = p[4]
                res['sSA2'] = np.sqrt(cov[3][3] * res.chisqr) * res.tval
                res['sSB2'] = np.sqrt(cov[4][4] * res.chisqr) * res.tval
            res['sK'] = np.sqrt(cov[0][0] * res.chisqr) * res.tval
            res['sSA'] = np.sqrt(cov[1][1] * res.chisqr) * res.tval
            res['sSB'] = np.sqrt(cov[2][2] * res.chisqr) * res.tval
        except TypeError:
            pass  # if some params are not successfully determined.
    return res


class Labelblock:
    """Parse a label block within a Tecan file.

    Parameters
    ----------
    tecanfile :
        Object containing (has-a) this Labelblock.
    lines :
        Lines for this Labelblock.

    Attributes
    ----------
    tecanfile

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

    def __init__(
        self,
        tecanfile: Optional['Tecanfile'],
        lines: list_of_lines,
    ) -> None:
        try:
            assert lines[14][0] == '<>' and lines[23] == lines[24] == [
                '',
                '',
                '',
                '',
                '',
                '',
                '',
                '',
                '',
                '',
                '',
                '',
                '',
            ]
        except AssertionError as err:
            raise Exception('Cannot build Labelblock: not 96 wells?') from err
        stripped = strip_lines(lines)
        stripped[14:23] = []
        self.tecanfile = tecanfile
        self.metadata = extract_metadata(stripped)
        self.data = self._extract_data(lines[15:23])

    def _extract_data(self, lines: list_of_lines) -> Dict[str, float]:
        """Convert data into a dictionary.

        {'A01' : value}
        :
        {'H12' : value}

        Parameters
        ----------
        lines
            xls file read into lines.

        Returns
        -------
        dict
            Data from a label block.

        Raises
        ------
        Exception
            When something went wrong. Possibly because not 96-well.

        Warns
        -----
            When a cell contains saturated signal (converted into np.nan).

        """
        rownames = tuple('ABCDEFGH')
        data = {}
        try:
            assert len(lines) == 8
            for i, row in enumerate(rownames):
                assert lines[i][0] == row  # e.g. "A" == "A"
                for col in range(1, 13):
                    try:
                        data[row + "{0:0>2}".format(col)] = float(lines[i][col])
                    except ValueError:
                        data[row + "{0:0>2}".format(col)] = np.nan
                        path = self.tecanfile.path if self.tecanfile else ""
                        warnings.warn(
                            "OVER value in {0}{1:0>2} well for {2} of tecanfile: {3}".format(
                                row, col, self.metadata['Label'], path
                            )
                        )
        except AssertionError as err:
            raise Exception("Cannot extract data in Labelblock: not 96 wells?") from err
        return data

    KEYS = [
        'Emission Bandwidth',
        'Emission Wavelength',
        'Excitation Bandwidth',
        'Excitation Wavelength',
        'Integration Time',
        'Mode',
        'Number of Flashes',
    ]

    def __eq__(self, other: object) -> bool:
        """Two labelblocks are equal when metadata KEYS are identical."""
        # Identical labelblocks can be grouped safely into the same titration; otherwise
        # some kind of normalization (# of flashes, gain, etc.) would be
        # necessary.
        if not isinstance(other, Labelblock):
            return NotImplemented
        eq: bool = True
        for k in Labelblock.KEYS:
            eq *= self.metadata[k] == other.metadata[k]
        # 'Gain': [81.0, 'Manual'] = 'Gain': [81.0, 'Optimal'] They are equal
        eq *= self.metadata['Gain'][0] == other.metadata['Gain'][0]
        # annotation error: Value of type "Union[str, float, List[str]]" is not indexable
        return eq


class Tecanfile:
    """Parse a .xls file as exported from Tecan.

    Parameters
    ----------
    path
        Name of the xls file.

    Attributes
    ----------
    path

    metadata : dict
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

    def __init__(self, path: str) -> None:
        csvl = Tecanfile.read_xls(path)
        idxs = Tecanfile.lookup_csv_lines(csvl, pattern='Label: Label', col=0)
        if len(idxs) == 0:
            raise Exception('No Labelblock found.')
        # path
        self.path = path
        # metadata
        self.metadata = extract_metadata(csvl[: idxs[0]])
        # labelblocks
        labelblocks = []
        n_labelblocks = len(idxs)
        idxs.append(len(csvl))
        for i in range(n_labelblocks):
            labelblocks.append(Labelblock(self, csvl[idxs[i] : idxs[i + 1]]))
        self.labelblocks = labelblocks

    def __eq__(self, other: object) -> bool:
        """Two Tecanfile are equal if their attributes are."""
        # never used thus far.
        # https://izziswift.com/compare-object-instances-for-equality-by-their-attributes/
        return self.__dict__ == other.__dict__

    def __hash__(self) -> int:
        """Define hash (related to __eq__) using self.path."""
        return hash(self.path)

    @classmethod
    def read_xls(cls, path: str) -> list_of_lines:
        """Read first sheet of an xls file.

        Parameters
        ----------
        path
            Path to .xls file.

        Returns
        -------
            Lines.

        """
        df = pd.read_excel(path)
        n0 = pd.DataFrame([[np.nan] * len(df.columns)], columns=df.columns)
        df = pd.concat([n0, df], ignore_index=True)
        df.fillna('', inplace=True)
        return df.values.tolist()

    @classmethod
    def lookup_csv_lines(
        cls,
        csvl: list_of_lines,
        pattern: str = 'Label: Label',
        col: int = 0,
    ) -> List[int]:
        """Lookup the line number where given pattern occurs.

        If nothing found return empty list.

        Parameters
        ----------
        csvl
            Lines of a csv/xls file.
        pattern
            Pattern to be searched for., default="Label: Label"
        col
            Column to search (line-by-line).

        Returns
        -------
            Row/line index for all occurrences of pattern.

        """
        idxs = []
        for i, line in enumerate(csvl):
            if pattern in line[col]:
                idxs.append(i)
        return idxs


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

    buffer: Dict[str, List[float]]
    data: Dict[str, List[float]]

    def __init__(self, labelblocks: List[Labelblock]) -> None:
        try:
            for lb in labelblocks[1:]:
                assert labelblocks[0] == lb
        except AssertionError as err:
            raise AssertionError('Creation of labelblock group failed.') from err
        # build common metadata only
        metadata = {}
        for k in Labelblock.KEYS:
            metadata[k] = labelblocks[0].metadata[k]
            # list with first element don't care about Manual/Optimal
        metadata['Gain'] = [labelblocks[0].metadata['Gain'][0]]
        self.metadata = metadata
        # temperatures
        temperatures = []
        for lb in labelblocks:
            temperatures.append(lb.metadata['Temperature'])
        self.temperatures = temperatures
        # data
        datagrp: Dict[str, List[float]] = {}
        for key in labelblocks[0].data.keys():
            datagrp[key] = []
            for lb in labelblocks:
                datagrp[key].append(lb.data[key])
        self.data = datagrp


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

    def __init__(self, filenames: List[str]) -> None:
        tecanfiles = []
        for f in filenames:
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
            nmax_labelblocks = max([len(tf.labelblocks) for tf in tecanfiles])
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
                raise Exception('No common labelblock in filenames' + str(filenames))
            else:
                warnings.warn(
                    'Different LabelblocksGroup among filenames.' + str(filenames)
                )
        self.metadata = tecanfiles[0].metadata
        self.labelblocksgroups = grps


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

    def __init__(self, listfile: str) -> None:
        try:
            df = pd.read_table(listfile, names=['filenames', 'conc'])
        except FileNotFoundError as err:
            raise FileNotFoundError('Cannot find: ' + listfile) from err
        try:
            assert df["filenames"].count() == df["conc"].count()
        except AssertionError as err:
            msg = 'Check format [filenames conc] for listfile: '
            raise Exception(msg + listfile) from err
        self.conc = df["conc"].tolist()
        dirname = os.path.dirname(listfile)
        filenames = [os.path.join(dirname, fn) for fn in df["filenames"]]
        super().__init__(filenames)

    def export_dat(self, path: str) -> None:
        """Export dat files [x,y1,..,yN] from labelblocksgroups.

        Parameters
        ----------
        path
            Path to output folder.

        """
        if not os.path.isdir(path):
            os.makedirs(path)
        for key, dy1 in self.labelblocksgroups[0].data.items():
            df = pd.DataFrame({'x': self.conc, 'y1': dy1})
            for n, lb in enumerate(self.labelblocksgroups[1:], start=2):
                dy = lb.data[key]
                df['y' + str(n)] = dy
            df.to_csv(os.path.join(path, key + '.dat'), index=False)


class TitrationAnalysis(Titration):
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

    def __init__(self, titration: Titration, schemefile: Optional[str] = None) -> None:
        if schemefile is None:
            self.scheme = pd.Series({'well': []})
        else:
            df = pd.read_table(schemefile)
            try:
                assert df.columns.tolist() == ['well', 'sample']
                assert df["well"].count() == df["sample"].count()
            except AssertionError as err:
                msg = 'Check format [well sample] for schemefile: '
                raise AssertionError(msg + schemefile) from err
            self.scheme = df.groupby('sample')["well"].unique()
        self.conc = np.array(titration.conc)
        self.labelblocksgroups = copy.deepcopy(titration.labelblocksgroups)

    def subtract_bg(self) -> None:
        """Subtract average buffer values for each titration point."""
        buffer_keys = self.scheme.pop('buffer')
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
        additionsfile
            File listing volume additions during titration.

        """
        if hasattr(self, 'additions'):
            warnings.warn('Dilution correction was already applied.')
            return
        df = pd.read_table(additionsfile, names=['add'])
        self.additions = df["add"]
        volumes = np.cumsum(self.additions)
        corr = volumes / volumes[0]
        for lbg in self.labelblocksgroups:
            for k in lbg.data:
                lbg.data[k] *= corr

    @classmethod
    def calculate_conc(
        cls,
        additions: Union[np.ndarray, List[float]],
        conc_stock: float,
        conc_ini: float = 0.0,
    ) -> np.ndarray:
        """Calculate concentration values.

        additions[0]=vol_ini; Stock concentration is a parameter.

        Parameters
        ----------
        additions
            Initial volume and all subsequent additions.
        conc_stock
            Concentration of the stock used for additions.
        conc_ini
            Initial concentration (default=0).

        Returns
        -------
            Concentrations as vector.

        Examples
        --------
        >>> additions = [112, 2, 2, 2, 2, 2, 2, 6, 4]
        >>> TitrationAnalysis.calculate_conc(additions, 1000)
        array([   0.        ,   17.54385965,   34.48275862,   50.84745763,
                 66.66666667,   81.96721311,   96.77419355,  138.46153846,
                164.17910448])

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
        if hasattr(self, 'normalized'):
            warnings.warn('Normalization using metadata was already applied.')
            return
        for lbg in self.labelblocksgroups:
            corr = 1000 / lbg.metadata['Gain'][0]
            corr /= lbg.metadata['Integration Time'][0]
            corr /= lbg.metadata['Number of Flashes'][0]
            for k in lbg.data:
                lbg.data[k] *= corr
        self.normalized = True

    def _get_keys(self) -> None:
        """Get plate positions of crtl and unk samples."""
        self.keys_ctrl = [k for ctr in self.scheme.tolist() for k in ctr]
        self.names_ctrl = list(self.scheme.to_dict())
        self.keys_unk = list(
            self.labelblocksgroups[0].data.keys() - set(self.keys_ctrl)
        )

    def fit(
        self: Any,
        kind: str,
        ini: int = 0,
        fin: Optional[int] = None,
        no_weight: bool = False,
        **kwargs: Any
    ) -> None:
        """Fit titrations.

        Here is less general. It is for 2 labelblocks.

        Parameters
        ----------
        kind
            Titration type {'pH'|'Cl'}
        ini
            Initial point (default: 0).
        fin
            Final point (default: None).
        no_weight
            Do not use residues from single Labelblock fit as weight for global fitting.
        **kwargs
            Only for tval different from default=0.95 for the confint calculation.

        """
        if kind == 'Cl':
            self.fz = fz_Kd_singlesite
        elif kind == 'pH':
            self.fz = fz_pK_singlesite
        x = self.conc
        fittings = []
        for lbg in self.labelblocksgroups:
            fitting = pd.DataFrame()
            for k, y in lbg.data.items():
                res = fit_titration(kind, x[ini:fin], np.array(y[ini:fin]), **kwargs)
                res.index = [k]
                # fitting = fitting.append(res, sort=False) DDD
                fitting = pd.concat([fitting, res], sort=False)
                # TODO assert (fitting.columns == res.columns).all()
                # better to refactor this function

            fittings.append(fitting)
        # global weighted on relative residues of single fittings
        fitting = pd.DataFrame()
        for k, y in self.labelblocksgroups[0].data.items():
            y2 = np.array(self.labelblocksgroups[1].data[k])
            y = np.array(y)
            residue = y - self.fz(
                fittings[0]['K'].loc[k],
                np.array([fittings[0]['SA'].loc[k], fittings[0]['SB'].loc[k]]),
                x,
            )
            residue /= y  # TODO residue or
            # log(residue/y) https://www.tandfonline.com/doi/abs/10.1080/00031305.1985.10479385
            residue2 = y2 - self.fz(
                fittings[1]['K'].loc[k],
                np.array([fittings[1]['SA'].loc[k], fittings[1]['SB'].loc[k]]),
                x,
            )
            residue2 /= y2
            if no_weight:
                for i, _rr in enumerate(residue):
                    residue[i] = 1  # TODO use np.ones() but first find a way to test
                    residue2[i] = 1
            res = fit_titration(
                kind,
                x[ini:fin],
                y[ini:fin],
                y2=y2[ini:fin],
                residue=residue[ini:fin],
                residue2=residue2[ini:fin],
                **kwargs
            )
            res.index = [k]
            # fitting = fitting.append(res, sort=False) DDD
            fitting = pd.concat([fitting, res], sort=False)
        fittings.append(fitting)
        for fitting in fittings:
            for ctrl, v in self.scheme.items():
                for k in v:
                    fitting.loc[k, 'ctrl'] = ctrl
        # self.fittings and self.fz
        self.fittings = fittings
        self._get_keys()

    def plot_K(
        self,
        lb: int,
        xlim: Optional[Tuple[float, float]] = None,
        title: Optional[str] = None,
    ) -> plt.figure:
        """Plot K values as stripplot.

        Parameters
        ----------
        lb
            Labelblock index.
        xlim
            Range.
        title
            To name the plot.

        Returns
        -------
            The figure.

        Raises
        ------
        Exception
            When no fitting results are available (in this object).

        """
        if not hasattr(self, 'fittings'):
            raise Exception('run fit first')
        sb.set(style="whitegrid")
        f = plt.figure(figsize=(12, 16))
        # Ctrls
        ax1 = plt.subplot2grid((8, 1), loc=(0, 0))
        if len(self.keys_ctrl) > 0:
            res_ctrl = self.fittings[lb].loc[self.keys_ctrl]
            sb.stripplot(
                x=res_ctrl['K'],
                y=res_ctrl.index,
                size=8,
                orient='h',
                hue=res_ctrl.ctrl,
                ax=ax1,
            )
            plt.errorbar(
                res_ctrl.K,
                range(len(res_ctrl)),
                xerr=res_ctrl.sK,  # xerr=res_ctrl.sK*res_ctrl.tval,
                fmt='.',
                c="lightgray",
                lw=8,
            )
            plt.grid(1, axis='both')
        # Unks
        #  FIXME keys_unk is an attribute or a property
        res_unk = self.fittings[lb].loc[self.keys_unk]
        ax2 = plt.subplot2grid((8, 1), loc=(1, 0), rowspan=7)
        sb.stripplot(
            x=res_unk['K'].sort_index(),
            y=res_unk.index,
            size=12,
            orient='h',
            palette="Greys",
            hue=res_unk['SA'].sort_index(),
            ax=ax2,
        )
        plt.legend('')
        plt.errorbar(
            res_unk['K'].sort_index(),
            range(len(res_unk)),
            xerr=res_unk['sK'].sort_index(),
            fmt='.',
            c="gray",
            lw=2,
        )
        plt.yticks(range(len(res_unk)), res_unk.index.sort_values())
        plt.ylim(-1, len(res_unk))
        plt.grid(1, axis='both')
        if not xlim:
            xlim = (
                0.99 * min(res_ctrl['K'].min(), res_unk['K'].min()),
                1.01 * max(res_ctrl['K'].max(), res_unk['K'].max()),
            )
        ax1.set_xlim(xlim)
        ax2.set_xlim(xlim)
        ax1.set_xticklabels([])
        ax1.set_xlabel('')
        title = title if title else ''
        title += '  label:' + str(lb)
        f.suptitle(title, fontsize=16)
        f.tight_layout(pad=1.2, w_pad=0.1, h_pad=0.5, rect=(0, 0, 1, 0.97))
        return f

    def plot_well(self, key: str) -> plt.figure:
        """Plot global fitting using 2 labelblocks.

        Here is less general. It is for 2 labelblocks.

        Parameters
        ----------
        key
            Well position as dictionary key like "A01".

        Returns
        -------
            Pointer to mpl.figure.

        Raises
        ------
        Exception
            When fit is not yet run.

        """
        if not hasattr(self, 'fittings'):
            raise Exception('run fit first')
        plt.style.use(['seaborn-ticks', 'seaborn-whitegrid'])
        out = ['K', 'sK', 'SA', 'sSA', 'SB', 'sSB']
        out2 = ['K', 'sK', 'SA', 'sSA', 'SB', 'sSB', 'SA2', 'sSA2', 'SB2', 'sSB2']
        x = self.conc
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
                x, y, 'o', color=colors[i], markersize=12, label='label' + str(i)
            )
            ax_data.plot(
                xfit,
                self.fz(df.K.loc[key], [df.SA.loc[key], df.SB.loc[key]], xfit),
                '-',
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
            line = ['%1.2f' % v for v in list(df[out].loc[key])]
            for _i in range(4):
                line.append('')
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
        lines.append(['%1.2f' % v for v in list(df[out2].loc[key])])
        ax_data.plot(
            xfit,
            self.fz(df.K.loc[key], [df.SA.loc[key], df.SB.loc[key]], xfit),
            'b--',
            lw=0.5,
        )
        ax_data.plot(
            xfit,
            self.fz(df.K.loc[key], [df.SA2.loc[key], df.SB2.loc[key]], xfit),
            'b--',
            lw=0.5,
        )
        ax_data.table(cellText=lines, colLabels=out2, loc='top')
        ax1.grid(0, axis='y')  # switch off horizontal
        ax2.grid(1, axis='both')
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
                "Ctrl: " + df['ctrl'].loc[key] + "  [" + key + "]", {'fontsize': 16}
            )
        else:
            plt.title(key, {'fontsize': 16})
        plt.close()
        return f

    def plot_all_wells(self, path: str) -> None:
        """Plot all wells into a pdf.

        Parameters
        ----------
        path
            Where the pdf file is saved.

        Raises
        ------
        Exception
            When fit is not yet run.

        """
        if not hasattr(self, 'fittings'):
            raise Exception('run fit first')
        out = PdfPages(path)
        for k in self.fittings[0].loc[self.keys_ctrl].index:
            out.savefig(self.plot_well(k))
        for k in self.fittings[0].loc[self.keys_unk].sort_index().index:
            out.savefig(self.plot_well(k))
        out.close()

    def plot_ebar(
        self,
        lb: int,
        x: str = 'K',
        y: str = 'SA',
        xerr: str = 'sK',
        yerr: str = 'sSA',
        xmin: Optional[float] = None,
        ymin: Optional[float] = None,
        xmax: Optional[float] = None,
        title: Optional[str] = None,
    ) -> plt.figure:
        """Plot SA vs. K with errorbar for the whole plate."""
        if not hasattr(self, 'fittings'):
            raise Exception('run fit first')
        df = self.fittings[lb]
        with plt.style.context('fivethirtyeight'):
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
                    fmt='o',
                    elinewidth=1,
                    markersize=10,
                    alpha=0.7,
                )
            except ValueError:
                pass
            if 'ctrl' not in df:
                df['ctrl'] = 0
            df = df[~np.isnan(df[x])]
            df = df[~np.isnan(df[y])]
            for idx, xv, yv, l in zip(df.index, df[x], df[y], df['ctrl']):
                # x or y do not exhist.# try:
                if type(l) == str:
                    color = '#' + hashlib.md5(l.encode()).hexdigest()[2:8]
                    plt.text(xv, yv, l, fontsize=13, color=color)
                else:
                    plt.text(xv, yv, idx, fontsize=12)
                # x or y do not exhist.# except:
                # x or y do not exhist.# continue
            plt.yscale('log')
            # min(x) can be = NaN
            min_x = min(max([0.01, df[x].min()]), 14)
            min_y = min(max([0.01, df[y].min()]), 5000)
            plt.xlim(0.99 * min_x, 1.01 * df[x].max())
            plt.ylim(0.90 * min_y, 1.10 * df[y].max())
            plt.grid(1, axis='both')
            plt.ylabel(y)
            plt.xlabel(x)
            title = title if title else ''
            title += '  label:' + str(lb)
            plt.title(title, fontsize=15)
            return f

    def print_fitting(self, lb: int) -> None:
        """Print fitting parameters for the whole plate."""

        def df_print(df: pd.DataFrame) -> None:
            for i, r in df.iterrows():
                print('{:s}'.format(i), end=' ')
                for k in out[:2]:
                    print('{:7.2f}'.format(r[k]), end=' ')
                for k in out[2:]:
                    print('{:7.0f}'.format(r[k]), end=' ')
                print()

        df = self.fittings[lb]
        if 'SA2' in df.keys():
            out = ['K', 'sK', 'SA', 'sSA', 'SB', 'sSB', 'SA2', 'sSA2', 'SB2', 'sSB2']
        else:
            out = ['K', 'sK', 'SA', 'sSA', 'SB', 'sSB']
        if len(self.keys_ctrl) > 0:
            res_ctrl = df.loc[self.keys_ctrl]
            gr = res_ctrl.groupby('ctrl')
            print('    ' + ' '.join(["{:>7s}".format(x) for x in out]))
            for g in gr:
                print(' ', g[0])
                df_print(g[1][out])
        res_unk = df.loc[self.keys_unk]
        print()
        print('    ' + ' '.join(["{:>7s}".format(x) for x in out]))
        print('  UNK')
        df_print(res_unk.sort_index())

    def plot_buffer(self, title: Optional[str] = None) -> plt.figure:
        """Plot buffers (indicated in scheme) for all labelblocksgroups."""
        x = self.conc
        f, ax = plt.subplots(2, 1, figsize=(10, 10))
        for i, lbg in enumerate(self.labelblocksgroups):
            buf = copy.deepcopy(lbg.buffer)
            bg = buf.pop('bg')
            bg_sd = buf.pop('bg_sd')
            rowlabel = ['Temp']
            lines = [['{:6.1f}'.format(x) for x in lbg.temperatures]]
            colors = plt.cm.Set3(np.linspace(0, 1, len(buf) + 1))
            for j, (k, v) in enumerate(buf.items(), start=1):
                rowlabel.append(k)
                lines.append(['{:6.1f}'.format(x) for x in v])
                ax[i].plot(x, v, 'o-', alpha=0.8, lw=2, markersize=3, color=colors[j])
            ax[i].errorbar(
                x,
                bg,
                yerr=bg_sd,
                fmt='o-.',
                markersize=15,
                lw=1,
                elinewidth=3,
                alpha=0.8,
                color='grey',
                label='label' + str(i),
            )
            plt.subplots_adjust(hspace=0.0)
            ax[i].legend(fontsize=22)
            if x[0] > x[-1]:  # reverse
                for line in lines:
                    line.reverse()
            ax[i].table(
                cellText=lines,
                rowLabels=rowlabel,
                loc='top',
                rowColours=colors,
                alpha=0.4,
            )
            ax[i].set_xlim(min(x) * 0.96, max(x) * 1.02)
            ax[i].set_yticks(ax[i].get_yticks()[:-1])
        ax[0].set_yticks(ax[0].get_yticks()[1:])
        ax[0].set_xticklabels('')
        if title:
            f.suptitle(title, fontsize=18)
        f.tight_layout(h_pad=5, rect=(0, 0, 1, 0.87))
        return f
