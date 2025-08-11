"""Core data structures in `clophfit`.

Classes:
--------
- DataArray: Represents matched `x`, `y`, and optional `w` arrays.
- Dataset: Extends `dict` to store `DataArray` as key-value pairs, with optional
  support for pH-specific datasets.
"""

import copy
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
from lmfit import MinimizerResult, Parameters
from matplotlib import figure
from scipy import odr
from uncertainties import ufloat  # type: ignore[import-untyped]

from clophfit.clophfit_types import ArrayF, ArrayMask


@dataclass
class DataArray:
    """Represent matched `x`, `y`, and optional `w` (weight) arrays."""

    #: x at creation
    xc: ArrayF
    #: y at creation
    yc: ArrayF
    #: x_err at creation
    x_errc: ArrayF = field(init=True, default_factory=lambda: np.array([]))
    #: y_err at creation
    y_errc: ArrayF = field(init=True, default_factory=lambda: np.array([]))
    _mask: ArrayMask = field(init=False)

    def __post_init__(self) -> None:
        """Ensure the x and y arrays are of equal length after initialization."""
        self._validate_lengths()
        self._validate_yerrc_lengths()
        self._validate_xerrc_lengths()
        self._mask = ~np.isnan(self.yc)

    def _validate_lengths(self) -> None:
        """Validate that xc and yc have the same length."""
        if len(self.xc) != len(self.yc):
            msg = "Length of 'xc' and 'yc' must be equal."
            raise ValueError(msg)

    def _validate_yerrc_lengths(self) -> None:
        """Validate that xc and wc have the same length."""
        if self.y_errc.size > 0 and len(self.xc) != len(self.y_errc):
            msg = "Length of 'xc' and 'y_errc' must be equal."
            raise ValueError(msg)

    def _validate_xerrc_lengths(self) -> None:
        """Validate that xc and wc have the same length."""
        if self.x_errc.size > 0 and len(self.xc) != len(self.x_errc):
            msg = "Length of 'xc' and 'x_errc' must be equal."
            raise ValueError(msg)

    @property
    def mask(self) -> ArrayMask:
        """Mask."""
        return self._mask

    @mask.setter
    def mask(self, mask: ArrayMask) -> None:
        """Only boolean where yc is not nan are considered."""
        self._mask = mask & ~np.isnan(self.yc)

    @property
    def x(self) -> ArrayF:
        """Masked x."""
        return self.xc[self.mask]

    @property
    def y(self) -> ArrayF:
        """Masked y."""
        return self.yc[self.mask]

    @property
    def y_err(self) -> ArrayF:
        """Masked y_err."""
        if self.y_errc.size == 0:
            self.y_errc = np.ones_like(self.xc)
        return self.y_errc[self.mask]

    @y_err.setter
    def y_err(self, y_errc: ArrayF) -> None:
        """Set y_err and validate its length."""
        if y_errc.ndim == 0:
            y_errc = np.ones_like(self.xc) * y_errc
        self.y_errc = y_errc
        self._validate_yerrc_lengths()

    @property
    def x_err(self) -> ArrayF:
        """Masked x_err."""
        if self.x_errc.size == 0:
            self.x_errc = np.ones_like(self.xc)
        return self.x_errc[self.mask]

    @x_err.setter
    def x_err(self, x_errc: ArrayF) -> None:
        """Set x_err and validate its length."""
        if x_errc.ndim == 0:
            x_errc = np.ones_like(self.xc) * x_errc
        self.x_errc = x_errc
        self._validate_xerrc_lengths()


class Dataset(dict[str, DataArray]):
    """A dictionary-like container for storing `DataArray`.

    Parameters
    ----------
    data : dict[str, DataArray]
        Maps string keys to `DataArray` instances.
    is_ph : bool, optional
        Indicates whether `x` values represent pH. Default is False.
    """

    is_ph: bool = False

    def __init__(self, data: dict[str, DataArray], is_ph: bool = False) -> None:
        super().__init__(data or {})
        self.is_ph = is_ph

    @classmethod
    def from_da(cls, da: DataArray | list[DataArray], is_ph: bool = False) -> "Dataset":
        """Alternative constructor to create Dataset from a list of DataArray.

        Parameters
        ----------
        da : DataArray | list[DataArray]
            The DataArray objects to populate the dataset.
        is_ph : bool, optional
            Indicate if x values represent pH (default is False).

        Returns
        -------
        Dataset
            The constructed Dataset object.
        """
        if not da:
            return cls({})
        if isinstance(da, list):
            data = {f"y{i}": da_item for i, da_item in enumerate(da)}
        elif isinstance(da, DataArray):
            data = {"default": da}
        return cls(data, is_ph)

    def apply_mask(self, combined_mask: ArrayMask) -> None:
        """Correctly distribute and apply the combined mask across all DataArrays.

        Parameters
        ----------
        combined_mask : ArrayMask
            Boolean array where True keeps the data point, and False masks it out.

        Raises
        ------
        ValueError
            If the length of the combined_mask does not match the total number of data points.
        """
        if combined_mask.size != sum(len(da.y) for da in self.values()):
            msg = "Length of combined_mask must match the total number of data points."
            raise ValueError(msg)
        start_idx = 0
        for da in self.values():
            end_idx = start_idx + len(da.y)
            da.mask[da.mask] &= combined_mask[start_idx:end_idx]
            start_idx = end_idx

    def copy(self, keys: list[str] | set[str] | None = None) -> "Dataset":
        """Return a copy of the Dataset.

        If keys are provided, only data associated with those keys are copied.

        Parameters
        ----------
        keys : list[str] | set[str] | None, optional
            List of keys to include in the copied dataset. If None (default),
            copies all data.

        Returns
        -------
        Dataset
            A copy of the dataset.

        Raises
        ------
        KeyError
            If a provided key does not exist in the Dataset.
        """
        if keys is None:
            return copy.deepcopy(self)
        copied = Dataset({}, is_ph=self.is_ph)
        for key in keys:
            if key in self:
                copied[key] = copy.deepcopy(self[key])
            else:
                msg = f"No such key: '{key}' in the Dataset."
                raise KeyError(msg)
        return copied

    def clean_data(self, n_params: int) -> None:
        """Remove too small datasets."""
        for key in list(
            self.keys()
        ):  # list() is used to avoid modifying dict during iteration
            if n_params > len(self[key].y):
                warnings.warn(
                    f"Removing key '{key}' from Dataset: number of parameters "
                    f"({n_params}) exceeds number of data points ({len(self[key].y)}).",
                    stacklevel=2,
                )
                del self[key]

    def concatenate_data(self) -> tuple[ArrayF, ArrayF, ArrayF, ArrayF]:
        """Concatenate x, y, x_err, and y_err across all datasets."""
        x_data = np.concatenate([v.x for v in self.values()])
        y_data = np.concatenate([v.y for v in self.values()])
        x_err = np.concatenate([v.x_err for v in self.values()])
        y_err = np.concatenate([v.y_err for v in self.values()])
        return x_data, y_data, x_err, y_err

    def export(self, filep: str | Path) -> None:
        """Export this dataset into a csv file."""
        fp = Path(filep)
        for lbl, da in self.items():
            data: dict[str, ArrayF | ArrayMask] = {"xc": da.xc, "yc": da.yc}
            if da.x_errc.size > 0:
                data["x_errc"] = da.x_errc
            if da.y_errc.size > 0:
                data["y_errc"] = da.y_errc
            data["mask"] = da.mask
            pd.DataFrame(data).to_csv(fp.with_stem(f"{fp.stem}_{lbl}"), index=False)


# --- Data Structures for Fit Results ---
@dataclass
class FitResult:
    """
    A container for the results of a fitting procedure.

    Attributes
    ----------
    figure : figure.Figure | None
        A matplotlib Figure object visualizing the fit.
    result : MinimizerResult | Parameters | az.InferenceData | odr.Output | None
        The primary result object from the fitting backend (e.g., lmfit, pymc).
    params : Parameters | None
        The fitted parameters, typically as an `lmfit.Parameters` object for
        easy access to values and uncertainties.
    dataset : Dataset | None
        The dataset that was used for the fitting.
    """

    figure: figure.Figure | None = None
    result: MinimizerResult | Parameters | az.InferenceData | odr.Output | None = None
    params: Parameters | None = None
    dataset: Dataset | None = None

    def summary(self) -> str:
        """Provide a brief summary of the fit, focusing on the K value."""
        if not self.params or "K" not in self.params:
            return "Fit result is invalid or does not contain a 'K' parameter."
        k_param = self.params["K"]
        k_val = k_param.value
        k_err = k_param.stderr
        if k_err:
            k_ufloat = ufloat(k_val, k_err)
            return f"K = {k_ufloat:.2u}"
        return f"K = {k_val:.3g} (no uncertainty available)"
