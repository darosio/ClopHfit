"""Core data structures in `clophfit`.

Classes:
--------
- DataArray: Represents matched `x`, `y`, and optional `w` arrays.
- Dataset: Extends `dict` to store `DataArray` as key-value pairs, with optional
  support for pH-specific datasets.
"""

import copy
import warnings
from collections import UserDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, TypeVar, runtime_checkable

import arviz as az
import numpy as np
import pandas as pd
from lmfit import Parameters  # type: ignore[import-untyped]
from lmfit.minimizer import Minimizer, MinimizerResult  # type: ignore[import-untyped]
from matplotlib.figure import Figure
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


class Dataset(UserDict[str, DataArray]):
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

    def __repr__(self) -> str:  # pragma: no cover
        """Readable, concise summary of the dataset with rounded values."""

        def _fmt_arr(a: list[float] | ArrayF, max_items: int = 6, prec: int = 3) -> str:
            arr = np.asarray(a)
            if arr.size == 0:
                return "[]"
            if np.issubdtype(arr.dtype, np.floating) or np.issubdtype(
                arr.dtype, np.integer
            ):
                arr = np.round(arr.astype(float), prec)

            def fmt(v: float) -> str:
                return f"{v:g}"

            if arr.size <= max_items:
                return "[" + ", ".join(fmt(v) for v in arr.tolist()) + "]"
            head = arr[: max_items - 2].tolist()
            tail = arr[-2:].tolist()
            return (
                "["
                + ", ".join(fmt(v) for v in head)
                + ", ..., "
                + ", ".join(fmt(v) for v in tail)
                + "]"
            )

        def _fmt_mask(m: ArrayMask, max_items: int = 12) -> str:
            if m.size == 0:
                return "[]"
            arr = m.astype(int)
            if arr.size <= max_items:
                return "[" + ", ".join(str(int(v)) for v in arr.tolist()) + "]"
            head = arr[: max_items - 2].tolist()
            tail = arr[-2:].tolist()
            return (
                "["
                + ", ".join(str(int(v)) for v in head)
                + ", ..., "
                + ", ".join(str(int(v)) for v in tail)
                + "]"
            )

        header = f"Dataset(is_ph={self.is_ph})"
        if len(self) == 0:
            return header
        lines = [header]
        for lbl, da in self.items():
            try:
                lines.extend(
                    (
                        f"  - {lbl}:",
                        f"        x={_fmt_arr(getattr(da, 'xc', np.array([])))}",
                        f"        y={_fmt_arr(getattr(da, 'yc', np.array([])))}",
                        f"        mask={_fmt_mask(da.mask)}",
                        f"        x_err={_fmt_arr(getattr(da, 'x_errc', np.array([])))}",
                        f"        y_err={_fmt_arr(getattr(da, 'y_errc', np.array([])))}",
                    )
                )
            except Exception:  # noqa: BLE001
                lines.append(f"  - {lbl}: <unavailable>")
        return "\n".join(lines)

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
class _Result:
    """Expose lmfit-like attributes."""

    params: Parameters
    residual: ArrayF = field(default_factory=lambda: np.array([]))
    redchi: float = 0.0
    success: bool = True


@runtime_checkable
class MiniProtocol(Protocol):
    """A very small common interface for all minimizers / backends."""


MiniType = TypeVar("MiniType", bound=MiniProtocol)
MiniT = Minimizer | odr.Output | az.InferenceData


@dataclass
class FitResult[MiniType: MiniProtocol]:
    """Result container of a fitting procedure.

    Attributes
    ----------
    figure : Figure | None
        Matplotlib figure visualizing the fit, if generated.
    result : MinimizerResult | _Result | None
        Backend-agnostic fit result exposing a .params attribute along with
        residual, redchi, and success fields (as in lmfit). For lmfit this is a
        MinimizerResult.
    mini : MiniT | None
        The primary backend object (e.g., lmfit.Minimizer, scipy.odr.Output, or
        az.InferenceData for PyMC).
    dataset : Dataset | None
        Dataset used for the fit (typically a deep copy of the input dataset).
    """

    figure: Figure | None = None
    result: MinimizerResult | _Result | None = None
    mini: MiniType | None = None
    dataset: Dataset | None = None

    def pprint(self) -> str:
        """Provide a brief summary of the fit, focusing on the K value."""
        if not self.result or "K" not in self.result.params:
            return "Fit result is invalid or does not contain a 'K' parameter."
        k_param = self.result.params["K"]
        k_val = k_param.value
        k_err = k_param.stderr
        if k_err:
            k_ufloat = ufloat(k_val, k_err)
            return f"K = {k_ufloat:.2u}"
        return f"K = {k_val:.3g} (no uncertainty available)"

    def is_valid(self) -> bool:
        """Whether figure, result, and minimizer exist."""
        return (
            self.figure is not None
            and self.result is not None
            and self.mini is not None
        )


@dataclass
class SpectraGlobResults:
    """A dataclass representing the results of both svd and bands fits.

    Attributes
    ----------
    svd : FitResult | None
        The `FitResult` object representing the outcome of the concatenated svd
        fit, or `None` if the svd fit was not performed.
    gsvd : FitResult | None
        The `FitResult` object representing the outcome of the svd fit, or
        `None` if the svd fit was not performed.
    bands : FitResult | None
        The `FitResult` object representing the outcome of the bands fit, or
        `None` if the bands fit was not performed.
    """

    svd: FitResult[Minimizer] | None = field(default=None)
    gsvd: FitResult[Minimizer] | None = field(default=None)
    bands: FitResult[Minimizer] | None = field(default=None)
