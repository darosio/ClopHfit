"""The MCR figure must survive a plate that was not read the same way twice.

A sample's two temperatures need not carry the same scans: a plate may
have been read with the emission scan at 20 C and without it at 37 C.
The figure draws one panel per scan, so a panel whose scan is missing at
one temperature has to skip that temperature rather than look up a scan
that is not there.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from clophfit.__main__ import (
    _plot_mcr,  # ruff: ignore[import-private-name]
)

if TYPE_CHECKING:
    from pathlib import Path


def _spectra_frame(scan: str, n: int) -> pd.DataFrame:
    """Build a resolved-spectra table for one scan.

    Parameters
    ----------
    scan : str
        The scan name the rows belong to.
    n : int
        How many wavelengths the scan holds.

    Returns
    -------
    pd.DataFrame
        A table with the columns the figure reads.
    """
    return pd.DataFrame({
        "scan": [scan] * n,
        "lambda": np.arange(400.0, 400.0 + n),
        "species_0": np.linspace(1.0, 0.0, n),
        "species_1": np.linspace(0.0, 1.0, n),
    })


def test_a_scan_missing_at_one_temperature_is_skipped(tmp_path: Path) -> None:
    """A panel must draw the temperatures that have the scan, and no others.

    The 37 C measurement has no emission scan, so the emission panel
    carries the 20 C curves alone - and the figure is still written.
    """
    x = np.linspace(5.5, 9.0, 8)
    c = np.column_stack([np.linspace(0.1, 0.9, 8), np.linspace(0.9, 0.1, 8)])
    profiles = {"sample": {"20": (x, c), "37": (x, c)}}
    spectra = {
        "sample": {
            "20": pd.concat(
                [_spectra_frame("excitation", 5), _spectra_frame("emission 420", 5)],
                ignore_index=True,
            ),
            "37": _spectra_frame("excitation", 5),
        }
    }
    fits = {
        "sample": {
            "20": (7.4, 0.1, 0.0, 1.0, True, 0.05),
            "37": (7.2, 0.1, 0.0, 1.0, True, 0.05),
        }
    }
    out = tmp_path / "mcr.pdf"
    _plot_mcr(profiles, spectra, fits, out)
    assert out.exists()
    assert out.stat().st_size > 1000
