"""The MCR data matrix must not claim one wavelength axis for two scans.

An excitation scan and an emission scan measure different physical
quantities: the first is emission at a fixed emission wavelength, the
second is emission at a fixed excitation wavelength. They overlap in
wavelength (440-500 nm on the G10 plate) but their values there are not
the same measurement, and the emission scan is an order of magnitude
weaker. Stacking them into one wavelength axis asserts they are - and
the axis then runs backwards across the seam (500 -> 440), so a figure
drawn against it misplaces every wavelength.

The fix is to give each scan its own column block, tagged with the scan
it came from, rather than pretending they share one axis.
"""

from __future__ import annotations

import numpy as np

from clophfit.prenspire.mcr import (
    mcr_titration,
    scan_blocks,
    species_by_scan,
    stack_scans,
)


def _scan(lo: float, hi: float, n_wells: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Build a synthetic scan: a Gaussian peak on a 1 nm grid."""
    lam = np.arange(lo, hi + 1.0, 1.0)
    y = np.exp(-0.5 * ((lam - (lo + hi) / 2) / (hi - lo) / 4) ** 2)
    return lam, np.tile(y, (n_wells, 1)).T


def test_two_scans_do_not_share_one_wavelength_axis() -> None:
    """The stacked axis must be tagged by scan, not glued into one ramp."""
    exc = _scan(275.0, 500.0)
    em = _scan(440.0, 650.0)
    stacked = stack_scans([("exc", exc), ("em", em)])
    assert stacked.data.shape == (4, exc[0].size + em[0].size)
    # every column is tagged with the scan it came from
    assert set(stacked.tags[: exc[0].size]) == {"exc"}
    assert set(stacked.tags[exc[0].size :]) == {"em"}


def test_the_stacked_axis_keeps_each_scan_monotone() -> None:
    """Each scan's own wavelengths must stay in order within its block."""
    exc = _scan(275.0, 500.0)
    em = _scan(440.0, 650.0)
    stacked = stack_scans([("exc", exc), ("em", em)])
    waves = stacked.wavelengths
    assert waves.size == stacked.tags.size
    # within the first block the wavelengths increase
    assert np.all(np.diff(waves[: exc[0].size]) > 0)
    # and within the second
    assert np.all(np.diff(waves[exc[0].size :]) > 0)


def test_a_single_scan_is_unchanged() -> None:
    """With one scan the matrix is the scan itself, tagged with its name."""
    exc = _scan(275.0, 500.0)
    stacked = stack_scans([("exc", exc)])
    assert stacked.data.shape == (4, exc[0].size)
    assert set(stacked.tags) == {"exc"}


def test_the_stacked_axis_says_which_scan_a_column_came_from() -> None:
    """The axis must be usable to draw the spectra panel per scan.

    The figure draws a spectrum per column of the data matrix; without
    the scan tag the two scans' spectra would be drawn on one ramp.
    """
    exc = _scan(275.0, 500.0)
    em = _scan(440.0, 650.0)
    stacked = stack_scans([("exc", exc), ("em", em)])
    blocks = scan_blocks(stacked)
    assert list(blocks) == ["exc", "em"]
    assert blocks["exc"] == slice(0, exc[0].size)
    assert blocks["em"] == slice(exc[0].size, exc[0].size + em[0].size)


def test_the_wavelengths_of_each_block_are_kept() -> None:
    """The wavelengths of every column must be recoverable, not just the tag."""
    exc = _scan(275.0, 500.0)
    em = _scan(440.0, 650.0)
    stacked = stack_scans([("exc", exc), ("em", em)])
    waves = stacked.wavelengths
    assert np.array_equal(waves[: exc[0].size], exc[0])
    assert np.array_equal(waves[exc[0].size :], em[0])


def test_the_spectra_table_says_which_scan_each_row_came_from() -> None:
    """The resolved species spectra must carry the scan tag.

    The table is written to CSV and read next to the figure, so a row
    without its scan would be a wavelength with no way to tell which
    measurement it belongs to.
    """
    x = np.linspace(5.5, 9.0, 8)
    exc = _scan(275.0, 300.0, n_wells=x.size)
    em = _scan(440.0, 465.0, n_wells=x.size)
    # make the emission scan track the titration so the fit has something to find
    frac = 1.0 / (1.0 + np.power(10.0, 7.2 - x))
    em = (em[0], em[1] * frac[None, :])
    table, spectra, _profiles = mcr_titration(
        {"20": [("exc", exc), ("em", em)]},
        {"20": x},
        n_species=2,
    )
    assert len(table) == 1
    spec = spectra["20"]
    assert "scan" in spec.columns
    assert set(spec["scan"]) == {"exc", "em"}
    assert (spec["scan"] == "exc").sum() == exc[0].size
    assert (spec["scan"] == "em").sum() == em[0].size


def test_each_scan_gets_its_own_species_spectra_to_draw() -> None:
    """The species spectra must come back split per scan.

    An excitation scan's species spectrum and an emission scan's are
    different observables; the figure draws them as such, each against
    its own wavelengths.
    """
    x = np.linspace(5.5, 9.0, 8)
    exc = _scan(275.0, 300.0, n_wells=x.size)
    em = _scan(440.0, 465.0, n_wells=x.size)
    frac = 1.0 / (1.0 + np.power(10.0, 7.2 - x))
    em = (em[0], em[1] * frac[None, :])
    _table, spectra, _profiles = mcr_titration(
        {"20": [("exc", exc), ("em", em)]},
        {"20": x},
        n_species=2,
    )
    by_scan = species_by_scan(spectra["20"], 2)
    assert list(by_scan) == ["exc", "em"]
    for name, block in (("exc", exc), ("em", em)):
        waves, curves = by_scan[name]
        assert np.array_equal(waves, block[0])
        assert len(curves) == 2
        assert all(c.size == block[0].size for c in curves)
