"""A scan must be named by the measurement it makes, not by its plate label.

The plate's labels are row names, and the same measurement gets a
different letter at each temperature: the excitation scan is "C" at
20 C and "E" at 37 C, the 420 nm emission scan is "D" and "B". A name
built from the label therefore says nothing about which measurement a
spectrum is, and the two temperatures of one sample end up looking like
different scans.

The monochromator and the wavelength it holds fixed do identify the
measurement, and they are the same at every temperature.
"""

from __future__ import annotations

from pathlib import Path

from clophfit.prenspire import EnspireFile
from clophfit.prenspire.spectral import scan_name

DATA = Path(__file__).parent / "EnSpire"


def _stub_plate(
    scans: list[tuple[str, str, str, str]],
) -> EnspireFile:
    """Build a file object whose measurements are the given scans.

    Parameters
    ----------
    scans : list[tuple[str, str, str, str]]
        (label, temperature, monochromator, fixed wavelength) per scan.

    Returns
    -------
    EnspireFile
        A file whose ``measurements`` hold only the given metadata.
    """
    ef = EnspireFile.__new__(EnspireFile)
    ef.measurements = {
        label: {
            "metadata": {
                "temp": temp,
                "Monochromator": mono,
                "Wavelength": wave,
            }
        }
        for label, temp, mono, wave in scans
    }
    return ef


def test_the_excitation_scan_is_named_by_its_measurement() -> None:
    """Both temperatures' excitation scans get the same name."""
    ef = EnspireFile(DATA / "G10.csv")
    assert scan_name(ef, "C") == "excitation"
    assert scan_name(ef, "E") == "excitation"


def test_the_emission_scans_are_named_by_their_fixed_wavelength() -> None:
    """An emission scan is named by the wavelength it excites at."""
    ef = EnspireFile(DATA / "G10.csv")
    assert scan_name(ef, "D") == "emission 420"
    assert scan_name(ef, "B") == "emission 420"
    assert scan_name(ef, "A") == "emission 278"


def test_a_scan_name_does_not_depend_on_the_temperature() -> None:
    """The same measurement must carry the same name at every temperature.

    This is the property the MCR figure needs: it draws one panel per
    scan, and the two temperatures of a sample must land on the same
    panel rather than on two panels of their own.
    """
    ef = EnspireFile(DATA / "G10.csv")
    for label, meas in ef.measurements.items():
        md = meas["metadata"]
        name = scan_name(ef, label)
        others = [
            scan_name(ef, other)
            for other, om in ef.measurements.items()
            if om["metadata"]["Monochromator"] == md["Monochromator"]
            and om["metadata"]["Wavelength"] == md["Wavelength"]
        ]
        assert others == [name] * len(others), label


def test_two_scans_of_one_kind_at_one_temperature_get_different_names() -> None:
    """A plate may hold two excitation scans; they must not share a name.

    Both are real measurements - the plate was read twice with the
    filter in different positions - and the MCR matrix keeps one column
    block per scan, so two blocks with one name would be indistinguishable.
    """
    ef = _stub_plate([
        ("C", "20", "Excitation", "520"),
        ("D", "20", "Excitation", "520"),
    ])
    names = {scan_name(ef, "C"), scan_name(ef, "D")}
    assert len(names) == 2


def test_a_name_is_not_made_unique_by_another_temperature() -> None:
    """Only scans of the same temperature compete for a name.

    The G10 plate holds the same three scans at 20 C and at 37 C under
    different labels; the two sets must keep the same names, or the two
    temperatures of one sample land on two different figure panels.
    """
    ef = _stub_plate([
        ("C", "20", "Excitation", "520"),
        ("E", "37", "Excitation", "520"),
    ])
    assert scan_name(ef, "C") == scan_name(ef, "E") == "excitation"
