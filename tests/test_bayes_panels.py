"""The whole-spectrum Bayesian panel figure, and the fractions it draws.

Every routine here is checked against a hand-computable case: the fraction at a
known K and a known titrant value (Henderson-Hasselbalch gives 0.5 at pH = pKa
and 10/11 one unit above), the unmixing of synthetic two-species spectra built
from a known mixture, and the premise of the per-well panel - that a row whose
*recorded* axis is off but whose data are good sits on the common curve when it
is plotted at its latent axis, and off it when plotted at the note's.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import xarray as xr

from clophfit.fitting.models import growth_fraction
from clophfit.fitting.spectral import observed_fraction_from_species
from clophfit.prenspire.figures import BayesPanelGroup, plot_global_bayes_panels

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

LAM = np.arange(400.0, 520.0)
K_TRUE = 7.4
X = np.array([5.8, 6.4, 6.9, 7.4, 7.9, 8.4, 8.9, 9.2])
NOTE_SHIFT = 0.4  # pH error recorded on row B while its spectra are unchanged


def _species() -> tuple[np.ndarray, np.ndarray]:
    """Two well-separated species spectra: anionic (basic) and protonated (acidic)."""
    basic = np.exp(-0.5 * ((LAM - 488.0) / 12.0) ** 2)
    acid = 0.4 * np.exp(-0.5 * ((LAM - 430.0) / 14.0) ** 2)
    return basic, acid


def _mixture(
    f_occupied: np.ndarray, amounts: np.ndarray, noise: float = 0.0
) -> np.ndarray:
    """Spectra of a known mix, as ``(n_lambda, n_wells)``.

    ``f_occupied`` is the share of ``species[1]`` - the protonated form on a pH
    plate, the bound form on a ligand one - which is the order both fitters use.
    The fraction the panels report is the *other* state on a pH plate.
    """
    basic, acid = _species()
    y = np.outer(basic, 1.0 - f_occupied) + np.outer(acid, f_occupied)
    y *= amounts
    if noise:
        y += np.random.default_rng(0).normal(0.0, noise, y.shape)
    return y


def test_growth_fraction_matches_hand_computed_values() -> None:
    """Henderson-Hasselbalch and Langmuir at their half points, plus one step either side."""
    assert growth_fraction(np.array([K_TRUE]), K_TRUE, is_ph=True)[0] == pytest.approx(
        0.5
    )
    assert growth_fraction(np.array([K_TRUE + 1.0]), K_TRUE, is_ph=True)[
        0
    ] == pytest.approx(10.0 / 11.0)
    assert growth_fraction(np.array([K_TRUE - 1.0]), K_TRUE, is_ph=True)[
        0
    ] == pytest.approx(1.0 / 11.0)
    assert growth_fraction(np.array([10.0]), 10.0)[0] == pytest.approx(0.5)
    assert growth_fraction(np.array([30.0]), 10.0)[0] == pytest.approx(0.75)
    assert growth_fraction(np.array([10.0 / 3.0]), 10.0)[0] == pytest.approx(0.25)


def test_growth_fraction_broadcasts_a_posterior_of_k() -> None:
    """A grid of wells against a column of draws gives the full (n_draws, n_wells) table."""
    k = np.array([[7.0], [7.4], [7.8]])
    x = np.array([6.4, 7.4, 8.4])  # one unit either side of K_TRUE
    out = growth_fraction(x, k, is_ph=True)
    assert out.shape == (3, 3)
    np.testing.assert_allclose(out[1], [1.0 / 11.0, 0.5, 10.0 / 11.0])
    np.testing.assert_allclose(out[0, 1], 10**0.4 / (1.0 + 10**0.4))
    np.testing.assert_allclose(out[2, 1], 10**-0.4 / (1.0 + 10**-0.4))


def test_observed_fraction_recovers_a_known_mixture() -> None:
    """Noiseless synthetic spectra built from a known mix give that mix back."""
    f = np.array([0.05, 0.35, 0.5, 0.8, 0.95])  # anionic share, what is reported
    species = {"excitation": np.stack(_species())}
    spectra = {"excitation": (LAM, _mixture(1.0 - f, np.ones(f.size)))}
    got = observed_fraction_from_species(species, spectra, is_ph=True)
    assert got is not None
    # the grid is 1e-3 .. 1 - 1e-3 in 401 steps, so a 0.01 tolerance is 4 grid steps
    np.testing.assert_allclose(got, f, atol=0.01)


def test_observed_fraction_ligand_reports_the_bound_share() -> None:
    """On a ligand plate the reported share is the bound one, not the complement."""
    f_bound = np.array([0.1, 0.5, 0.9])  # the bound form is species[1]
    species = {"excitation": np.stack(_species())}
    spectra = {"excitation": (LAM, _mixture(f_bound, np.ones(f_bound.size)))}
    got = observed_fraction_from_species(species, spectra, is_ph=False)
    assert got is not None
    np.testing.assert_allclose(got, f_bound, atol=0.01)


def test_per_well_panel_premise_a_shifted_note_axis_does_not_move_the_data() -> None:
    """The panel's claim, checked arithmetically.

    Row B's spectra are generated at the true axis while its note records
    ``X + NOTE_SHIFT``. Unmixed against the true species, its fractions sit on the
    common curve when plotted at the latent axis and off it by more than 0.2 when
    plotted at the note's - which is exactly what the two fraction panels are for.
    """
    f_true = growth_fraction(X, K_TRUE, is_ph=True)
    y = _mixture(1.0 - f_true, np.ones(X.size))
    species = {"excitation": np.stack(_species())}
    f_obs = observed_fraction_from_species(
        species, {"excitation": (LAM, y)}, is_ph=True
    )
    assert f_obs is not None
    on_latent = growth_fraction(X, K_TRUE, is_ph=True)
    on_note = growth_fraction(X + NOTE_SHIFT, K_TRUE, is_ph=True)
    assert np.max(np.abs(f_obs - on_latent)) < 0.05
    assert np.max(np.abs(f_obs - on_note)) > 0.2


def _trace(
    extra: Mapping[str, tuple[np.ndarray, np.ndarray, np.ndarray]] | None = None,
    *,
    n_chains: int = 2,
    n_draws: int = 40,
) -> tuple[xr.DataTree, np.ndarray, np.ndarray]:
    """Build a posterior with the structure of a global-Bayes trace.

    ``extra`` adds scans beyond the excitation one as ``label -> (wavelengths,
    E0, E1)``, which is how a two-scan fit is built here without running NUTS.
    """
    basic, acid = _species()
    f_true = growth_fraction(X, K_TRUE, is_ph=True)
    x_true = np.concatenate([X, X + NOTE_SHIFT])
    amounts = np.concatenate([np.ones(X.size), np.full(X.size, 1.1)])
    shape2 = (n_chains, n_draws)
    draws: dict[str, tuple[tuple[str, ...], np.ndarray]] = {
        "K": (("chain", "draw"), np.full(shape2, K_TRUE)),
        "x_true": (("chain", "draw", "well"), np.tile(x_true, (*shape2, 1))),
        "amplitude": (("chain", "draw", "well"), np.tile(amounts, (*shape2, 1))),
        # one length-dimension per scan, as the real trace has: two scans need
        # not share a wavelength grid (excitation 275-500, emission 450-650)
        "E0_excitation": (("chain", "draw", "wl_exc"), np.tile(basic, (*shape2, 1))),
        "E1_excitation": (("chain", "draw", "wl_exc"), np.tile(acid, (*shape2, 1))),
    }
    coords = {
        "chain": np.arange(n_chains),
        "draw": np.arange(n_draws),
        "well": np.arange(x_true.size),
    }
    for label, (_lam, e0, e1) in (extra or {}).items():
        dim = f"wl_{label}"
        draws[f"E0_{label}"] = (("chain", "draw", dim), np.tile(e0, (*shape2, 1)))
        draws[f"E1_{label}"] = (("chain", "draw", dim), np.tile(e1, (*shape2, 1)))
    trace = xr.DataTree.from_dict({"posterior": xr.Dataset(draws, coords=coords)})
    return trace, x_true, np.concatenate([f_true, f_true])


def _group(
    extra: Mapping[str, tuple[np.ndarray, np.ndarray, np.ndarray]] | None = None,
) -> BayesPanelGroup:
    """One synthetic group: two rows, the second recorded on a shifted axis."""
    trace, _x_true, f_true = _trace(extra)
    wells = [f"A0{i + 1}" for i in range(X.size)] + [
        f"B0{i + 1}" for i in range(X.size)
    ]
    spectra = {"excitation": (LAM, _mixture(1.0 - f_true, np.ones(f_true.size)))}
    for label, (lam, e0, e1) in (extra or {}).items():
        # an emission scan whose two states differ only in brightness: the shape
        # carries no fraction, so it is the shared model's scale that pins it
        spectra[label] = (lam, np.outer(e0, 1.0 - f_true) + np.outer(e1, f_true))
    return BayesPanelGroup(
        label="synthetic at 20.0",
        is_ph=True,
        wells=wells,
        x_note=np.concatenate([X, X + NOTE_SHIFT]),
        spectra=spectra,
        trace=trace,
    )


def test_panels_are_written_and_carry_both_fraction_views(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Five panels per group, both fraction panels present, PDF on disk."""
    import matplotlib.pyplot as plt  # ruff: ignore[import-outside-top-level]

    closed: list[bool] = []
    monkeypatch.setattr(plt, "close", lambda *_a, **_k: closed.append(True))
    pdf = tmp_path / "panels.pdf"
    out = plot_global_bayes_panels(pdf, [_group()])
    assert out == pdf
    assert pdf.exists()
    assert pdf.stat().st_size > 5_000
    assert closed == [True]
    fig = plt.gcf()
    axes = fig.get_axes()
    titles = [ax.get_title(loc="left") for ax in axes]
    # one scan + one fraction panel + the K posterior
    assert len(axes) >= 3
    assert any(t.startswith("fraction per well") for t in titles)
    assert any(t.startswith("K ") for t in titles)
    frac = next(
        ax for ax, t in zip(axes, titles, strict=True) if t.startswith("fraction")
    )
    legend_artist = frac.get_legend()
    assert legend_artist is not None
    legend = [text.get_text() for text in legend_artist.get_texts()]
    assert "note pH" in legend  # the recorded axis is drawn, not applied
    assert not any("only" in text for text in legend)  # one scan: nothing to compare
    k_ax = next(ax for ax, t in zip(axes, titles, strict=True) if t.startswith("K "))
    assert k_ax.get_box_aspect() == 1  # square, so it reads as an inset


def test_two_scans_draw_a_shared_and_a_per_scan_fraction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With two scans the fraction panel names both, one shape per scan."""
    import matplotlib.pyplot as plt  # ruff: ignore[import-outside-top-level]

    monkeypatch.setattr(plt, "close", lambda *_a, **_k: None)
    emission = np.arange(450.0, 600.0)
    e0 = np.exp(-0.5 * ((emission - 515.0) / 20.0) ** 2)
    e1 = 0.25 * e0  # the protonated state is dimmer, same shape: amplitude only
    two = _group({"emission 420": (emission, e0, e1)})
    pdf = tmp_path / "two.pdf"
    plot_global_bayes_panels(pdf, [two])
    assert pdf.exists()
    axes = plt.gcf().get_axes()
    titles = [ax.get_title(loc="left") for ax in axes]
    assert any("emission 420" in t for t in titles)  # one scan panel per scan
    frac = next(
        ax for ax, t in zip(axes, titles, strict=True) if t.startswith("fraction")
    )
    legend_artist = frac.get_legend()
    assert legend_artist is not None
    legend = [text.get_text() for text in legend_artist.get_texts()]
    assert "all scans (shared)" in legend
    assert "excitation only" in legend
    assert "emission 420 only" in legend


def test_acid_state_panel_fit_uses_the_third_species(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The overlay for acid-state fits includes E2, not just the first two species."""
    import matplotlib.pyplot as plt  # ruff: ignore[import-outside-top-level]

    monkeypatch.setattr(plt, "close", lambda *_a, **_k: None)
    base = _group()
    posterior = base.trace.posterior.copy()
    delta = 1.2
    posterior["delta"] = xr.DataArray(np.full((2, 40), delta), dims=("chain", "draw"))
    third = 0.6 * np.exp(-0.5 * ((LAM - 415.0) / 9.0) ** 2)
    posterior["E2_excitation"] = xr.DataArray(
        np.tile(third, (2, 40, 1)), dims=("chain", "draw", "wl_exc")
    )
    trace = xr.DataTree.from_dict({"posterior": posterior})
    x_true = np.concatenate([X, X + NOTE_SHIFT])
    amplitudes = np.concatenate([np.ones(X.size), np.full(X.size, 1.1)])
    ln10 = np.log(10.0)
    logs = np.column_stack([
        np.zeros_like(x_true),
        (K_TRUE - x_true) * ln10,
        ((K_TRUE - x_true) + (K_TRUE - delta - x_true)) * ln10,
    ])
    weights = np.exp(logs - np.max(logs, axis=1, keepdims=True))
    fr = weights / np.sum(weights, axis=1, keepdims=True)
    basic, acid = _species()
    species = (basic, acid, third)
    measured = amplitudes[None, :] * sum(
        fr[:, state][None, :] * species[state][:, None] for state in range(3)
    )
    group = BayesPanelGroup(
        label="acid-state synthetic",
        is_ph=True,
        wells=base.wells,
        x_note=x_true,
        spectra={"excitation": (LAM, measured)},
        trace=trace,
    )
    pdf = tmp_path / "acid-state.pdf"
    plot_global_bayes_panels(pdf, [group])
    assert pdf.exists()
    scan_ax = plt.gcf().axes[0]
    # Each well contributes an observation line and then a dashed posterior fit.
    np.testing.assert_allclose(
        np.asarray(scan_ax.lines[1].get_ydata(), dtype=np.float64), measured[:, 0]
    )
    assert any(
        "third state was fitted" in ax.get_title(loc="left") for ax in plt.gcf().axes
    )


def test_panels_of_nothing_are_not_written(tmp_path: Path) -> None:
    """No groups means no figure, and no directory created for it."""
    pdf = tmp_path / "empty.pdf"
    assert plot_global_bayes_panels(pdf, []) == pdf
    assert not pdf.exists()
