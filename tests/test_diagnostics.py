"""Tests for clophfit.fitting.diagnostics."""

import pandas as pd
import pytest

from clophfit.fitting.diagnostics import detect_bad_wells


@pytest.fixture
def simple_ffit() -> pd.DataFrame:
    """Minimal ffit DataFrame with one good well, one at each bound, one poor fit."""
    return pd.DataFrame({
        "well": ["A01", "B06", "E10", "C11", "D01"],
        "K": [7.1, 3.0, 11.0, 4.5, 7.2],
        "sK": [0.06, 400.0, 35.0, 0.08, 0.05],
        "S0_1": [600.0, 45.0, 5890.0, 590.0, 610.0],
        "S1_1": [1100.0, -7800.0, 475.0, 1080.0, 1120.0],
    })


def test_k_at_bound(simple_ffit: pd.DataFrame) -> None:
    """K at optimizer bound (3.0 or 11.0) must be flagged."""
    flags = detect_bad_wells(simple_ffit, k_min=3.0, k_max=11.0)
    at_bound = set(flags[flags.flag_k_at_bound]["well"])
    assert "B06" in at_bound, "K=3.0 (lower bound) must be flagged"
    assert "E10" in at_bound, "K=11.0 (upper bound) must be flagged"
    assert "A01" not in at_bound
    assert "D01" not in at_bound


def test_poor_fit(simple_ffit: pd.DataFrame) -> None:
    """sK/K above threshold must be flagged as poor_fit."""
    flags = detect_bad_wells(simple_ffit, k_min=3.0, k_max=11.0, max_sk_ratio=0.3)
    poor = set(flags[flags.flag_poor_fit]["well"])
    assert "B06" in poor, "sK/K >> 1 must be flagged as poor fit"
    assert "E10" in poor, "sK/K = 3.18 must be flagged"
    assert "A01" not in poor


def test_inverted_curve(simple_ffit: pd.DataFrame) -> None:
    """S1 < S0 for pH assay must be flagged as inverted."""
    flags = detect_bad_wells(simple_ffit, k_min=3.0, k_max=11.0, is_ph=True)
    inverted = set(flags[flags.flag_inverted]["well"])
    assert "B06" in inverted, "S1=-7800 < S0=45 must be flagged as inverted"
    assert "E10" in inverted, "S1=475 < S0=5890 must be flagged as inverted"
    assert "A01" not in inverted


def test_k_outlier(simple_ffit: pd.DataFrame) -> None:
    """Normal K values near plate median must not be flagged as outliers."""
    flags = detect_bad_wells(simple_ffit, k_min=3.0, k_max=11.0, k_mad_factor=5.0)
    # C11 has K=4.5 vs median ~7.1; whether it's flagged depends on plate spread
    # at least, A01 and D01 (K≈7.1-7.2) should not be outliers
    assert not flags.loc[flags["well"] == "A01", "flag_k_outlier"].to_numpy()[0]
    assert not flags.loc[flags["well"] == "D01", "flag_k_outlier"].to_numpy()[0]


def test_flag_any_and_count(simple_ffit: pd.DataFrame) -> None:
    """flag_any summarises all flags; flag_count accumulates individual flags."""
    flags = detect_bad_wells(simple_ffit, k_min=3.0, k_max=11.0)
    assert "flag_any" in flags.columns
    assert "flag_count" in flags.columns
    # B06 should have multiple flags
    b06 = flags[flags["well"] == "B06"].iloc[0]
    assert b06["flag_any"]
    assert b06["flag_count"] >= 2
    # Good well should have no flags
    a01 = flags[flags["well"] == "A01"].iloc[0]
    assert not a01["flag_any"]


def test_sorted_by_flag_count(simple_ffit: pd.DataFrame) -> None:
    """Output rows must be ordered by flag_count descending."""
    flags = detect_bad_wells(simple_ffit, k_min=3.0, k_max=11.0)
    counts = flags["flag_count"].tolist()
    assert counts == sorted(counts, reverse=True), (
        "Output must be sorted by flag_count DESC"
    )


def test_flat_curve() -> None:
    """Well with |S1-S0|/max(|S0|,|S1|) < 0.05 must be flagged as flat."""
    df = pd.DataFrame({
        "well": ["A01", "A02"],
        "K": [7.0, 7.0],
        "sK": [0.05, 0.05],
        "S0_1": [1000.0, 1000.0],
        "S1_1": [1001.0, 1500.0],  # A01: barely changes, A02: normal
    })
    flags = detect_bad_wells(df, k_min=3.0, k_max=11.0)
    assert flags.loc[flags["well"] == "A01", "flag_flat_curve"].to_numpy()[0]
    assert not flags.loc[flags["well"] == "A02", "flag_flat_curve"].to_numpy()[0]


def test_residual_stats_with_well_column() -> None:
    """Per-well residual MAD exceeding plate median by residual_mad_factor must be flagged."""
    df = pd.DataFrame({
        "well": ["A01", "A02", "A03"],
        "K": [7.0, 7.0, 7.0],
        "sK": [0.05, 0.05, 0.05],
        "S0_1": [600.0, 610.0, 605.0],
        "S1_1": [1100.0, 1110.0, 1105.0],
    })
    resid = pd.DataFrame({
        "well": ["A01", "A02", "A03"],
        "label": ["1", "1", "1"],
        "mad": [1.0, 1.0, 200.0],  # A03 has huge residuals (>5x median of 1.0)
    })
    flags = detect_bad_wells(df, k_min=3.0, k_max=11.0, residual_stats=resid)
    assert "flag_high_residuals" in flags.columns
    assert flags.loc[flags["well"] == "A03", "flag_high_residuals"].to_numpy()[0]
    assert not flags.loc[flags["well"] == "A01", "flag_high_residuals"].to_numpy()[0]


def test_cl_polarity() -> None:
    """For Cl assay expect S0 > S1; flag if S0 < S1."""
    df = pd.DataFrame({
        "well": ["A01", "A02"],
        "K": [100.0, 100.0],
        "sK": [2.0, 2.0],
        "S0_1": [1000.0, 500.0],
        "S1_1": [500.0, 1000.0],  # A02 inverted for Cl
    })
    flags = detect_bad_wells(df, k_min=1.0, k_max=999.0, is_ph=False)
    assert not flags.loc[flags["well"] == "A01", "flag_inverted"].to_numpy()[0]
    assert flags.loc[flags["well"] == "A02", "flag_inverted"].to_numpy()[0]


def test_multi_label_ffit() -> None:
    """Multi-label ffit (y1, y2 columns) should be handled."""
    df = pd.DataFrame({
        "well": ["A01", "B01"],
        "K": [7.0, 7.0],
        "sK": [0.05, 0.05],
        "S0_y1": [600.0, 600.0],
        "S1_y1": [1100.0, 1100.0],
        "S0_y2": [400.0, 400.0],
        "S1_y2": [800.0, 800.0],
    })
    flags = detect_bad_wells(df, k_min=3.0, k_max=11.0)
    assert not flags["flag_any"].any(), "Both wells should be clean"


def test_check_polarity_false() -> None:
    """When check_polarity=False, inverted flag must not appear."""
    df = pd.DataFrame({
        "well": ["A01"],
        "K": [7.0],
        "sK": [0.05],
        "S0_1": [1000.0],
        "S1_1": [100.0],  # inverted, but polarity disabled
    })
    flags = detect_bad_wells(df, k_min=3.0, k_max=11.0, check_polarity=False)
    assert "flag_inverted" not in flags.columns
