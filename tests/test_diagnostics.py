"""Tests for clophfit.fitting.diagnostics."""

from pathlib import Path

import pandas as pd
import pytest

from clophfit.fitting.diagnostics import detect_bad_wells, detect_bad_wells_from_dat


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
    """sK/K above threshold must be flagged as poor_fit, but not when K is at bound."""
    flags = detect_bad_wells(simple_ffit, k_min=3.0, k_max=11.0, max_sk_ratio=0.3)
    poor = set(flags[flags.flag_poor_fit]["well"])
    # B06 (K=3.0 at bound) and E10 (K=11.0 at bound): sK undefined → not flagged
    assert "B06" not in poor, "K at bound: sK undefined, poor_fit must be suppressed"
    assert "E10" not in poor, "K at bound: sK undefined, poor_fit must be suppressed"
    assert "A01" not in poor  # sK/K = 0.06/7.1 ≈ 0.008, well below threshold
    # Explicit check: well with high sK NOT at bound must be flagged
    df = pd.DataFrame({
        "well": ["A02"],
        "K": [6.5],
        "sK": [2.5],  # sK/K = 0.38 > 0.3
        "S0_1": [600.0],
        "S1_1": [1100.0],
    })
    flags2 = detect_bad_wells(df, k_min=3.0, k_max=11.0, max_sk_ratio=0.3)
    assert flags2.loc[flags2["well"] == "A02", "flag_poor_fit"].to_numpy()[0]


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


def test_low_signal() -> None:
    """Wells with max(|S0|,|S1|) < min_signal_fraction * plate_median must be flagged."""
    df = pd.DataFrame({
        "well": ["A01", "A02", "G12"],
        "K": [7.0, 7.1, 8.4],
        "sK": [0.05, 0.06, 1.57],
        "S0_1": [600.0, 610.0, 2.8],  # G12 tiny signal
        "S1_1": [1100.0, 1120.0, 9.96],
    })
    flags = detect_bad_wells(df, k_min=3.0, k_max=11.0)
    assert flags.loc[flags["well"] == "G12", "flag_low_signal"].to_numpy()[0], (
        "G12 with max_sig≈10 vs median≈855 must be flagged as low_signal"
    )
    assert not flags.loc[flags["well"] == "A01", "flag_low_signal"].to_numpy()[0]


def test_ctr_cols_k_flags_suppressed() -> None:
    """CTR wells must not be flagged for k_at_bound / k_outlier; other flags still apply."""
    df = pd.DataFrame({
        "well": ["A01", "H12", "B02", "C02", "D02"],
        "K": [3.0, 5.3, 7.1, 7.15, 7.2],  # A01 at bound, H12 free; both CTR
        "sK": [300.0, 2.5, 0.06, 0.05, 0.07],  # H12 has poor fit (sK/K > 0.3)
        "S0_1": [975.0, 1126.0, 600.0, 610.0, 605.0],
        "S1_1": [-83000.0, 900.0, 1100.0, 1120.0, 1115.0],
    })
    flags = detect_bad_wells(df, k_min=3.0, k_max=11.0, ctr_cols=[1, 12])

    # K population flags must be suppressed for CTR
    for ctr_well in ("A01", "H12"):
        row = flags[flags["well"] == ctr_well].iloc[0]
        assert not row["flag_k_at_bound"], (
            f"CTR {ctr_well}: k_at_bound must be suppressed"
        )
        assert not row["flag_k_outlier"], (
            f"CTR {ctr_well}: k_outlier must be suppressed"
        )

    # A01 is at bound → sK meaningless → poor_fit suppressed; H12 not at bound → applies
    assert not flags[flags["well"] == "A01"].iloc[0]["flag_poor_fit"], (
        "CTR A01 at bound: poor_fit must be suppressed (sK undefined at bound)"
    )
    assert flags[flags["well"] == "H12"].iloc[0]["flag_poor_fit"], (
        "CTR H12 not at bound: sK/K=0.47 > 0.3 must be flagged"
    )

    # inverted now applies to CTR wells too: A01 S1=-83000 < S0=975
    assert not flags[flags["well"] == "A01"].iloc[0]["flag_inverted"], (
        "CTR A01: flag_inverted suppressed (CTR polarity differs by design)"
    )

    # Sample wells with good K must not be flagged as k_outlier
    for good_well in ("B02", "C02", "D02"):
        row = flags[flags["well"] == good_well].iloc[0]
        assert not row["flag_k_outlier"], (
            f"Good sample {good_well} must not be K-outlier"
        )


def test_ctr_flat_curve_still_flagged() -> None:
    """CTR wells with very low signal must still be flagged as flat_curve."""
    df = pd.DataFrame({
        "well": ["A01", "B02", "C02"],
        "K": [5.0, 7.1, 7.2],  # A01 is CTR col 1
        "sK": [0.2, 0.06, 0.07],
        "S0_1": [1000.0, 600.0, 605.0],
        "S1_1": [1001.0, 1100.0, 1115.0],  # A01: flat (bad CTR signal)
    })
    flags = detect_bad_wells(df, k_min=3.0, k_max=11.0, ctr_cols=[1])
    assert flags[flags["well"] == "A01"].iloc[0]["flag_flat_curve"], (
        "CTR well with flat signal must still be flagged"
    )
    assert not flags[flags["well"] == "B02"].iloc[0]["flag_flat_curve"]


def test_ctr_cols_k_stats_use_samples_only() -> None:
    """K outlier threshold must be computed from sample wells, not CTR wells."""
    df = pd.DataFrame({
        "well": ["A01", "B02", "C02", "D02", "E02"],
        #         CTR    sample sample sample sample
        "K": [5.0, 7.1, 7.15, 7.2, 14.0],  # E02 is an outlier sample
        "sK": [0.1, 0.06, 0.05, 0.07, 0.5],
        "S0_1": [500.0, 600.0, 610.0, 605.0, 590.0],
        "S1_1": [900.0, 1100.0, 1120.0, 1115.0, 950.0],
    })
    flags = detect_bad_wells(df, k_min=3.0, k_max=14.0, ctr_cols=[1])

    # A01 is CTR — must not be flagged despite K=5 being far from sample median
    assert not flags[flags["well"] == "A01"].iloc[0]["flag_any"]
    # E02 is a sample outlier — must be flagged
    assert flags[flags["well"] == "E02"].iloc[0]["flag_k_outlier"]


# ---------------------------------------------------------------------------
# detect_bad_wells_from_dat tests
# ---------------------------------------------------------------------------


@pytest.fixture
def dat_dir(tmp_path: Path) -> str:
    """Create a minimal plate with 4 wells: 2 good, 1 low-signal, 1 flat."""
    wells = {
        # Good wells: strong signal, clear dynamic range
        "A02": "x,y1,y2\n8,100,200\n7,90,190\n6,50,120\n5,10,30\n",
        "B02": "x,y1,y2\n8,110,210\n7,95,195\n6,55,130\n5,15,35\n",
        # Low-signal: ~1% of plate median (should fire flag_low_signal)
        "G12": "x,y1,y2\n8,1.0,0.5\n7,1.1,0.6\n6,0.9,0.4\n5,0.8,0.3\n",
        # Flat curve: no dynamic range in y1 (should fire flag_flat_curve)
        "H05": "x,y1,y2\n8,500,200\n7,499,190\n6,500,120\n5,501,30\n",
    }
    for name, content in wells.items():
        (tmp_path / f"{name}.dat").write_text(content)
    return str(tmp_path)


def test_from_dat_low_signal(dat_dir: str) -> None:
    """G12 with tiny signal must be caught by flag_low_signal."""
    flags = detect_bad_wells_from_dat(dat_dir)
    row = flags[flags["well"] == "G12"].iloc[0]
    assert row["flag_low_signal"], "G12 tiny signal must be flagged"
    assert row["flag_any"]


def test_from_dat_flat_curve(dat_dir: str) -> None:
    """H05 with flat y1 must be caught by flag_flat_curve."""
    flags = detect_bad_wells_from_dat(dat_dir)
    row = flags[flags["well"] == "H05"].iloc[0]
    assert row["flag_flat_curve"], "H05 flat y1 must be flagged"
    assert row["flag_any"]


def test_from_dat_good_wells_not_flagged(dat_dir: str) -> None:
    """Good wells A02 and B02 must not be flagged."""
    flags = detect_bad_wells_from_dat(dat_dir)
    for well in ["A02", "B02"]:
        row = flags[flags["well"] == well].iloc[0]
        assert not row["flag_any"], f"{well} is a good well and must not be flagged"


def test_from_dat_no_dat_files(tmp_path: Path) -> None:
    """Empty directory must raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        detect_bad_wells_from_dat(str(tmp_path))


def test_from_dat_ctr_cols_logged(dat_dir: str) -> None:
    """ctr_cols parameter adds is_ctr column to the output."""
    flags = detect_bad_wells_from_dat(dat_dir, ctr_cols=[1, 12])
    assert "flag_low_signal" in flags.columns
    assert "is_ctr" in flags.columns, "is_ctr column must appear when ctr_cols is set"
    # G12 is col 12 → is_ctr=True
    row = flags[flags["well"] == "G12"].iloc[0]
    assert row["is_ctr"]
    # A02 is col 2 → is_ctr=False
    row = flags[flags["well"] == "A02"].iloc[0]
    assert not row["is_ctr"]


def test_from_dat_empty_series_flagged(tmp_path: Path) -> None:
    """Header-only .dat files must be treated as bad wells instead of crashing."""
    (tmp_path / "A01.dat").write_text("x,y1,y2\n8,100,200\n7,90,190\n")
    (tmp_path / "A02.dat").write_text("x,y1,y2\n")

    flags = detect_bad_wells_from_dat(tmp_path)

    row = flags[flags["well"] == "A02"].iloc[0]
    assert row["flag_low_signal"]
    assert row["flag_flat_curve"]
    assert row["flag_any"]
