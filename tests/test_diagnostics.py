"""Tests for clophfit.fitting.diagnostics."""

import numpy as np
import pandas as pd
import pytest

from clophfit.fitting.diagnostics import curve_turnover, detect_bad_wells, screen_wells


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
        "well": ["A01", "A02", "A03", "A04"],
        "K": [7.0, 7.0, 7.0, 7.0],
        "sK": [0.05, 0.05, 0.05, 0.05],
        "S0_1": [1000.0, 1000.0, 1000.0, 1000.0],
        "S1_1": [1001.0, 1500.0, 1510.0, 1490.0],  # A01: barely changes, others normal
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
        "S0_1": [580.0, 600.0, 610.0, 605.0, 590.0],
        "S1_1": [1080.0, 1100.0, 1120.0, 1115.0, 1090.0],
    })
    flags = detect_bad_wells(df, k_min=3.0, k_max=14.0, ctr_cols=[1])

    # A01 is CTR — must not be flagged despite K=5 being far from sample median
    assert not flags[flags["well"] == "A01"].iloc[0]["flag_any"]
    # E02 is a sample outlier — must be flagged
    assert flags[flags["well"] == "E02"].iloc[0]["flag_k_outlier"]


class TestCurveTurnover:
    """``curve_turnover``: how far a curve comes back down from its own peak.

    The 400 nm neutral-form channel does not fall monotonically with pH -- it
    turns over at the acid end. Any rule that infers a channel's direction from
    its endpoints is therefore wrong on those wells, which is how a
    plate-by-plate direction test produced 44 phantom "inverted" wells whose
    maxima were simply interior.

    The measure is the smaller of the two drops from the peak, over the
    observed range. A monotone curve has its peak at an end, so one drop is
    zero and so is the metric; a genuine interior peak scores by how far it
    returns on the side it need not. Normalising by the range rather than by
    the peak keeps it meaningful when the signal is negative, which it is on
    the dimmest wells.
    """

    @staticmethod
    def _x() -> np.ndarray:
        return np.array([5.0, 6.0, 7.0, 8.0, 9.0])

    def test_monotone_decreasing_curve_has_no_turnover(self) -> None:
        """A healthy label-1 curve falls across the range and scores zero."""
        y = np.array([1000.0, 800.0, 500.0, 200.0, 100.0])
        assert curve_turnover(self._x(), y) == pytest.approx(0.0)

    def test_monotone_increasing_curve_has_no_turnover(self) -> None:
        """A healthy label-2 curve rises across the range and scores zero."""
        y = np.array([100.0, 200.0, 500.0, 800.0, 1000.0])
        assert curve_turnover(self._x(), y) == pytest.approx(0.0)

    def test_symmetric_peak_scores_one(self) -> None:
        """A curve returning fully to both ends is entirely turnover."""
        y = np.array([100.0, 500.0, 1000.0, 500.0, 100.0])
        assert curve_turnover(self._x(), y) == pytest.approx(1.0)

    def test_acid_turnover_is_measured_against_the_shallower_side(self) -> None:
        """L4/G01: rises 640 -> 1570, returns to 1130. 440/930 of its range."""
        x = np.array([5.0, 6.0, 7.08, 8.0, 8.97])
        y = np.array([640.0, 1050.0, 1570.0, 1300.0, 1130.0])
        assert curve_turnover(x, y) == pytest.approx(440.0 / 930.0, rel=1e-6)

    def test_a_rising_curve_that_barely_dips_scores_low(self) -> None:
        """L4/E05 is concordant with label 2, not peaked: 90/1838 of its range."""
        x = np.array([5.0, 6.0, 7.0, 8.32, 8.97])
        y = np.array([132.0, 600.0, 1200.0, 1970.0, 1880.0])
        assert curve_turnover(x, y) == pytest.approx(90.0 / 1838.0, rel=1e-6)

    def test_negative_signals_do_not_break_the_metric(self) -> None:
        """L2/B03's label 1 sits below zero; normalising by range keeps it finite."""
        y = np.array([-8.9, -6.0, -11.5, -13.0, -2.5])
        got = curve_turnover(self._x(), y)
        assert np.isfinite(got)
        assert 0.0 <= got <= 1.0

    def test_unsorted_x_is_handled(self) -> None:
        """Tecan files store pH descending; the metric must not depend on order."""
        x = np.array([9.0, 8.0, 7.0, 6.0, 5.0])
        y = np.array([100.0, 500.0, 1000.0, 500.0, 100.0])
        assert curve_turnover(x, y) == pytest.approx(1.0)

    def test_a_flat_curve_has_no_turnover(self) -> None:
        """A dead channel has no range, so there is nothing to come back from."""
        y = np.full(5, 42.0)
        assert curve_turnover(self._x(), y) == pytest.approx(0.0)


class TestScreenWells:
    """``screen_wells``: quality signatures read from the data, before any fit.

    Every badly-fitted well in the eleven-plate campaign has a dim 485 nm
    channel -- all 55 with sK/K > 0.3, without exception -- so a poor fit is a
    consequence of weak signal and the signal can be read directly. Screening
    first therefore avoids having to fit a well to learn whether it was worth
    fitting.

    Quality is judged on the 485 nm anion channel, not the 400 nm neutral one.
    Label 1's amplitude varies for reasons that are properties of the construct
    rather than faults -- it turns over at the acid end in 60% of wells -- and
    on adjudicated data a label-1 brightness rule cost roughly half the
    precision for no extra recall.

    The amplitude is the upper quartile against the *background level* the
    signal sits on. Against the level rather than its scatter, because the
    ratio is then dimensionless and needs no per-plate normalisation across a
    campaign whose plate brightness spans fifteenfold; the upper quartile
    rather than the maximum, because a single spike put a dead well (L5a E12,
    label 2 reading 4.5, 12, 21, 11, 1, 0.5) above a maximum-based threshold.
    """

    @staticmethod
    def _ph() -> np.ndarray:
        return np.array([5.0, 6.0, 7.0, 8.0, 9.0])

    def _well(self, y1: list[float], y2: list[float]) -> dict[str, dict[str, object]]:
        return {"A01": {"1": np.array(y1, dtype=float), "2": np.array(y2, dtype=float)}}

    def test_a_healthy_well_is_not_flagged(self) -> None:
        """Label 1 falls, label 2 rises well above background: nothing fires."""
        wells = self._well([900, 800, 500, 200, 100], [100, 200, 500, 800, 900])
        row = (
            screen_wells(wells, self._ph(), bg_level={"1": 10.0, "2": 10.0})
            .set_index("well")
            .loc["A01"]
        )
        assert not row["flag_low_signal"]
        assert not row["flag_concordant"]
        assert not row["flag_flat_curve"]

    def test_a_channel_near_its_background_is_flagged(self) -> None:
        """The 485 nm upper quartile below twice its background is too dim."""
        wells = self._well([900, 800, 500, 200, 100], [8, 9, 10, 11, 12])
        row = (
            screen_wells(wells, self._ph(), bg_level={"1": 10.0, "2": 10.0})
            .set_index("well")
            .loc["A01"]
        )
        assert bool(row["flag_low_signal"])

    def test_the_ratio_is_dimensionless_so_plate_brightness_cancels(self) -> None:
        """Scaling signal and background together must not change the verdict.

        This is what lets one threshold serve plates whose brightness spans
        fifteenfold without a per-plate percentile.
        """
        dim = self._well([90, 80, 50, 20, 10], [10, 20, 50, 80, 90])
        bright = self._well(
            [9000, 8000, 5000, 2000, 1000], [1000, 2000, 5000, 8000, 9000]
        )
        a = (
            screen_wells(dim, self._ph(), bg_level={"1": 1.0, "2": 1.0})
            .set_index("well")
            .loc["A01"]
        )
        b = (
            screen_wells(bright, self._ph(), bg_level={"1": 100.0, "2": 100.0})
            .set_index("well")
            .loc["A01"]
        )
        assert bool(a["flag_low_signal"]) == bool(b["flag_low_signal"])
        assert a["signal_ratio_2"] == pytest.approx(b["signal_ratio_2"])

    def test_a_single_spike_does_not_rescue_a_dead_channel(self) -> None:
        """One bright point must not lift a flat channel over the threshold.

        The maximum is not a safe amplitude: a channel sitting at background
        with a single excursion has a healthy-looking maximum. The upper
        quartile ignores it.

        Note this does not rescue every such well -- L5a E12 reads
        4.5/12/21/11/1/0.5, whose upper quartile is still 11.75, or 3.6x its
        background. Wells that are noisy rather than flat need the fit to
        expose them, which is what ``sK/K`` is for.
        """
        wells = self._well([900, 800, 500, 200, 100], [10, 11, 10, 11, 60])
        row = (
            screen_wells(wells, self._ph(), bg_level={"1": 10.0, "2": 10.0})
            .set_index("well")
            .loc["A01"]
        )
        # max would be 60 -> 6.0x and pass; the upper quartile is 11 -> 1.1x.
        assert row["signal_ratio_2"] == pytest.approx(1.1)
        assert bool(row["flag_low_signal"])

    def test_quality_is_judged_on_label_2_not_label_1(self) -> None:
        """A dim 400 nm channel beside a healthy 485 nm one is not a bad well.

        L6b F04 is exactly this: label 1 at 0.02x its background while label 2
        titrates, and the reviewer kept the well with only label 1 excluded.
        """
        wells = self._well([1, 1, 1, 1, 1], [100, 200, 500, 800, 900])
        row = (
            screen_wells(wells, self._ph(), bg_level={"1": 50.0, "2": 10.0})
            .set_index("well")
            .loc["A01"]
        )
        assert not row["flag_low_signal"]
        assert bool(row["flag_low_signal_1"])

    def test_concordant_labels_are_flagged(self) -> None:
        """Both channels moving together is notable: they should be opposed.

        The neutral form falls with pH as the anion rises, so a healthy well is
        strongly anti-correlated -- median -0.955 across the campaign. This
        needs no direction test on either channel alone, which is what the old
        inverted-curve check could not survive.
        """
        wells = self._well([100, 200, 500, 800, 900], [100, 200, 500, 800, 900])
        row = (
            screen_wells(wells, self._ph(), bg_level={"1": 10.0, "2": 10.0})
            .set_index("well")
            .loc["A01"]
        )
        assert bool(row["flag_concordant"])

    def test_a_flat_channel_is_flagged_but_is_not_low_signal(self) -> None:
        """Flatness and dimness are different questions and must not collapse.

        They were one Series under two names, which is why a flat but bright
        channel and a dim one were indistinguishable.
        """
        wells = self._well([500, 500, 501, 500, 499], [100, 200, 500, 800, 900])
        row = (
            screen_wells(wells, self._ph(), bg_level={"1": 10.0, "2": 10.0})
            .set_index("well")
            .loc["A01"]
        )
        assert bool(row["flag_flat_curve_1"])
        assert not row["flag_low_signal"]

    def test_turnover_is_reported_per_label(self) -> None:
        """The acid turnover is a property worth carrying, not a fault."""
        wells = self._well([100, 500, 1000, 500, 100], [100, 200, 500, 800, 900])
        row = (
            screen_wells(wells, self._ph(), bg_level={"1": 10.0, "2": 10.0})
            .set_index("well")
            .loc["A01"]
        )
        assert row["turnover_1"] == pytest.approx(1.0)
        assert row["turnover_2"] == pytest.approx(0.0)
