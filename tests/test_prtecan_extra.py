"""Additional tests for prtecan module to improve coverage."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal

from clophfit import prtecan


def test_dilution_correction_standard():
    """Test dilution correction function returns cumulative volume ratio."""
    additions = [100.0, 50.0, 50.0]
    # volumes = [100,150,200] -> corrections = [1.0,1.5,2.0]
    corr = prtecan.dilution_correction(additions)
    assert_array_equal(corr, np.array([1.0, 1.5, 2.0]))


def test_plate_scheme_discard_and_nofit_keys():
    """Test discard setter and nofit_keys property of PlateScheme."""
    ps = prtecan.PlateScheme()
    ps.buffer = ["A01", "B01"]
    ps.discard = ["C01"]
    # nofit_keys is union of buffer and discard
    assert set(ps.nofit_keys) == {"A01", "B01", "C01"}
    with pytest.raises(TypeError):
        ps.discard = [1, 2]  # non-str entries


def test_titration_config_callback_trigger():
    """Test that TitrationConfig triggers callback on attribute change."""
    cfg = prtecan.TitrationConfig()
    events = []
    cfg.set_callback(lambda: events.append(True))
    # change boolean attribute
    cfg.bg = not cfg.bg
    assert events == [True]
    # setting same value does not re-trigger
    cfg.bg = cfg.bg
    assert events == [True]
    # change string attribute
    cfg.bg_mth = "fit"
    assert len(events) == 2


def test_bufferfit_empty_flag():
    """Test BufferFit.empty property for NaN and non-NaN values."""
    bf = prtecan.BufferFit()
    assert bf.empty
    bf2 = prtecan.BufferFit(m=1.0, q=0.0, m_err=0.1, q_err=0.2)
    assert not bf2.empty


def test_generate_combinations_and_prepare_folder(tmp_path):
    """Test generation of parameter combinations and output folder naming."""
    data_tests = Path(__file__).parent / "Tecan"
    # use existing list file for minimal Titration
    tit = prtecan.Titration.fromlistfile(data_tests / "L1" / "list.pH.csv", is_ph=True)
    combos = tit._generate_combinations()
    # 2^4 boolean flags times 3 methods
    assert len(combos) == 16 * 3
    flags, method = combos[0]
    assert isinstance(flags, tuple)
    assert len(flags) == 4
    assert method in ("mean", "meansd", "fit")
    # test prepare output folder naming
    # set all flags to True and bg_mth to 'fit'
    tit.params.bg = True
    tit.params.bg_adj = True
    tit.params.dil = True
    tit.params.nrm = True
    tit.params.bg_mth = "fit"
    out = tit._prepare_output_folder(tmp_path)
    name = out.name
    assert "_bg" in name
    assert "_adj" in name
    assert "_dil" in name
    assert "_nrm" in name
    assert "_fit" in name


def test_determine_xlim():
    """Test _determine_xlim method adjusts limits correctly."""
    # dummy dataframes
    import pandas as pd

    df_ctr = pd.DataFrame({"K": [2.0, 4.0]})
    df_unk = pd.DataFrame({"K": [1.0, 5.0]})

    # use dummy tit for method, no state required
    class Dummy:
        pass

    dummy = Dummy()
    # bind method
    lim = prtecan.Titration._determine_xlim(dummy, df_ctr, df_unk)
    # lower should be 0.99*min, upper 1.01*max
    assert pytest.approx(lim[0], rel=1e-3) == 0.99 * 1.0
    assert pytest.approx(lim[1], rel=1e-3) == 1.01 * 5.0


def test_extract_xls_roundtrip(tmp_path):
    """Test read_xls and strip_lines integration with a temporary CSV file."""
    # create a small Excel file
    df = pd.DataFrame([[1, None, "a"], [None, 2, None]], columns=["x", "y", "z"])
    path = tmp_path / "test.xls"
    df.to_excel(path, index=False)
    lines = prtecan.read_xls(path)
    # strip_lines should remove blanks
    stripped = prtecan.strip_lines(lines)
    # each line has no empty elements
    assert all(all(e != "" for e in row) for row in stripped)
