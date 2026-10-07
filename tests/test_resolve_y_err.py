"""Missing per-point errors must take the label's scale, never unity.

Regression test for the defect found in the library campaign of 2026-10-07:
two labels fitted jointly, a handful of (step, well) rows arriving with no
usable ``y_errc``, and those rows being filled with 1.0 counts. On L2 label 1,
whose scatter is ~98 counts, that weighted 9 of 616 points ~96x more strongly
than their neighbours - invisible to r-hat, ESS and divergences.
"""

from __future__ import annotations

import numpy as np
import pytest

from clophfit.fitting.utils import resolve_y_err

BASE = 98.1786


def test_all_usable_is_untouched() -> None:
    """A complete error vector comes back unchanged."""
    err = np.full(7, BASE)
    out = resolve_y_err(err, label="1", well="C04")
    assert np.array_equal(out, err)


def test_missing_entries_take_the_label_scale() -> None:
    """NaN and zero entries inherit the label scale, never 1.0."""
    err = np.full(7, BASE)
    err[[0, 2]] = np.nan
    err[5] = 0.0
    out = resolve_y_err(err, label="1", well="C04")
    assert np.all(out > 0)
    assert np.all(np.isfinite(out))
    assert np.allclose(out, BASE)
    assert 1.0 not in set(out.tolist()), "unity must never become a weight"


def test_negative_is_replaced_too() -> None:
    """Negative and infinite entries are unusable too."""
    err = np.array([BASE, -1.0, BASE, np.inf])
    out = resolve_y_err(err, label="2", well="H09")
    assert np.allclose(out, BASE)


def test_no_scale_at_all_uses_fallback() -> None:
    """An explicit fallback serves when the vector is empty of scale."""
    out = resolve_y_err(np.full(4, np.nan), label="1", fallback=BASE)
    assert np.allclose(out, BASE)


def test_no_scale_and_no_fallback_raises() -> None:
    """Silence is not an option when nothing can be learned."""
    with pytest.raises(ValueError, match="no usable y_err"):
        resolve_y_err(np.zeros(4), label="1", well="A01")


def test_shape_is_preserved() -> None:
    """The 2-D (step, well) grid keeps its shape."""
    err = np.full((7, 3), BASE)
    err[0, 0] = np.nan
    out = resolve_y_err(err, label="1")
    assert out.shape == err.shape
    assert np.allclose(out, BASE)
