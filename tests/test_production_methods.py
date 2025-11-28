"""Tests for production-recommended fitting methods.

These tests validate the methods recommended in FITTING_METHODS_SUMMARY.md:
1. ODR-Recursive (rank #2, best speed/precision)
2. ODR-Recursive+Outlier (rank #1, best precision)
3. outlier2 (rank #3, good alternative)
"""

from typing import TYPE_CHECKING, Any

from clophfit.fitting.core import (
    fit_binding_glob,
    outlier2,
)
from clophfit.fitting.data_structures import Dataset, FitResult
from clophfit.fitting.odr import (
    fit_binding_odr_recursive,
    fit_binding_odr_recursive_outlier,
)

if TYPE_CHECKING:
    from collections.abc import Callable


def test_odr_recursive_basic_convergence(ph_dataset: Dataset) -> None:
    """Test that ODR recursive method converges on pH data."""
    # First get LM result
    lm_result = fit_binding_glob(ph_dataset, robust=False)
    assert lm_result.result is not None
    assert lm_result.result.success, "LM fit failed"

    # Then refine with ODR
    odr_result = fit_binding_odr_recursive(lm_result)
    assert odr_result.result is not None
    assert odr_result.result.success, "ODR recursive fit failed"
    assert "K" in odr_result.result.params, "Missing K parameter"
    assert odr_result.result.params["K"].stderr is not None
    assert odr_result.result.params["K"].stderr > 0, "Invalid stderr"


def test_odr_recursive_outlier_basic_convergence(ph_dataset: Dataset) -> None:
    """Test that ODR recursive+outlier method converges on pH data."""
    lm_result = fit_binding_glob(ph_dataset, robust=False)
    assert lm_result.result is not None
    assert lm_result.result.success, "LM fit failed"

    odr_result = fit_binding_odr_recursive_outlier(lm_result)
    assert odr_result.result is not None
    assert odr_result.result.success, "ODR recursive+outlier fit failed"
    assert "K" in odr_result.result.params, "Missing K parameter"
    assert odr_result.result.params["K"].stderr is not None
    assert odr_result.result.params["K"].stderr > 0, "Invalid stderr"


def test_outlier2_basic_convergence(ph_dataset: Dataset) -> None:
    """Test that outlier2 method converges on pH data."""
    result = outlier2(ph_dataset, key="default")
    assert result.result is not None
    assert result.result.success, "outlier2 fit failed"
    assert "K" in result.result.params, "Missing K parameter"
    assert result.result.params["K"].stderr is not None
    assert result.result.params["K"].stderr > 0, "Invalid stderr"


def test_all_production_methods_converge(ph_dataset: Dataset) -> None:
    """Verify all recommended production methods succeed on same data."""
    # outlier2 works directly on Dataset
    result = outlier2(ph_dataset, key="default")
    assert result.result is not None, "outlier2 returned None result"
    assert result.result.success, "outlier2 failed to converge"
    assert "K" in result.result.params, "outlier2 missing K parameter"

    # ODR methods need LM result first
    lm_result = fit_binding_glob(ph_dataset, robust=False)
    assert lm_result.result is not None
    assert lm_result.result.success, "LM failed"

    odr_methods: list[tuple[str, Callable[[FitResult[Any]], FitResult[Any]]]] = [
        ("ODR-Recursive", fit_binding_odr_recursive),
        ("ODR-Recursive+Outlier", fit_binding_odr_recursive_outlier),
    ]

    for name, odr_method in odr_methods:
        result = odr_method(lm_result)
        assert result.result is not None, f"{name} returned None result"
        assert result.result.success, f"{name} failed to converge"
        assert "K" in result.result.params, f"{name} missing K parameter"


def test_odr_vs_lm_both_succeed(ph_dataset: Dataset) -> None:
    """Verify ODR doesn't break when LM works."""
    lm_result = fit_binding_glob(ph_dataset, robust=False)
    assert lm_result.result is not None
    assert lm_result.result.success

    odr_result = fit_binding_odr_recursive(lm_result)
    assert odr_result.result is not None
    assert odr_result.result.success

    # Both should give reasonable K values (within ballpark)
    lm_k = lm_result.result.params["K"].value
    odr_k = odr_result.result.params["K"].value

    # ODR should be close to LM (not a wild number)
    # Allow 2x difference (generous, usually much closer)
    assert 0.5 * lm_k < odr_k < 2.0 * lm_k, (
        f"ODR K ({odr_k:.3f}) too far from LM K ({lm_k:.3f})"
    )


def test_multi_label_support(multi_dataset: Dataset) -> None:
    """Test production methods work with multi-label data."""
    # outlier2 supports multi-label
    # Note: outlier2 iterates over all labels if key is not provided, or specific key if provided.
    # The fixture 'multi_dataset' has keys "y1" and "y2".

    result = outlier2(multi_dataset, key="y1")
    assert result.result is not None
    assert result.result.success
    assert "K" in result.result.params


def test_odr_preserves_lm_structure(ph_dataset: Dataset) -> None:
    """Verify ODR output maintains FitResult structure."""
    lm_result = fit_binding_glob(ph_dataset, robust=False)
    assert lm_result.result is not None
    odr_result = fit_binding_odr_recursive(lm_result)
    assert odr_result.result is not None

    # Should have same parameters
    for param_name in lm_result.result.params:
        assert param_name in odr_result.result.params, (
            f"Missing parameter: {param_name}"
        )

    # Should have stderr for all parameters
    for param_name, param in odr_result.result.params.items():
        assert param.stderr is not None, f"{param_name} missing stderr"
        assert param.stderr >= 0, f"{param_name} has negative stderr"


def test_outlier2_robust_to_noise(ph_dataset: Dataset) -> None:
    """Test that outlier2 handles realistic data."""
    result = outlier2(ph_dataset, key="default")

    assert result.result is not None
    assert result.result.success, "outlier2 failed on realistic data"
    assert result.result.params["K"].stderr is not None
    assert result.result.params["K"].stderr > 0, "outlier2 invalid stderr"

    # K should be in reasonable range for pH (3-12)
    k_val = result.result.params["K"].value
    assert 3.0 < k_val < 12.0, f"outlier2 K ({k_val:.2f}) out of pH range"
