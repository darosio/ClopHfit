"""Tests for supported production fitting methods."""

from typing import TYPE_CHECKING, Any

from clophfit.fitting.core import fit_binding_glob
from clophfit.fitting.data_structures import Dataset, FitResult
from clophfit.fitting.odr import fit_binding_odr

if TYPE_CHECKING:
    from collections.abc import Callable


def _fit_huber_outlier(ds: Dataset) -> FitResult[Any]:
    """Run the supported robust LM configuration with outlier removal."""
    return fit_binding_glob(ds, method="huber", remove_outliers="zscore:2.5:5")


def test_odr_recursive_basic_convergence(ph_dataset: Dataset) -> None:
    """Test that ODR recursive method converges on pH data."""
    lm_result = fit_binding_glob(ph_dataset)
    assert lm_result.result is not None
    assert lm_result.result.success, "LM fit failed"

    odr_result = fit_binding_odr(lm_result, reweight=True)
    assert odr_result.result is not None
    assert odr_result.result.success, "ODR recursive fit failed"
    assert "K" in odr_result.result.params, "Missing K parameter"
    assert odr_result.result.params["K"].stderr is not None
    assert odr_result.result.params["K"].stderr > 0, "Invalid stderr"


def test_odr_recursive_outlier_basic_convergence(ph_dataset: Dataset) -> None:
    """Test that ODR recursive+outlier method converges on pH data."""
    lm_result = fit_binding_glob(ph_dataset)
    assert lm_result.result is not None
    assert lm_result.result.success, "LM fit failed"

    odr_result = fit_binding_odr(lm_result, remove_outliers="zscore:2.0")
    assert odr_result.result is not None
    assert odr_result.result.success, "ODR recursive+outlier fit failed"
    assert "K" in odr_result.result.params, "Missing K parameter"
    assert odr_result.result.params["K"].stderr is not None
    assert odr_result.result.params["K"].stderr > 0, "Invalid stderr"


def test_huber_outlier_basic_convergence(ph_dataset: Dataset) -> None:
    """Test that the supported robust LM method converges on pH data."""
    result = _fit_huber_outlier(ph_dataset)
    assert result.result is not None
    assert result.result.success, "Huber+outlier fit failed"
    assert "K" in result.result.params, "Missing K parameter"
    assert result.result.params["K"].stderr is not None
    assert result.result.params["K"].stderr > 0, "Invalid stderr"


def test_all_production_methods_converge(ph_dataset: Dataset) -> None:
    """Verify all supported production methods succeed on same data."""
    result = _fit_huber_outlier(ph_dataset)
    assert result.result is not None, "Huber+outlier returned None result"
    assert result.result.success, "Huber+outlier failed to converge"
    assert "K" in result.result.params, "Huber+outlier missing K parameter"

    lm_result = fit_binding_glob(ph_dataset)
    assert lm_result.result is not None
    assert lm_result.result.success, "LM failed"

    odr_methods: list[tuple[str, Callable[[FitResult[Any]], FitResult[Any]]]] = [
        ("ODR-Recursive", lambda fr: fit_binding_odr(fr, reweight=True)),
        (
            "ODR-Recursive+Outlier",
            lambda fr: fit_binding_odr(fr, remove_outliers="zscore:2.0"),
        ),
    ]

    for name, odr_method in odr_methods:
        odr_result = odr_method(lm_result)
        assert odr_result.result is not None, f"{name} returned None result"
        assert odr_result.result.success, f"{name} failed to converge"
        assert "K" in odr_result.result.params, f"{name} missing K parameter"


def test_odr_vs_lm_both_succeed(ph_dataset: Dataset) -> None:
    """Verify ODR doesn't break when LM works."""
    lm_result = fit_binding_glob(ph_dataset)
    assert lm_result.result is not None
    assert lm_result.result.success

    odr_result = fit_binding_odr(lm_result, reweight=True)
    assert odr_result.result is not None
    assert odr_result.result.success

    lm_k = lm_result.result.params["K"].value
    odr_k = odr_result.result.params["K"].value
    assert 0.5 * lm_k < odr_k < 2.0 * lm_k, (
        f"ODR K ({odr_k:.3f}) too far from LM K ({lm_k:.3f})"
    )


def test_multi_label_support(multi_dataset: Dataset) -> None:
    """Test production methods work with multi-label data."""
    result = _fit_huber_outlier(multi_dataset)
    assert result.result is not None
    assert result.result.success
    assert "K" in result.result.params


def test_odr_preserves_lm_structure(ph_dataset: Dataset) -> None:
    """Verify ODR output maintains FitResult structure."""
    lm_result = fit_binding_glob(ph_dataset)
    assert lm_result.result is not None
    odr_result = fit_binding_odr(lm_result, reweight=True)
    assert odr_result.result is not None

    for param_name in lm_result.result.params:
        assert param_name in odr_result.result.params, (
            f"Missing parameter: {param_name}"
        )

    for param_name, param in odr_result.result.params.items():
        assert param.stderr is not None, f"{param_name} missing stderr"
        assert param.stderr >= 0, f"{param_name} has negative stderr"


def test_huber_outlier_robust_to_noise(ph_dataset: Dataset) -> None:
    """Test that the supported robust LM method handles realistic data."""
    result = _fit_huber_outlier(ph_dataset)

    assert result.result is not None
    assert result.result.success, "Huber+outlier failed on realistic data"
    assert result.result.params["K"].stderr is not None
    assert result.result.params["K"].stderr > 0, "Huber+outlier invalid stderr"

    k_val = result.result.params["K"].value
    assert 3.0 < k_val < 12.0, f"Huber+outlier K ({k_val:.2f}) out of pH range"
