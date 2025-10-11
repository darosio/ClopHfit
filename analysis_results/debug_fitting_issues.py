#!/usr/bin/env python3
"""
Debug script to identify why all fitting functions are failing.
"""

import traceback

import numpy as np

from realistic_synthetic_data import RealisticSimulationParameters, generate_realistic_dataset
from src.clophfit.fitting.core import fit_binding_glob, fit_lm


def test_data_generation():
    """Test if data generation works correctly."""
    print("🧪 Testing data generation...")

    try:
        params = RealisticSimulationParameters(
            random_seed=42,
            K_true=7.0,
            outlier_probability=0.1,
            y1_base_error=100.0,
            y2_base_error=10.0
        )

        dataset, true_params = generate_realistic_dataset(params)

        print(f"✅ Data generation successful!")
        print(f"Dataset labels: {list(dataset.keys())}")
        print(f"True parameters: {true_params}")

        # Check dataset structure
        for label, da in dataset.items():
            print(f"\n{label} DataArray:")
            print(f"  x shape: {da.xc.shape}, values: {da.xc}")
            print(f"  y shape: {da.yc.shape}, values: {da.yc}")
            print(f"  x_err shape: {da.x_errc.shape}, values: {da.x_errc}")
            print(f"  y_err shape: {da.y_errc.shape}, values: {da.y_errc}")
            print(f"  mask: {da.mask}")
            print(f"  Has NaN in y: {np.isnan(da.yc).any()}")
            print(f"  Has Inf in y: {np.isinf(da.yc).any()}")
            print(f"  Has zero errors: {(da.y_errc == 0).any()}")

        return dataset, true_params

    except Exception as e:
        print(f"❌ Data generation failed: {e}")
        traceback.print_exc()
        return None, None


def test_simple_fitting(dataset):
    """Test simple fitting functions."""
    print("\n🔧 Testing simple fitting functions...")

    if dataset is None:
        print("❌ No dataset to test")
        return

    # Test fit_binding_glob
    try:
        print("Testing fit_binding_glob...")
        result = fit_binding_glob(dataset)
        print(f"Result: {result}")
        print(f"Success: {result.result.success if result.result else False}")
        if result.result:
            print(f"Parameters: {result.result.params}")
    except Exception as e:
        print(f"❌ fit_binding_glob failed: {e}")
        traceback.print_exc()

    # Test fit_lm
    try:
        print("\nTesting fit_lm...")
        result = fit_lm(dataset)
        print(f"Result: {result}")
        print(f"Success: {result.result.success if result.result else False}")
        if result.result:
            print(f"Parameters: {result.result.params}")
    except Exception as e:
        print(f"❌ fit_lm failed: {e}")
        traceback.print_exc()


def test_dataset_properties(dataset):
    """Test dataset properties that might cause fitting issues."""
    print("\n🔍 Testing dataset properties...")

    if dataset is None:
        print("❌ No dataset to test")
        return

    # Test weighting function
    try:
        from src.clophfit.fitting.core import weight_multi_ds_titration
        print("Testing weight_multi_ds_titration...")
        weight_multi_ds_titration(dataset)
        print("✅ Dataset weighting successful")

        # Check weights after weighting
        for label, da in dataset.items():
            print(f"{label} after weighting:")
            print(f"  y_err: {da.y_err}")
            print(f"  Has zero y_err: {(da.y_err == 0).any()}")

    except Exception as e:
        print(f"❌ Dataset weighting failed: {e}")
        traceback.print_exc()

    # Test parameter building
    try:
        from src.clophfit.fitting.core import _build_params_1site
        print("\nTesting parameter building...")
        params = _build_params_1site(dataset)
        print(f"✅ Parameter building successful: {params}")

    except Exception as e:
        print(f"❌ Parameter building failed: {e}")
        traceback.print_exc()


def main():
    """Run debug tests."""
    print("🐛 DEBUG: FITTING FUNCTION FAILURES")
    print("=" * 50)

    # Test data generation
    dataset, true_params = test_data_generation()

    # Test dataset properties
    test_dataset_properties(dataset)

    # Test simple fitting
    test_simple_fitting(dataset)


if __name__ == "__main__":
    main()
