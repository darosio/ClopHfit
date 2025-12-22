#!/usr/bin/env python
"""Test GPU acceleration for PyMC fitting.

Run this script to compare CPU vs GPU performance.
"""

import os
import sys
import time
from pathlib import Path

import numpy as np

# Test configurations
CONFIGS = {
    "cpu_default": {},  # Default PyTensor backend (CPU)
    "jax_cpu": {"backend": "jax"},  # JAX backend on CPU
    "jax_gpu": {"backend": "jax", "device": "gpu"},  # JAX backend on GPU
}


def setup_backend(config_name: str, config: dict) -> None:
    """Configure PyTensor backend before importing PyMC."""
    # Clear any cached imports
    for mod in list(sys.modules.keys()):
        if mod.startswith(("pytensor", "pymc", "aesara")):
            del sys.modules[mod]

    # Set JAX backend if requested
    if config.get("backend") == "jax":
        os.environ["PYTENSOR_FLAGS"] = "floatX=float32"

        # Configure JAX device
        if config.get("device") == "gpu":
            # JAX will use GPU by default if available
            pass
        else:
            # Force CPU
            os.environ["JAX_PLATFORMS"] = "cpu"
    else:
        # Use default PyTensor backend
        os.environ["PYTENSOR_FLAGS"] = "device=cpu,floatX=float64"
        if "JAX_PLATFORMS" in os.environ:
            del os.environ["JAX_PLATFORMS"]


def run_benchmark(config_name: str, config: dict, n_wells: int = 24) -> dict:
    """Run a benchmark with given configuration."""
    print(f"\n{'='*70}")
    print(f"Testing: {config_name}")
    print(f"  Backend: {config.get('backend', 'default')}")
    print(f"  Device: {config.get('device', 'cpu')}")
    print(f"  Wells: {n_wells}")
    print(f"{'='*70}")

    # Setup
    setup_backend(config_name, config)

    # Import after configuration
    import pymc as pm
    import pytensor

    print(f"PyTensor config: {pytensor.config.device}")

    # Check if using JAX
    if config.get("backend") == "jax":
        try:
            import jax
            print(f"JAX devices: {jax.devices()}")
        except ImportError:
            print("JAX not available!")
            return {"success": False, "error": "JAX not installed"}

    # Create synthetic data
    np.random.seed(42)
    x = np.linspace(5, 9, 8)
    true_K = 7.0
    y_true = 100 + 900 / (1 + 10**(true_K - x))

    results = {}

    # Build model
    print("\nBuilding model...")
    build_start = time.time()

    with pm.Model() as model:
        # Priors
        K = pm.Normal("K", mu=7.0, sigma=1.0, shape=n_wells)
        ymin = pm.Normal("ymin", mu=0, sigma=500, shape=n_wells)
        ymax = pm.Normal("ymax", mu=1000, sigma=500, shape=n_wells)

        # Model for all wells
        for i in range(n_wells):
            y_noise = y_true + np.random.normal(0, 20, len(x))
            mu = ymin[i] + (ymax[i] - ymin[i]) / (1 + 10**(K[i] - x))
            pm.Normal(f"y_{i}", mu=mu, sigma=20, observed=y_noise)

    build_time = time.time() - build_start
    results["build_time"] = build_time
    print(f"Model built in {build_time:.2f}s")

    # Sample
    print("\nSampling...")
    sample_start = time.time()

    try:
        with model:
            # Use nutpie if using JAX backend for better performance
            if config.get("backend") == "jax":
                try:
                    import nutpie
                    print("Using nutpie sampler (recommended for JAX)...")
                    trace = nutpie.sample(
                        model,
                        draws=500,
                        tune=250,
                        chains=2,
                        progress_bar=True,
                    )
                except ImportError:
                    print("Nutpie not available, using pm.sample...")
                    trace = pm.sample(
                        draws=500,
                        tune=250,
                        chains=2,
                        cores=1,
                        progressbar=True,
                        return_inferencedata=True,
                    )
            else:
                trace = pm.sample(
                    draws=500,
                    tune=250,
                    chains=2,
                    cores=1,
                    progressbar=True,
                    return_inferencedata=True,
                )

        sample_time = time.time() - sample_start
        results["sample_time"] = sample_time
        results["success"] = True
        results["total_time"] = build_time + sample_time

        print(f"\n✅ Sampling completed in {sample_time:.2f}s")
        print(f"   Total time: {results['total_time']:.2f}s")

        # Check convergence
        import arviz as az
        if hasattr(trace, 'posterior'):
            summary = az.summary(trace, var_names=["K"])
        else:
            # Nutpie returns dict, convert to InferenceData
            trace_idata = az.from_dict(trace)
            summary = az.summary(trace_idata, var_names=["K"])

        rhat_max = summary["r_hat"].max()
        results["rhat_max"] = float(rhat_max)
        print(f"   Max R-hat: {rhat_max:.3f}")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        results["success"] = False
        results["error"] = str(e)
        results["sample_time"] = None
        results["total_time"] = None

    return results


def main():
    """Run all benchmarks and compare."""
    print("="*70)
    print("CloPHfit GPU Acceleration Benchmark")
    print("="*70)

    # Check GPU availability
    print("\nChecking GPU availability...")
    gpu_available = False
    try:
        import jax
        devices = jax.devices()
        print(f"JAX devices: {devices}")
        gpu_available = any(d.platform == 'gpu' for d in devices)
        if gpu_available:
            print("✅ GPU detected via JAX")
    except ImportError:
        print("❌ JAX not installed")
    except Exception as e:
        print(f"❌ Error checking JAX: {e}")

    # Select configurations to test
    configs_to_test = {
        "cpu_default": CONFIGS["cpu_default"],
    }

    if gpu_available:
        configs_to_test["jax_cpu"] = CONFIGS["jax_cpu"]
        configs_to_test["jax_gpu"] = CONFIGS["jax_gpu"]
    else:
        print("\nℹ️  GPU tests will be skipped (no GPU available)")
        print("   Install JAX with GPU support:")
        print("   uv pip install jax[cuda12]")

    # Run benchmarks
    all_results = {}
    for config_name, config in configs_to_test.items():
        try:
            results = run_benchmark(config_name, config, n_wells=24)
            all_results[config_name] = results
        except Exception as e:
            print(f"\n❌ Benchmark failed: {e}")
            import traceback
            traceback.print_exc()
            all_results[config_name] = {
                "success": False,
                "error": str(e),
            }

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    baseline = None
    for config_name, results in all_results.items():
        if results.get("success"):
            total = results["total_time"]
            sample = results["sample_time"]
            rhat = results.get("rhat_max", "N/A")

            if baseline is None:
                baseline = total
                speedup = "1.0x (baseline)"
            else:
                speedup = f"{baseline / total:.1f}x"

            print(f"\n{config_name}:")
            print(f"  Total time:    {total:.2f}s")
            print(f"  Sampling time: {sample:.2f}s")
            print(f"  Speedup:       {speedup}")
            print(f"  Max R-hat:     {rhat:.3f}" if isinstance(rhat, float) else f"  Max R-hat:     {rhat}")
        else:
            print(f"\n{config_name}:")
            print(f"  ❌ Failed: {results.get('error', 'Unknown error')}")

    # Recommendations
    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)

    if gpu_available and "jax_gpu" in all_results:
        gpu_results = all_results.get("jax_gpu", {})
        cpu_results = all_results.get("cpu_default", {})

        if gpu_results.get("success") and cpu_results.get("success"):
            speedup = cpu_results["total_time"] / gpu_results["total_time"]

            if speedup > 2.0:
                print(f"\n✅ JAX GPU backend is HIGHLY RECOMMENDED ({speedup:.1f}x faster)")
                print("\nTo enable JAX GPU permanently, add to your code:")
                print("  import os")
                print("  os.environ['PYTENSOR_FLAGS'] = 'floatX=float32'")
                print("  # JAX will use GPU by default")
            elif speedup > 1.2:
                print(f"\n✅ JAX GPU backend recommended ({speedup:.1f}x faster)")
            else:
                print(f"\n⚠️  JAX GPU shows limited benefit ({speedup:.1f}x)")
                print("   This is normal for small models.")
                print("   GPU will be more beneficial with more wells (>50).")

        print("\nFor even better performance, consider:")
        print("  • NumPyro (JAX-native, 10-50x faster for large models)")
        print("  • Nutpie sampler (pip install nutpie)")
    else:
        print("\n💡 Install JAX with GPU support for better performance:")
        print("   uv pip install jax[cuda12]")
        print("\n   See docs/GPU_ACCELERATION.md for details")


if __name__ == "__main__":
    main()
