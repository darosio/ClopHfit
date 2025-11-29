# Robust Fitting Benchmarks

This directory contains scripts for evaluating and benchmarking the robust fitting methods in ClopHfit.

## 📂 Scripts

### 1. Synthetic Data Benchmark (`run_benchmark.py`)

The primary benchmark script. It uses `realistic_synthetic_data.py` to generate datasets that mimic real experimental conditions (noise, outliers, correlation) and evaluates all fitting methods.

**Usage:**

```bash
python benchmarks/run_benchmark.py
```

### 2. Real Data Comparison (`compare_real_data.py`)

Compares fitting methods on **real** experimental data (Tecan datasets). It uses the full `Titration` pipeline (buffer subtraction + dilution correction) to ensure production-grade evaluation.

**Usage:**

```bash
python benchmarks/compare_real_data.py
```

### 3. Stress Testing (`stress_test.py`)

Tests fitting methods under extreme conditions (high outlier rates, signal drops, missing data) to identify failure modes.

**Usage:**

```bash
python benchmarks/stress_test.py
```

### 4. Data Generation (`realistic_synthetic_data.py`)

Module for generating realistic synthetic titration data. It is based on statistical analysis of real Tecan datasets (L1, L2, L4, 140220).

## 📊 Key Findings

- **IRLS (Iteratively Reweighted Least Squares)** is generally the most robust method for standard experimental noise.
- **Bayesian Methods (PyMC)** provide the most accurate results when x-errors (pH uncertainty) are significant, but are slower.
- **Outlier Detection** is critical for datasets with "acidic tail collapse" (signal drop at low pH).

## 🛠️ Dependencies

These scripts require the `clophfit` package to be installed (or available in `PYTHONPATH`).
