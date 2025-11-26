# Robust Fitting Evaluation - Development Scripts

This directory contains development and analysis scripts created during the evaluation and improvement of robust fitting methods for the ClopHfit project.

## Purpose

These scripts were used to:

- Evaluate different robust fitting approaches
- Generate synthetic datasets for testing
- Compare performance of various methods
- Produce visualizations and reports

## Organization

### Main Analysis Scripts

- **`final_robust_evaluation.py`** - Comprehensive evaluation of all robust fitting methods
- **`enhanced_robust_testing.py`** - Enhanced testing framework for robust methods
- **`simple_enhanced_robust.py`** - Simplified implementation of enhanced robust fitting
- **`error_scaling_impact_demo.py`** - Demonstration of error scaling effects
- **`show_simulated_data.py`** - Visualization of simulated datasets
- **`cleanup_analysis_files.py`** - Utility to organize analysis outputs

### Results and Supporting Files

- **`analysis_results/`** - Generated results, plots, and archived experiments
  - `comprehensive_fitting_evaluation.py` - Comprehensive evaluation script
  - `debug_fitting_issues.py` - Debugging utilities
  - `realistic_synthetic_data.py` - Realistic data generation
  - `archive/` - Archived previous versions and experiments

## Status

⚠️ **WORK IN PROGRESS** - This benchmark work is incomplete.

The synthetic data generation needs refinement to better match real experimental datasets before final recommendations can be made. See `BENCHMARK_STATUS.md` for detailed status and next steps.

### Preliminary Findings Available

Initial benchmarks identified:

- Best performing methods (`fit_lm_standard`, `fit_binding_glob_reweighted`)
- Broken methods that need fixing (`outlier2`, `api_fit_lm_outlier`)
- Functional duplicates that could be consolidated

**Note**: These findings are based on preliminary synthetic data and should be validated against real experimental datasets.

## Note

These are **development/research scripts** and are not part of the main ClopHfit package. They are excluded from linting and testing in the main build process.

## Next Steps

1. Analyze real experimental data characteristics
1. Refine synthetic data generation to match real data
1. Run comprehensive benchmarks
1. Validate recommendations
1. Implement cleanup based on validated results

See `BENCHMARK_STATUS.md` for complete details.

______________________________________________________________________

*Last Updated: 2025-11-25*
