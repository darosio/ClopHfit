# Noise Model Development - Quick Reference

## 🎯 Start Here

```bash
jupyter notebook 00_noise_model_dashboard.ipynb
```

## 📊 Workflow

```
01_noise_characterization → dev/*.csv → 02_synthetic_data_generator → 03_fitting_comparison
```

## 🔧 Import Functions

```python
from dev.noise_models import (
    compute_residual_covariance,
    analyze_label_bias,
    detect_adjacent_correlation,
    estimate_x_shift_statistics,
    simulate_correlated_noise,
    export_noise_parameters,
)
```

## 📝 Your Issues & Solutions

| Issue | What                        | Where | Function                        |
| ----- | --------------------------- | ----- | ------------------------------- |
| 1     | y1 at low pH negative (>3σ) | NB 01 | `analyze_label_bias()`          |
| 2     | Adjacent +/- alternation    | NB 01 | `detect_adjacent_correlation()` |
| 3     | pH values may be wrong      | NB 01 | `estimate_x_shift_statistics()` |
| 4     | GLS vs PyMC comparison      | NB 03 | Full analysis                   |

**NB** = Notebook

## 📂 Files

### Notebooks

- `00_noise_model_dashboard.ipynb` - Dashboard
- `01_noise_characterization.ipynb` - Analyze real data → exports to dev/
- `02_synthetic_data_generator.ipynb` - Build synthetic with realistic noise
- `03_fitting_method_comparison.ipynb` - Compare GLS vs PyMC

### Code

- `dev/noise_models.py` - All reusable functions
- `dev/__init__.py` - Package imports

### Docs

- `dev/README.md` - Full documentation
- `NOISE_MODEL_ORGANIZATION.md` - This guide

## 🎨 Typical Analysis Flow

```python
# In notebook 01
from dev.noise_models import *

# 1. Compute covariance
cov_by_label = compute_residual_covariance(all_res)

# 2. Analyze bias (Issue 1)
bias_summary, label_bias = analyze_label_bias(all_res)

# 3. Check adjacent correlation (Issue 2)
corr_stats, corr_by_label = detect_adjacent_correlation(all_res)

# 4. Test x-shift hypothesis (Issue 3)
shift_stats = estimate_x_shift_statistics(all_res, fit_results)

# 5. Export for synthetic data
export_noise_parameters(cov_by_label, label_bias, corr_stats)
```

```python
# In notebook 02
import pandas as pd

# Load parameters
cov_y1 = pd.read_csv("dev/cov_matrix_y1.csv", index_col=0)
label_bias = pd.read_csv("dev/label_bias.csv", index_col=0)

# Generate correlated noise
noise = simulate_correlated_noise(cov_y1.values, n_samples=100)
```

## 💡 Tips

- **Lost?** Check `00_noise_model_dashboard.ipynb` for overview
- **Reference?** Original `prtecan_devel.ipynb` preserved
- **Functions?** All in `dev/noise_models.py` with docstrings
- **Outputs?** Look in `dev/` for CSV files

## ✅ Validation Checklist

After notebook 01:

- [ ] `dev/cov_matrix_y1.csv` exists
- [ ] `dev/cov_matrix_y2.csv` exists
- [ ] `dev/label_bias.csv` exists
- [ ] `dev/correlation_summary.csv` exists
- [ ] Issue 1 findings documented
- [ ] Issue 2 findings documented
- [ ] Issue 3 findings documented

After notebook 02:

- [ ] Synthetic data matches real covariance
- [ ] Synthetic data matches real bias pattern
- [ ] Validation plots look good

After notebook 03:

- [ ] GLS implemented
- [ ] PyMC tested
- [ ] Performance compared
- [ ] Final recommendation made

______________________________________________________________________

**Quick Start:** `jupyter notebook 00_noise_model_dashboard.ipynb`
