#!/usr/bin/env python
"""Comprehensive comparison of ALL fitting methods including ODR.

This script compares fitting methods on synthetic and real data using 3 key metrics:
1. Residual distribution normality (Gaussian) - especially for real data
2. Bias (synthetic only - requires known true pKa)
3. 95% CI coverage (pKa uncertainty should ensure 95% coverage)

Methods compared:
- LM standard (fit_binding_glob)
- LM robust (Huber loss)
- outlier2 uniform
- outlier2 shot-noise
- ODR single pass (fit_binding_odr)
- ODR recursive (fit_binding_odr_recursive)
- ODR recursive with outlier removal (fit_binding_odr_recursive_outlier)
"""

import tempfile
from pathlib import Path

from clophfit.testing.synthetic import make_dataset
import copy
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from clophfit.fitting.core import fit_binding_glob, outlier2
from clophfit.fitting.data_structures import DataArray, Dataset, FitResult
from clophfit.fitting.models import binding_1site
from clophfit.fitting.odr import (
    fit_binding_odr,
    fit_binding_odr_recursive,
    fit_binding_odr_recursive_outlier,
)
import seaborn as sns

def main():
    output_dir = Path(tempfile.mkdtemp(prefix="synthetic_data_", dir="."))
    print(f"Saving plots to: {output_dir}")
    n_repeats = 150
    Ks = []
    for i in range(1, n_repeats + 1):
        name = f"{i:02d}_uniform_err_ph_drop"
        title = f"Repeat {i}: 2 Labels, Uniform Err (y1=10, y2=3), pH Drop"
        print(f"[{i}/{n_repeats}] Generating: {name}")
        ds, truth = make_dataset(7, 1000,100, noise=0.1, error_model="simple") #uniform simple realistic physics
        output_path = output_dir / f"{name}.png"
        output_path_fit = output_dir / f"{name}_fit.png"
        fig = ds.plot(title=title)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        fr = outlier2(ds, error_model="uniform")
        Ks.append(fr.result.params["K"].value)
        fr.figure.savefig(output_path_fit, dpi=150, bbox_inches="tight")
        print(f"       Saved: {output_path}")
        print(f"       Truth: pKa={truth.K:.2f}, S0={truth.S0}, S1={truth.S1}")
        Ks.append(fr.result.params["K"].value)

        for label, da in sorted(ds.items()):
            print(f"       {label}: n={len(da.y)}, "
                  f"y_err={da.y_err.mean():.1f}±{da.y_err.std():.1f}")

    print(f"\nAll plots saved to: {output_dir}")
    print("To view: open the folder or run:")
    print(f"  ls {output_dir}")
    plt.figure()
    g = sns.histplot(Ks, kde=True)
    fig = g.get_figure()
    fig.savefig(output_dir / "residue_distribution.png", dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    main()
