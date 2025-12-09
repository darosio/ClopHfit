#!/usr/bin/env python
"""Manual test script for synthetic data generation.

Generates 10 repeats of 1 example with 2 labels, uniform errors, pH drop.
"""

import tempfile
from pathlib import Path

from clophfit.testing.synthetic import make_dataset


def main():
    """Generate 10 repeats of 1 example with 2 labels, uniform errors, pH drop."""
    output_dir = Path(tempfile.mkdtemp(prefix="synthetic_data_", dir="."))
    print(f"Saving plots to: {output_dir}")

    n_repeats = 10
    base_seed = 42

    for i in range(1, n_repeats + 1):
        name = f"{i:02d}_uniform_err_ph_drop"
        title = f"Repeat {i}: 2 Labels, Uniform Err (y1=10, y2=3), pH Drop"

        print(f"[{i}/{n_repeats}] Generating: {name}")
        ds, truth = make_dataset(
            randomize_signals=True,
            n_labels=2,
            error_model="uniform",
            y_err={"y1": 10.0, "y2": 3.0},
            low_ph_drop=True,
            low_ph_drop_magnitude=0.4,
            low_ph_drop_label="y1",
            seed=base_seed + i,
        )

        output_path = output_dir / f"{name}.png"
        fig = ds.plot(title=title)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"       Saved: {output_path}")
        print(f"       Truth: pKa={truth.K:.2f}, S0={truth.S0}, S1={truth.S1}")

        for label, da in sorted(ds.items()):
            print(f"       {label}: n={len(da.y)}, "
                  f"y_err={da.y_err.mean():.1f}±{da.y_err.std():.1f}")

    print(f"\nAll plots saved to: {output_dir}")
    print("To view: open the folder or run:")
    print(f"  ls {output_dir}")


if __name__ == "__main__":
    main()
