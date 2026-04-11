"""Numerically compares our Python gravity coefficients to the shipped Stata
coefficients in Raw_data/Gravity1919_LZ.xls."""
import numpy as np
import pandas as pd

from utils import DIST_BINS, OTHER_VARS, RAW, OUT


def main():
    stata = pd.read_csv(RAW / "Gravity1919_LZ.xls", sep="\t").set_index(
        pd.read_csv(RAW / "Gravity1919_LZ.xls", sep="\t").columns[0]
    )
    rows = DIST_BINS + OTHER_VARS
    stata_coefs = stata.loc[rows].astype(float)

    mine = pd.read_csv(OUT / "gravity_coefs.csv")
    mine_wide = mine.pivot(index="var", columns="sector", values="coef")
    mine_wide.columns = [f"est{c}" for c in mine_wide.columns]
    mine_wide = mine_wide.reindex(rows)

    diff = stata_coefs.values - mine_wide.values
    abs_diff = np.abs(diff)
    print(f"Cells compared: {abs_diff.size}")
    print(f"Max abs diff:    {abs_diff.max():.3e}")
    print(f"Mean abs diff:   {abs_diff.mean():.3e}")
    print(f"Median abs diff: {np.median(abs_diff):.3e}")
    print(f"Cells with diff > 1e-4: {(abs_diff > 1e-4).sum()}")

    # Per-variable max diff
    per_var = pd.DataFrame(
        {"max_abs_diff": abs_diff.max(axis=1), "mean_abs_diff": abs_diff.mean(axis=1)},
        index=rows,
    )
    print("\nPer-variable diff summary:")
    print(per_var.to_string())

    per_var.to_csv(OUT / "gravity_diff_vs_stata.csv")


if __name__ == "__main__":
    main()
