"""Data audit of the cleaned quarterly US-DE yield-curve/exchange-rate panel.

Checks panel coverage, monotonicity of the yield curve, outliers, and the
consistency of forward rates, returns, and yields. These are all sanity
checks on the inputs the predictability regressions and MLE estimation rely
on.
"""

import numpy as np
import pandas as pd

from utils import MATURITIES, in_sample, load_quarterly


def audit(df, label):
    print(f"\n=== {label}: n={len(df)}, "
          f"{df['yq'].min().date()} -> {df['yq'].max().date()} ===")

    # 1. Monotone quarterly spacing
    diff = df["yq"].diff().dropna().dt.days
    print(f"quarter gap (days) -- min={diff.min()}, max={diff.max()}")

    # 2. Missing data
    ys = [f"yH_{m}" for m in MATURITIES] + [f"yF_{m}" for m in MATURITIES]
    miss = df[ys + ["log_E"]].isna().sum().sum()
    print(f"missing yields+log_E cells: {miss}")

    # 3. Yield ranges (percent, since yields are stored in %-per-year)
    for m in [12, 60, 120, 240]:
        for c in "HF":
            col = f"y{c}_{m}"
            print(f"  {col}: min={df[col].min():.2f}, "
                  f"max={df[col].max():.2f}, mean={df[col].mean():.2f}")

    # 4. Monotonicity: long yields vs short — we DO NOT expect strict
    #    monotonicity (curve can invert), but we expect the mean slope
    #    to be positive over the post-1986 sample.
    slope_H = df["yH_240"] - df["yH_12"]
    slope_F = df["yF_240"] - df["yF_12"]
    print(f"slope yH_240-yH_12: mean={slope_H.mean():.2f}  negative share={np.mean(slope_H<0):.2f}")
    print(f"slope yF_240-yF_12: mean={slope_F.mean():.2f}  negative share={np.mean(slope_F<0):.2f}")

    # 5. Forward vs yield identity: f_{m} should equal
    #    ((m+12)*y_{m+12} - m*y_m)/12  given 12-month step.
    # (Just a loose check -- the cleaned data uses NSS/GSW curves.)
    lhs = df["fH_24"]
    rhs = (2 * df["yH_24"] - 1 * df["yH_12"]) / 1  # dτ=12months
    mean_gap = (lhs - rhs).abs().mean() if "fH_24" in df.columns else np.nan
    print(f"  (forward-yield identity check at 24m): mean|gap|={mean_gap:.3f}")

    # 6. Log-exchange-rate range & innovation size
    log_E = df["log_E"]
    d_log_E = log_E.diff()
    print(f"log_E: mean={log_E.mean():.2f}, std={log_E.std():.2f}")
    print(f"Δlog_E (quarterly): std={d_log_E.std():.3f}, "
          f"min={d_log_E.min():.3f}, max={d_log_E.max():.3f}")

    # 7. Duplicate quarters
    dups = df["yq"].duplicated().sum()
    print(f"duplicate quarters: {dups}")


def main():
    full = load_quarterly()
    audit(full, "Full panel 1961Q2-2021Q1")
    audit(in_sample(full), "Paper sample 1986Q2-2021Q1")


if __name__ == "__main__":
    main()
