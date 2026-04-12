"""Load the cleaned quarterly panel and verify its shape and sample.

The upstream Stata pipeline already produces us_de_yc_exchange_quarterly.dta;
we only need to confirm that our Python read gives the same quarterly frame
the predictability regressions will run on.
"""

from utils import load_quarterly, in_sample, MATURITIES


def main():
    df = load_quarterly()
    print(f"Full panel: {len(df)} quarterly obs, "
          f"{df['yq'].min().date()} -> {df['yq'].max().date()}")

    sub = in_sample(df)
    print(f"In-sample (>=1986Q2): {len(sub)} obs, "
          f"{sub['yq'].min().date()} -> {sub['yq'].max().date()}")

    needed = [f"yH_{m}" for m in MATURITIES] + [f"yF_{m}" for m in MATURITIES] + ["log_E"]
    miss = sub[needed].isna().sum()
    print(f"Missing counts across key series (max): {miss.max()}")

    print(f"Short-rate 1y: yH_12 mean={sub['yH_12'].mean():.3f}%, "
          f"yF_12 mean={sub['yF_12'].mean():.3f}%")
    print(f"log(E) range: {sub['log_E'].min():.3f} -> {sub['log_E'].max():.3f}")


if __name__ == "__main__":
    main()
