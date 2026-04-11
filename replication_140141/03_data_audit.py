"""Phase 3: data audit for the Cravino/Levchenko/Rojas macro panel."""
import pandas as pd
import numpy as np
from utils import OUT


def main():
    df = pd.read_parquet(OUT / "macro_panel.parquet")
    print("=" * 70)
    print("DATA AUDIT - Cravino/Levchenko/Rojas 2021 (macro panel)")
    print("=" * 70)

    # 1. Coverage
    print("\n[1] Coverage")
    print(f"  Rows: {len(df)}")
    print(f"  Countries: {df['code'].nunique()} ({sorted(df['code'].unique())})")
    print(f"  Years: {df['year'].min()}–{df['year'].max()}")

    # 2. Completeness by variable
    print("\n[2] Missingness")
    miss = df.isnull().mean().sort_values(ascending=False)
    print(miss[miss > 0].to_string())

    # 3. Sector share sums to 1
    print("\n[3] Sector share consistency (should sum to 1)")
    for prefix in ["va_share", "hw_share"]:
        s = df[[f"{prefix}_agr", f"{prefix}_man", f"{prefix}_ser"]].sum(axis=1)
        print(f"  {prefix}: mean={s.mean():.6f}, "
              f"min={s.min():.6f}, max={s.max():.6f}")

    # 4. Plausibility bounds
    print("\n[4] Plausibility of key variables")
    for v in ["va_share_agr", "va_share_man", "va_share_ser",
              "hw_share_ser", "share_65plus", "gdp_pc"]:
        s = df[v].dropna()
        print(f"  {v:<15} min={s.min():.4g}  p50={s.median():.4g}  "
              f"max={s.max():.4g}  mean={s.mean():.4g}")

    # 5. share_65plus bounds
    sp = df["share_65plus"].dropna()
    oob = ((sp < 0) | (sp > 1)).sum()
    print(f"\n  share_65plus out of [0,1]: {oob}")

    # 6. Panel balance
    print("\n[5] Panel balance (years per country)")
    print(df.groupby("code").size().describe().to_string())

    # 7. Outliers - top and bottom 5 on services share
    print("\n[6] Extreme values of va_share_ser")
    top = df.nlargest(5, "va_share_ser")[["code", "year", "va_share_ser"]]
    bot = df.nsmallest(5, "va_share_ser")[["code", "year", "va_share_ser"]]
    print(" Top 5:\n", top.to_string(index=False))
    print(" Bottom 5:\n", bot.to_string(index=False))

    # 8. Duplicates
    dups = df.duplicated(["code", "year"]).sum()
    print(f"\n[7] Duplicate (code, year) rows: {dups}")

    # 9. Time trend of services share (headline fact)
    print("\n[8] Average VA services share over time (1970→2005)")
    ts = df.groupby("year")["va_share_ser"].mean()
    print(f"  1970: {ts.loc[1970]:.4f}")
    print(f"  1985: {ts.loc[1985]:.4f}")
    print(f"  2000: {ts.loc[2000]:.4f}")
    print(f"  2005: {ts.loc[2005]:.4f}")

    # 10. Aging trend
    print("\n[9] Average share_65plus over time")
    ts2 = df.groupby("year")["share_65plus"].mean()
    print(f"  1970: {ts2.loc[1970]:.4f}")
    print(f"  2005: {ts2.loc[2005]:.4f}")

    # 11. Correlation of aging and services share
    print("\n[10] Cross-sectional ρ(share_65plus, va_share_ser)")
    dd = df.dropna(subset=["share_65plus", "va_share_ser"])
    r = dd[["share_65plus", "va_share_ser"]].corr().iloc[0, 1]
    print(f"  Pooled correlation: {r:.4f}")


if __name__ == "__main__":
    main()
