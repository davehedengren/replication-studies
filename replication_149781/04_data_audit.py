"""Phase 3: data audit for the CZ sample."""
import numpy as np
import pandas as pd
from utils import BASELINE_CONTROLS, X_OLS, X_IV

df = pd.read_parquet("replication_149781/cz_clean.parquet")

print("=" * 70)
print("DATA AUDIT — Derenoncourt 2022 AER CZ sample")
print("=" * 70)
print(f"\nSample size: {len(df)} commuting zones")
print(f"Unique cz IDs: {df['cz'].nunique()}")
print(f"Region distribution:")
print(df["region"].value_counts().sort_index())

print("\n--- Treatment and instrument distribution ---")
for c in [X_OLS, X_IV, "GM_hat1940", "GM_hat8"]:
    s = df[c]
    print(f"  {c:12s} n={s.count()} mean={s.mean():.2f} sd={s.std(ddof=1):.2f} "
          f"min={s.min():.2f} p25={s.quantile(.25):.2f} p50={s.median():.2f} "
          f"p75={s.quantile(.75):.2f} max={s.max():.2f}")

print("\n--- Controls ---")
for c in BASELINE_CONTROLS:
    s = df[c]
    print(f"  {c:30s} n={s.count()} mean={s.mean():.3f} sd={s.std(ddof=1):.3f} "
          f"range=[{s.min():.3f},{s.max():.3f}]")

print("\n--- Outcome coverage ---")
outcomes = ["perm_res_p25_kr26", "causal_p25_czkr26", "causal_p25_czkir26",
            "kfr_black_pooled_p252015", "kfr_white_pooled_p252015",
            "kir_black_male_p252015", "kir_black_female_p252015"]
for c in outcomes:
    s = df[c]
    print(f"  {c:30s} n={s.count()} missing={len(df)-s.count()}")

print("\n--- Check: GM is a percentile (should span 1-100) ---")
print(f"  GM       range = {df[X_OLS].min():.1f} – {df[X_OLS].max():.1f}")
print(f"  GM_hat2  range = {df[X_IV].min():.1f} – {df[X_IV].max():.1f}")

print("\n--- Correlations among GM, GM_hat2, and controls ---")
corr_cols = [X_OLS, X_IV, "GM_hat1940", *BASELINE_CONTROLS]
corr = df[corr_cols].corr()
print(corr.round(3))

print("\n--- Missing causal outcomes by CZ ---")
miss = df[df["kfr_black_pooled_p252015"].isna()][["cz_name", "cz", X_OLS, X_IV]]
print(miss)

print("\n--- Weighted-regression standard errors distribution ---")
for c in ["causal_p25_czkr26_se", "causal_p25_czkir26_se"]:
    s = df[c]
    print(f"  {c:30s} mean={s.mean():.3f} range=[{s.min():.4f},{s.max():.4f}]")
w = 1.0 / (df["causal_p25_czkr26_se"] ** 2)
print(f"  implied weights: min={w.min():.3f} max={w.max():.3f} "
      f"ratio max/min={w.max()/w.min():.1f}")

print("\n--- Region dummies consistency ---")
reg_sum = df[["reg2", "reg3", "reg4"]].sum(axis=1)
print(f"  Count with reg2+reg3+reg4==0 (region 1): {(reg_sum == 0).sum()}")
print(f"  Count with reg2+reg3+reg4==1:            {(reg_sum == 1).sum()}")
print(f"  Any rows with >1 region?                  {(reg_sum > 1).any()}")

print("\nAudit complete.")
