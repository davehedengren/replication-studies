"""Phase 1/2 step 1: load the CZ analytic dataset and verify key variables."""
from utils import load_cz, BASELINE_CONTROLS, X_OLS, X_IV

df = load_cz()
print(f"CZ rows: {len(df)}   (paper: 130)")
print(f"Unique CZs: {df['cz'].nunique()}")
print("\nKey variables (non-missing counts):")
key = [X_OLS, X_IV, "GM_hat1940", "GM_hat8",
       "perm_res_p25_kr26", "causal_p25_czkr26", "causal_p25_czkir26",
       "kfr_black_pooled_p252015", "kir_black_male_p252015", "kir_black_female_p252015",
       "kfr_white_pooled_p252015"]
for c in key + BASELINE_CONTROLS:
    n = df[c].count() if c in df.columns else "MISSING"
    print(f"  {c:<35s} n={n}")

print("\nSD of GM and GM_hat2 (unweighted):")
print(f"  sd(GM)       = {df['GM'].std(ddof=1):.4f}   (paper reports ~28.98)")
print(f"  sd(GM_hat2)  = {df['GM_hat2'].std(ddof=1):.4f}")
print(f"  mean(GM)     = {df['GM'].mean():.4f}")
print(f"  mean(GM_hat2)= {df['GM_hat2'].mean():.4f}")

# Save as parquet for downstream scripts
out = df[[X_OLS, X_IV, "GM_hat1940", "GM_hat8", "cz", "cz_name",
          "perm_res_p25_kr26", "causal_p25_czkr26", "causal_p25_czkr26_se",
          "causal_p25_czkir26", "causal_p25_czkir26_se",
          "causal_p25_czkr26_m", "causal_p25_czkr26_m_se",
          "causal_p25_czkr26_f", "causal_p25_czkr26_f_se",
          "causal_p25_czkir26_m", "causal_p25_czkir26_m_se",
          "causal_p25_czkir26_f", "causal_p25_czkir26_f_se",
          "kfr_black_pooled_p252015", "kfr_black_pooled_p752015",
          "kir_black_male_p252015", "kir_black_male_p752015",
          "kir_black_female_p252015", "kir_black_female_p752015",
          "kfr_white_pooled_p252015", "kir_white_male_p252015",
          "kir_white_female_p252015",
          "region", *BASELINE_CONTROLS]].copy()
out.to_parquet("replication_149781/cz_clean.parquet", index=False)
print("\nWrote cz_clean.parquet with", len(out), "rows,", len(out.columns), "cols")
