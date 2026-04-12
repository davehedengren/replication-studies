"""Phase 2: main tables / cited statistics from the paper.

Reproduces:
  - First stage (Fig 5): GM on GM_hat2 + controls  ->  0.30 / F ~ 15.3
  - Reduced form (Fig 6): perm_res_p25_kr26 on GM_hat2
  - Reduced form (Fig 7): causal_p25_czkr26 on GM_hat2 (weighted)
  - 2SLS main outcomes (Table 5 / Fig 8 heterogeneity)
  - Effect on kir_black_male_p252015 etc.
  - Table 1: GM impact on perm_res and causal, SD-rescaled
"""
import numpy as np
import pandas as pd
from utils import ols, tsls, BASELINE_CONTROLS, X_OLS, X_IV

df = pd.read_parquet("replication_149781/cz_clean.parquet")
X_ctrl = df[BASELINE_CONTROLS].to_numpy(dtype=float)

def run(y_name, weighted=False, instrument=X_IV):
    sub = df.dropna(subset=[y_name]).copy()
    y = sub[y_name].to_numpy(dtype=float)
    x = sub[X_OLS].to_numpy(dtype=float)
    z = sub[instrument].to_numpy(dtype=float)
    X_c = sub[BASELINE_CONTROLS].to_numpy(dtype=float)
    if weighted:
        se_col = y_name + "_se"
        w = 1.0 / (sub[se_col].to_numpy(dtype=float) ** 2)
    else:
        w = None
    return sub, y, x, z, X_c, w

print("=" * 78)
print("FIRST STAGE (Figure 5):  GM on GM_hat2 + baseline controls")
print("=" * 78)
_, _, x, z, X_c, _ = run("perm_res_p25_kr26")
const = np.ones(len(x))
Z1 = np.column_stack([const, z, X_c])
fs = ols(x, Z1)
print(f"  coeff on GM_hat2 = {fs['b'][1]:.4f}  se = {fs['se'][1]:.4f}")
# F stat on z
Zr = np.column_stack([const, X_c])
fs_r = ols(x, Zr)
rss_r, rss_u = (fs_r["resid"]**2).sum(), (fs["resid"]**2).sum()
F = ((rss_r - rss_u) / 1) / (rss_u / (len(x) - Z1.shape[1]))
print(f"  F-stat           = {F:.3f}")
print(f"  Published:  coeff = 0.30 , F-stat = 15.3")

print()
print("=" * 78)
print("REDUCED FORM (Figure 6):  perm_res_p25_kr26 on GM_hat2 + controls (no weights)")
print("=" * 78)
sub, y, x, z, X_c, _ = run("perm_res_p25_kr26")
Zrf = np.column_stack([np.ones(len(y)), z, X_c])
rf = ols(y, Zrf)
print(f"  coeff on GM_hat2 = {rf['b'][1]:.4f}  se = {rf['se'][1]:.4f}  N={len(y)}")
print(f"  Published (from figure): coeff -0.08 (no controls), reduced-form with controls also negative")

print()
print("=" * 78)
print("TABLE 5 (2SLS): causal_p25_czkr26 on GM (iv=GM_hat2, weighted 1/se^2)")
print("=" * 78)
table5_targets = [
    ("causal_p25_czkr26",   "-0.0087", "0.0028"),
    ("causal_p25_czkr26_f", None, None),
    ("causal_p25_czkr26_m", None, None),
    ("causal_p25_czkir26",  "-0.0072", "0.0027"),
    ("causal_p25_czkir26_f", "-0.0042", "0.0037"),
    ("causal_p25_czkir26_m", "-0.0103", "0.0040"),
]
for yv, pub_b, pub_se in table5_targets:
    sub, y, x, z, X_c, w = run(yv, weighted=True)
    res = tsls(y, x, z, X_c, w=w)
    pub = f"(pub {pub_b} / {pub_se})" if pub_b else ""
    print(f"  {yv:<26s} b={res['b'][1]:+.4f} se={res['se'][1]:.4f} F={res['f_stat']:.2f}  {pub}")

print()
print("=" * 78)
print("2SLS TABLE 5: kfr_black_pooled_p252015 x100 on GM (iv=GM_hat2)")
print("=" * 78)
sub, y, x, z, X_c, _ = run("kfr_black_pooled_p252015")
y100 = y * 100.0
res = tsls(y100, x, z, X_c)
# fixed index: [const, GM, ...controls]
print(f"  GM (IV)   b = {res['b'][1]:+.4f}  se = {res['se'][1]:.4f}  N={res['n']}  F={res['f_stat']:.2f}")
print(f"  Published in Fig 8 text: -0.059 se 0.026")

print()
print("=" * 78)
print("2SLS heterogeneity outcomes x100 (from Figure 8 / social outcomes tables)")
print("=" * 78)
hetero = [
    "kfr_black_pooled_p252015", "kfr_black_pooled_p752015",
    "kir_black_male_p252015",   "kir_black_male_p752015",
    "kir_black_female_p252015", "kir_black_female_p752015",
    "kfr_white_pooled_p252015",
]
rows = []
for out in hetero:
    sub, y, x, z, X_c, _ = run(out)
    res = tsls(y * 100, x, z, X_c)
    rows.append((out, res["b"][1], res["se"][1], res["n"], res["f_stat"]))
    print(f"  {out:<35s} b={res['b'][1]:+.4f} se={res['se'][1]:.4f} N={res['n']}")

print()
print("=" * 78)
print("TABLE 1: SD-rescaled IV effect on perm_res_p25_kr26 and causal_p25_czkr26")
print("=" * 78)
# Unweighted SD rescale
sub, y, x, z, X_c, _ = run("perm_res_p25_kr26")
sd_gm_hat2 = x.std(ddof=1)  # but Stata rescales GM_hat2 by its own sd
sd_GMhat2 = sub[X_IV].std(ddof=1)
sd_GM = sub[X_OLS].std(ddof=1)
z_sd = z / sd_GMhat2
x_sd = x / sd_GM
res = tsls(y, x_sd, z_sd, X_c)
print(f"  perm_res_p25_kr26: GM_imp_upm  = {res['b'][1]:.3f} se={res['se'][1]:.3f}  (Stata local GM_imp_upm)")
# Weighted for causal
sub, y, x, z, X_c, w = run("causal_p25_czkr26", weighted=True)
def wmean(a, w): return (a*w).sum()/w.sum()
def wsd(a, w):
    m = wmean(a, w)
    return np.sqrt((w * (a - m)**2).sum() / w.sum())
sd_GMhat2_w = wsd(z, w)
sd_GM_w = wsd(x, w)
z_sd = z / sd_GMhat2_w
x_sd = x / sd_GM_w
res_w = tsls(y, x_sd, z_sd, X_c, w=w)
print(f"  causal_p25_czkr26: causal SD-rescaled IV = {res_w['b'][1]:.4f} se={res_w['se'][1]:.4f}")
print(f"  *20 (effect over 20 yrs)  = {res_w['b'][1]*20:.3f}    published GM_imp_loc_20 uses factor 20")
print(f"  ratio loc/upm *20         = {res_w['b'][1]*20/res['b'][1]*100:.1f}%")
