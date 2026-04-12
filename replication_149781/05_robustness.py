"""Phase 4: robustness checks for the main 2SLS and Table 5 causal result."""
import numpy as np
import pandas as pd
from utils import ols, tsls, BASELINE_CONTROLS, X_OLS, X_IV

df = pd.read_parquet("replication_149781/cz_clean.parquet")


def run_iv(sub, y_name, weighted=False, controls=None):
    sub = sub.dropna(subset=[y_name]).copy()
    y = sub[y_name].to_numpy(dtype=float)
    x = sub[X_OLS].to_numpy(dtype=float)
    z = sub[X_IV].to_numpy(dtype=float)
    if controls is None:
        controls = BASELINE_CONTROLS
    X_c = sub[controls].to_numpy(dtype=float)
    w = None
    if weighted:
        w = 1.0 / (sub[y_name + "_se"].to_numpy(dtype=float) ** 2)
    return tsls(y, x, z, X_c, w=w), len(sub)


def reduced_form(sub, y_name, weighted=False, controls=None):
    sub = sub.dropna(subset=[y_name]).copy()
    y = sub[y_name].to_numpy(dtype=float)
    z = sub[X_IV].to_numpy(dtype=float)
    if controls is None:
        controls = BASELINE_CONTROLS
    X_c = sub[controls].to_numpy(dtype=float)
    w = None
    if weighted:
        w = 1.0 / (sub[y_name + "_se"].to_numpy(dtype=float) ** 2)
    Z = np.column_stack([np.ones(len(y)), z, X_c])
    return ols(y, Z, w=w), len(sub)


OUTCOME_A = "causal_p25_czkr26"       # Table 5 main outcome, weighted
OUTCOME_B = "kfr_black_pooled_p252015"  # Figure 8 / Table 6, unweighted x100
header_a = f"IV on {OUTCOME_A} (weighted, Table 5)"
header_b = f"IV on {OUTCOME_B} x100 (unweighted, Table 6 Black HH low-inc)"


def row(label, res, n, scale=1.0, extra=""):
    b = res["b"][1] * scale
    se = res["se"][1] * scale
    print(f"  {label:<45s} b={b:+.4f}  se={se:.4f}  N={n}  F={res['f_stat']:5.2f} {extra}")


print("=" * 78)
print("BASELINE")
print("=" * 78)
r, n = run_iv(df, OUTCOME_A, weighted=True)
row("Baseline (weighted)", r, n)
r, n = run_iv(df, OUTCOME_B)
row("Baseline x100", r, n, scale=100)
print()

print("=" * 78)
print("1. Alternative instrument versions (GM_hat1940, GM_hat8)")
print("=" * 78)
for alt in ["GM_hat1940", "GM_hat8"]:
    sub = df.dropna(subset=[OUTCOME_A]).copy()
    y = sub[OUTCOME_A].to_numpy(float)
    x = sub[X_OLS].to_numpy(float)
    z = sub[alt].to_numpy(float)
    X_c = sub[BASELINE_CONTROLS].to_numpy(float)
    w = 1.0 / (sub[OUTCOME_A + "_se"].to_numpy(float) ** 2)
    r = tsls(y, x, z, X_c, w=w)
    row(f"IV={alt} on {OUTCOME_A}", r, len(sub))
print()

print("=" * 78)
print("2. Drop each census region (reg1–reg4)")
print("=" * 78)
for r_ in [1, 2, 3, 4]:
    sub = df[df["region"] != r_].copy()
    keep = [c for c in BASELINE_CONTROLS
            if not (c.startswith("reg") and sub[c].sum() == 0)]
    r, n = run_iv(sub, OUTCOME_A, weighted=True, controls=keep)
    row(f"drop region {r_}", r, n)
print()

print("=" * 78)
print("3. Drop top-5 largest-GM CZs (highest treatment)")
print("=" * 78)
top5 = df.nlargest(5, X_OLS)
sub = df.drop(top5.index)
r, n = run_iv(sub, OUTCOME_A, weighted=True)
row("drop top-5 GM", r, n, extra="dropped: "+", ".join(top5["cz_name"].astype(str).tolist()))
print()

print("=" * 78)
print("4. Drop top-5 largest-instrument CZs")
print("=" * 78)
top5 = df.nlargest(5, X_IV)
sub = df.drop(top5.index)
r, n = run_iv(sub, OUTCOME_A, weighted=True)
row("drop top-5 GM_hat2", r, n, extra="dropped: "+", ".join(top5["cz_name"].astype(str).tolist()))
print()

print("=" * 78)
print("5. Drop bottom-5 weight CZs (largest-se exposure effects)")
print("=" * 78)
sub = df[df["causal_p25_czkr26_se"] <= df["causal_p25_czkr26_se"].quantile(0.95)]
r, n = run_iv(sub, OUTCOME_A, weighted=True)
row("drop top-5% noisy causal estimates", r, n)
print()

print("=" * 78)
print("6. Unweighted specification (equal weights)")
print("=" * 78)
r, n = run_iv(df, OUTCOME_A, weighted=False)
row("unweighted", r, n)
print()

print("=" * 78)
print("7. Placebo: white upward mobility outcome should be ~0")
print("=" * 78)
r, n = run_iv(df, "kfr_white_pooled_p252015")
row("IV effect on kfr_white_pooled_p252015 x100", r, n, scale=100)
r, n = run_iv(df, "kir_white_male_p252015")
row("IV effect on kir_white_male_p252015 x100", r, n, scale=100)
r, n = run_iv(df, "kir_white_female_p252015")
row("IV effect on kir_white_female_p252015 x100", r, n, scale=100)
print()

print("=" * 78)
print("8. Controls off (no baseline controls)")
print("=" * 78)
# Table 1/Fig 6 permres_GM_nocontrols cites -0.08 reduced-form w/o controls
sub = df.dropna(subset=["perm_res_p25_kr26"])
y = sub["perm_res_p25_kr26"].to_numpy(float)
x = sub[X_OLS].to_numpy(float)
z = sub[X_IV].to_numpy(float)
X_c = np.zeros((len(sub), 0))
r_rf, _ = reduced_form(sub, "perm_res_p25_kr26", controls=[])
print(f"  reduced-form no controls on perm_res_p25_kr26: b={r_rf['b'][1]:+.4f} se={r_rf['se'][1]:.4f}  (pub -0.08)")
# Also IV with no controls
r = tsls(y, x, z, X_c)
row("2SLS no controls perm_res", r, len(sub))
print()

print("=" * 78)
print("9. Reduced-form placebo: shuffle GM_hat2 (1000 draws)")
print("=" * 78)
rng = np.random.default_rng(20260410)
sub = df.dropna(subset=[OUTCOME_A]).copy().reset_index(drop=True)
y = sub[OUTCOME_A].to_numpy(float)
z_obs = sub[X_IV].to_numpy(float)
X_c = sub[BASELINE_CONTROLS].to_numpy(float)
w = 1.0 / (sub[OUTCOME_A + "_se"].to_numpy(float) ** 2)
n_sub = len(sub)
Z_base = np.column_stack([np.ones(n_sub), z_obs, X_c])
rf_obs = ols(y, Z_base, w=w)["b"][1]
draws = np.empty(1000)
for i in range(1000):
    z_perm = rng.permutation(z_obs)
    Zp = np.column_stack([np.ones(n_sub), z_perm, X_c])
    draws[i] = ols(y, Zp, w=w)["b"][1]
p_two = float(np.mean(np.abs(draws) >= abs(rf_obs)))
print(f"  observed RF coeff={rf_obs:+.4f}  two-sided permutation p={p_two:.3f}  (pub ri p≈0.054)")
print()

print("=" * 78)
print("10. Leave-one-CZ-out (min/max sensitivity)")
print("=" * 78)
sub = df.dropna(subset=[OUTCOME_A]).copy().reset_index(drop=True)
base = run_iv(sub, OUTCOME_A, weighted=True)[0]["b"][1]
coefs = []
for i in range(len(sub)):
    s = sub.drop(sub.index[i])
    r = run_iv(s, OUTCOME_A, weighted=True)[0]
    coefs.append(r["b"][1])
coefs = np.array(coefs)
order_min = int(np.argmin(coefs))
order_max = int(np.argmax(coefs))
print(f"  baseline                    b={base:+.4f}")
print(f"  min coef (drop {sub.iloc[order_min]['cz_name']:>25s}) b={coefs[order_min]:+.4f}")
print(f"  max coef (drop {sub.iloc[order_max]['cz_name']:>25s}) b={coefs[order_max]:+.4f}")
print(f"  range = [{coefs.min():+.4f}, {coefs.max():+.4f}]")
print()

print("=" * 78)
print("11. Heterogeneity gender: Table 5 sub-columns")
print("=" * 78)
for yv in ["causal_p25_czkr26_m", "causal_p25_czkr26_f",
           "causal_p25_czkir26_m", "causal_p25_czkir26_f"]:
    r, n = run_iv(df, yv, weighted=True)
    row(yv, r, n)
print()

print("=" * 78)
print("12. Winsorize GM_hat2 at 5/95 percentiles")
print("=" * 78)
sub = df.dropna(subset=[OUTCOME_A]).copy()
lo, hi = np.quantile(sub[X_IV], [0.05, 0.95])
z_w = np.clip(sub[X_IV].to_numpy(float), lo, hi)
y = sub[OUTCOME_A].to_numpy(float)
x = sub[X_OLS].to_numpy(float)
X_c = sub[BASELINE_CONTROLS].to_numpy(float)
w = 1.0 / (sub[OUTCOME_A + "_se"].to_numpy(float) ** 2)
r = tsls(y, x, z_w, X_c, w=w)
row("Winsorized GM_hat2 @ 5/95", r, len(sub))

print("\nRobustness complete.")
