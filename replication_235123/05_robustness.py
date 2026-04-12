"""Robustness checks for Table 2 Panel A of Artiles (2025).

Base spec: reghdfe y d_div g_all_idi_grouplevel_4_n_w inter_d_div_all <controls>,
           abs(par_id) vce(cluster par_id)
Primary outcome: l_nl_pc13_viirs (log nightlights per capita 2013).
The coefficient of interest is `inter_d_div_all` (published +0.042*, SE 0.021).

Checks:
1. Different primary outcomes (all 5 Panel A columns)
2. Drop controls — raw spec
3. Drop one bishopric (obi_id) at a time (leave-one-group-out)
4. Alternative cluster: obi_id (5) and none/HC1
5. Winsorize outcome at 1%/99%
6. Drop treated-outlier parishes (top/bottom x,y coords)
7. Placebo treatment: shuffle d_div within province, re-run 500 times
8. Restrict to provinces with both treated and control
9. Add quadratic terms on key controls
10. Alternative FE: bishopric-absorbed instead of province-absorbed
11. Drop the most extreme province by exchange variance
12. Drop zero-nightlight parishes (ln = 0)
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import load_doctrinas, reghdfe, CONTROLS_ALL, PANEL_A_OUTCOMES


XVARS = ["d_div", "g_all_idi_grouplevel_4_n_w", "inter_d_div_all"] + CONTROLS_ALL


def baseline_inter(df, y="l_nl_pc13_viirs", absorb="par_id"):
    r = reghdfe(df, y, XVARS, absorb=absorb)
    return (r["params"]["inter_d_div_all"],
            r["se"]["inter_d_div_all"],
            r["params"]["d_div"],
            r["se"]["d_div"],
            r["N"])


def main():
    df = load_doctrinas()
    print("=" * 78)
    print("ROBUSTNESS CHECKS — Table 2 Panel A (primary: l_nl_pc13_viirs)")
    print("=" * 78)
    print("\nBaseline (published): d_div -0.036** [0.016], interaction +0.042** [0.021]")

    b, se, bd, sd, n = baseline_inter(df)
    print(f"Replication baseline: d_div {bd:+.3f} [{sd:.3f}], interaction {b:+.3f} [{se:.3f}] (N={n})")

    print("\n--- [1] All Panel A outcomes ---")
    for y in PANEL_A_OUTCOMES:
        b, se, bd, sd, n = baseline_inter(df, y)
        print(f"  {y:22s}: int {b:+8.3f} [{se:7.3f}]  d_div {bd:+8.3f} [{sd:7.3f}]")

    y = "l_nl_pc13_viirs"
    print("\n--- [2] Without controls (only d_div, g_all, inter) ---")
    r = reghdfe(df, y, ["d_div", "g_all_idi_grouplevel_4_n_w", "inter_d_div_all"], absorb="par_id")
    print(f"  int {r['params']['inter_d_div_all']:+.3f} [{r['se']['inter_d_div_all']:.3f}]"
          f"  d_div {r['params']['d_div']:+.3f} [{r['se']['d_div']:.3f}]  N={r['N']}")

    print("\n--- [3] Drop one bishopric at a time (leave-one-group-out) ---")
    for obi in sorted(df["obi_id"].unique()):
        sub = df[df["obi_id"] != obi]
        b, se, bd, sd, n = baseline_inter(sub, y)
        print(f"  drop obi_id={obi}: int {b:+.3f} [{se:.3f}]  d_div {bd:+.3f} [{sd:.3f}]  N={n}")

    print("\n--- [4] Alternative cluster levels ---")
    r = reghdfe(df, y, XVARS, absorb="par_id", cluster="obi_id")
    print(f"  cluster obi_id (5): int {r['params']['inter_d_div_all']:+.3f} [{r['se']['inter_d_div_all']:.3f}]")
    # HC1: no FE absorb, explicit dummies
    import statsmodels.api as sm
    prov_fe = [c for c in df.columns if c.startswith("prov_fe_")]
    d = df[[y] + XVARS + prov_fe].dropna().copy()
    X = sm.add_constant(d[XVARS + prov_fe])
    res = sm.OLS(d[y], X).fit(cov_type="HC1")
    print(f"  HC1 with prov dummies: int {res.params['inter_d_div_all']:+.3f} [{res.bse['inter_d_div_all']:.3f}]")

    print("\n--- [5] Winsorize outcome at 1%/99% ---")
    y_win = df[y].copy()
    lo, hi = y_win.quantile([0.01, 0.99])
    y_win = y_win.clip(lo, hi)
    d = df.copy(); d["_y"] = y_win
    r = reghdfe(d, "_y", XVARS, absorb="par_id")
    print(f"  int {r['params']['inter_d_div_all']:+.3f} [{r['se']['inter_d_div_all']:.3f}]"
          f"  d_div {r['params']['d_div']:+.3f}")

    print("\n--- [6] Drop geographic extreme parishes (top/bottom 2.5% x,y) ---")
    keep = (
        (df["x"] > df["x"].quantile(0.025)) & (df["x"] < df["x"].quantile(0.975)) &
        (df["y"] > df["y"].quantile(0.025)) & (df["y"] < df["y"].quantile(0.975))
    )
    b, se, bd, sd, n = baseline_inter(df[keep], y)
    print(f"  int {b:+.3f} [{se:.3f}]  d_div {bd:+.3f} [{sd:.3f}]  N={n}")

    print("\n--- [7] Placebo: shuffle d_div within province (500 reps) ---")
    rng = np.random.default_rng(42)
    placebo = []
    for _ in range(500):
        d = df.copy()
        d["d_div"] = d.groupby("par_id")["d_div"].transform(lambda s: rng.permutation(s.values))
        d["inter_d_div_all"] = d["d_div"] * d["g_all_idi_grouplevel_4_n_w"]
        r = reghdfe(d, y, XVARS, absorb="par_id")
        placebo.append(r["params"]["inter_d_div_all"])
    placebo = np.array(placebo)
    b_obs, *_ = baseline_inter(df, y)
    pval = (np.abs(placebo) >= abs(b_obs)).mean()
    print(f"  observed int = {b_obs:+.3f}, placebo mean = {placebo.mean():+.4f}, "
          f"placebo sd = {placebo.std():.4f}")
    print(f"  two-sided permutation p-value = {pval:.3f}")

    print("\n--- [8] Restrict to provinces with both treated & control ---")
    mixed = df.groupby("par_id")["d_div"].nunique() == 2
    mixed_parids = mixed[mixed].index.tolist()
    sub = df[df["par_id"].isin(mixed_parids)]
    b, se, bd, sd, n = baseline_inter(sub, y)
    print(f"  {len(mixed_parids)} mixed provinces, N={n}: int {b:+.3f} [{se:.3f}]  d_div {bd:+.3f} [{sd:.3f}]")

    print("\n--- [9] Add quadratic elevation & ruggedness ---")
    d = df.copy()
    d["mean_el_sq"] = d["mean_el_hwsd"] ** 2
    d["std_el_sq"] = d["std_el_hwsd"] ** 2
    r = reghdfe(d, y, XVARS + ["mean_el_sq", "std_el_sq"], absorb="par_id")
    print(f"  int {r['params']['inter_d_div_all']:+.3f} [{r['se']['inter_d_div_all']:.3f}]"
          f"  d_div {r['params']['d_div']:+.3f} [{r['se']['d_div']:.3f}]")

    print("\n--- [10] Bishopric FE instead of province FE ---")
    r = reghdfe(df, y, XVARS, absorb="obi_id", cluster="obi_id")
    print(f"  int {r['params']['inter_d_div_all']:+.3f} [{r['se']['inter_d_div_all']:.3f}]"
          f"  d_div {r['params']['d_div']:+.3f}")

    print("\n--- [11] Drop province with highest exchange variance ---")
    var_by = df.groupby("par_id")["g_all_idi_grouplevel_4_n_w"].var().sort_values(ascending=False)
    drop_prov = var_by.index[0]
    sub = df[df["par_id"] != drop_prov]
    b, se, bd, sd, n = baseline_inter(sub, y)
    print(f"  drop par_id={drop_prov}: int {b:+.3f} [{se:.3f}]  d_div {bd:+.3f}  N={n}")

    print("\n--- [12] Drop zero-nightlight parishes (y == 0) ---")
    sub = df[df[y] > 0]
    b, se, bd, sd, n = baseline_inter(sub, y)
    print(f"  int {b:+.3f} [{se:.3f}]  d_div {bd:+.3f}  N={n}")


if __name__ == "__main__":
    main()
