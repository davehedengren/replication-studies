"""Data audit for Artiles (2025) parish-level data.

Checks:
- Coverage: obs count, unique parishes, province/bishopric grouping
- Distributions: summary stats, plausibility bounds
- Logical: shares in [0,1], interaction = d_div * exchange
- Missing data patterns by treatment
- Duplicates
- Geographic coverage
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import load_doctrinas, CONTROLS_ALL, PANEL_A_OUTCOMES


def main():
    df = load_doctrinas()
    print("=" * 78)
    print("DATA AUDIT — data_doctrinas.dta")
    print("=" * 78)

    # 1. Coverage
    print(f"\n[1] Coverage")
    print(f"    N rows         : {len(df)}")
    print(f"    N cols         : {df.shape[1]}")
    print(f"    Unique u_id    : {df['u_id'].nunique()}")
    print(f"    Unique par_id  : {df['par_id'].nunique()} (provinces)")
    print(f"    Unique obi_id  : {df['obi_id'].nunique()} (bishoprics)")
    # Longitude/latitude range
    print(f"    x range (lon)  : [{df['x'].min():.3f}, {df['x'].max():.3f}]")
    print(f"    y range (lat)  : [{df['y'].min():.3f}, {df['y'].max():.3f}]")

    # 2. Treatment balance
    print(f"\n[2] Treatment distribution (d_div)")
    print(f"    mean(d_div)    : {df['d_div'].mean():.4f}")
    print(f"    N treated      : {int(df['d_div'].sum())}")
    print(f"    N control      : {int((1 - df['d_div']).sum())}")

    # 3. Crop exchange variable
    print(f"\n[3] Crop exchange (g_all_idi_grouplevel_4_n_w)")
    ex = df["g_all_idi_grouplevel_4_n_w"]
    print(f"    range          : [{ex.min():.3f}, {ex.max():.3f}]")
    print(f"    mean (sd)      : {ex.mean():.3f} ({ex.std():.3f})")

    # 4. Interaction consistency
    print(f"\n[4] Interaction (inter_d_div_all) consistency")
    computed = df["d_div"] * df["g_all_idi_grouplevel_4_n_w"]
    diff = (computed - df["inter_d_div_all"]).abs().max()
    print(f"    max |inter - d_div*g_all| = {diff:.2e}")
    if diff < 1e-8:
        print("    -> Interaction variable exactly equals d_div * g_all. OK.")
    else:
        print("    -> WARNING: interaction does NOT equal d_div * g_all")

    # 5. Outcome distributions
    print(f"\n[5] Outcome distributions")
    cols = PANEL_A_OUTCOMES + ["l_mean_nl_pc00_03", "CN94_d_ua_venta",
                                "CS93_sh_hig_red", "CS93_sh_agua_red",
                                "CS17_sh_hig_red", "CS17_sh_agua_red"]
    for c in cols:
        if c not in df.columns:
            continue
        s = df[c]
        print(f"    {c:22s}  n={s.notna().sum():3d}  "
              f"mean={s.mean():8.3f}  sd={s.std():8.3f}  "
              f"min={s.min():8.3f}  max={s.max():8.3f}")

    # 6. Shares in [0,1]
    print(f"\n[6] Share variables in [0,1]?")
    share_vars = [c for c in df.columns if ("_sh_" in c or c.endswith("_sh"))
                  and df[c].dtype != "O"]
    for c in share_vars[:15]:
        s = df[c].dropna()
        if len(s) == 0:
            continue
        ok = (s >= 0).all() and (s <= 1).all()
        if not ok:
            print(f"    {c:25s}: min={s.min():.3f} max={s.max():.3f}  <- out of [0,1]!")

    # 7. Missingness by treatment
    print(f"\n[7] Missing rates of outcomes by d_div")
    for y in PANEL_A_OUTCOMES:
        if y not in df.columns:
            continue
        na_treat = df.loc[df["d_div"] == 1, y].isna().mean()
        na_ctrl = df.loc[df["d_div"] == 0, y].isna().mean()
        print(f"    {y:22s}: treated miss={na_treat:.3f}  control miss={na_ctrl:.3f}")

    # 8. Covariate balance summary (treated vs control)
    print(f"\n[8] Covariate means: d_div==1 vs d_div==0")
    for c in CONTROLS_ALL:
        m1 = df.loc[df["d_div"] == 1, c].mean()
        m0 = df.loc[df["d_div"] == 0, c].mean()
        sd0 = df.loc[df["d_div"] == 0, c].std()
        std_diff = (m1 - m0) / sd0 if sd0 > 0 else np.nan
        print(f"    {c:25s}: treat={m1:9.2f}  ctrl={m0:9.2f}  std.diff={std_diff:+.3f}")

    # 9. Duplicates
    print(f"\n[9] Duplicates")
    dup = df.duplicated(subset=["u_id"]).sum()
    print(f"    duplicate u_id : {dup}")
    dup = df.duplicated(subset=["x", "y"]).sum()
    print(f"    duplicate (x,y): {dup}")

    # 10. Parish grouping sizes
    print(f"\n[10] Parishes per province (par_id) and bishopric (obi_id)")
    g = df.groupby("par_id").size()
    print(f"    par_id: min={g.min()}, median={int(g.median())}, max={g.max()}, "
          f"n_groups={len(g)}")
    g = df.groupby("obi_id").size()
    print(f"    obi_id: min={g.min()}, median={int(g.median())}, max={g.max()}, "
          f"n_groups={len(g)}")


if __name__ == "__main__":
    main()
