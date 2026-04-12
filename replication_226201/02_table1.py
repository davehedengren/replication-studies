"""Table 1 — NIH IV regressions (Alon, Capelle, Matsuda 2025).

Replicates all six specifications from `3 - NIH Regressions.do` and compares
each coefficient and SE to the published values in
`226201-V1/empirical/output/Table_1.xlsx`.
"""
import numpy as np
import pandas as pd

from utils import OUT, aw_cluster_iv, build_regression_panel, dummies_from

PANEL = build_regression_panel()

# --- Published values from Table_1.xlsx ----------------------------------
PUBLISHED = {
    "(1)":     {"rd": 0.15039, "rd_se": 0.05324, "n": 1565, "r2": 0.16899},
    "(2)":     {"rd": 0.10438, "rd_se": 0.03249,
                "rd_x_pnp": 0.06313, "rd_x_pnp_se": 0.07258,
                "pnp": 2811.084, "pnp_se": 371.881,
                "n": 1565, "r2": 0.41318},
    "(3)":     {"rd": 0.10452, "rd_se": 0.04693,
                "rd_x_pnp": 0.06498, "rd_x_pnp_se": 0.07126,
                "pnp": 2877.047, "pnp_se": 355.832,
                "n": 1565, "r2": 0.41247},
    "(NLA)":   {"rd": 0.11670, "rd_se": 0.04700,
                "rd_x_pnp": 0.04566, "rd_x_pnp_se": 0.07455,
                "pnp": 3166.481, "pnp_se": 431.292,
                "n": 1062, "r2": 0.43759},
    "(LAC)":   {"rd": 4.038,  "rd_se": 16.221,
                "rd_x_pnp": -3.969, "rd_x_pnp_se": 16.248,
                "pnp": 3805.973, "pnp_se": 1500.670,
                "n": 503, "r2": 0.47906},
    "(Placebo)": {"rd": 3.820, "rd_se": 2.790,
                  "rd_x_pnp": -1.252, "rd_x_pnp_se": 3.061,
                  "pnp": 1374.018, "pnp_se": 3094.988,
                  "n": 1565},
}


def build_state_dummies(df):
    D = dummies_from(df["geo_state"], "state", drop_first=True)
    return D.values, list(D.columns)


def build_sector_dummies(df):
    # Baseline = sector==1; include sector==2 dummy (private non-profit)
    pnp = (df["sector"] == 2).astype(float).values.reshape(-1, 1)
    return pnp, ["pnp"]


def spec_1(df):
    """Benchmark IV: xx instrumented by zz; controls = state FE."""
    dfr = df.dropna(subset=["yy", "xx", "zz", "fte_wts", "geo_state"]).copy()
    states, state_names = build_state_dummies(dfr)
    n = len(dfr)
    X_exog = np.column_stack([np.ones(n), states])
    X_endog = dfr[["xx"]].values
    Z_instr = dfr[["zz"]].values
    res = aw_cluster_iv(
        y=dfr["yy"].values,
        X_exog=X_exog, X_endog=X_endog, Z_instr=Z_instr,
        weights=dfr["fte_wts"].values,
        cluster=dfr["geo_state"].values,
    )
    res["names"] = ["xx"] + ["const"] + state_names
    return res


def spec_sectoral(df, xvar="xx", controls=None):
    """
    Matches `ivregress 2sls yy (c.xx c.xx#i.sector = c.zz c.zz#i.sector)
                               i.sector i.state [aw=fte_wts] ...`

    Endogenous: xx, xx*pnp
    Instruments: zz, zz*pnp
    Exogenous: const, pnp, state dummies, (optional controls)
    """
    req = ["yy", xvar, "zz", "fte_wts", "geo_state", "sector"]
    if controls:
        req += controls
    dfr = df.dropna(subset=req).copy()
    states, state_names = build_state_dummies(dfr)
    pnp = (dfr["sector"] == 2).astype(float).values.reshape(-1, 1)
    xx_main = dfr[[xvar]].values
    xx_int = xx_main * pnp
    zz_main = dfr[["zz"]].values
    zz_int = zz_main * pnp

    exog_blocks = [np.ones(len(dfr)).reshape(-1, 1), pnp, states]
    ctrl_names = []
    if controls:
        ctrl_mat = dfr[controls].values
        exog_blocks.append(ctrl_mat)
        ctrl_names = list(controls)

    X_exog = np.column_stack(exog_blocks)
    X_endog = np.column_stack([xx_main, xx_int])
    Z_instr = np.column_stack([zz_main, zz_int])

    res = aw_cluster_iv(
        y=dfr["yy"].values,
        X_exog=X_exog, X_endog=X_endog, Z_instr=Z_instr,
        weights=dfr["fte_wts"].values,
        cluster=dfr["geo_state"].values,
    )
    res["names"] = [xvar, f"{xvar}*pnp", "const", "pnp"] + state_names + ctrl_names
    return res


def fmt(p, r, key):
    pub = p.get(key, "")
    return f"{pub:>12}" if isinstance(pub, str) else f"{pub:>12.4f}"


def pick(res, name):
    i = res["names"].index(name)
    return res["beta"][i], res["se"][i]


rows = []

# (1) benchmark
r = spec_1(PANEL)
b_rd, se_rd = pick(r, "xx")
pub = PUBLISHED["(1)"]
rows.append(("(1) Benchmark", "R&D", pub["rd"], pub["rd_se"], b_rd, se_rd))
print(f"(1) N={r['n']} (pub 1565)  R²={r['R2']:.5f} (pub 0.16899)")
print(f"    R&D: {b_rd:.5f} ({se_rd:.5f})  vs pub 0.15039 (0.05324)")

# (2) sector IV
r = spec_sectoral(PANEL)
b_rd, se_rd = pick(r, "xx")
b_int, se_int = pick(r, "xx*pnp")
b_pnp, se_pnp = pick(r, "pnp")
pub = PUBLISHED["(2)"]
rows.append(("(2) Sector IV", "R&D", pub["rd"], pub["rd_se"], b_rd, se_rd))
rows.append(("(2) Sector IV", "R&D*PNP", pub["rd_x_pnp"], pub["rd_x_pnp_se"], b_int, se_int))
rows.append(("(2) Sector IV", "PNP", pub["pnp"], pub["pnp_se"], b_pnp, se_pnp))
print(f"(2) N={r['n']}  R²={r['R2']:.5f}")
print(f"    R&D: {b_rd:.5f} ({se_rd:.5f})  vs pub 0.10438 (0.03249)")
print(f"    R&D*PNP: {b_int:.5f} ({se_int:.5f})  vs pub 0.06313 (0.07258)")
print(f"    PNP: {b_pnp:.3f} ({se_pnp:.3f})  vs pub 2811.08 (371.88)")

# (3) Sector IV + pre-trend
ctrls = ["init_life_sci_uni", "init_uni_size", "init_fac_stud_ratio"]
# The Stata spec uses `i.init_life_sci_uni` — dummy. Since it's 0/1 already
# it is identical to including the raw variable.
r = spec_sectoral(PANEL, controls=ctrls)
b_rd, se_rd = pick(r, "xx")
b_int, se_int = pick(r, "xx*pnp")
b_pnp, se_pnp = pick(r, "pnp")
pub = PUBLISHED["(3)"]
rows.append(("(3) +controls", "R&D", pub["rd"], pub["rd_se"], b_rd, se_rd))
rows.append(("(3) +controls", "R&D*PNP", pub["rd_x_pnp"], pub["rd_x_pnp_se"], b_int, se_int))
rows.append(("(3) +controls", "PNP", pub["pnp"], pub["pnp_se"], b_pnp, se_pnp))
print(f"(3) N={r['n']}  R²={r['R2']:.5f}")
print(f"    R&D: {b_rd:.5f} ({se_rd:.5f})  vs pub 0.10452 (0.04693)")
print(f"    R&D*PNP: {b_int:.5f} ({se_int:.5f})  vs pub 0.06498 (0.07126)")
print(f"    PNP: {b_pnp:.3f} ({se_pnp:.3f})  vs pub 2877.05 (355.83)")

# (NLA) non-liberal-arts
nla = PANEL[PANEL["flag_lib_art_col"] == 0]
r = spec_sectoral(nla, controls=ctrls)
b_rd, se_rd = pick(r, "xx")
b_int, se_int = pick(r, "xx*pnp")
b_pnp, se_pnp = pick(r, "pnp")
pub = PUBLISHED["(NLA)"]
rows.append(("(NLA)", "R&D", pub["rd"], pub["rd_se"], b_rd, se_rd))
rows.append(("(NLA)", "R&D*PNP", pub["rd_x_pnp"], pub["rd_x_pnp_se"], b_int, se_int))
rows.append(("(NLA)", "PNP", pub["pnp"], pub["pnp_se"], b_pnp, se_pnp))
print(f"(NLA) N={r['n']} (pub 1062)")
print(f"    R&D: {b_rd:.5f} ({se_rd:.5f})  vs pub 0.11670 (0.04700)")

# (LAC) liberal-arts-only
lac = PANEL[PANEL["flag_lib_art_col"] == 1]
r = spec_sectoral(lac, controls=ctrls)
b_rd, se_rd = pick(r, "xx")
b_int, se_int = pick(r, "xx*pnp")
b_pnp, se_pnp = pick(r, "pnp")
pub = PUBLISHED["(LAC)"]
rows.append(("(LAC)", "R&D", pub["rd"], pub["rd_se"], b_rd, se_rd))
rows.append(("(LAC)", "R&D*PNP", pub["rd_x_pnp"], pub["rd_x_pnp_se"], b_int, se_int))
rows.append(("(LAC)", "PNP", pub["pnp"], pub["pnp_se"], b_pnp, se_pnp))
print(f"(LAC) N={r['n']} (pub 503)")
print(f"    R&D: {b_rd:.3f} ({se_rd:.3f})  vs pub 4.038 (16.221)")

# (Placebo) amenity spending as endogenous variable
r = spec_sectoral(PANEL, xvar="placebo_xx", controls=ctrls)
b_rd, se_rd = pick(r, "placebo_xx")
b_int, se_int = pick(r, "placebo_xx*pnp")
b_pnp, se_pnp = pick(r, "pnp")
pub = PUBLISHED["(Placebo)"]
rows.append(("(Placebo)", "Amenity", pub["rd"], pub["rd_se"], b_rd, se_rd))
rows.append(("(Placebo)", "Amenity*PNP", pub["rd_x_pnp"], pub["rd_x_pnp_se"], b_int, se_int))
rows.append(("(Placebo)", "PNP", pub["pnp"], pub["pnp_se"], b_pnp, se_pnp))
print(f"(Placebo) N={r['n']}")
print(f"    Amenity: {b_rd:.3f} ({se_rd:.3f})  vs pub 3.820 (2.790)")

out_df = pd.DataFrame(rows, columns=["spec", "coef", "pub_b", "pub_se", "rep_b", "rep_se"])
out_df["diff_b_pct"] = (out_df["rep_b"] - out_df["pub_b"]) / out_df["pub_b"] * 100
out_df.to_csv(OUT / "table1_comparison.csv", index=False)
print("\nFull comparison:")
print(out_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print(f"\nSaved to {OUT / 'table1_comparison.csv'}")
