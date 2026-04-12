"""Replicate Table 1 from Artiles (2025).

Table 1 uses Kling-Liebman-Katz (KLK) average standardized effects via the
Stata `avg_effect` package. The coefficient is the simple average of the
standardized per-outcome coefficients:

  For each outcome y_j: standardize y_std = (y - mu_c)/sigma_c using the
  control-group (d_div==0) mean/sd. Run the regression (with same X, FE,
  cluster). The reported "ae_<var>" is the MEAN across the J outcome
  regressions of the coefficient on <var>.

For joint SEs, we use the SUR-style stacked regression where each outcome
has its own intercept, its own FE (via interacted FE dummies), and shares
X. The linear combination average(X_coef) over outcomes is computed from
the joint cluster-robust variance.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import load_doctrinas, reghdfe, CONTROLS_ALL, OUTCOMES_1, OUTCOMES_2

CONTROLS_AUG = [
    "l_dist_fort", "l_dist_urban", "l_dist_monum_1", "l_dist_resid_elite",
    "l_dist_qhapac_nan", "l_dist_canal", "l_dist_puente",
    "mean_maizelo", "mean_whitepotatolo",
]
XBASE = ["d_div", "g_all_idi_grouplevel_4_n_w", "inter_d_div_all"]


def standardize_by_control(df, outcomes):
    out = df.copy()
    ctrl = df["d_div"] == 0
    for y in outcomes:
        mu = df.loc[ctrl, y].mean()
        sd = df.loc[ctrl, y].std()
        out[y] = (df[y] - mu) / sd
    return out


def avg_effect(df, outcomes, xvars, absorb=None, cluster="par_id", cov="robust",
               effect_vars=None):
    """Run per-outcome regression, average coefs. Returns dict with b, se, N.

    For SEs when cov='cluster', we approximate via stacked regression cluster SE.
    Point estimates exactly replicate Stata avg_effect.
    """
    if effect_vars is None:
        effect_vars = xvars
    d_std = standardize_by_control(df, outcomes)
    # Per-outcome regressions (point estimates)
    betas = {v: [] for v in effect_vars}
    for y in outcomes:
        if absorb is not None:
            r = reghdfe(d_std, y, xvars, absorb=absorb, cluster=cluster)
            for v in effect_vars:
                betas[v].append(r["params"][v])
        else:
            X = sm.add_constant(d_std[xvars])
            yv = d_std[y]
            if cov == "cluster":
                res = sm.OLS(yv, X).fit(cov_type="cluster",
                                        cov_kwds={"groups": d_std[cluster].values})
            else:
                res = sm.OLS(yv, X).fit(cov_type="HC1")
            for v in effect_vars:
                betas[v].append(res.params[v])
    avg = {v: float(np.mean(betas[v])) for v in effect_vars}
    # Stacked regression for SE via SUR framework (outcome-interacted design)
    long = []
    keep_cols = list(dict.fromkeys([cluster] + xvars + ([absorb] if absorb else [])))
    for j, y in enumerate(outcomes):
        tmp = d_std[[y] + keep_cols].copy()
        tmp = tmp.rename(columns={y: "__y"})
        tmp["__o"] = j
        long.append(tmp)
    L = pd.concat(long, axis=0, ignore_index=True).dropna().reset_index(drop=True)
    J = len(outcomes)
    # Build interacted design: for each of X vars, one column per outcome
    X_parts = []
    col_names = []
    for v in xvars:
        for j in range(J):
            col = (L["__o"] == j).astype(float) * L[v]
            X_parts.append(col.rename(f"{v}__o{j}"))
            col_names.append(f"{v}__o{j}")
    # Outcome intercepts
    for j in range(J):
        X_parts.append(((L["__o"] == j).astype(float)).rename(f"__int_o{j}"))
        col_names.append(f"__int_o{j}")
    # FE dummies interacted with outcome (mimics per-outcome FE absorption)
    if absorb is not None:
        fe_vals = sorted(L[absorb].dropna().unique())
        for j in range(J):
            for k, fv in enumerate(fe_vals[1:]):  # drop first
                col = ((L["__o"] == j) & (L[absorb] == fv)).astype(float)
                X_parts.append(col.rename(f"fe_{k}_o{j}"))
                col_names.append(f"fe_{k}_o{j}")
    X = pd.concat(X_parts, axis=1)
    # Drop collinear
    y = L["__y"]
    res = sm.OLS(y, X).fit(cov_type="cluster",
                           cov_kwds={"groups": L[cluster].values})
    # Compute linear combo: avg over outcomes of coef on v
    out_se = {}
    cov_mat = res.cov_params()
    for v in effect_vars:
        R = np.zeros(len(res.params))
        for j in range(J):
            name = f"{v}__o{j}"
            if name in res.params.index:
                R[list(res.params.index).index(name)] = 1.0 / J
        var = float(R @ cov_mat.values @ R)
        out_se[v] = float(np.sqrt(var))
    return {
        "b": avg, "se": out_se, "N": int(L.shape[0]), "n_parishes": int(L[cluster].nunique() * J if False else len(L) // J),
        "per_outcome": betas,
    }


def print_col(label, res, pub):
    print(f"\n[{label}]  N={res['N']} (obs x outcomes)")
    for v, pname in [("d_div", "d_div"), ("g_all_idi_grouplevel_4_n_w", "g_all"), ("inter_d_div_all", "inter")]:
        if v in res["b"]:
            b = res["b"][v]
            s = res["se"][v]
            pub_b, pub_s = pub.get(pname, (None, None))
            pub_str = f" (pub {pub_b:+.3f} [{pub_s:.3f}])" if pub_b is not None else ""
            print(f"  {pname:12s}: {b:+.3f} [{s:.3f}]{pub_str}")


def main():
    df = load_doctrinas()

    print("Table 1: Average Standardized Effects (KLK)")
    print("=" * 78)

    # C1: x=d_div only, robust (HC1)
    r = avg_effect(df, OUTCOMES_1, ["d_div"], absorb=None, cov="robust")
    print_col("C1", r, {"d_div": (-0.200, 0.070)})
    # C2: x=d_div + controls_all, robust
    r = avg_effect(df, OUTCOMES_1, ["d_div"] + CONTROLS_ALL, absorb=None, cov="robust",
                   effect_vars=["d_div"])
    print_col("C2", r, {"d_div": (-0.165, 0.066)})
    # C3: full XBASE + controls_all, robust
    r = avg_effect(df, OUTCOMES_1, XBASE + CONTROLS_ALL, absorb=None, cov="robust",
                   effect_vars=XBASE)
    print_col("C3", r, {"d_div": (-0.689, 0.243), "g_all": (0.080, 0.213), "inter": (0.781, 0.349)})
    # C4: + obi FE, cluster obi_id
    r = avg_effect(df, OUTCOMES_1, XBASE + CONTROLS_ALL, absorb="obi_id", cluster="obi_id",
                   effect_vars=XBASE)
    print_col("C4", r, {"d_div": (-0.556, 0.247), "g_all": (0.253, 0.235), "inter": (0.659, 0.321)})
    # C5: + prov FE, cluster par_id
    r = avg_effect(df, OUTCOMES_1, XBASE + CONTROLS_ALL, absorb="par_id", cluster="par_id",
                   effect_vars=XBASE)
    print_col("C5", r, {"d_div": (-0.604, 0.206), "g_all": (0.226, 0.226), "inter": (0.789, 0.303)})
    # C8: outcomes2, prov FE, cluster par_id
    r = avg_effect(df, OUTCOMES_2, XBASE + CONTROLS_ALL, absorb="par_id", cluster="par_id",
                   effect_vars=XBASE)
    print_col("C8", r, {"d_div": (-0.610, 0.225), "g_all": (0.299, 0.318), "inter": (0.846, 0.350)})
    # C10: outcomes2, prov FE + controls_aug + idi, cluster par_id
    r = avg_effect(df, OUTCOMES_2,
                   XBASE + CONTROLS_ALL + CONTROLS_AUG + ["idi_parishlevel_4_n"],
                   absorb="par_id", cluster="par_id",
                   effect_vars=XBASE + ["idi_parishlevel_4_n"])
    pub_c10 = {"d_div": (-0.561, 0.209), "g_all": (-0.074, 0.331), "inter": (0.836, 0.343)}
    print_col("C10", r, pub_c10)
    print(f"  idi_parish  : {r['b']['idi_parishlevel_4_n']:+.3f} [{r['se']['idi_parishlevel_4_n']:.3f}]"
          f" (pub +0.403 [0.294])")


if __name__ == "__main__":
    main()
