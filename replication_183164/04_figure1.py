"""Replicate the coefficient estimates behind Figure 1 of the paper.

Figure 1, Panel A: edyrs ~ race_cat6 + age FE, one regression per Census year
  among US-born men ages 25-59 (hisp / white / black only).
Figure 1, Panel B: log(annual earnings) ~ race_cat6 + age FE + state FE
  (spec 1) and spec 1 + edyrs (spec 2).

The do-file uses aweight=perwt for 1990 onward and unweighted for 1970/1980
(because those IPUMS samples already pre-apply case weights).  We mirror that.

Base levels: race_cat6=6 (non-Hispanic white), age=35, statefip=6 (CA).
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
import pyreadstat

from utils import OUT, DATA_COEF

YEARS = [1970, 1980, 1990, 2000, 2010, 2019]
BASE_RACE = 6
BASE_AGE = 35
BASE_STATE = 6


def build_X(df, include_state_fe, include_edyrs):
    parts = [pd.Series(1.0, index=df.index, name="const")]
    race = pd.get_dummies(df["race_cat6"], prefix="race")
    race = race.drop(columns=[f"race_{BASE_RACE}"], errors="ignore")
    parts.append(race.astype(float))
    age = pd.get_dummies(df["age"], prefix="age")
    age = age.drop(columns=[f"age_{BASE_AGE}"], errors="ignore")
    parts.append(age.astype(float))
    if include_state_fe:
        st = pd.get_dummies(df["statefip"], prefix="st")
        st = st.drop(columns=[f"st_{BASE_STATE}"], errors="ignore")
        parts.append(st.astype(float))
    if include_edyrs:
        parts.append(df["edyrs"].astype(float).rename("edyrs"))
    return pd.concat(parts, axis=1)


def run_reg(df, y, weighted, include_state_fe=False, include_edyrs=False):
    d = df.dropna(subset=[y]).copy()
    y_vec = d[y].astype(float).to_numpy()
    X = build_X(d, include_state_fe, include_edyrs)
    X_np = X.to_numpy(dtype=float)
    if weighted:
        w = d["perwt"].astype(float).to_numpy()
        w = w * len(w) / w.sum()
        res = sm.WLS(y_vec, X_np, weights=w).fit(cov_type="HC1")
    else:
        res = sm.OLS(y_vec, X_np).fit(cov_type="HC1")
    res.exog_names = list(X.columns)
    return res


def get_param(res, key):
    if key in res.exog_names:
        i = res.exog_names.index(key)
        return float(res.params[i]), float(res.bse[i])
    return np.nan, np.nan


def load_author_coefs(tag, year):
    """Author-saved parmest coefficients for cross-check."""
    path = DATA_COEF / f"JEP-HispanicsLaborMarket_{tag}_{year}.dta"
    df, _ = pyreadstat.read_dta(str(path))
    df = df[df["parm"].str.contains("race_cat6", na=False)].copy()
    df["race_cat6"] = df["parm"].str[0].astype(int)
    return df[["race_cat6", "estimate", "stderr"]].set_index("race_cat6")


def main():
    df = pd.read_parquet(OUT / "census_figure1.parquet")
    df = df[df["sex"] == 1].copy()
    for c in ["year", "age", "statefip", "race_cat6"]:
        df[c] = df[c].astype(int)
    print(f"[04] figure1 men N={len(df):,}")

    panel_a_rows = []
    for yr in YEARS:
        sub = df[df["year"] == yr]
        weighted = yr >= 1990
        res = run_reg(sub, "edyrs", weighted=weighted)
        b_mex, se_mex = get_param(res, "race_1")
        b_blk, se_blk = get_param(res, "race_5")
        auth = load_author_coefs("Figure1A", yr)
        panel_a_rows.append({
            "year": yr,
            "N": int(res.nobs),
            "mex_repl": b_mex, "mex_se_repl": se_mex,
            "mex_auth": auth.loc[1, "estimate"], "mex_se_auth": auth.loc[1, "stderr"],
            "blk_repl": b_blk, "blk_se_repl": se_blk,
            "blk_auth": auth.loc[5, "estimate"], "blk_se_auth": auth.loc[5, "stderr"],
        })

    tA = pd.DataFrame(panel_a_rows)
    print("\n=== Figure 1, Panel A (Mexican and Black edyrs differentials, US-born men) ===")
    print(tA.to_string(index=False, float_format=lambda x: f"{x:,.4f}" if isinstance(x, float) else str(x)))

    panel_b_rows = []
    for yr in YEARS:
        sub = df[df["year"] == yr]
        weighted = yr >= 1990
        # spec 1: controls for age + state FE
        r1 = run_reg(sub, "ln_annualearnings", weighted=weighted, include_state_fe=True, include_edyrs=False)
        # spec 2: spec 1 + edyrs
        r2 = run_reg(sub, "ln_annualearnings", weighted=weighted, include_state_fe=True, include_edyrs=True)
        row = {"year": yr, "N_s1": int(r1.nobs), "N_s2": int(r2.nobs)}
        for spec_tag, res in [("s1", r1), ("s2", r2)]:
            b, se = get_param(res, "race_1")
            row[f"mex_{spec_tag}"] = b
            row[f"mex_{spec_tag}_se"] = se
            b, se = get_param(res, "race_5")
            row[f"blk_{spec_tag}"] = b
            row[f"blk_{spec_tag}_se"] = se

        # cross-check against author coefs
        auth1 = load_author_coefs("Figure1B_reg1", yr)
        auth2 = load_author_coefs("Figure1B_reg2", yr)
        row["mex_s1_auth"] = auth1.loc[1, "estimate"]
        row["mex_s2_auth"] = auth2.loc[1, "estimate"]
        row["blk_s1_auth"] = auth1.loc[5, "estimate"]
        row["blk_s2_auth"] = auth2.loc[5, "estimate"]
        panel_b_rows.append(row)

    tB = pd.DataFrame(panel_b_rows)
    print("\n=== Figure 1, Panel B (log earnings differentials) ===")
    print(tB.to_string(index=False, float_format=lambda x: f"{x:,.4f}" if isinstance(x, float) else str(x)))

    tA.to_csv(OUT / "figure1_panelA.csv", index=False)
    tB.to_csv(OUT / "figure1_panelB.csv", index=False)
    print("\n[04] done.")


if __name__ == "__main__":
    main()
