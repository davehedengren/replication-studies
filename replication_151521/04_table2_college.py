"""Reproduce Table 2 (rows 10-17) of Cai & Heathcote (2020): college-level moments.

This is a Python translation of 151521-V1/main.do. Inputs are College Scorecard
2013-14 (`mostrecent.dta`) merged with Chetty et al. (2020) mobility data
(`mrc_merged.dta`) on opeid6 == super_opeid/opeid.

Paper targets (Table 2, Data column, rows 10-17):
    Std/Mean   Net tuition            0.99
    Std/Mean   Sticker tuition        0.77
    Std/Mean   Avg. family income     0.51
    Std/Mean   Fraction of high abil. 0.26
    Corr       Sticker vs Net         0.83
    Corr       Net vs Family income   0.60
    Corr       Net vs Frac high abil. 0.22
    Corr       Family inc vs Frac HA  0.59
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).parent))
from utils import load_mostrecent, load_mrc_merged, OUT, weighted_mean, weighted_std, weighted_corr


# SAT/ACT national averages used by Cai & Heathcote (hard-coded in main.do)
NAT_AVG = dict(math=510, verbal=496, writing=482, act=20)


def compute_fraction_high_ability(df):
    """Replicate the `eta` construction from main.do.

    Assumes SAT subscores are normally distributed within a college and recovers
    sigma from the 25/75 percentile gaps, then eta = 1 - Phi((national_avg - mid)/sigma).
    eta is the average of (math, verbal, writing) and ACT-derived shares.
    """
    def eta_from(mid, p25, p75, nat_avg):
        sig1 = (p75 - mid) / 0.675
        sig2 = -(p25 - mid) / 0.675
        sigma = 0.5 * (sig1 + sig2)
        return 1 - norm.cdf((nat_avg - mid) / sigma)

    df = df.copy()
    df["eta_math"] = eta_from(df["satmtmid"], df["satmt25"], df["satmt75"], NAT_AVG["math"])
    df["eta_verbal"] = eta_from(df["satvrmid"], df["satvr25"], df["satvr75"], NAT_AVG["verbal"])
    df["eta_writing"] = eta_from(df["satwrmid"], df["satwr25"], df["satwr75"], NAT_AVG["writing"])
    df["eta_sat"] = df[["eta_math", "eta_verbal", "eta_writing"]].mean(axis=1)

    # ACT: main.do uses (act75 - act25) / 0.675 / 2  (note: one sigma, not symmetric average)
    df["eta_act"] = 1 - norm.cdf(
        (NAT_AVG["act"] - df["actcmmid"]) / ((df["actcm75"] - df["actcm25"]) / 0.675 / 2)
    )
    # rmean takes row-wise mean ignoring NaN (Stata `egen eta=rmean(etasat etaact)`)
    df["eta"] = df[["eta_sat", "eta_act"]].mean(axis=1, skipna=True)
    return df


def build_college_panel():
    ms = load_mostrecent()
    mrc = load_mrc_merged()

    # main.do: merge m:1 opeid6 using mrc_merged; drop if _merge==1 or 2  -> inner join
    mrc_cols = ["opeid6", "par_mean"]
    merged = ms.merge(mrc[mrc_cols], on="opeid6", how="inner")

    # Filters
    merged = merged[merged["preddeg"] == 3]
    merged = merged[merged["region"] != 9]
    merged = merged[merged["ugds"] > 0]
    # private nonprofit (2) or public (1) — matches "keep if control==1 | control==2"
    merged = merged[merged["control"].isin([1, 2])]

    # eta construction
    merged = compute_fraction_high_ability(merged)

    # Net tuition: private if available else public
    merged["t"] = merged["npt4_priv"]
    merged.loc[merged["t"].isna(), "t"] = merged.loc[merged["t"].isna(), "npt4_pub"]

    # Subtract room and board (cost of attendance minus in-state tuition = RB proxy)
    merged["rb"] = merged["costt4_a"] - merged["tuitionfee_in"]
    merged["t_net"] = merged["t"] - merged["rb"]

    # Sticker tuition s = tuitionfee_in (in-state for publics, full for privates)
    merged["s"] = merged["tuitionfee_in"]

    # Family income (Chetty) and college scorecard family income
    merged["i2"] = merged["par_mean"]

    return merged


def summarize(merged, out_csv):
    # Only rows with all the vars needed for Table 2 rows 10-17 are used (not every
    # college has SAT/ACT + Chetty + tuition). main.do uses ugds as weight.
    need = ["t_net", "s", "i2", "eta", "ugds"]
    df = merged.dropna(subset=need).copy()
    df = df[df["ugds"] > 0]

    w = df["ugds"].values
    print(f"\nTable 2 (rows 10-17) — N = {len(df)} colleges, weighted enrollment = {w.sum():,.0f}")

    rows = []

    def std_over_mean(var, label, target):
        m = weighted_mean(df[var], w)
        s = weighted_std(df[var], w)
        val = s / m
        rows.append({"stat": label, "paper": target, "repl": round(val, 3)})
        print(f"  {label:38s}  paper={target:<5}  repl={val:6.3f}")

    def corr(v1, v2, label, target):
        c = weighted_corr(df[v1], df[v2], w)
        rows.append({"stat": label, "paper": target, "repl": round(c, 3)})
        print(f"  {label:38s}  paper={target:<5}  repl={c:6.3f}")

    print("\n  Std / Mean:")
    std_over_mean("t_net", "Net tuition std/mean", 0.99)
    std_over_mean("s", "Sticker tuition std/mean", 0.77)
    std_over_mean("i2", "Avg. family income std/mean", 0.51)
    std_over_mean("eta", "Fraction of high ability std/mean", 0.26)

    print("\n  Correlations (enrollment weighted):")
    corr("s", "t_net", "Sticker vs Net", 0.83)
    corr("t_net", "i2", "Net vs Family income", 0.60)
    corr("t_net", "eta", "Net vs Frac high ability", 0.22)
    corr("i2", "eta", "Family income vs Frac HA", 0.59)

    # Also show some levels for context
    print("\n  Levels (for reference):")
    print(f"    Mean net tuition (enrollment-wt):     ${weighted_mean(df['t_net'], w):,.0f}")
    print(f"    Mean sticker tuition (enrollment-wt): ${weighted_mean(df['s'], w):,.0f}")
    print(f"    Mean Chetty par_mean income:          ${weighted_mean(df['i2'], w):,.0f}")
    print(f"    Mean fraction high ability:            {weighted_mean(df['eta'], w):.3f}")

    result = pd.DataFrame(rows)
    result.to_csv(out_csv, index=False)
    print(f"\nSaved {out_csv}")
    return df, result


def main():
    print("=" * 70)
    print("04_table2_college.py — College Scorecard × Chetty moments (Table 2)")
    print("=" * 70)
    panel = build_college_panel()
    panel_out = OUT / "college_panel.parquet"
    panel.drop(columns=[c for c in panel.columns if panel[c].dtype == object]).to_parquet(panel_out)
    print(f"Saved college panel -> {panel_out} ({len(panel)} rows)")
    summarize(panel, OUT / "table2_moments.csv")


if __name__ == "__main__":
    main()
