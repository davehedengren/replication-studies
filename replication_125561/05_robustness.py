"""Robustness checks for 125561 main turnout specification
(Panel B of Table 1, col 1: ln_turnout on ln_fine with district + date×province×cat FE).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
from utils import feols_hdfe, iterative_singleton_drop, OUT

BASELINE = {"y": "ln_turnout", "x": ["ln_fine"], "absorb": ["ubigeo", "dpc_id"],
            "cluster": "province_id", "weight": "electores_01"}


def refit(df):
    d = df.copy()
    d = iterative_singleton_drop(d, ["ubigeo", "dpc_id"])
    for col in ["ubigeo", "dpc_id"]:
        d[col] = pd.Categorical(d[col]).codes.astype(int)
    return feols_hdfe(d, BASELINE["y"], BASELINE["x"], BASELINE["absorb"],
                      BASELINE["cluster"], BASELINE["weight"])


def rep(label, res, paper_b=None):
    b = res["beta"][0]
    se = res["se"][0]
    t = b / se
    star = "***" if abs(t) > 2.58 else ("**" if abs(t) > 1.96 else ("*" if abs(t) > 1.645 else ""))
    cmp = f"  (baseline={paper_b:+.4f})" if paper_b is not None else ""
    print(f"  {label:<42} β={b:+.4f}[{se:.4f}]{star}  N={res['N']}{cmp}")


def main():
    df = pd.read_parquet(OUT / "main_sample.parquet")

    # Baseline
    base = feols_hdfe(df, BASELINE["y"], BASELINE["x"], BASELINE["absorb"],
                      BASELINE["cluster"], BASELINE["weight"])
    paper_b = base["beta"][0]
    print("\nBaseline: ln_turnout on ln_fine")
    rep("Baseline (Table 1B col 1)", base)

    print("\n========== Robustness checks ==========")

    # 1. HC1 robust SE (unclustered)
    d = df.copy()
    res_hc1 = feols_hdfe(d, "ln_turnout", ["ln_fine"], ["ubigeo", "dpc_id"],
                         cluster="ubigeo", weight="electores_01")
    rep("1. Cluster at district", res_hc1, paper_b)

    # 2. Cluster at region (coarser)
    res_reg = feols_hdfe(d, "ln_turnout", ["ln_fine"], ["ubigeo", "dpc_id"],
                         cluster="region", weight="electores_01")
    rep("2. Cluster at region (25 groups)", res_reg, paper_b)

    # 3. Unweighted
    res_uw = feols_hdfe(d, "ln_turnout", ["ln_fine"], ["ubigeo", "dpc_id"],
                        cluster="province_id", weight=None)
    rep("3. Unweighted OLS", res_uw, paper_b)

    # 4. Drop Lima/Callao (largest provinces / capital)
    d4 = df[df["province_id"] != df.groupby("province_id").size().idxmax()].copy()
    res_lima = refit(d4)
    rep("4. Drop largest province", res_lima, paper_b)

    # 5. Drop 2016 (most-lagged reform effect; test long-run robustness)
    d5 = df[df["year"] != 2016].copy()
    res_no16 = refit(d5)
    rep("5. Drop 2016 elections", res_no16, paper_b)

    # 6. Drop 2001 (pre-reform baseline elections)
    d6 = df[df["year"] != 2001].copy()
    res_no01 = refit(d6)
    rep("6. Drop 2001 elections", res_no01, paper_b)

    # 7. General elections only
    d7 = df[df["runoff"] == 0].copy()
    res_gen = refit(d7)
    rep("7. General elections only", res_gen, paper_b)

    # 8. Runoff elections only
    d8 = df[df["runoff"] == 1].copy()
    res_run = refit(d8)
    rep("8. Runoff elections only", res_run, paper_b)

    # 9. Winsorize turnout at 1st/99th pct
    d9 = df.copy()
    lo, hi = d9["turnout"].quantile([0.01, 0.99])
    d9["turnout"] = d9["turnout"].clip(lo, hi)
    d9["ln_turnout"] = np.log(d9["turnout"])
    res_win = refit(d9)
    rep("9. Winsorize turnout at 1/99 pct", res_win, paper_b)

    # 10. Leave-one-region-out (report range)
    outs = []
    for r in sorted(df["region"].unique()):
        if r == -1 or pd.isna(r):
            continue
        sub = df[df["region"] != r]
        try:
            o = refit(sub)
            outs.append((r, o["beta"][0], o["se"][0]))
        except Exception:
            pass
    if outs:
        coefs = [o[1] for o in outs]
        print(f"  10. Leave-one-region-out (n={len(outs)})         "
              f"β range: [{min(coefs):+.4f}, {max(coefs):+.4f}], "
              f"mean={np.mean(coefs):+.4f}")

    # 11. Permutation placebo: shuffle fine within date groups
    rng = np.random.default_rng(42)
    null = []
    d11 = df.copy()
    fine_orig = d11["ln_fine"].to_numpy()
    for _ in range(200):
        # Shuffle fine_a within each election date (preserves date margin)
        shuf = d11.groupby("date_id")["ln_fine"].transform(
            lambda s: s.sample(frac=1, random_state=rng.integers(0, 1_000_000)).to_numpy()
        )
        d11["ln_fine"] = shuf
        res_p = feols_hdfe(d11, "ln_turnout", ["ln_fine"], ["ubigeo", "dpc_id"],
                           "province_id", "electores_01")
        null.append(res_p["beta"][0])
    d11["ln_fine"] = fine_orig
    null = np.array(null)
    p_perm = np.mean(np.abs(null) >= abs(paper_b))
    print(f"  11. Permutation p-value (shuffle fine w/in date, 200 reps)   "
          f"null mean={null.mean():+.4f}, sd={null.std():.4f}, p={p_perm:.3f}")

    # 12. Placebo outcome: vote share of top-2 candidates (should not respond)
    res_pl = refit(df.dropna(subset=["vs_2candidates"]).assign(
        ln_turnout=lambda d: d["vs_2candidates"]))
    rep("12. Placebo outcome: vs_2candidates", res_pl, paper_b)


if __name__ == "__main__":
    main()
