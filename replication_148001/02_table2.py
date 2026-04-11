"""Table 2: impact of selection reform on State Wins.

Target (from paper):
  col (1): Retirements_2010 x Post2010 = -0.237 [0.0342], no case/bench controls
  col (2): Retirements_2010 x Post2010 = -0.210 [0.0383], all controls  <-- baseline
  col (3): Retirements_2009 x Post2010 =  0.0677 [0.0589], placebo
  col (4): Retirements_2008 x Post2010 =  0.143 [0.112], placebo
  col (5): Retirements_2007 x Post2010 =  0.115 [0.0968], placebo
N = 8,446 for all columns, mean dep var = 0.482.
"""

import numpy as np
import pandas as pd
import pyhdfe
import statsmodels.api as sm

from utils import (
    load_team1, main_sample_mask,
    CONTROLS_CASE, CONTROLS_JUDGE, CONTROLS_DISTRICT, CONTROLS_BENCH,
)


def run(df, treat, controls):
    """OLS with district-by-year FE absorbed, cluster at district-bench."""
    y = df["StateWins"].to_numpy(dtype=float)
    X = df[[treat] + controls].to_numpy(dtype=float)
    fe = df["yeardecisionXdistrict_encode"].to_numpy().reshape(-1, 1)
    cluster = df["districtXbench_enco"].to_numpy()

    algo = pyhdfe.create(fe, drop_singletons=False)
    yr = algo.residualize(y.reshape(-1, 1))[:, 0]
    Xr = algo.residualize(X)

    res = sm.OLS(yr, Xr).fit(
        cov_type="cluster", cov_kwds={"groups": cluster}
    )

    coef = res.params[0]
    se = res.bse[0]
    n = len(y)
    r2 = 1 - (res.resid ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return coef, se, n, r2


def main():
    df = load_team1()
    mask = main_sample_mask(df)
    sub = df.loc[mask].copy()
    assert len(sub) == 8446, f"Sample size {len(sub)} != 8446"

    # Per Tables.do, the published Table 2 reports the second regression,
    # which uses bench + case controls on the e(sample)==1 subset from the
    # first regression (district + case + judge + case-type controls).
    reported_controls = CONTROLS_BENCH + CONTROLS_CASE

    rows = []
    # (1) FE only on the e(sample) defined by the first regression
    c, se, n, r2 = run(sub, "Retirements_2010_Post2010", [])
    rows.append(("(1) Retirements_2010_Post2010 (FE only)", c, se, n, r2,
                 -0.237, 0.0342))
    # (2) reported baseline: bench + case controls
    c, se, n, r2 = run(sub, "Retirements_2010_Post2010", reported_controls)
    rows.append(("(2) Retirements_2010_Post2010 (bench+case)", c, se, n, r2,
                 -0.210, 0.0383))
    # (3)-(5) placebo pre-reform mandatory retirements, with controls
    for yr_, pub_c, pub_se in [
        (2009, 0.0677, 0.0589),
        (2008, 0.143, 0.112),
        (2007, 0.115, 0.0968),
    ]:
        var = f"Retirements_{yr_}_Post2010"
        c, se, n, r2 = run(sub, var, reported_controls)
        rows.append((f"({3 + (2009 - yr_)}) {var}", c, se, n, r2, pub_c, pub_se))

    print(f"\n{'Spec':<45} {'coef':>9} {'se':>9} {'pub coef':>10} {'pub se':>10}  N")
    for label, c, se, n, r2, pc, ps in rows:
        print(f"{label:<45} {c:>9.4f} [{se:>7.4f}] {pc:>10.4f} {ps:>10.4f}  {n}")

    out = pd.DataFrame(
        rows,
        columns=["spec", "coef", "se", "n", "r2", "pub_coef", "pub_se"],
    )
    out.to_csv("replication_148001/output/table2.csv", index=False)
    print("\nMean dep var:", sub["StateWins"].mean())


if __name__ == "__main__":
    main()
