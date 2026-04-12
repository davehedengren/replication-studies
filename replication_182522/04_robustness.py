"""Phase 4 robustness checks for the R&D efficiency calibration (equation 46).

The paper's headline empirical input into the structural calibration is the vector
b_s of country R&D efficiencies. This script perturbs the recipe used in
02_rd_efficiency.do and reports how much the 25-country ordering and magnitudes
change.

Checks
------
 1. Baseline (median, DEU reference, 2010-14, >=14 industries) — paper spec
 2. Mean instead of median within country
 3. Reference country = USA instead of DEU (should be a level shift only)
 4. Alternative year window 2005-09
 5. Extended year window 2005-14
 6. Tighter industry threshold (>=18 of 20 industries)
 7. Looser industry threshold (>=10 of 20 industries)
 8. Drop the petroleum industry (D19) which has extreme outliers
 9. Trimmed mean (drop top/bottom 10% of rel_lrdint within cou)
10. Include services-adjacent construction (but we use same 20 — skipped)
11. Use Belgium as reference country
12. Leave-one-industry-out: max spread across 20 holdouts

For each check, we report Spearman rank correlation with the baseline and max
absolute deviation. A robust headline quantity means these should be close to 1.0
and small respectively.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr

from utils import BASELINE_COUS, compute_rd_intensity


def rd_eff(
    m: pd.DataFrame,
    reference: str = "DEU",
    years: tuple[int, int] = (2010, 2014),
    min_inds: int = 14,
    agg: str = "median",
    normalise_to: str = "USA",
    exclude_inds: tuple[str, ...] = (),
    trim: float = 0.0,
) -> pd.Series:
    s = m.copy()
    if exclude_inds:
        s = s[~s["ind"].isin(exclude_inds)]
    s["cou_obs_man"] = s.groupby(["cou", "year"])["cou"].transform("count")
    s = s[(s["cou_obs_man"] >= min_inds) & s["year"].between(years[0], years[1])].copy()
    ref = (
        s[s["cou"] == reference][["ind", "year", "lrd_intensity"]]
        .rename(columns={"lrd_intensity": "lref"})
    )
    s = s.merge(ref, on=["ind", "year"], how="left")
    s["rel"] = s["lrd_intensity"] - s["lref"]
    s = s[s["rel"].notna()]

    def agg_fn(x: pd.Series) -> float:
        if trim > 0:
            lo, hi = x.quantile(trim), x.quantile(1 - trim)
            x = x[(x >= lo) & (x <= hi)]
        if agg == "median":
            return x.median()
        return x.mean()

    g = s.groupby("cou")["rel"].apply(agg_fn)
    g = g[g.index.isin(BASELINE_COUS)]
    if normalise_to in g.index:
        g = g - g.loc[normalise_to]
    return g.sort_index()


def compare(label: str, baseline: pd.Series, alt: pd.Series, rows: list):
    common = baseline.index.intersection(alt.index)
    if len(common) < len(baseline):
        missing = sorted(set(baseline.index) - set(alt.index))
        note = f" (missing: {','.join(missing)})"
    else:
        note = ""
    rho_s = spearmanr(baseline.loc[common], alt.loc[common]).statistic
    rho_p = pearsonr(baseline.loc[common], alt.loc[common]).statistic
    max_dev = float((alt.loc[common] - baseline.loc[common]).abs().max())
    rows.append({
        "check": label + note,
        "n": len(common),
        "spearman_vs_baseline": round(rho_s, 4),
        "pearson_vs_baseline": round(rho_p, 4),
        "max_abs_dev": round(max_dev, 4),
    })


def main():
    m = compute_rd_intensity()
    base = rd_eff(m)
    print(f"Baseline: {len(base)} countries, USA={base.loc['USA']:.4f}, "
          f"min={base.min():.3f} ({base.idxmin()}), "
          f"max={base.max():.3f} ({base.idxmax()})")

    rows = []

    # 1. (baseline self-check)
    compare("1. baseline (median, DEU ref, 2010-14, >=14)", base, base, rows)

    # 2. mean within cou
    compare("2. mean (instead of median)", base, rd_eff(m, agg="mean"), rows)

    # 3. USA reference country
    compare("3. reference country = USA", base, rd_eff(m, reference="USA"), rows)

    # 4. earlier window 2005-09
    compare("4. years = 2005-2009", base, rd_eff(m, years=(2005, 2009)), rows)

    # 5. extended window
    compare("5. years = 2005-2014", base, rd_eff(m, years=(2005, 2014)), rows)

    # 6. tighter industry threshold
    compare("6. min_inds = 18 of 20", base, rd_eff(m, min_inds=18), rows)

    # 7. looser industry threshold
    compare("7. min_inds = 10 of 20", base, rd_eff(m, min_inds=10), rows)

    # 8. drop petroleum (D19 has outliers)
    compare("8. drop D19 petroleum", base, rd_eff(m, exclude_inds=("D19",)), rows)

    # 9. trimmed median (10% each tail)
    compare("9. 10% trim on rel_lrdint", base, rd_eff(m, trim=0.10), rows)

    # 10. reference country = Belgium
    compare("10. reference country = BEL", base, rd_eff(m, reference="BEL"), rows)

    # 11. reference country = Japan
    compare("11. reference country = JPN", base, rd_eff(m, reference="JPN"), rows)

    # 12. leave-one-industry-out: worst case
    all_inds = sorted(m["ind"].unique())
    worst_dev = 0.0
    worst_ind = None
    worst_spearman = 1.0
    for ind in all_inds:
        alt = rd_eff(m, exclude_inds=(ind,))
        common = base.index.intersection(alt.index)
        if len(common) < 25:
            continue
        dev = float((alt.loc[common] - base.loc[common]).abs().max())
        rho = spearmanr(base.loc[common], alt.loc[common]).statistic
        if dev > worst_dev:
            worst_dev = dev
            worst_ind = ind
            worst_spearman = rho
    rows.append({
        "check": f"12. leave-one-industry-out WORST ({worst_ind})",
        "n": 25,
        "spearman_vs_baseline": round(float(worst_spearman), 4),
        "pearson_vs_baseline": None,
        "max_abs_dev": round(worst_dev, 4),
    })

    df = pd.DataFrame(rows)
    print("\n" + df.to_string(index=False))

    from utils import OUT
    df.to_csv(OUT / "robustness_checks.csv", index=False)
    print(f"\nSaved {OUT / 'robustness_checks.csv'}")


if __name__ == "__main__":
    main()
