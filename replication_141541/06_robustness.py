"""Phase 4 robustness checks tailored to a simulation-only matching paper.

The paper's three headline claims are:
  (i)  Int-DA Pareto-dominates Tr-DA on matching outcomes (higher top-1 / top-3
       share and lower unmatched share) across the common-weight grid.
  (ii) Int-DA is essentially indistinguishable from full-DA in stability
       (same-partner under proposer change) and is "close" to DA in welfare.
  (iii) The qualitative conclusion survives when rescaled to NRMP size via a
        log-linear logit extrapolation.

We run a battery of 12 checks that stress every link in this chain without
rerunning the expensive raw simulations.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from utils import (K_VEC, LAMBDA_FULL, MARKET_SIZES, MARKET_SUM, N_VEC,
                   VAR_FNS, load_balanced, load_k, load_rank_diffs, load_sigs,
                   load_unbalanced)


def nrmp_rescale(bal, stem, lamD, lamH, model="logit", leave_out=None):
    xs, ys, raw = [], [], []
    for N in N_VEC:
        row = bal[(bal["N"] == N) & np.isclose(bal["lambdaD"], lamD) & np.isclose(bal["lambdaH"], lamH)].iloc[0]
        v = float(VAR_FNS[stem](row, N))
        raw.append(v)
        xs.append(np.log(N))
        if model == "logit":
            if v >= 100:
                ys.append(np.log(20000))
            else:
                ys.append(np.log(v / (100 - v)))
        else:  # ols in percent
            ys.append(v)
    b, a = np.polyfit(np.asarray(xs), np.asarray(ys), 1)
    sizes = MARKET_SIZES if leave_out is None else [n for n in MARKET_SIZES if n != leave_out]
    total = sum(sizes)
    weighted = 0.0
    for n in sizes:
        pred = a + b * np.log(n)
        if model == "logit":
            e = np.exp(pred)
            f = 100 * e / (1 + e)
        else:
            f = pred
        weighted += (n / total) * f
    return weighted, raw


def check_1_pareto_matching(bal):
    """Int-DA ≥ Tr-DA on {top, top3} and ≤ Tr-DA on unmatched at every grid cell."""
    viol = 0
    for _, r in bal.iterrows():
        if r["top.tags"] < r["top.trgs"] - 1e-9: viol += 1
        if r["top3.tags"] < r["top3.trgs"] - 1e-9: viol += 1
        if r["unmatched.tags"] > r["unmatched.trgs"] + 1e-9: viol += 1
    return f"violating cells (of {3*len(bal)}): {viol}"


def check_2_stability(bal):
    """Same-partner under proposer change in Int-DA should be ≥ 99% conditional on matched for all cells."""
    min_ = 100.0
    arg = None
    for _, r in bal.iterrows():
        v = 100 * r["samepartner.tags"] / (1 - r["unmatched.tags"])
        if v < min_:
            min_ = v
            arg = (int(r["N"]), r["lambdaD"], r["lambdaH"])
    return f"min {min_:.2f}% at (N,λD,λH)={arg}"


def check_3_blocking_matched(bal):
    """Blocking-pair share for matched doctors in Int-DA should stay below 5%."""
    worst = 0.0
    arg = None
    for _, r in bal.iterrows():
        v = 100 * (r["bp.match.tags"] / r["N"]) / (1 - r["unmatched.tags"])
        if v > worst:
            worst = v
            arg = (int(r["N"]), r["lambdaD"], r["lambdaH"])
    return f"max {worst:.2f}% at (N,λD,λH)={arg}"


def check_4_loo_nrmp(bal):
    """Leave-one-out the largest submarket (n=9127) and recompute Table 1 cell."""
    base, _ = nrmp_rescale(bal, "UnmatchedIntDA", 0.5, 0.5)
    lo, _ = nrmp_rescale(bal, "UnmatchedIntDA", 0.5, 0.5, leave_out=9127)
    return f"Int-DA unmatched at (.5,.5): full {base:.3f}% vs drop-9127 {lo:.3f}% (Δ={lo-base:+.3f})"


def check_5_ols_vs_logit(bal):
    """Compare logit rescaling to a naive OLS-in-percent rescaling."""
    rows = []
    for stem in ["TopIntDA", "UnmatchedIntDA", "Top3IntDA"]:
        lg, _ = nrmp_rescale(bal, stem, 0.5, 0.5, model="logit")
        ol, _ = nrmp_rescale(bal, stem, 0.5, 0.5, model="ols")
        rows.append(f"{stem}: logit {lg:.2f}% vs OLS {ol:.2f}%")
    return " | ".join(rows)


def check_6_large_N_extrapolation(bal):
    """At (λD,λH)=(0.5,0.5) the logit fit predicts virtually constant values
    across N — if Int-DA is really an asymptotic Pareto improvement, the
    logit should show a slope close to zero in magnitude."""
    slopes = []
    for stem in ["TopIntDA", "UnmatchedIntDA", "SamePartnerIntDA"]:
        xs, ys = [], []
        for N in N_VEC:
            row = bal[(bal["N"] == N) & np.isclose(bal["lambdaD"], 0.5) & np.isclose(bal["lambdaH"], 0.5)].iloc[0]
            v = float(VAR_FNS[stem](row, N))
            xs.append(np.log(N))
            ys.append(np.log(max(min(v, 99.9), 0.1) / (100 - max(min(v, 99.9), 0.1))))
        b, _ = np.polyfit(xs, ys, 1)
        slopes.append(f"{stem} slope={b:+.3f}")
    return " | ".join(slopes)


def check_7_k_sensitivity(ks):
    """Does Int-DA still dominate Tr-DA at k=2 (tight interview cap)?"""
    viol = 0
    for _, r in ks[ks["k"] == 2].iterrows():
        if r["top.tags"] < r["top.trgs"]: viol += 1
    return f"k=2 cells with Int-DA < Tr-DA on top match: {viol}/{int((ks['k']==2).sum())}"


def check_8_k_monotone(ks):
    """Unmatched share under Int-DA should weakly decrease in k."""
    viol = 0
    for lamD in LAMBDA_FULL:
        for lamH in LAMBDA_FULL:
            sub = ks[(np.isclose(ks["lambdaD"], lamD)) & (np.isclose(ks["lambdaH"], lamH))].sort_values("k")
            vals = sub["unmatched.tags"].tolist()
            for i in range(len(vals) - 1):
                if vals[i + 1] > vals[i] + 1e-3:
                    viol += 1
    return f"pairs violating k-monotone unmatched: {viol}"


def check_9_unbalanced(ub):
    """At nD=600, nH=500 Int-DA still should beat Tr-DA on top-match."""
    viol = 0
    for _, r in ub[ub["nD"] == 600].iterrows():
        if r["top.tags"] < r["top.trgs"]: viol += 1
    return f"nD=600 cells Int-DA<Tr-DA top match: {viol}"


def check_10_sigs(sigs, bal):
    """SIGS (doctor-common interviews only) should be weakly worse than full Int-DA."""
    worse = 0
    tot = 0
    for _, r in sigs.iterrows():
        ref = bal[(bal["N"] == 500) & np.isclose(bal["lambdaD"], r["lambdaD"]) & np.isclose(bal["lambdaH"], r["lambdaH"])].iloc[0]
        if r["top.sigs"] < ref["top.tags"] - 1e-4:
            worse += 1
        tot += 1
    return f"SIGS worse than full Int-DA on top match: {worse}/{tot}"


def check_11_rank_diff(rank):
    """Aggregate Int-DA→DA rank-gap statistic should shrink with market size.

    In ``rankDiffsx.csv`` each row encodes a single (Int-DA rank, DA rank)
    observation with a weight of 1/N, so ``sum(diff) == Σ(tags_rank - gs_rank)
    / N`` within a cell. Positive ⇒ Int-DA gives worse true-rank on average
    at (λD,λH)=(.5,.5). The paper's welfare-gap claim predicts this summary
    should decrease monotonically in N.
    """
    vals = []
    for n in sorted(rank["Nd"].unique()):
        sub = rank[(rank["Nd"] == n) & np.isclose(rank["cud"], 0.5) & np.isclose(rank["cuh"], 0.5)]
        vals.append((int(n), float(sub["diff"].sum())))
    pretty = ", ".join(f"N={n}: {v:+.1f}" for n, v in vals)
    monotone = all(vals[i + 1][1] <= vals[i][1] + 1e-6 for i in range(len(vals) - 1))
    return f"{pretty} (monotone shrinking: {monotone})"


def check_12_identical_bound(bal):
    """Identical-partner rate (Int-DA match == DA match | matched) should lie
    in [50, 90]% across the balanced grid — a rough sanity bound."""
    vals = []
    for _, r in bal.iterrows():
        v = 100 * r["gs.tags.identical"] / (1 - r["unmatched.tags"])
        vals.append(v)
    return f"identical-partner range: [{min(vals):.1f}%, {max(vals):.1f}%]"


def main() -> None:
    bal = load_balanced()
    ks = load_k()
    ub = load_unbalanced()
    sigs = load_sigs()
    rank = load_rank_diffs()
    checks = [
        ("1.  Int-DA Pareto-dominates Tr-DA (balanced grid)", check_1_pareto_matching(bal)),
        ("2.  Int-DA stability ≥99% (min same-partner | matched)", check_2_stability(bal)),
        ("3.  Matched blocking pairs < 5%", check_3_blocking_matched(bal)),
        ("4.  NRMP rescale leave-one-out (drop largest submarket)", check_4_loo_nrmp(bal)),
        ("5.  Logit vs OLS rescaling at (λD,λH)=(.5,.5)", check_5_ols_vs_logit(bal)),
        ("6.  Log-N slope diagnostic at (.5,.5)", check_6_large_N_extrapolation(bal)),
        ("7.  k=2 sensitivity (top match)", check_7_k_sensitivity(ks)),
        ("8.  k-monotone unmatched in Int-DA", check_8_k_monotone(ks)),
        ("9.  Unbalanced nD=600 sensitivity", check_9_unbalanced(ub)),
        ("10. SIGS ≤ Int-DA on top match", check_10_sigs(sigs, bal)),
        ("11. Rank improvement under Int-DA (Fig B8 cell)", check_11_rank_diff(rank)),
        ("12. Identical-partner rate bound", check_12_identical_bound(bal)),
    ]
    print("== Robustness checks ==")
    for name, result in checks:
        print(f"{name}: {result}")


if __name__ == "__main__":
    main()
