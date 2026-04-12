"""Robustness checks for the reduced-form Experiment 1 results.

Each check revisits a headline finding (piece-rate sensitivity, consequential
work, or null gift effects) and asks whether a reasonable perturbation moves
the point estimate or its significance."""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from utils import clean_long

long, _ = clean_long()

long["output58mean"] = long.groupby("id").apply(
    lambda g: g.loc[(g["time"] > 4) & (g["time"] <= 8), "output"].mean()
).reindex(long["id"]).values


def reg(df, lhs_log, controls):
    d = df.copy()
    d["const"] = 1.0
    y = np.log(d["output"].values) if lhs_log else d["output"].values
    rhs = ["const", "d_pos_gift", "d_neg_gift", "d_knd_gift"] + list(controls)
    X = d[rhs].values
    return sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": d["group"].values})


def row(label, res):
    pos_b, pos_se = res.params[1], res.bse[1]
    neg_b, neg_se = res.params[2], res.bse[2]
    knd_b, knd_se = res.params[3], res.bse[3]
    return (f"{label:46s}  pos={pos_b:+6.3f}({pos_se:.3f})  "
            f"neg={neg_b:+6.3f}({neg_se:.3f})  inkind={knd_b:+6.3f}({knd_se:.3f})  "
            f"N={int(res.nobs)}")


print("=" * 110)
print("ROBUSTNESS CHECKS — Gift effects (rounds 9-10) unless noted")
print("=" * 110)
print("Benchmark (col 3 of Table 4, levels, with output58mean control)\n")

rd910 = long[(long["time"] == 9) | (long["time"] == 10)]

# 1. Benchmark
res = reg(rd910, False, ["output58mean"])
print(row("1. Benchmark (levels, control)", res))

# 2. Log output
res = reg(rd910, True, ["output58mean"])
print(row("2. Log output", res))

# 3. HC1 (heteroskedastic-robust) SE instead of cluster
d = rd910.copy(); d["const"] = 1.0
rhs = ["const", "d_pos_gift", "d_neg_gift", "d_knd_gift", "output58mean"]
res_hc1 = sm.OLS(d["output"].values, d[rhs].values).fit(cov_type="HC1")
print(row("3. HC1 robust SE", res_hc1))

# 4. Cluster by individual instead of session
d = rd910.copy(); d["const"] = 1.0
res_id = sm.OLS(d["output"].values, d[rhs].values).fit(
    cov_type="cluster", cov_kwds={"groups": d["id"].values}
)
print(row("4. Cluster by worker id", res_id))

# 5. Drop in-kind (added late, smaller sample)
res = reg(rd910[rd910["d_knd_gift"] == 0], False, ["output58mean"])
print(row("5. Drop in-kind treatment", res))

# 6. Winsorize output at 1/99 per round
w = long.copy()
w["output_w"] = w.groupby("time")["output"].transform(
    lambda s: s.clip(lower=s.quantile(0.01), upper=s.quantile(0.99))
)
w_rd = w[(w["time"] == 9) | (w["time"] == 10)]
w_rd = w_rd.assign(output=w_rd["output_w"])
res = reg(w_rd, False, ["output58mean"])
print(row("6. Winsorize 1/99 by round", res))

# 7. Drop Phase 1 (pre-pre-registration, N=127)
res = reg(rd910[rd910["phase"] == 2], False, ["output58mean"])
print(row("7. Phase 2 only (post-pre-reg)", res))

# 8. Control with individual FEs (no output58mean) using demeaning
d = rd910.copy()
d["y_dm"] = d["output"] - d.groupby("id")["output"].transform("mean")
# Within-id the gift dummies are constant (gift is assigned at worker level),
# so FEs swallow them — use output58mean as the absorbing control and
# add order FE instead:
d["const"] = 1.0
d["order2"] = (d["order"] == 2).astype(int)
rhs2 = ["const", "d_pos_gift", "d_neg_gift", "d_knd_gift", "output58mean", "order2"]
res = sm.OLS(d["output"].values, d[rhs2].values).fit(
    cov_type="cluster", cov_kwds={"groups": d["group"].values}
)
print(row("8. + order FE", res))

# 9. Placebo — run the gift regression on rounds 5-6 (before gifts are announced)
placebo = long[(long["time"] == 5) | (long["time"] == 6)]
res_p = reg(placebo, False, [])  # no output58mean (it would be mechanically zero on same rounds)
print(row("9. Placebo: rounds 5-6 pre-gift", res_p))

# 10. Leave-one-out by charity (charity_order 1..3)
for co in [1, 2, 3]:
    res = reg(rd910[rd910["charity_order"] != co], False, ["output58mean"])
    print(row(f"10. Drop charity_order = {co}", res))

# 11. Randomization inference for positive gift effect (permute gift assignment)
rng = np.random.default_rng(0)
obs_t = reg(rd910, False, ["output58mean"]).tvalues[1]
perm_tstats = []
for _ in range(999):
    perm = rd910.copy()
    # Permute gift type at worker level (sessions remain balanced because
    # we shuffle across all 446 workers and then expand to long).
    worker_gift = perm.groupby("id")["gift_type"].first().reset_index()
    worker_gift["gift_type_perm"] = rng.permutation(worker_gift["gift_type"].values)
    perm = perm.merge(worker_gift[["id", "gift_type_perm"]], on="id")
    perm["d_pos_gift"] = (perm["gift_type_perm"] == "pos").astype(int)
    perm["d_neg_gift"] = (perm["gift_type_perm"] == "neg").astype(int)
    perm["d_knd_gift"] = (perm["gift_type_perm"] == "knd").astype(int)
    r = reg(perm, False, ["output58mean"])
    perm_tstats.append(r.tvalues[1])
perm_tstats = np.array(perm_tstats)
p_rand = (np.abs(perm_tstats) >= np.abs(obs_t)).mean()
print(f"\n11. Permutation p-value for positive-gift (999 draws): t_obs={obs_t:.2f}  p_perm={p_rand:.3f}")

# 12. Piece-rate effect robustness  (check the headline +12% finding)
print("\n" + "=" * 110)
print("ROBUSTNESS — Piece-rate effect (type 1=0c, 3=10c, 5=20c in Figure 2a)")
print("=" * 110)
d = long.copy()
d["type"] = np.nan
d.loc[((d["time"] == 7) & (d["order"] == 1)) | ((d["time"] == 3) & (d["order"] == 2)), "type"] = 1
d.loc[((d["time"] == 3) & (d["order"] == 1)) | ((d["time"] == 7) & (d["order"] == 2)), "type"] = 3
d.loc[((d["time"] == 6) & (d["order"] == 1)) | ((d["time"] == 4) & (d["order"] == 2)), "type"] = 5
sub = d.dropna(subset=["type"]).copy()
sub["t3"] = (sub["type"] == 3).astype(int)
sub["t5"] = (sub["type"] == 5).astype(int)
sub["const"] = 1.0

for label, filt in [
    ("Full sample", sub),
    ("Phase 2 only", sub[sub["phase"] == 2]),
    ("Employed only", sub[sub["dempl"] == 1]),
    ("Female only", sub[sub["female"] == 1]),
    ("Male only", sub[sub["female"] == 0]),
]:
    X = filt[["const", "t3", "t5"]].values
    y = filt["output"].values
    r = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": filt["group"].values})
    b0, b10, b20 = r.params
    se10, se20 = r.bse[1], r.bse[2]
    print(f"  {label:18s} 0c={b0:5.2f}  10c-0c=+{b10:4.2f}({se10:.2f})  "
          f"20c-0c=+{b20:4.2f}({se20:.2f})  N={int(r.nobs)}")

print("\nRobustness complete.")
