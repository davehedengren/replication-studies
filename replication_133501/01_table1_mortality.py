"""Replicate Table 1 Panel B (mortality) from Huh & Reif (2021).

For each outcome and scenario we compute the rdrobust MSE-optimal
conventional point estimate and the robust bias-corrected 95% CI, mirroring
the Stata code in 4_analysis.do (rd_mortality block)."""
import pandas as pd

from utils import MORTALITY_OUTCOMES, OUT, load_mortality, mean_before, rd_mse_opt

PUBLISHED = {
    ("cod_any", "none"): (5.84, (1.99, 9.36)),
    ("cod_internal", "none"): (0.406, (-0.120, 1.17)),
    ("cod_external", "none"): (5.20, (1.42, 8.47)),
    ("cod_MVA", "none"): (4.92, (2.36, 7.07)),
    ("cod_sa", "none"): (0.167, (-0.680, 0.924)),
    ("cod_sa_firearms", "none"): (0.0914, (-0.326, 0.474)),
    ("cod_sa_poisoning", "none"): (0.314, (0.183, 0.522)),
    ("cod_sa_poisoning_subst", "none"): (0.315, (0.233, 0.496)),
    ("cod_sa_poisoning_gas", "none"): (0.103, (-0.0301, 0.215)),
    ("cod_sa_drowning", "none"): (-0.294, (-0.576, -0.0967)),
    ("cod_sa_other", "none"): (0.105, (-0.316, 0.463)),
    ("cod_homicide", "none"): (-0.0423, (-0.623, 0.534)),
    ("cod_extother", "none"): (0.00608, (-0.148, 0.154)),
    ("cod_any", "Male"): (5.72, (-0.809, 11.3)),
    ("cod_MVA", "Male"): (5.67, (2.76, 8.10)),
    ("cod_sa_poisoning", "Male"): (0.133, (-0.218, 0.458)),
    ("cod_any", "Female"): (5.76, (4.35, 7.53)),
    ("cod_MVA", "Female"): (4.46, (2.41, 6.14)),
    ("cod_sa_poisoning", "Female"): (0.747, (0.591, 1.07)),
    ("cod_sa_poisoning_subst", "Female"): (0.646, (0.476, 0.999)),
    ("cod_sa_poisoning_gas", "Female"): (0.127, (0.0333, 0.243)),
}

rows = []
for scenario in ["none", "Male", "Female"]:
    df = load_mortality(scenario)
    for y in MORTALITY_OUTCOMES:
        est = rd_mse_opt(df, y, covs=True)
        mean_y = mean_before(df, y)
        rows.append({
            "scenario": scenario,
            "outcome": y,
            "mean_y_pub": mean_y,
            "conv": est["conv"],
            "robust_ci_lo": est["robust_ci"][0],
            "robust_ci_hi": est["robust_ci"][1],
            "h": est["h"],
        })

out = pd.DataFrame(rows)
out.to_csv(OUT / "table1_panelB_mortality.csv", index=False)

print("=" * 90)
print("Table 1 Panel B — replication of RD mortality estimates (per 100k)")
print("=" * 90)
print(f"{'Outcome':<26} {'Scen':<7} {'Mean':>8} {'RD':>9} "
      f"{'Lo':>9} {'Hi':>9}  {'Pub RD':>8} {'Match?':>7}")
for _, r in out.iterrows():
    key = (r.outcome, r.scenario)
    pub = PUBLISHED.get(key)
    pub_str = f"{pub[0]:>8.3f}" if pub else f"{'':>8}"
    match = "  —  "
    if pub:
        match = "  ok " if abs(pub[0] - r.conv) < 0.01 else " DIFF"
    print(f"{r.outcome:<26} {r.scenario:<7} {r.mean_y_pub:>8.3f} "
          f"{r.conv:>9.4f} {r.robust_ci_lo:>9.3f} {r.robust_ci_hi:>9.3f}  "
          f"{pub_str} {match:>7}")

# Headline
print("\nHeadline matches:")
for key, (pub_rd, pub_ci) in PUBLISHED.items():
    hit = out[(out.outcome == key[0]) & (out.scenario == key[1])].iloc[0]
    lo_diff = hit.robust_ci_lo - pub_ci[0]
    hi_diff = hit.robust_ci_hi - pub_ci[1]
    print(f"  {key[0]} ({key[1]}): RD {hit.conv:.3f} vs {pub_rd:.3f}  "
          f"CI [{hit.robust_ci_lo:.3f},{hit.robust_ci_hi:.3f}] "
          f"(Δlo={lo_diff:+.3f}, Δhi={hi_diff:+.3f})")
