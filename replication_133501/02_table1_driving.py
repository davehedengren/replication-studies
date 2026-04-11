"""Table 1 Panel A — driving outcomes from Add Health."""
import pandas as pd

from utils import OUT, load_addhealth, mean_before, rd_mse_opt

PUBLISHED = {
    ("DriverLicense", "none"): (0.186, (0.124, 0.231)),
    ("VehicleMiles_150", "none"): (375, (159, 530)),
    ("VehicleMiles_265", "none"): (575, (231, 856)),
    ("DriverLicense", "Male"): (0.193, (0.139, 0.231)),
    ("VehicleMiles_150", "Male"): (486, (195, 734)),
    ("VehicleMiles_265", "Male"): (753, (328, 1194)),
    ("DriverLicense", "Female"): (0.179, (0.103, 0.232)),
    ("VehicleMiles_150", "Female"): (234, (-105, 479)),
    ("VehicleMiles_265", "Female"): (327, (-144, 676)),
}

rows = []
for scenario in ["none", "Male", "Female"]:
    df = load_addhealth(scenario)
    for y in ["DriverLicense", "VehicleMiles_150", "VehicleMiles_265"]:
        est = rd_mse_opt(df, y, covs=True)
        rows.append({
            "scenario": scenario,
            "outcome": y,
            "mean_y": mean_before(df, y),
            "conv": est["conv"],
            "robust_ci_lo": est["robust_ci"][0],
            "robust_ci_hi": est["robust_ci"][1],
            "h": est["h"],
        })

out = pd.DataFrame(rows)
out.to_csv(OUT / "table1_panelA_driving.csv", index=False)

print("=" * 90)
print("Table 1 Panel A — driving outcomes (Add Health)")
print("=" * 90)
print(f"{'Outcome':<20} {'Scen':<7} {'Mean':>10} {'RD':>10} "
      f"{'Lo':>10} {'Hi':>10}  {'Pub RD':>10}  Match?")
for _, r in out.iterrows():
    key = (r.outcome, r.scenario)
    pub_rd, pub_ci = PUBLISHED[key]
    match = "ok" if abs(pub_rd - r.conv) < max(0.005, 0.02 * abs(pub_rd)) else "DIFF"
    print(f"{r.outcome:<20} {r.scenario:<7} {r.mean_y:>10.3f} {r.conv:>10.4f} "
          f"{r.robust_ci_lo:>10.3f} {r.robust_ci_hi:>10.3f}  "
          f"{pub_rd:>10.3f}  {match}")
