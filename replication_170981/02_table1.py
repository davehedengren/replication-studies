"""Reproduce Table 1 (columns 1-5) from Leonardi & Moretti (2022)."""
import pandas as pd
from utils import build_analysis_frame, ols

df = build_analysis_frame()
y = "d1204_r"

base = [
    "prezzo2004Abitaz",
    "prezzo2004Negozi",
    "metro_dum",
    "univ",
    "attraction_dum",
    "day_pop",
    "lat2004",
    "prezzo2004",
    "cucina2004",
    "mmichelin2004",
]

specs = {
    "c1": base,
    "c2": base + ["lat2004_d", "lat2004_a"],
    "c3": base + ["d0004_d", "d0004_a"],
    "c4": base + ["lat2004_d", "lat2004_a", "d0004_d", "d0004_a"],
}

print("=" * 90)
print("TABLE 1, columns 1-4 (multivariate)")
print("=" * 90)

published = {
    "prezzo2004Abitaz": [(0.053, 0.110), (0.065, 0.111), (0.068, 0.111), (0.077, 0.113)],
    "prezzo2004Negozi": [(-0.010, 0.095), (-0.010, 0.096), (-0.023, 0.096), (-0.025, 0.098)],
    "metro_dum": [(-0.004, 0.026), (-0.001, 0.026), (-0.002, 0.027), (0.002, 0.027)],
    "univ": [(0.063, 0.072), (0.060, 0.072), (0.058, 0.072), (0.057, 0.073)],
    "attraction_dum": [(-0.134, 0.140), (-0.168, 0.156), (-0.121, 0.142), (-0.146, 0.158)],
    "day_pop": [(-0.000, 0.000), (-0.000, 0.000), (-0.000, 0.000), (-0.000, 0.000)],
    "lat2004": [(0.005, 0.006), (0.002, 0.010), (0.003, 0.006), (0.001, 0.010)],
    "prezzo2004": [(-0.003, 0.002), (-0.003, 0.002), (-0.003, 0.002), (-0.003, 0.002)],
    "cucina2004": [(0.012, 0.028), (0.010, 0.028), (0.011, 0.028), (0.009, 0.028)],
    "mmichelin2004": [(0.019, 0.028), (0.024, 0.029), (0.019, 0.029), (0.022, 0.029)],
    "lat2004_d": [None, (-0.002, 0.003), None, (-0.001, 0.003)],
    "lat2004_a": [None, (0.014, 0.012), None, (0.010, 0.015)],
    "d0004_d": [None, None, (-0.135, 0.102), (-0.109, 0.110)],
    "d0004_a": [None, None, (0.079, 0.070), (0.044, 0.087)],
}

results = {}
for name, xs in specs.items():
    m = ols(df, y, xs)
    results[name] = m
    print(f"\n--- {name} ---  N={int(m.nobs)}  R²={m.rsquared:.3f}")
    for v in xs:
        pub = published.get(v)
        pub_str = ""
        if pub:
            col_idx = int(name[1]) - 1
            if pub[col_idx] is not None:
                pub_str = f"  [pub: {pub[col_idx][0]:+.3f} ({pub[col_idx][1]:.3f})]"
        print(f"  {v:<20} {m.params[v]:+.3f} ({m.bse[v]:.3f}){pub_str}")
    print(f"  const                {m.params['const']:+.3f} ({m.bse['const']:.3f})")

# Column 5 — bivariate regressions
print("\n" + "=" * 90)
print("TABLE 1, column 5 (bivariate regressions)")
print("=" * 90)

biv_vars = [
    ("prezzo2004Abitaz", 0.143, 0.056, 180),
    ("prezzo2004Negozi", 0.137, 0.045, 180),
    ("metro_dum", -0.043, 0.029, 180),
    ("univ", 0.068, 0.085, 180),
    ("attraction_dum", -0.082, 0.109, 180),
    ("day_pop", 0.000, 0.000, 180),
    ("lat2004", 0.021, 0.006, 180),
    ("prezzo2004", -0.003, 0.001, 140),
    ("cucina2004", -0.021, 0.020, 140),
    ("mmichelin2004", 0.052, 0.032, 180),
    ("lat2004_d", 0.004, 0.001, 180),
    ("lat2004_a", 0.022, 0.008, 180),
    ("d0004_d", 0.073, 0.093, 180),
    ("d0004_a", 0.049, 0.064, 180),
]

for v, pub_b, pub_se, pub_n in biv_vars:
    m = ols(df, y, [v])
    flag = "✓" if (abs(m.params[v] - pub_b) < 0.002 and abs(m.bse[v] - pub_se) < 0.002) else "~"
    print(
        f"  {v:<20} β={m.params[v]:+.3f} ({m.bse[v]:.3f}) "
        f"N={int(m.nobs):3d}  R²={m.rsquared:.3f}  "
        f"[pub: {pub_b:+.3f} ({pub_se:.3f}) N={pub_n}]  {flag}"
    )

print("\nAll replications use homoskedastic OLS SEs (Stata default).")
