"""
05_robustness.py — Robustness checks for the main DiD results.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import *
from scipy import stats

def section(title):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")

# Load data
sdf = load_state_data()
states = sorted(sdf["state"].unique())
state_dummies = ["stnum1"] + [f"stnum{i}" for i in range(3, len(states) + 1)]
q_vals = sorted(sdf["qnum"].unique())
for q in q_vals:
    sdf[f"q_{q}"] = (sdf["qnum"] == q).astype(int)
q_dummies = [f"q_{q}" for q in q_vals[1:]]
interaction_vars_s = ["MAXpost1", "MAXpost2", "MAXpost3"]
post_vars_s = ["post1", "post2", "post3"]
xvars_s = interaction_vars_s + post_vars_s + ["ue"] + state_dummies + q_dummies

cdf, q_vals_county = load_county_data()
q_dummies_c = [f"qnum{i}" for i in range(2, len(q_vals_county) + 1)]
counties = sorted(cdf["county"].unique())
for c in counties:
    cdf[f"cty_{c}"] = (cdf["county"] == c).astype(int)
county_dummies = [f"cty_{c}" for c in counties[1:]]
interaction_vars_c = ["MAXpost1", "MAXpost2"]
post_vars_c = ["post1", "post2"]
xvars_c = interaction_vars_c + post_vars_c + ["ue"] + q_dummies_c + county_dummies

# ─────────────────────────────────────────────────────────────────────
# 1. Drop Maine and Vermont (other health reform states)
# ─────────────────────────────────────────────────────────────────────
section("1. Drop Maine (ME) and Vermont (VT)")
sdf_nomevt = sdf[~sdf["state"].isin(["ME", "VT"])].copy()
# Rebuild state dummies
remaining_states = sorted(sdf_nomevt["state"].unique())
for i, s in enumerate(remaining_states, 1):
    sdf_nomevt[f"st2_{i}"] = (sdf_nomevt["state"] == s).astype(int)
# MA is whichever index
ma_idx = remaining_states.index("MA") + 1
st2_dummies = [f"st2_{i}" for i in range(1, len(remaining_states) + 1) if i != ma_idx]
xvars_nomevt = interaction_vars_s + post_vars_s + ["ue"] + st2_dummies + q_dummies

for depvar in ["allapps", "DIonly"]:
    res, _ = run_wls(sdf_nomevt, depvar, xvars_nomevt)
    print_did_results(res, interaction_vars_s, f"{depvar} (drop ME & VT)")

# ─────────────────────────────────────────────────────────────────────
# 2. Restrict to FY2005-FY2008 (shorter window)
# ─────────────────────────────────────────────────────────────────────
section("2. Shorter pre-period (FY2005 onward, q_fld >= 179)")
sdf_short = sdf[sdf["qnum"] >= 179].copy()
q_vals_short = sorted(sdf_short["qnum"].unique())
for q in q_vals_short:
    sdf_short[f"qs_{q}"] = (sdf_short["qnum"] == q).astype(int)
qs_dummies = [f"qs_{q}" for q in q_vals_short[1:]]
xvars_short = interaction_vars_s + post_vars_s + ["ue"] + state_dummies + qs_dummies

for depvar in ["allapps", "DIonly"]:
    res, _ = run_wls(sdf_short, depvar, xvars_short)
    print_did_results(res, interaction_vars_s, f"{depvar} (FY2005+ only)")

# ─────────────────────────────────────────────────────────────────────
# 3. Drop outlier counties (top/bottom 5% by application rate)
# ─────────────────────────────────────────────────────────────────────
section("3. Drop outlier counties (top/bottom 5% by mean allapps)")
county_means = cdf.groupby("county")["allapps"].mean()
lo, hi = county_means.quantile(0.05), county_means.quantile(0.95)
keep_counties = county_means[(county_means >= lo) & (county_means <= hi)].index
cdf_trim = cdf[cdf["county"].isin(keep_counties)].copy()
trim_counties = sorted(cdf_trim["county"].unique())
for c in trim_counties:
    cdf_trim[f"cty_{c}"] = (cdf_trim["county"] == c).astype(int)
trim_county_dummies = [f"cty_{c}" for c in trim_counties[1:]]
xvars_trim = interaction_vars_c + post_vars_c + ["ue"] + q_dummies_c + trim_county_dummies

print(f"Dropped {len(counties) - len(trim_counties)} counties, keeping {len(trim_counties)}")
for depvar in ["allapps", "DIonly"]:
    res, _ = run_wls(cdf_trim, depvar, xvars_trim)
    print_did_results(res, interaction_vars_c, f"{depvar} (trimmed counties)")

# ─────────────────────────────────────────────────────────────────────
# 4. Placebo test: shuffle MA treatment across states
# ─────────────────────────────────────────────────────────────────────
section("4. Placebo/Permutation test (state level, allapps)")
np.random.seed(42)
n_perms = 500
actual_res, _ = run_wls(sdf, "allapps", xvars_s)
actual_coef = actual_res.params["MAXpost2"]  # Focus on FY2008 (strongest result)

placebo_coefs = []
unique_states = sdf["state"].unique()
for _ in range(n_perms):
    fake_ma = np.random.choice(unique_states, 1)[0]
    sdf_perm = sdf.copy()
    sdf_perm["MA_fake"] = (sdf_perm["state"] == fake_ma).astype(int)
    sdf_perm["MAXpost1_fake"] = sdf_perm["MA_fake"] * sdf_perm["post1"]
    sdf_perm["MAXpost2_fake"] = sdf_perm["MA_fake"] * sdf_perm["post2"]
    sdf_perm["MAXpost3_fake"] = sdf_perm["MA_fake"] * sdf_perm["post3"]
    fake_interaction = ["MAXpost1_fake", "MAXpost2_fake", "MAXpost3_fake"]
    fake_xvars = fake_interaction + post_vars_s + ["ue"] + state_dummies + q_dummies
    try:
        res_p, _ = run_wls(sdf_perm, "allapps", fake_xvars)
        placebo_coefs.append(res_p.params["MAXpost2_fake"])
    except:
        pass

placebo_coefs = np.array(placebo_coefs)
p_value = (np.abs(placebo_coefs) >= np.abs(actual_coef)).mean()
print(f"Actual MA*FY2008 coefficient: {actual_coef:.4f}")
print(f"Placebo distribution: mean={placebo_coefs.mean():.4f}, sd={placebo_coefs.std():.4f}")
print(f"Permutation p-value (two-sided): {p_value:.4f}")
print(f"  (Fraction of |placebo| >= |actual|, from {len(placebo_coefs)} permutations)")

# ─────────────────────────────────────────────────────────────────────
# 5. Winsorize outcome variables
# ─────────────────────────────────────────────────────────────────────
section("5. Winsorize outcomes at 1st/99th percentile")
from scipy.stats import mstats

for depvar in ["allapps", "DIonly"]:
    sdf_w = sdf.copy()
    lo_p, hi_p = sdf_w[depvar].quantile(0.01), sdf_w[depvar].quantile(0.99)
    sdf_w[depvar] = sdf_w[depvar].clip(lo_p, hi_p)
    res, _ = run_wls(sdf_w, depvar, xvars_s)
    print_did_results(res, interaction_vars_s, f"{depvar} (winsorized)")

# ─────────────────────────────────────────────────────────────────────
# 6. Placebo outcome: unemployment rate
# ─────────────────────────────────────────────────────────────────────
section("6. Placebo outcome: unemployment rate (should not be affected by MA reform)")
# Remove ue from RHS for this test
xvars_noue = interaction_vars_s + post_vars_s + state_dummies + q_dummies
res_ue, _ = run_wls(sdf, "ue", xvars_noue)
print_did_results(res_ue, interaction_vars_s, "Unemployment rate as outcome")

# ─────────────────────────────────────────────────────────────────────
# 7. Subgroup heterogeneity by state
# ─────────────────────────────────────────────────────────────────────
section("7. Leave-one-state-out sensitivity (state level, allapps, MA*FY2008)")
baseline_res, _ = run_wls(sdf, "allapps", xvars_s)
baseline_coef = baseline_res.params["MAXpost2"]
print(f"Baseline MA*FY2008: {baseline_coef:.4f}")

for drop_state in [s for s in unique_states if s != "MA"]:
    sdf_loo = sdf[sdf["state"] != drop_state].copy()
    # Rebuild state dummies
    loo_states = sorted(sdf_loo["state"].unique())
    for i, s in enumerate(loo_states, 1):
        sdf_loo[f"loo_{i}"] = (sdf_loo["state"] == s).astype(int)
    ma_idx_loo = loo_states.index("MA") + 1
    loo_dummies = [f"loo_{i}" for i in range(1, len(loo_states) + 1) if i != ma_idx_loo]
    loo_xvars = interaction_vars_s + post_vars_s + ["ue"] + loo_dummies + q_dummies
    try:
        res_loo, _ = run_wls(sdf_loo, "allapps", loo_xvars)
        coef = res_loo.params["MAXpost2"]
        print(f"  Drop {drop_state}: MA*FY2008 = {coef:.4f} (delta = {coef - baseline_coef:+.4f})")
    except Exception as e:
        print(f"  Drop {drop_state}: ERROR — {e}")

# ─────────────────────────────────────────────────────────────────────
# 8. Alternative clustering: HC1 robust SEs (no clustering)
# ─────────────────────────────────────────────────────────────────────
section("8. Alternative SEs: HC1 (robust, no clustering)")
for depvar in ["allapps", "DIonly"]:
    subset = sdf.dropna(subset=[depvar] + xvars_s + ["wapop"]).copy()
    y = subset[depvar].astype(float)
    X = sm.add_constant(subset[xvars_s].astype(float))
    w = subset["wapop"].astype(float)
    mod = sm.WLS(y, X, weights=w)
    res_hc1 = mod.fit(cov_type="HC1")
    print(f"\n  {depvar} with HC1 SEs:")
    for v in interaction_vars_s:
        print(f"    {v}: coef={res_hc1.params[v]:.4f} se=({res_hc1.bse[v]:.4f})")

# ─────────────────────────────────────────────────────────────────────
# 9. Log transformation of outcomes
# ─────────────────────────────────────────────────────────────────────
section("9. Log transformation of outcome (state level)")
for depvar in ["allapps", "DIonly"]:
    sdf_log = sdf.copy()
    sdf_log[f"ln_{depvar}"] = np.log(sdf_log[depvar].clip(lower=0.001))
    res, _ = run_wls(sdf_log, f"ln_{depvar}", xvars_s)
    print_did_results(res, interaction_vars_s, f"ln({depvar})")

# ─────────────────────────────────────────────────────────────────────
# 10. Single post-period (pooled post)
# ─────────────────────────────────────────────────────────────────────
section("10. Pooled post-period (single DiD)")
sdf_pooled = sdf.copy()
sdf_pooled["post"] = ((sdf_pooled["qnum"] >= 187)).astype(int)
sdf_pooled["MAXpost"] = sdf_pooled["MA"] * sdf_pooled["post"]
xvars_pooled = ["MAXpost", "post", "ue"] + state_dummies + q_dummies

for depvar in ["allapps", "DIonly", "SSItotal", "SSDItotal"]:
    res, _ = run_wls(sdf_pooled, depvar, xvars_pooled)
    coef = res.params["MAXpost"]
    se = res.bse["MAXpost"]
    pval = res.pvalues["MAXpost"]
    stars = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
    print(f"  {depvar:12s}  MAXpost={coef:.4f} ({se:.4f}) {stars}")

# ─────────────────────────────────────────────────────────────────────
# 11. Alternative lowHI threshold (county level)
# ─────────────────────────────────────────────────────────────────────
section("11. Alternative lowHI thresholds (county level, allapps)")
for threshold in [0.10, 0.11, 0.12, 0.13, 0.14, 0.15]:
    cdf_alt = cdf.copy()
    cdf_alt["lowHI_alt"] = (cdf_alt["nohi05"] >= threshold).astype(int)
    for hi_val, label in [(1, "Low-ins"), (0, "High-ins")]:
        sub = cdf_alt[cdf_alt["lowHI_alt"] == hi_val].copy()
        sub_counties = sorted(sub["county"].unique())
        for c in sub_counties:
            sub[f"cty_{c}"] = (sub["county"] == c).astype(int)
        sub_cty_dummies = [f"cty_{c}" for c in sub_counties[1:]]
        sub_xvars = interaction_vars_c + post_vars_c + ["ue"] + q_dummies_c + sub_cty_dummies
        try:
            res, _ = run_wls(sub, "allapps", sub_xvars)
            c1 = res.params["MAXpost1"]
            c2 = res.params["MAXpost2"]
            print(f"  threshold={threshold:.2f} {label:>8s}: MA*FY2007={c1:7.4f}  MA*FY2008={c2:7.4f}  N={int(res.nobs)}")
        except Exception as e:
            print(f"  threshold={threshold:.2f} {label:>8s}: ERROR — {e}")

# ─────────────────────────────────────────────────────────────────────
# 12. Event study (state level, quarterly coefficients)
# ─────────────────────────────────────────────────────────────────────
section("12. Event study: quarterly MA interactions (state level, allapps)")
sdf_es = sdf.copy()
# Create quarterly interactions (exclude one pre-period quarter as reference)
# Reference: q_fld=186 (last pre-reform quarter, 2006Q2)
ref_q = 186
es_qs = [q for q in sorted(sdf_es["qnum"].unique()) if q != ref_q]
for q in es_qs:
    sdf_es[f"MA_q{q}"] = (sdf_es["MA"] * (sdf_es["qnum"] == q)).astype(int)
es_interaction = [f"MA_q{q}" for q in es_qs]
xvars_es = es_interaction + ["ue"] + state_dummies + q_dummies

res_es, _ = run_wls(sdf_es, "allapps", xvars_es)
print(f"{'Quarter':>10s}  {'Coef':>10s}  {'SE':>10s}  {'p':>8s}")
print("-" * 45)
for q in es_qs:
    v = f"MA_q{q}"
    coef = res_es.params[v]
    se = res_es.bse[v]
    pval = res_es.pvalues[v]
    stars = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
    marker = " <-- post" if q >= 187 else ""
    print(f"  q={q:4d}  {coef:10.4f}  ({se:8.4f})  p={pval:.3f} {stars}{marker}")

print("\n\nRobustness checks complete.")
