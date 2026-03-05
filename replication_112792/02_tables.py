"""
02_tables.py — Replicate Table 1 (Panels A-D), Table 2, and Table A-1.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import *

# ─────────────────────────────────────────────────────────────────────
# TABLE A-1: Descriptive Statistics
# ─────────────────────────────────────────────────────────────────────
print("=" * 70)
print("  TABLE A-1: Descriptive Statistics")
print("=" * 70)

sdf = load_state_data()
# Create fiscal year
sdf["fy"] = np.nan
for yr, lo, hi in [(2003, 171, 174), (2004, 175, 178), (2005, 179, 182),
                    (2006, 183, 186), (2007, 187, 190), (2008, 191, 194),
                    (2009, 195, 198)]:
    sdf.loc[(sdf["qnum"] >= lo) & (sdf["qnum"] <= hi), "fy"] = yr

# Weighted collapse by quarter and treatment
sdf["treat"] = np.where(sdf["state"] == "MA", "MA", "Other")

for panel_label, var in [("A. All applications", "allapps"),
                          ("B. SSDI only", "DIonly"),
                          ("C. SSI total", "SSItotal"),
                          ("D. SSDI total", "SSDItotal"),
                          ("E. Unemployment rate", "ue")]:
    print(f"\nPanel {panel_label}")
    print(f"{'FY':>6s}  {'MA':>8s}  {'Other NE':>8s}")
    for yr in [2003, 2004, 2005, 2006, 2007, 2008, 2009]:
        mask_fy = sdf["fy"] == yr
        for treat_label in ["MA", "Other"]:
            mask = mask_fy & (sdf["treat"] == treat_label)
            sub = sdf.loc[mask]
            if len(sub) > 0:
                wmean = np.average(sub[var], weights=sub["wapop"])
            else:
                wmean = np.nan
            if treat_label == "MA":
                ma_val = wmean
            else:
                other_val = wmean
        print(f"{yr:6d}  {ma_val:8.2f}  {other_val:8.2f}")

# Uninsurance rates
print("\nPanel F. Uninsurance rate")
for q, label in [(180, "CY 2005"), (196, "CY 2010")]:
    sub = sdf[sdf["qnum"] == q]
    for treat_label in ["MA", "Other"]:
        mask = sub["treat"] == treat_label
        vals = sub.loc[mask, "unin"]
        if len(vals) > 0:
            if treat_label == "MA":
                print(f"  {label} MA: {vals.values[0]:.1f}")
            else:
                wmean = np.average(sub.loc[mask, "unin"], weights=sub.loc[mask, "wapop"])
                print(f"  {label} Other NE: {wmean:.1f}")

# ─────────────────────────────────────────────────────────────────────
# TABLE 1, PANEL A: State-level regressions
# ─────────────────────────────────────────────────────────────────────
print(f"\n\n{'=' * 70}")
print("  TABLE 1, PANEL A: State-level regressions (applications)")
print("=" * 70)

sdf = load_state_data()
states = sorted(sdf["state"].unique())
# State dummies: drop state 2 (MA) — use stnum1, stnum3-stnum9
state_dummies = ["stnum1"] + [f"stnum{i}" for i in range(3, len(states) + 1)]
# Quarter dummies
q_vals = sorted(sdf["qnum"].unique())
for q in q_vals:
    sdf[f"q_{q}"] = (sdf["qnum"] == q).astype(int)
# Drop first quarter as reference
q_dummies = [f"q_{q}" for q in q_vals[1:]]

interaction_vars = ["MAXpost1", "MAXpost2", "MAXpost3"]
post_vars = ["post1", "post2", "post3"]
xvars = interaction_vars + post_vars + ["ue"] + state_dummies + q_dummies

print(f"\n{'Outcome':>12s}  {'MA*FY2007':>12s}  {'SE':>10s}  {'MA*FY2008':>12s}  {'SE':>10s}  {'MA*FY2009':>12s}  {'SE':>10s}  {'N':>5s}")
print("-" * 100)

for depvar in APP_VARS_STATE:
    res, sub = run_wls(sdf, depvar, xvars)
    coefs = [res.params[v] for v in interaction_vars]
    ses = [res.bse[v] for v in interaction_vars]
    print(f"{depvar:>12s}  {coefs[0]:12.4f}  ({ses[0]:8.4f})  {coefs[1]:12.4f}  ({ses[1]:8.4f})  {coefs[2]:12.4f}  ({ses[2]:8.4f})  {int(res.nobs):5d}")

print("\nPublished Table 1 Panel A:")
print("                  All apps    SSDI only   SSI total   SSDI total")
print("MA*FY2007          0.0407*     0.0295*     0.0113      0.0295")
print("                  (0.0198)    (0.0134)    (0.0254)    (0.0247)")
print("MA*FY2008          0.0800***   0.0647***   0.0153      0.0703**")
print("                  (0.0152)    (0.0170)    (0.0189)    (0.0264)")
print("MA*FY2009          0.0148      0.0405     -0.0257      0.0234")
print("                  (0.0560)    (0.0325)    (0.0374)    (0.0272)")

# ─────────────────────────────────────────────────────────────────────
# TABLE 1, PANELS B-D: County-level regressions
# ─────────────────────────────────────────────────────────────────────
print(f"\n\n{'=' * 70}")
print("  TABLE 1, PANELS B-D: County-level regressions (applications)")
print("=" * 70)

cdf, q_vals_county = load_county_data()
# Quarter dummies — drop first as reference
q_dummies_c = [f"qnum{i}" for i in range(2, len(q_vals_county) + 1)]
# County dummies
counties = sorted(cdf["county"].unique())
for c in counties:
    cdf[f"cty_{c}"] = (cdf["county"] == c).astype(int)
# Drop first county as reference
county_dummies = [f"cty_{c}" for c in counties[1:]]

interaction_vars_c = ["MAXpost1", "MAXpost2"]
post_vars_c = ["post1", "post2"]
xvars_c = interaction_vars_c + post_vars_c + ["ue"] + q_dummies_c + county_dummies

for panel, label, condition in [
    ("B", "All counties", None),
    ("C", "Low-insurance counties (lowHI==1)", 1),
    ("D", "High-insurance counties (lowHI==0)", 0),
]:
    print(f"\n--- Panel {panel}: {label} ---")
    if condition is not None:
        sub_df = cdf[cdf["lowHI"] == condition].copy()
        # Rebuild county dummies for subset
        sub_counties = sorted(sub_df["county"].unique())
        for c in sub_counties:
            sub_df[f"cty_{c}"] = (sub_df["county"] == c).astype(int)
        sub_county_dummies = [f"cty_{c}" for c in sub_counties[1:]]
        sub_xvars = interaction_vars_c + post_vars_c + ["ue"] + q_dummies_c + sub_county_dummies
    else:
        sub_df = cdf.copy()
        sub_xvars = xvars_c

    print(f"{'Outcome':>12s}  {'MA*FY2007':>12s}  {'SE':>10s}  {'MA*FY2008':>12s}  {'SE':>10s}  {'N':>5s}")
    print("-" * 70)
    for depvar in APP_VARS_COUNTY:
        res, _ = run_wls(sub_df, depvar, sub_xvars)
        coefs = [res.params[v] for v in interaction_vars_c]
        ses = [res.bse[v] for v in interaction_vars_c]
        print(f"{depvar:>12s}  {coefs[0]:12.4f}  ({ses[0]:8.4f})  {coefs[1]:12.4f}  ({ses[1]:8.4f})  {int(res.nobs):5d}")

print("\nPublished Table 1 Panels B-D:")
print("Panel B (Counties):        All apps    SSDI only   SSI total   SSDI total")
print("MA*FY2007                  -0.00724     0.00347    -0.0107     -0.0050")
print("                           (0.0121)    (0.0068)    (0.0133)    (0.0184)")
print("MA*FY2008                   0.0340*     0.0469***  -0.0129      0.0483")
print("                           (0.0161)    (0.0085)    (0.0164)    (0.0285)")
print("Panel C (Low-ins):         All apps    SSDI only   SSI total   SSDI total")
print("MA*FY2007                  -0.0617***   0.00482    -0.0665***  -0.0470***")
print("                           (0.0048)    (0.0111)    (0.0121)    (0.0084)")
print("MA*FY2008                  -0.0610***   0.0448***  -0.106***   -0.0266**")
print("                           (0.0090)    (0.0108)    (0.0073)    (0.0080)")
print("Panel D (High-ins):        All apps    SSDI only   SSI total   SSDI total")
print("MA*FY2007                   0.0405      0.00782     0.0327      0.0372")
print("                           (0.0361)    (0.0082)    (0.0300)    (0.0351)")
print("MA*FY2008                   0.133**     0.0540***   0.0794*     0.137*")
print("                           (0.0393)    (0.0134)    (0.0396)    (0.0458)")

# ─────────────────────────────────────────────────────────────────────
# TABLE 2: Time to Filing
# ─────────────────────────────────────────────────────────────────────
print(f"\n\n{'=' * 70}")
print("  TABLE 2: Effect of MA Health Insurance Reform on Time to Filing")
print("=" * 70)

for panel, label, condition in [
    ("A", "Low-insurance counties (lowHI==1)", 1),
    ("B", "High-insurance counties (lowHI==0)", 0),
]:
    print(f"\n--- Panel {panel}: {label} ---")
    sub_df = cdf[cdf["lowHI"] == condition].copy()
    sub_counties = sorted(sub_df["county"].unique())
    for c in sub_counties:
        sub_df[f"cty_{c}"] = (sub_df["county"] == c).astype(int)
    sub_county_dummies = [f"cty_{c}" for c in sub_counties[1:]]
    sub_xvars = interaction_vars_c + post_vars_c + ["ue"] + q_dummies_c + sub_county_dummies

    print(f"{'Outcome':>18s}  {'MA*FY2007':>12s}  {'SE':>10s}  {'MA*FY2008':>12s}  {'SE':>10s}  {'N':>5s}")
    print("-" * 75)
    for depvar in TIME_VARS:
        try:
            res, _ = run_wls(sub_df, depvar, sub_xvars)
            coefs = [res.params[v] for v in interaction_vars_c]
            ses = [res.bse[v] for v in interaction_vars_c]
            print(f"{depvar:>18s}  {coefs[0]:12.4f}  ({ses[0]:8.4f})  {coefs[1]:12.4f}  ({ses[1]:8.4f})  {int(res.nobs):5d}")
        except Exception as e:
            # Check actual column names for time variables
            print(f"{depvar:>18s}  ERROR: {e}")

    # Also try alternate column names
    for alt_name in ["DI_mntime", "DIonly_mntime", "Allapps_mntime",
                     "SSItotal_mtime", "SSItotal_mntime"]:
        if alt_name in sub_df.columns and alt_name not in TIME_VARS:
            try:
                res, _ = run_wls(sub_df, alt_name, sub_xvars)
                coefs = [res.params[v] for v in interaction_vars_c]
                ses = [res.bse[v] for v in interaction_vars_c]
                print(f"  (alt: {alt_name:>18s})  {coefs[0]:12.4f}  ({ses[0]:8.4f})  {coefs[1]:12.4f}  ({ses[1]:8.4f})  {int(res.nobs):5d}")
            except:
                pass

print("\nPublished Table 2:")
print("Panel A (Low-ins):         SSDI only   SSI total")
print("MA*FY2007                   1.099***    5.378***")
print("                           (0.267)     (0.366)")
print("MA*FY2008                   0.511***    9.199***")
print("                           (0.134)     (0.695)")
print("Panel B (High-ins):        SSDI only   SSI total")
print("MA*FY2007                  -1.438***    3.733***")
print("                           (0.318)     (0.254)")
print("MA*FY2008                  -2.098**     6.697***")
print("                           (0.586)     (0.487)")
