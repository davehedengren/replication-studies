"""
04_data_audit.py — Comprehensive data audit for all datasets.

Danieli et al. (2025) "Negative Control Falsification Tests for IV Designs"
Replication package: openICPSR 238658-V1

Audits:
  1. ADH (Autor, Dorn, Hanson 2013) — China trade shock panel
  2. Deming (2014) — School choice lottery
  3. Ashraf & Galor (2013) — Genetic diversity cross-section
  4. Nunn & Qian (2014) — Food aid and conflict panel
  5. Literature Survey — 140 IV papers
"""

import numpy as np
import pandas as pd
from utils import (OUTPUT_DIR, load_adh, load_adh_preperiod, load_deming,
                   load_ashraf_galor, load_nunn_qian, load_literature_survey)


# ══════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════

def describe_var(series, name, indent=4):
    """Print summary statistics for a single variable."""
    prefix = " " * indent
    valid = series.dropna()
    n_miss = series.isna().sum()
    n_total = len(series)
    pct_miss = 100 * n_miss / n_total if n_total > 0 else 0

    print(f"{prefix}{name}:")
    print(f"{prefix}  dtype={series.dtype}, N={n_total}, "
          f"valid={len(valid)}, missing={n_miss} ({pct_miss:.1f}%)")

    if len(valid) == 0:
        print(f"{prefix}  (all missing)")
        return

    if pd.api.types.is_numeric_dtype(series):
        print(f"{prefix}  mean={valid.mean():.6f}, sd={valid.std():.6f}")
        print(f"{prefix}  min={valid.min():.6f}, median={valid.median():.6f}, "
              f"max={valid.max():.6f}")
        n_unique = valid.nunique()
        if n_unique <= 10:
            print(f"{prefix}  unique values ({n_unique}): "
                  f"{sorted(valid.unique().tolist())}")
    else:
        n_unique = valid.nunique()
        print(f"{prefix}  unique values: {n_unique}")
        if n_unique <= 20:
            vc = valid.value_counts()
            for val, cnt in vc.items():
                print(f"{prefix}    {val}: {cnt}")


def missing_summary(df, label=""):
    """Print a summary of missing values across all columns."""
    n_rows = len(df)
    miss = df.isnull().sum()
    cols_with_miss = miss[miss > 0].sort_values(ascending=False)

    if len(cols_with_miss) == 0:
        print(f"    No missing values in any column.")
    else:
        print(f"    Columns with missing values ({len(cols_with_miss)} of {len(df.columns)}):")
        for col, n_miss in cols_with_miss.items():
            pct = 100 * n_miss / n_rows
            print(f"      {col}: {n_miss}/{n_rows} ({pct:.1f}%)")


def dtype_summary(df):
    """Print variable type distribution."""
    dtype_counts = df.dtypes.value_counts()
    print(f"    Variable types:")
    for dtype, count in dtype_counts.items():
        print(f"      {dtype}: {count} columns")


# ══════════════════════════════════════════════════════════════════════
# 1. ADH (Autor, Dorn, Hanson 2013)
# ══════════════════════════════════════════════════════════════════════

print("=" * 70)
print("04_data_audit.py — Comprehensive Data Audit")
print("=" * 70)

print("\n" + "=" * 70)
print("1. ADH (Autor, Dorn, Hanson 2013) — China Trade Shock")
print("=" * 70)

adh = load_adh()
adh_pre = load_adh_preperiod()

# --- Dimensions ---
print(f"\n  ADH main panel:")
print(f"    Dimensions: {adh.shape[0]} rows x {adh.shape[1]} columns")
print(f"    Expected: 1444 rows (722 czones x 2 years)")
assert adh.shape[0] == 1444, (
    f"FAIL: Expected 1444 rows, got {adh.shape[0]}")
print(f"    PASS: Row count matches expectation (1444)")

# --- Variable types ---
dtype_summary(adh)

# --- Panel structure ---
print(f"\n  Panel structure:")
years = sorted(adh['yr'].unique())
print(f"    Years: {years}")
assert set(years) == {1990, 2000}, f"FAIL: Expected years {{1990, 2000}}, got {set(years)}"
print(f"    PASS: Years are 1990 and 2000")

for y in years:
    n = len(adh[adh['yr'] == y])
    print(f"    Year {y}: {n} commuting zones")
    assert n == 722, f"FAIL: Expected 722 czones in year {y}, got {n}"
print(f"    PASS: Each year has 722 commuting zones")

# --- Cluster structure ---
n_states = adh['statefip'].nunique()
print(f"\n  Cluster variable (statefip):")
print(f"    Unique state FIPS codes: {n_states}")

# Czones per state
czones_per_state = adh[adh['yr'] == 1990].groupby('statefip')['czone'].nunique()
print(f"    Czones per state: min={czones_per_state.min()}, "
      f"max={czones_per_state.max()}, "
      f"mean={czones_per_state.mean():.1f}")

# --- Missing values ---
print(f"\n  Missing values (main panel):")
missing_summary(adh)

# --- Key variables ---
print(f"\n  Key variable distributions (full panel):")

key_adh = ['d_tradeotch_pw_lag', 'd_tradeusch_pw', 'd_sh_empl_mfg',
           'timepwt48', 'statefip']
for v in key_adh:
    if v in adh.columns:
        describe_var(adh[v], v)

# --- Cross-section distributions by year ---
print(f"\n  Key variable distributions by year:")
for y in [1990, 2000]:
    print(f"\n    Year {y}:")
    sub = adh[adh['yr'] == y]
    for v in ['d_tradeotch_pw_lag', 'd_tradeusch_pw', 'd_sh_empl_mfg']:
        if v in sub.columns:
            s = sub[v].dropna()
            print(f"      {v}: mean={s.mean():.4f}, sd={s.std():.4f}, "
                  f"min={s.min():.4f}, max={s.max():.4f}")

# --- Weights ---
print(f"\n  Weights (timepwt48):")
w = adh['timepwt48'].dropna()
print(f"    All positive: {(w > 0).all()}")
print(f"    Sum: {w.sum():.2f}")
print(f"    Range: [{w.min():.6f}, {w.max():.6f}]")

# --- Pre-period data ---
print(f"\n  ADH pre-period panel:")
print(f"    Dimensions: {adh_pre.shape[0]} rows x {adh_pre.shape[1]} columns")
pre_years = sorted(adh_pre['yr'].unique())
print(f"    Years: {pre_years}")
for y in pre_years:
    print(f"    Year {y}: {len(adh_pre[adh_pre['yr'] == y])} czones")

# --- Logical consistency: czones match across datasets ---
czones_main = set(adh['czone'].unique())
czones_pre = set(adh_pre['czone'].unique())
print(f"\n  Logical consistency:")
print(f"    Main panel czones: {len(czones_main)}")
print(f"    Pre-period czones: {len(czones_pre)}")
print(f"    Overlap: {len(czones_main & czones_pre)}")
print(f"    In main only: {len(czones_main - czones_pre)}")
print(f"    In pre only: {len(czones_pre - czones_main)}")


# ══════════════════════════════════════════════════════════════════════
# 2. DEMING (2014)
# ══════════════════════════════════════════════════════════════════════

print("\n\n" + "=" * 70)
print("2. Deming (2014) — School Choice Lottery")
print("=" * 70)

dem = load_deming()

# --- Dimensions ---
print(f"\n  Dimensions: {dem.shape[0]} rows x {dem.shape[1]} columns")

# --- Variable types ---
dtype_summary(dem)

# --- Missing values ---
print(f"\n  Missing values:")
missing_summary(dem)

# --- Key variables ---
print(f"\n  Key variable distributions:")
key_dem = ['lottery', 'testz2003', 'lottery_FE']
for v in key_dem:
    if v in dem.columns:
        describe_var(dem[v], v)

# --- Lottery balance ---
print(f"\n  Lottery balance:")
if 'lottery' in dem.columns:
    lottery_vc = dem['lottery'].value_counts().sort_index()
    for val, cnt in lottery_vc.items():
        pct = 100 * cnt / len(dem)
        print(f"    lottery={val}: {cnt} ({pct:.1f}%)")

    # Balance on pre-treatment variables
    pre_vars = ['testz2002', 'math_2002_imp', 'read_2002_imp']
    pre_vars = [v for v in pre_vars if v in dem.columns]
    if pre_vars:
        print(f"\n  Pre-treatment balance across lottery groups:")
        for v in pre_vars:
            for lott_val in sorted(dem['lottery'].unique()):
                sub = dem[dem['lottery'] == lott_val][v].dropna()
                if len(sub) > 0:
                    print(f"    lottery={lott_val}, {v}: "
                          f"N={len(sub)}, mean={sub.mean():.4f}, sd={sub.std():.4f}")

# --- Lottery FE groups ---
print(f"\n  Lottery FE groups:")
if 'lottery_FE' in dem.columns:
    fe_valid = dem['lottery_FE'].dropna()
    n_fe = fe_valid.nunique()
    print(f"    Unique groups: {n_fe}")
    print(f"    Missing lottery_FE: {dem['lottery_FE'].isna().sum()}")

    fe_sizes = dem.groupby('lottery_FE').size()
    print(f"    Group sizes: min={fe_sizes.min()}, max={fe_sizes.max()}, "
          f"mean={fe_sizes.mean():.1f}, median={fe_sizes.median():.1f}")

# --- Sample sizes after common restrictions ---
print(f"\n  Sample sizes under common restrictions:")
dem_nofe_miss = dem[dem['lottery_FE'].notna()]
print(f"    After dropping missing lottery_FE: {len(dem_nofe_miss)}")

dem_no14 = dem_nofe_miss[dem_nofe_miss['lottery_FE'] != 14]
print(f"    After also dropping lottery_FE == 14: {len(dem_no14)}")

key_needed = ['lottery', 'testz2003', 'lottery_FE']
key_available = [c for c in key_needed if c in dem.columns]
dem_complete = dem_no14.dropna(subset=key_available)
print(f"    After also dropping missing key vars: {len(dem_complete)}")

# --- Outcome distribution ---
print(f"\n  Outcome (testz2003) distribution:")
if 'testz2003' in dem.columns:
    describe_var(dem['testz2003'], 'testz2003')

# --- Additional test score variables ---
testz_cols = [c for c in dem.columns if c.startswith('testz')]
if testz_cols:
    print(f"\n  All testz* columns ({len(testz_cols)}):")
    for v in sorted(testz_cols):
        n_valid = dem[v].notna().sum()
        n_miss = dem[v].isna().sum()
        print(f"    {v}: valid={n_valid}, missing={n_miss}")


# ══════════════════════════════════════════════════════════════════════
# 3. ASHRAF & GALOR (2013)
# ══════════════════════════════════════════════════════════════════════

print("\n\n" + "=" * 70)
print("3. Ashraf & Galor (2013) — Genetic Diversity")
print("=" * 70)

ag = load_ashraf_galor()

# --- Dimensions ---
print(f"\n  Dimensions: {ag.shape[0]} rows x {ag.shape[1]} columns")

# --- Variable types ---
dtype_summary(ag)

# --- Missing values ---
print(f"\n  Missing values:")
missing_summary(ag)

# --- Clean sample flag ---
print(f"\n  Sample flags:")
if 'cleanpd1500' in ag.columns:
    flag_vc = ag['cleanpd1500'].value_counts().sort_index()
    for val, cnt in flag_vc.items():
        print(f"    cleanpd1500={val}: {cnt}")

    ag_clean = ag[ag['cleanpd1500'] == 1].copy()
    print(f"\n  Clean sample (cleanpd1500 == 1): {len(ag_clean)} countries")
else:
    ag_clean = ag.copy()
    print(f"  No cleanpd1500 column; using full data: {len(ag_clean)}")

# --- Key variables in clean sample ---
print(f"\n  Key variable distributions (clean sample):")
key_ag = ['mdist_hgdp', 'ln_pd1500', 'ln_yst', 'ln_arable',
          'ln_abslat', 'ln_suitavg']
for v in key_ag:
    if v in ag_clean.columns:
        describe_var(ag_clean[v], v)

# --- Critical check: mdist_hgdp missingness ---
print(f"\n  CRITICAL: mdist_hgdp (IV) missingness in clean sample:")
if 'mdist_hgdp' in ag_clean.columns:
    n_valid_iv = ag_clean['mdist_hgdp'].notna().sum()
    n_miss_iv = ag_clean['mdist_hgdp'].isna().sum()
    print(f"    Non-missing: {n_valid_iv}")
    print(f"    Missing: {n_miss_iv}")
    print(f"    Expected non-missing: ~20")
    if n_valid_iv <= 25:
        print(f"    NOTE: Very small IV sample. Analyses conditioning on mdist_hgdp "
              f"use only {n_valid_iv} observations.")

# --- Fake IVs (negative control IVs) ---
print(f"\n  Fake IV distributions (clean sample):")
fake_iv_vars = ['mdist_addis', 'mdist_london', 'mdist_tokyo', 'mdist_mexico']
for v in fake_iv_vars:
    if v in ag_clean.columns:
        describe_var(ag_clean[v], v)

# --- Check: countries with IV also have controls ---
print(f"\n  Logical consistency: IV + controls availability in clean sample:")
ctrl_ag = ['ln_yst', 'ln_arable', 'ln_abslat', 'ln_suitavg']
for combo_label, subset_filter in [
    ("All clean", ag_clean),
    ("Clean + all controls non-missing",
     ag_clean.dropna(subset=['ln_pd1500'] + ctrl_ag)),
    ("Clean + controls + mdist_hgdp non-missing",
     ag_clean.dropna(subset=['ln_pd1500'] + ctrl_ag + ['mdist_hgdp']))
]:
    if isinstance(subset_filter, pd.DataFrame):
        n = len(subset_filter)
    else:
        n = len(combo_label)
    print(f"    {combo_label}: {n} countries")

# --- Correlation among fake IVs ---
print(f"\n  Pairwise correlations among fake IVs (clean sample, complete cases):")
fake_iv_present = [v for v in fake_iv_vars if v in ag_clean.columns]
if len(fake_iv_present) >= 2:
    fake_iv_data = ag_clean[fake_iv_present].dropna()
    if len(fake_iv_data) > 2:
        corr = fake_iv_data.corr()
        for i in range(len(fake_iv_present)):
            for j in range(i + 1, len(fake_iv_present)):
                print(f"    cor({fake_iv_present[i]}, {fake_iv_present[j]}): "
                      f"{corr.iloc[i, j]:.4f}")


# ══════════════════════════════════════════════════════════════════════
# 4. NUNN & QIAN (2014)
# ══════════════════════════════════════════════════════════════════════

print("\n\n" + "=" * 70)
print("4. Nunn & Qian (2014) — Food Aid and Conflict")
print("=" * 70)

nq = load_nunn_qian()

# --- Dimensions ---
print(f"\n  Dimensions: {nq.shape[0]} rows x {nq.shape[1]} columns")

# --- Variable types ---
dtype_summary(nq)

# --- Missing values ---
print(f"\n  Missing values:")
missing_summary(nq)

# --- Panel structure ---
print(f"\n  Panel structure:")
if 'year' in nq.columns:
    years_nq = sorted(nq['year'].unique())
    print(f"    Years: {years_nq}")
    print(f"    Number of years: {len(years_nq)}")
    print(f"    Year range: {min(years_nq)} to {max(years_nq)}")

if 'risocode' in nq.columns:
    n_countries = nq['risocode'].nunique()
    print(f"    Unique countries (risocode): {n_countries}")

    # Country-year balance
    if 'year' in nq.columns:
        obs_per_country = nq.groupby('risocode')['year'].nunique()
        print(f"    Years per country: min={obs_per_country.min()}, "
              f"max={obs_per_country.max()}, "
              f"mean={obs_per_country.mean():.1f}")

        obs_per_year = nq.groupby('year')['risocode'].nunique()
        print(f"    Countries per year: min={obs_per_year.min()}, "
              f"max={obs_per_year.max()}, "
              f"mean={obs_per_year.mean():.1f}")

        # Check for balanced panel
        expected_balanced = n_countries * len(years_nq)
        is_balanced = len(nq) == expected_balanced
        print(f"    Balanced panel: {is_balanced} "
              f"(actual={len(nq)}, balanced would be={expected_balanced})")

# --- Region coverage ---
region_col = 'wb_region' if 'wb_region' in nq.columns else (
    'region' if 'region' in nq.columns else None)
if region_col:
    print(f"\n  Region coverage ({region_col}):")
    region_vc = nq[region_col].value_counts().sort_index()
    for val, cnt in region_vc.items():
        n_ctry = nq[nq[region_col] == val]['risocode'].nunique() if 'risocode' in nq.columns else '?'
        print(f"    {val}: {cnt} obs ({n_ctry} countries)")

# --- Key variables ---
print(f"\n  Key variable distributions:")
key_nq = ['instrument', 'intra_state', 'fadum_avg']
for v in key_nq:
    if v in nq.columns:
        describe_var(nq[v], v)

# --- NCIV columns: l_USprod_* ---
print(f"\n  Negative Control IV columns (l_USprod_*):")
nciv_cols = [c for c in nq.columns if c.startswith('l_USprod_')]
print(f"    Found {len(nciv_cols)} l_USprod_* columns:")
for c in sorted(nciv_cols):
    s = nq[c].dropna()
    miss = nq[c].isna().sum()
    print(f"      {c}: valid={len(s)}, missing={miss}, "
          f"mean={s.mean():.4f}, sd={s.std():.4f}")

# --- Expected 10 crop NCIV columns ---
expected_crops = ['Oranges', 'Grapes', 'Lettuce', 'Cotton_lint', 'Onions_dry',
                  'Grapefruit', 'Cabbages', 'Watermelons', 'Carrots_turnips',
                  'Peaches_nectarines']
expected_nciv = [f'l_USprod_{crop}' for crop in expected_crops]
found = [c for c in expected_nciv if c in nq.columns]
missing_nciv = [c for c in expected_nciv if c not in nq.columns]
print(f"\n    Expected 10 crop NCIVs: {len(found)} found, {len(missing_nciv)} missing")
if missing_nciv:
    print(f"    Missing: {missing_nciv}")

# --- fadum_avg for NCIV transformation ---
print(f"\n  fadum_avg (used for NCIV transformation):")
if 'fadum_avg' in nq.columns:
    describe_var(nq['fadum_avg'], 'fadum_avg')
    print(f"    Zero values: {(nq['fadum_avg'] == 0).sum()}")

# --- Logical consistency: instrument construction ---
print(f"\n  Logical consistency:")
if 'instrument' in nq.columns and 'fadum_avg' in nq.columns:
    # Check if instrument and fadum_avg are related
    both_valid = nq[['instrument', 'fadum_avg']].dropna()
    corr_inst_fad = both_valid['instrument'].corr(both_valid['fadum_avg'])
    print(f"    cor(instrument, fadum_avg): {corr_inst_fad:.4f}")

if 'intra_state' in nq.columns:
    # Outcome range check (should be 0/1 or proportion)
    intra = nq['intra_state'].dropna()
    print(f"    intra_state range: [{intra.min():.6f}, {intra.max():.6f}]")
    print(f"    intra_state == 0: {(intra == 0).sum()} "
          f"({100 * (intra == 0).sum() / len(intra):.1f}%)")
    print(f"    intra_state > 0: {(intra > 0).sum()} "
          f"({100 * (intra > 0).sum() / len(intra):.1f}%)")


# ══════════════════════════════════════════════════════════════════════
# 5. LITERATURE SURVEY
# ══════════════════════════════════════════════════════════════════════

print("\n\n" + "=" * 70)
print("5. Literature Survey — 140 IV Papers")
print("=" * 70)

lit = load_literature_survey()
lit.columns = lit.columns.str.strip()

# --- Dimensions ---
print(f"\n  Dimensions: {lit.shape[0]} rows x {lit.shape[1]} columns")
print(f"    Expected: 140 papers")
assert len(lit) == 140, f"FAIL: Expected 140 papers, got {len(lit)}"
print(f"    PASS: Paper count matches expectation (140)")

# --- Variable types ---
dtype_summary(lit)

# --- All columns ---
print(f"\n  Column names:")
for i, c in enumerate(lit.columns):
    print(f"    [{i}] {c} (dtype={lit[c].dtype})")

# --- Missing values ---
print(f"\n  Missing values:")
missing_summary(lit)

# --- Key variable: is_falsification ---
print(f"\n  is_falsification:")
if 'is_falsification' in lit.columns:
    describe_var(lit['is_falsification'], 'is_falsification')
    n_fals = lit['is_falsification'].sum()
    print(f"    Papers using falsification: {n_fals}/{len(lit)} "
          f"({100 * n_fals / len(lit):.1f}%)")

# --- Journal distribution ---
print(f"\n  Journal distribution:")
if 'journal' in lit.columns:
    journal_vc = lit['journal'].value_counts().sort_values(ascending=False)
    for journal, cnt in journal_vc.items():
        pct = 100 * cnt / len(lit)
        # Also show falsification rate by journal
        if 'is_falsification' in lit.columns:
            sub = lit[lit['journal'] == journal]
            fals_rate = sub['is_falsification'].mean()
            print(f"    {journal}: {cnt} papers ({pct:.1f}%), "
                  f"falsification rate: {fals_rate:.0%}")
        else:
            print(f"    {journal}: {cnt} papers ({pct:.1f}%)")

# --- Falsification type distributions ---
print(f"\n  Falsification type distributions:")

# NCO type
if 'falsification_type_nco' in lit.columns:
    print(f"\n    falsification_type_nco:")
    describe_var(lit['falsification_type_nco'], 'falsification_type_nco', indent=6)

# NCI type
if 'falsification_type_nci' in lit.columns:
    print(f"\n    falsification_type_nci:")
    describe_var(lit['falsification_type_nci'], 'falsification_type_nci', indent=6)

# Any columns with 'nco' or 'nci' or 'falsification' in name
print(f"\n  All falsification/NC-related columns:")
nc_related = [c for c in lit.columns
              if any(pat in c.lower() for pat in ['nco', 'nci', 'falsif', 'placebo', 'nc_'])]
for c in nc_related:
    if pd.api.types.is_numeric_dtype(lit[c]):
        n_pos = (lit[c] > 0).sum()
        print(f"    {c}: sum={lit[c].sum()}, non-zero={n_pos}/{len(lit)} "
              f"({100 * n_pos / len(lit):.1f}%)")
    else:
        n_valid = lit[c].notna().sum()
        n_unique = lit[c].nunique()
        print(f"    {c}: valid={n_valid}, unique={n_unique}")

# --- Cross-tabulation: journal x is_falsification ---
if 'journal' in lit.columns and 'is_falsification' in lit.columns:
    print(f"\n  Cross-tabulation: journal x is_falsification:")
    ct = pd.crosstab(lit['journal'], lit['is_falsification'], margins=True)
    ct.columns = [f"is_fals={c}" for c in ct.columns]
    print(ct.to_string(index=True))


# ══════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════

print("\n\n" + "=" * 70)
print("AUDIT SUMMARY")
print("=" * 70)

print(f"""
  Dataset               Rows      Cols    Key Check
  -----------------------------------------------------------
  ADH (main)            {adh.shape[0]:>6}    {adh.shape[1]:>4}    722 czones x 2 years = 1444
  ADH (pre-period)      {adh_pre.shape[0]:>6}    {adh_pre.shape[1]:>4}    Pre-1990 outcomes
  Deming                {dem.shape[0]:>6}    {dem.shape[1]:>4}    Binary lottery IV
  Ashraf-Galor          {ag.shape[0]:>6}    {ag.shape[1]:>4}    ~20 non-missing IV in clean
  Nunn-Qian             {nq.shape[0]:>6}    {nq.shape[1]:>4}    Country-year panel, 10 NCIVs
  Literature Survey     {lit.shape[0]:>6}    {lit.shape[1]:>4}    140 papers surveyed
""")

print("=" * 70)
print("04_data_audit.py — DONE")
print("=" * 70)
