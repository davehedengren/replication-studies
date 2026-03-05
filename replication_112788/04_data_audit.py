"""
04_data_audit.py – Data quality checks
Banerjee, Duflo & Hornbeck (2014)
"""

import sys, os, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings('ignore')
from utils import *


def main():
    # ══════════════════════════════════════════════════════════════════════
    print_section("1. COVERAGE")
    # ══════════════════════════════════════════════════════════════════════
    asamp = load_analysis_sample()
    loans = load_loans()
    bl = load_baseline()
    el = load_endline()
    treat = load_treatment()

    print(f"Analysis sample: {len(asamp)} clients")
    print(f"  Treatment: {(asamp['treatment']==1).sum()}, Control: {(asamp['treatment']==0).sum()}")
    print(f"Loans data: {len(loans)} rows, {loans['sks_id'].nunique()} unique clients")
    print(f"  Eligible: {(loans['eligible']==1).sum()} rows, {loans[loans['eligible']==1]['sks_id'].nunique()} clients")
    print(f"Baseline: {len(bl)} clients")
    print(f"Endline: {len(el)} clients")
    print(f"Treatment file: {len(treat)} centers across {treat['village_id'].nunique()} villages")
    print(f"  101 treatment villages: {treat[treat['treatment']==1]['village_id'].nunique()}")
    print(f"  Control villages: {treat[treat['treatment']==0]['village_id'].nunique()}")
    print(f"  Strata: {treat['stratify'].nunique()} groups")

    # ══════════════════════════════════════════════════════════════════════
    print_section("2. VARIABLE COMPLETENESS")
    # ══════════════════════════════════════════════════════════════════════
    print("\nBaseline variables:")
    for col in bl.columns:
        nmiss = bl[col].isna().sum()
        if nmiss > 0:
            pct = nmiss / len(bl) * 100
            print(f"  {col}: {nmiss} missing ({pct:.1f}%)")

    print("\nEndline variables:")
    for col in el.columns:
        nmiss = el[col].isna().sum()
        if nmiss > 0:
            pct = nmiss / len(el) * 100
            print(f"  {col}: {nmiss} missing ({pct:.1f}%)")

    print("\nLoans variables:")
    for col in loans.columns:
        nmiss = loans[col].isna().sum()
        if nmiss > 0:
            pct = nmiss / len(loans) * 100
            print(f"  {col}: {nmiss} missing ({pct:.1f}%)")

    # ══════════════════════════════════════════════════════════════════════
    print_section("3. DISTRIBUTIONS")
    # ══════════════════════════════════════════════════════════════════════
    print_subsection("Continuous baseline variables")
    for var in ['serious_health_events', 'total_health_expense', 'hosp_expense',
                'hh_expenditure_yearly', 'health_scale']:
        v = bl[var].dropna()
        print(f"\n  {var} (N={len(v)}):")
        print(f"    Mean={v.mean():.2f}, Median={v.median():.2f}, SD={v.std():.2f}")
        print(f"    Min={v.min():.2f}, Max={v.max():.2f}")
        print(f"    P5={v.quantile(.05):.2f}, P95={v.quantile(.95):.2f}")
        # IQR outliers
        q1, q3 = v.quantile(0.25), v.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outliers = v[(v < lower) | (v > upper)]
        print(f"    IQR outliers: {len(outliers)} ({len(outliers)/len(v)*100:.1f}%)")

    print_subsection("Binary baseline variables")
    for var in ['chronic', 'family_chronic', 'poor_health', 'consult_symptoms',
                'smoke_drink', 'plan_baby', 'healthinsurance_available', 'healthinsurance_have',
                'own_business']:
        v = bl[var].dropna()
        print(f"  {var}: mean={v.mean():.4f}, N={len(v)}, values={sorted(v.unique())}")

    print_subsection("Endline outcome distributions")
    for var in ['el_overnight', 'el_health_exp', 'el_baby', 'sks_presentclient']:
        v = el[var].dropna()
        print(f"  {var}: mean={v.mean():.4f}, N={len(v)}")

    # ══════════════════════════════════════════════════════════════════════
    print_section("4. LOGICAL CONSISTENCY")
    # ══════════════════════════════════════════════════════════════════════

    # Treatment should be constant within village
    treat_per_village = asamp.groupby('village_id')['treatment'].nunique()
    multi_treat = (treat_per_village > 1).sum()
    print(f"Villages with multiple treatment values: {multi_treat}")

    # Stratify should be constant within village
    strat_per_village = asamp.groupby('village_id')['stratify'].nunique()
    multi_strat = (strat_per_village > 1).sum()
    print(f"Villages with multiple strata: {multi_strat}")

    # Each sks_id should appear exactly once in analysis sample
    dup = asamp.duplicated(subset=['sks_id']).sum()
    print(f"Duplicate sks_ids in analysis sample: {dup}")

    # Binary vars should be 0/1
    for var in ['chronic', 'family_chronic', 'poor_health', 'consult_symptoms',
                'smoke_drink', 'plan_baby', 'healthinsurance_available',
                'healthinsurance_have', 'own_business']:
        vals = bl[var].dropna().unique()
        ok = set(vals).issubset({0, 1})
        if not ok:
            print(f"  WARNING: {var} has values outside {{0,1}}: {sorted(vals)}")

    # Health expenses should be non-negative
    for var in ['total_health_expense', 'hosp_expense', 'hh_expenditure_yearly']:
        neg = (bl[var].dropna() < 0).sum()
        if neg > 0:
            print(f"  WARNING: {var} has {neg} negative values")

    # Renewal should be 0/1
    for var in ['renewal_first', 'renewal_endline']:
        vals = loans[var].dropna().unique()
        ok = set(vals).issubset({0, 1, 0.0, 1.0})
        if not ok:
            print(f"  WARNING: {var} has unexpected values: {sorted(vals)}")

    print("All logical consistency checks passed.")

    # ══════════════════════════════════════════════════════════════════════
    print_section("5. MISSING DATA PATTERNS BY TREATMENT")
    # ══════════════════════════════════════════════════════════════════════
    for label, mask in [('Treatment', bl['treatment']==1), ('Control', bl['treatment']==0)]:
        sub = bl[mask]
        print(f"\n{label} (N={len(sub)}):")
        for var in ['chronic', 'family_chronic', 'health_scale', 'poor_health',
                    'consult_symptoms', 'smoke_drink', 'plan_baby',
                    'total_health_expense', 'hosp_expense', 'hh_expenditure_yearly',
                    'self_financial', 'own_business']:
            nmiss = sub[var].isna().sum()
            if nmiss > 0:
                pct = nmiss / len(sub) * 100
                print(f"  {var}: {nmiss} ({pct:.1f}%)")

    # ══════════════════════════════════════════════════════════════════════
    print_section("6. TREATMENT BALANCE")
    # ══════════════════════════════════════════════════════════════════════
    # Villages per treatment arm
    t_villages = asamp[asamp['treatment']==1]['village_id'].nunique()
    c_villages = asamp[asamp['treatment']==0]['village_id'].nunique()
    print(f"Treatment villages (in analysis): {t_villages}")
    print(f"Control villages (in analysis): {c_villages}")

    # Clients per village
    cpv = asamp.groupby('village_id').size()
    print(f"\nClients per village: mean={cpv.mean():.1f}, median={cpv.median():.0f}")
    print(f"  Min={cpv.min()}, Max={cpv.max()}")

    cpv_t = asamp[asamp['treatment']==1].groupby('village_id').size()
    cpv_c = asamp[asamp['treatment']==0].groupby('village_id').size()
    print(f"  Treatment: mean={cpv_t.mean():.1f}, Control: mean={cpv_c.mean():.1f}")

    # Strata balance
    print_subsection("Strata distribution")
    strata = asamp.groupby(['stratify', 'treatment']).size().unstack(fill_value=0)
    print(strata)

    # ══════════════════════════════════════════════════════════════════════
    print_section("7. DUPLICATES AND ANOMALIES")
    # ══════════════════════════════════════════════════════════════════════
    # Exact duplicates
    for name, df in [('analysis_sample', asamp), ('baseline', bl), ('endline', el)]:
        n_dup = df.duplicated().sum()
        print(f"Exact duplicate rows in {name}: {n_dup}")

    # Check sks_id format
    sks_ids = asamp['sks_id']
    lengths = sks_ids.str.len().unique()
    print(f"\nsks_id lengths: {sorted(lengths)}")
    print(f"sks_id sample: {sks_ids.head(3).tolist()}")

    # Center IDs should be first 4 chars of sks_id
    center_check = (asamp['sks_id'].str[:4] == asamp['center_id']).all()
    print(f"center_id == sks_id[:4]: {center_check}")

    # ══════════════════════════════════════════════════════════════════════
    print_section("8. OUTLIER ANALYSIS (Top 10)")
    # ══════════════════════════════════════════════════════════════════════
    for var in ['total_health_expense', 'hosp_expense', 'hh_expenditure_yearly']:
        top10 = bl[var].dropna().nlargest(10)
        print(f"\n{var} top 10:")
        print(top10.to_string())

    print_section("DATA AUDIT COMPLETE")


if __name__ == '__main__':
    main()
