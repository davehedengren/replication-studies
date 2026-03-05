"""
01_clean.py – Data validation
Banerjee, Duflo & Hornbeck (2014)
"""

import sys, os, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings('ignore')
from utils import *


def main():
    print_section("DATA VALIDATION")

    # ── Analysis sample ───────────────────────────────────────────────────
    print_subsection("Analysis Sample")
    asamp = load_analysis_sample()
    print(f"Shape: {asamp.shape}  (expected 5366 × 7)")
    print(f"Columns: {list(asamp.columns)}")
    print(f"Treatment: {asamp['treatment'].value_counts().sort_index().to_dict()}")
    print(f"Unique villages: {asamp['village_id'].nunique()}")
    print(f"Unique strata: {asamp['stratify'].nunique()}")
    print(f"Unique centers: {asamp['center_id'].nunique()}")

    # ── Loans ─────────────────────────────────────────────────────────────
    print_subsection("Loans Data")
    loans = load_loans()
    print(f"Shape: {loans.shape}  (expected 56060)")
    print(f"Eligible == 1: {(loans['eligible'] == 1).sum()}  (expected 29860)")
    print(f"Columns: {list(loans.columns)}")
    print(f"renewal_first: mean={loans['renewal_first'].mean():.3f}, missing={loans['renewal_first'].isna().sum()}")
    print(f"renewal_endline: mean={loans['renewal_endline'].dropna().mean():.3f}, missing={loans['renewal_endline'].isna().sum()}")

    # ── Treatment ─────────────────────────────────────────────────────────
    print_subsection("Treatment Assignment")
    treat = load_treatment()
    print(f"Shape: {treat.shape}  (expected 447 centers)")
    print(f"Treatment: {treat['treatment'].value_counts().sort_index().to_dict()}")
    n_treat_villages = treat[treat['treatment'] == 1]['village_id'].nunique()
    n_control_villages = treat[treat['treatment'] == 0]['village_id'].nunique()
    print(f"Treatment villages: {n_treat_villages}  (expected 101)")
    print(f"Control villages: {n_control_villages}  (expected 100)")
    missing_rollout = treat.loc[treat['treatment'] == 1, 'rollout_date'].isna().sum()
    print(f"Treatment centers with missing rollout_date: {missing_rollout}  (expected 32)")

    # ── Baseline ──────────────────────────────────────────────────────────
    print_subsection("Clean Baseline")
    bl = load_baseline()
    print(f"Shape: {bl.shape}  (expected 5366)")
    print(f"Columns: {list(bl.columns)}")
    for v in ['serious_health_events', 'total_health_expense', 'hosp_expense',
              'hh_expenditure_yearly', 'healthinsurance_available', 'healthinsurance_have']:
        n_miss = bl[v].isna().sum()
        print(f"  {v}: mean={bl[v].mean():.3f}, missing={n_miss}")

    # ── Endline ───────────────────────────────────────────────────────────
    print_subsection("Clean Endline")
    el = load_endline()
    print(f"Shape: {el.shape}  (expected 5366)")
    print(f"Columns: {list(el.columns)}")
    for v in ['el_overnight', 'el_health_exp', 'el_baby', 'sks_presentclient']:
        n_miss = el[v].isna().sum()
        print(f"  {v}: mean={el[v].dropna().mean():.3f}, missing={n_miss}")

    # ── Attrition sample ──────────────────────────────────────────────────
    print_subsection("Attrition Sample")
    att = load_attrition()
    print(f"Shape: {att.shape}")
    # Replicate attrition calculation from do-file
    att['attrition'] = np.nan
    mask_att = (att['merge_base_end'] == 1) & (att['merge_surveys_admin'] == 3) & (att['merge_treatment'] == 3)
    mask_no_att = (att['merge_base_end'] == 3) & (att['merge_surveys_admin'] == 3) & (att['merge_treatment'] == 3)
    att.loc[mask_att, 'attrition'] = 1
    att.loc[mask_no_att, 'attrition'] = 0
    att_valid = att.dropna(subset=['attrition'])
    print(f"Attrition-valid N: {len(att_valid)}  (expected 5436)")
    print(f"Attrition rate: {att_valid['attrition'].mean():.6f}  (expected 0.012877)")

    # ── Raw data crosswalks ───────────────────────────────────────────────
    print_subsection("Raw Data Crosswalks")
    bcw = load_baseline_crosswalk()
    print(f"Baseline crosswalk: {len(bcw)} sks_ids  (expected 5680)")
    ecw = load_endline_crosswalk()
    print(f"Endline crosswalk: {len(ecw)} sks_ids  (expected 5643)")

    # ── Merge validation ──────────────────────────────────────────────────
    print_subsection("Merge Validation")
    # All analysis sample sks_ids should be in baseline and endline
    asamp_ids = set(asamp['sks_id'])
    bl_ids = set(bl['sks_id'])
    el_ids = set(el['sks_id'])
    print(f"Analysis IDs in baseline: {len(asamp_ids & bl_ids)}/{len(asamp_ids)}")
    print(f"Analysis IDs in endline: {len(asamp_ids & el_ids)}/{len(asamp_ids)}")

    # Loans merge
    loans_ids = set(loans[loans['eligible'] == 1]['sks_id'])
    print(f"Analysis IDs in eligible loans: {len(asamp_ids & loans_ids)}/{len(asamp_ids)}")

    print_section("DATA VALIDATION COMPLETE")


if __name__ == '__main__':
    main()
