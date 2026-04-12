"""04_data_audit.py — Data quality audit for data_BL and data_EL.

Checks sample coverage, treatment balance, missing data patterns,
distribution of outcomes, panel balance, and duplicates.
"""
import os
import numpy as np
import pandas as pd
from utils import load_endline, load_baseline, OUTPUT_DIR


def line(s=""):
    print(s)


def audit():
    bl = load_baseline()
    el = load_endline()

    line("=" * 70)
    line("1. COVERAGE")
    line("=" * 70)
    line(f"Baseline rows:          {len(bl):>6}")
    line(f"Baseline HH_IDs:        {bl['HH_ID'].nunique():>6}")
    line(f"Baseline communities:   {bl['COM_ID'].nunique():>6}")
    line(f"Baseline districts:     {bl['DIST_ID'].nunique():>6}")
    line(f"Endline rows:           {len(el):>6}")
    line(f"Endline HH_IDs:         {el['HH_ID'].nunique():>6}")
    line(f"Endline communities:    {el['COM_ID'].nunique():>6}")
    line(f"Endline districts:      {el['DIST_ID'].nunique():>6}")
    line(f"Endline target children (istarget35==1): {(el['istarget35']==1).sum()}")

    treat_com = el.drop_duplicates('COM_ID')['treatment_ecd'].value_counts().sort_index()
    line(f"\nCommunities by treatment (endline):")
    line(treat_com.to_string())
    line(f"Total communities: {int(treat_com.sum())} (paper reports 218)")

    line("\n" + "=" * 70)
    line("2. MAIN OUTCOME DISTRIBUTIONS (istarget35==1)")
    line("=" * 70)
    d = el[el['istarget35'] == 1]
    outcomes = ['wenttopreschl', 'enroledatschool', 'schl_repetition_uncon',
                'app_grade1', 'enrolment_zi', 'Preliteracy_zi', 'Premath_zi',
                'Executive_funct_zi', 'Socioemo_cogn_zi', 'Fine_motor_zi',
                'literacyinterest_zi', 'skills_zi']
    print(d[outcomes].describe().T[['count', 'mean', 'std', 'min', 'max']]
          .to_string(float_format='%.3f'))

    line("\n" + "=" * 70)
    line("3. BINARY-OUTCOME SANITY CHECK")
    line("=" * 70)
    for v in ['wenttopreschl', 'enroledatschool', 'schl_repetition_uncon',
              'app_grade1']:
        vals = sorted(d[v].dropna().unique())
        line(f"  {v}: values = {vals}  (expect {{0,1}})")

    line("\n" + "=" * 70)
    line("4. MISSING RATES BY TREATMENT (target children)")
    line("=" * 70)
    miss = d.groupby('treatment_ecd')[outcomes].apply(lambda g: g.isna().mean())
    print(miss.T.rename(columns={0: 'ctrl_miss', 1: 'trt_miss'})
          .to_string(float_format='%.3f'))

    # Diff in missingness (simple t of miss dummy on treatment)
    line("\nDifference in missingness (trt - ctrl, prop pts):")
    diff = (d.groupby('treatment_ecd')[outcomes].apply(lambda g: g.isna().mean())
             .T)
    diff['diff'] = diff[1] - diff[0]
    print(diff[['diff']].to_string(float_format='%.4f'))

    line("\n" + "=" * 70)
    line("5. PANEL BALANCE / ATTRITION")
    line("=" * 70)
    bl_hhs = set(bl.loc[bl['target_35'] == 1, 'HH_ID'].unique())
    el_hhs = set(el.loc[el['istarget35'] == 1, 'HH_ID'].unique())
    line(f"Baseline target HHs:     {len(bl_hhs)}")
    line(f"Endline target HHs:      {len(el_hhs)}")
    line(f"BL ∩ EL:                 {len(bl_hhs & el_hhs)}")
    line(f"BL only (attrited):      {len(bl_hhs - el_hhs)}")
    line(f"EL only (new in endline):{len(el_hhs - bl_hhs)}")

    # Attrition by treatment
    bl_tgt = bl.loc[bl['target_35'] == 1, ['HH_ID', 'treatment_ecd']].drop_duplicates('HH_ID')
    bl_tgt['attrited'] = ~bl_tgt['HH_ID'].isin(el_hhs)
    attr_by = bl_tgt.groupby('treatment_ecd')['attrited'].agg(['mean', 'sum', 'count'])
    line("\nAttrition by treatment (baseline target HH sample):")
    print(attr_by.to_string(float_format='%.4f'))

    line("\n" + "=" * 70)
    line("6. COMMUNITY-LEVEL SAMPLE SIZES")
    line("=" * 70)
    com_n = d.groupby('COM_ID').size()
    line(f"Target children per community: min={com_n.min()}, p5={com_n.quantile(.05):.0f}, "
         f"median={com_n.median():.0f}, mean={com_n.mean():.1f}, max={com_n.max()}")
    line(f"Communities with fewer than 10 target children: {(com_n<10).sum()}")

    line("\n" + "=" * 70)
    line("7. equal_weights CONSTRUCTION CHECK")
    line("=" * 70)
    # equal_weights should equal 1 / (n_community_within_district * n_obs_in_comm)?
    # Verify the product weights sum to something sensible per community.
    w_sum = d.groupby('COM_ID')['equal_weights'].sum()
    line(f"equal_weights sum by community: min={w_sum.min():.3f}, "
         f"median={w_sum.median():.3f}, max={w_sum.max():.3f}")
    # For each community, weight*n_in_comm should be ≈ constant
    w_times_n = d.groupby('COM_ID').apply(lambda g: g['equal_weights'].iloc[0] * len(g))
    line(f"weight * N per community: min={w_times_n.min():.4f}, "
         f"median={w_times_n.median():.4f}, max={w_times_n.max():.4f}")

    line("\n" + "=" * 70)
    line("8. DUPLICATE CHECKS")
    line("=" * 70)
    dup_id = el.duplicated(subset=['ID']).sum()
    dup_hhmem = el.duplicated(subset=['HH_ID', 'MemberId']).sum()
    line(f"Endline rows with duplicate ID:          {dup_id}")
    line(f"Endline rows with duplicate HH_ID/Mem:   {dup_hhmem}")
    dup_target = d.duplicated(subset=['HH_ID']).sum()
    line(f"Target-child rows with duplicate HH_ID:  {dup_target}")

    line("\n" + "=" * 70)
    line("9. BALANCE AT BASELINE (subset of vars, sanity check)")
    line("=" * 70)
    bl2 = bl.rename(columns={'wealth_index': 'wealth_index_gsem',
                             'stimul_index': 'stimul_index_gsem'})
    bl2 = bl2.rename(columns={'wealth_index_zi': 'wealth_index',
                              'stimul_index_zi': 'stimul_index'})
    check_vars = ['female', 'ageinmonth', 'zhfa_35', 'wenttopreschl',
                  'wealth_index', 'stimul_index']
    for v in check_vars:
        if v not in bl2.columns:
            continue
        sub = bl2.dropna(subset=[v, 'treatment_ecd'])
        c = sub[sub['treatment_ecd'] == 0][v].mean()
        t = sub[sub['treatment_ecd'] == 1][v].mean()
        line(f"  {v}: ctrl={c:.3f}, trt={t:.3f}, diff={t-c:+.3f}")

    with open(os.path.join(OUTPUT_DIR, 'data_audit.txt'), 'w') as f:
        f.write("Data audit completed — see console output.\n")


if __name__ == '__main__':
    audit()
