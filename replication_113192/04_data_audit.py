"""
04_data_audit.py — Data quality audit for 113192.

Checks: coverage, distributions, attrition patterns, balance on baseline,
        logical consistency, duplicates.
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from utils import load_wide, load_attendance, load_school_results, OUTPUT_DIR

def section(title):
    print(f'\n========== {title} ==========')

def main():
    df = load_wide()

    section('1. Coverage')
    print(f'Baseline obs:     {df.shape[0]}')
    print(f'Treatment:        {int((df["treat"]==1).sum())}')
    print(f'Control:          {int((df["treat"]==0).sum())}')
    print(f'Strata:           {df["strata"].nunique()}')
    print(f'Endline math (m_theta_mle2): {df["m_theta_mle2"].notna().sum()}')
    print(f'Endline hindi (h_theta_mle2): {df["h_theta_mle2"].notna().sum()}')
    att_math = 1 - df["m_theta_mle2"].notna().mean()
    att_hindi = 1 - df["h_theta_mle2"].notna().mean()
    print(f'Attrition math:   {att_math:.1%}')
    print(f'Attrition hindi:  {att_hindi:.1%}')

    section('2. Attrition by treatment status')
    for arm in [0, 1]:
        sub = df[df['treat']==arm]
        print(f'treat={arm}: baseline={len(sub)}, '
              f'math endline={sub["m_theta_mle2"].notna().sum()} '
              f'({sub["m_theta_mle2"].notna().mean():.1%}), '
              f'hindi endline={sub["h_theta_mle2"].notna().sum()} '
              f'({sub["h_theta_mle2"].notna().mean():.1%})')

    section('3. Baseline balance (mean by arm, t-test)')
    from scipy import stats
    for var in ['m_theta_mle1','h_theta_mle1','st_age1','st_female1','ses_index']:
        t = df[df['treat']==1][var].dropna()
        c = df[df['treat']==0][var].dropna()
        tstat, p = stats.ttest_ind(t, c, equal_var=False)
        print(f'{var:18s}  T={t.mean():7.3f} (sd {t.std():.3f})  '
              f'C={c.mean():7.3f} (sd {c.std():.3f})  diff={t.mean()-c.mean():+.4f}  p={p:.3f}')

    section('4. Test score distributions')
    for v in ['m_theta_mle1','m_theta_mle2','h_theta_mle1','h_theta_mle2']:
        s = df[v].dropna()
        print(f'{v}: N={len(s)}, mean={s.mean():.3f}, sd={s.std():.3f}, '
              f'min={s.min():.2f}, max={s.max():.2f}')

    section('5. Outliers (|z|>3 relative to baseline SD)')
    for v in ['m_theta_mle1','m_theta_mle2','h_theta_mle1','h_theta_mle2']:
        s = df[v].dropna()
        n_out = int((s.abs() > 3).sum())
        print(f'{v}: {n_out} outliers with |z|>3 (of {len(s)})')

    section('6. Logical consistency')
    # Baseline and endline should be correlated
    for base, end in [('m_theta_mle1','m_theta_mle2'), ('h_theta_mle1','h_theta_mle2')]:
        r = df[[base, end]].dropna().corr().iloc[0,1]
        print(f'corr({base}, {end}) = {r:.3f}')
    # Grade dummies sum to 1?
    gd = df[[c for c in df.columns if c.startswith('d_sch_grade')]].sum(axis=1)
    print(f'grade dummy row-sum: min={gd.min()} max={gd.max()} '
          f'(should all be 1) — {(gd==1).sum()}/{len(df)} are 1')

    section('7. Duplicates')
    print(f'duplicate st_id: {df["st_id"].duplicated().sum()}')

    section('8. Attendance (ms_ei)')
    ei = load_attendance()
    print(f'rows: {len(ei)}')
    print(f'att_tot: mean={ei["att_tot"].mean():.2f}, median={ei["att_tot"].median():.2f}, '
          f'sd={ei["att_tot"].std():.2f}, min={ei["att_tot"].min()}, max={ei["att_tot"].max()}')
    print(f'students with 0 days:   {(ei["att_tot"]==0).sum()}')
    print(f'students with ≥60 days: {(ei["att_tot"]>=60).sum()}')
    print(f'max possible days per paper = 86')
    # Treatment-only?
    merged = ei.merge(df[['st_id','treat']], on='st_id', how='left')
    print(f'merged treat values: {merged["treat"].value_counts().to_dict()}')

    section('9. Panel balance over 2 waves')
    n_both = df[['m_theta_mle1','m_theta_mle2']].notna().all(axis=1).sum()
    n_base_only = ((df['m_theta_mle1'].notna()) & (df['m_theta_mle2'].isna())).sum()
    print(f'Math: both waves = {n_both}, baseline only = {n_base_only}')
    n_both_h = df[['h_theta_mle1','h_theta_mle2']].notna().all(axis=1).sum()
    n_base_only_h = ((df['h_theta_mle1'].notna()) & (df['h_theta_mle2'].isna())).sum()
    print(f'Hindi: both waves = {n_both_h}, baseline only = {n_base_only_h}')

    section('10. School results (sc_results)')
    sc = load_school_results()
    print(f'rows: {len(sc)}')
    print(f'cols: {len(sc.columns)}')

    print('\n[done] 04_data_audit.py')

if __name__ == '__main__':
    main()
