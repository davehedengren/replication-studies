"""
01_clean.py — Load, validate, and construct the analysis sample.

Creates:
  - output/analysis_wide.parquet  (619 rows, with normalized test scores)
  - output/attendance.parquet     (313 rows, with treatment merge)
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from utils import (load_wide, load_attendance, load_school_results,
                   normalize_to_baseline, OUTPUT_DIR)

def main():
    df = load_wide()
    print(f'[load] wide: {df.shape}')
    print(f'[load] treat counts: {df["treat"].value_counts().to_dict()}')
    print(f'[load] strata nunique: {df["strata"].nunique()}')

    # Normalize test scores using the full baseline sample mean/SD (standard
    # practice for reporting SD-unit treatment effects in Table 2).
    df, m_mu, m_sd = normalize_to_baseline(df, 'm_theta_mle1', 'm_theta_mle2')
    df, h_mu, h_sd = normalize_to_baseline(df, 'h_theta_mle1', 'h_theta_mle2')
    print(f'[norm] math baseline mean={m_mu:.4f} sd={m_sd:.4f}')
    print(f'[norm] hindi baseline mean={h_mu:.4f} sd={h_sd:.4f}')

    # Endline sample sizes (for ITT)
    n_math  = df[['m_theta_mle2_z','m_theta_mle1_z','treat','strata']].dropna().shape[0]
    n_hindi = df[['h_theta_mle2_z','h_theta_mle1_z','treat','strata']].dropna().shape[0]
    print(f'[sample] math endline regression N = {n_math}  (paper: 529)')
    print(f'[sample] hindi endline regression N = {n_hindi} (paper: 533)')

    df.to_csv(os.path.join(OUTPUT_DIR, 'analysis_wide.csv'), index=False)

    # Attendance file for IV (Table 8)
    ei = load_attendance()
    print(f'[load] ms_ei: {ei.shape}')
    print(f'[att] mean att_tot = {ei["att_tot"].mean():.2f} (paper: ~50 of 86)')
    # Merge treatment status from wide
    ei = ei.merge(df[['st_id','treat','strata','m_theta_mle1_z','h_theta_mle1_z',
                      'm_theta_mle2_z','h_theta_mle2_z','st_female1','ses_index',
                      'd_sch_grade4','d_sch_grade5','d_sch_grade6','d_sch_grade7',
                      'd_sch_grade8','d_sch_grade9']],
                  on='st_id', how='left')
    print(f'[att] after merge: {ei.shape}; treat values: {ei["treat"].value_counts().to_dict()}')
    ei.to_csv(os.path.join(OUTPUT_DIR, 'attendance.csv'), index=False)

    # School results
    sc = load_school_results()
    print(f'[load] sc_results: {sc.shape}')
    sc.to_csv(os.path.join(OUTPUT_DIR, 'sc_results.csv'), index=False)

    print('[done] 01_clean.py')

if __name__ == '__main__':
    main()
