"""
01_clean.py – Load and validate dataset for Nielsen & Rehbeck (2022)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import *

def main():
    df = load_data()
    print_section("DATASET OVERVIEW")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nDtypes:\n{df.dtypes}")

    # ── Subject counts by treatment ──────────────────────────────────────
    print_section("SUBJECT COUNTS BY TREATMENT")
    lab_nc = filter_lab_nocost(df)
    lab_c = filter_lab_cost(df)
    onl = filter_online(df)

    n_lab_nc = n_subjects(lab_nc)
    n_lab_c = n_subjects(lab_c)
    n_onl = n_subjects(onl)
    n_total = n_subjects(df)

    print(f"Lab no-cost:  {n_lab_nc}  (paper claims 110)")
    print(f"Lab cost:     {n_lab_c}")
    print(f"Online:       {n_onl}")
    print(f"Total unique: {n_total}")

    # ── Panel structure: rows per subject ─────────────────────────────────
    print_section("PANEL STRUCTURE")
    rows_per_subj = df.groupby('subject').size()
    expected_rows = sum(QUESTIONS_PER_AXIOM.values())  # 17
    print(f"Expected rows per subject: {expected_rows}")
    print(f"Actual rows per subject: min={rows_per_subj.min()}, "
          f"max={rows_per_subj.max()}, median={rows_per_subj.median()}")

    # Check per axiom
    for ax in AXIOM_ORDER:
        ax_df = df[df['axiomnumber'] == ax]
        qs = ax_df.groupby('subject')['questionnumber'].nunique()
        expected = QUESTIONS_PER_AXIOM[ax]
        mismatch = (qs != expected).sum()
        print(f"  {axiom_label(ax)}: expected {expected} questions, "
              f"mismatches={mismatch}")

    # ── Uniqueness check ─────────────────────────────────────────────────
    print_section("UNIQUENESS CHECK")
    key = ['subject', 'axiomnumber', 'questionnumber']
    n_rows = len(df)
    n_unique = df[key].drop_duplicates().shape[0]
    print(f"Rows: {n_rows}, Unique (subject, axiom, question): {n_unique}")
    print(f"Duplicates: {n_rows - n_unique}")

    # ── Variable ranges ──────────────────────────────────────────────────
    print_section("VARIABLE RANGES")
    for var, expected in [('rulechoice', '{0,1}'), ('crulechoice', '{0,1}'),
                          ('cost', '{0,1}'), ('online', '{0,1}'),
                          ('violation', '{-1,0,1}'), ('cviolation', '{-1,0,1}')]:
        vals = sorted(df[var].dropna().unique())
        print(f"  {var}: {vals}  (expected {expected})")

    print(f"\n  whichreconcile unique: {sorted(df['whichreconcile'].dropna().unique())}")
    print(f"  cwhichreconcile unique: {sorted(df['cwhichreconcile'].dropna().unique())}")
    print(f"  rulerevise unique: {sorted(df['rulerevise'].dropna().unique())}")

    # CRT and test index
    print(f"\n  CRTscore range: [{df['CRTscore'].min()}, {df['CRTscore'].max()}]  (expected [0,10])")
    print(f"  totaltestindex range: [{df['totaltestindex'].min()}, {df['totaltestindex'].max()}]  (expected [0,8])")

    # ── Final profit summary ─────────────────────────────────────────────
    print_section("FINAL PROFIT (lab no-cost, per subject)")
    profit = lab_nc.drop_duplicates('subject')['finalprofit']
    print(f"  Mean:   ${profit.mean():.2f}  (paper says ~$14 avg including $7 show-up)")
    print(f"  Std:    ${profit.std():.2f}")
    print(f"  Min:    ${profit.min():.2f}")
    print(f"  Max:    ${profit.max():.2f}")
    print(f"  Median: ${profit.median():.2f}")

    # ── Logical consistency checks ───────────────────────────────────────
    print_section("LOGICAL CONSISTENCY")

    # rulechoice should be constant within subject-axiom
    rc_var = df.groupby(['subject', 'axiomnumber'])['rulechoice'].nunique()
    rc_varies = (rc_var > 1).sum()
    print(f"  rulechoice varies within subject-axiom: {rc_varies} cases")

    crc_var = df.groupby(['subject', 'axiomnumber'])['crulechoice'].nunique()
    crc_varies = (crc_var > 1).sum()
    print(f"  crulechoice varies within subject-axiom: {crc_varies} cases")

    # whichreconcile should be non-missing only when rulechoice==1 & violation==1
    wr_present = df['whichreconcile'].notna()
    should_have_wr = (df['rulechoice'] == 1) & (df['violation'] == 1)
    unexpected_wr = wr_present & ~should_have_wr
    missing_wr = ~wr_present & should_have_wr
    print(f"  whichreconcile present but shouldn't be: {unexpected_wr.sum()}")
    print(f"  whichreconcile missing but should be present: {missing_wr.sum()}")

    cwr_present = df['cwhichreconcile'].notna()
    should_have_cwr = (df['crulechoice'] == 1) & (df['cviolation'] == 1)
    unexpected_cwr = cwr_present & ~should_have_cwr
    missing_cwr = ~cwr_present & should_have_cwr
    print(f"  cwhichreconcile present but shouldn't be: {unexpected_cwr.sum()}")
    print(f"  cwhichreconcile missing but should be present: {missing_cwr.sum()}")

    # wtp only for cost treatment
    wtp_noncost = df[(df['cost'] == 0) & df['wtp'].notna()]
    print(f"  wtp present for non-cost subjects: {len(wtp_noncost)}")

    # rank only for cost treatment
    rank_noncost = df[(df['cost'] == 0) & df['rank'].notna()]
    print(f"  rank present for non-cost subjects: {len(rank_noncost)}")

    print_section("VALIDATION COMPLETE")


if __name__ == '__main__':
    main()
