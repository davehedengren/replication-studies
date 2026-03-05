"""
04_data_audit.py – Data quality checks
Nielsen & Rehbeck (2022) "When Choices Are Mistakes"
"""

import sys, os, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings('ignore')
from utils import *


def main():
    df = load_data()

    # ══════════════════════════════════════════════════════════════════════
    print_section("1. COVERAGE")
    # ══════════════════════════════════════════════════════════════════════
    print(f"Total observations: {len(df)}")
    print(f"Total unique subjects: {df['subject'].nunique()}")

    for label, filt in [('Lab no-cost', (df['online']==0) & (df['cost']==0)),
                         ('Lab cost', (df['online']==0) & (df['cost']==1)),
                         ('Online', df['online']==1)]:
        sub = df[filt]
        print(f"  {label}: {sub['subject'].nunique()} subjects, {len(sub)} rows")

    # Rows per subject
    rps = df.groupby('subject').size()
    print(f"\nRows per subject: all have {rps.unique()} (expected 24 = 6 axioms × 4 questions)")

    # Variables present
    var_completeness = df.notna().sum() / len(df) * 100
    print(f"\nVariable completeness (% non-null):")
    for col in df.columns:
        print(f"  {col}: {var_completeness[col]:.1f}%")

    # ══════════════════════════════════════════════════════════════════════
    print_section("2. DISTRIBUTIONS")
    # ══════════════════════════════════════════════════════════════════════
    print("\nSummary statistics:")
    print(df.describe().T.to_string())

    # Key variable distributions
    print_subsection("CRTscore distribution")
    crt = df.drop_duplicates('subject')['CRTscore']
    print(crt.value_counts().sort_index())
    print(f"Mean={crt.mean():.2f}, Median={crt.median():.1f}")

    print_subsection("totaltestindex distribution")
    tti = df.drop_duplicates('subject')['totaltestindex']
    print(tti.value_counts().sort_index())

    print_subsection("finalprofit distribution")
    profit = df.drop_duplicates('subject')['finalprofit']
    print(f"Mean=${profit.mean():.2f}, Std=${profit.std():.2f}")
    print(f"Min=${profit.min():.2f}, Max=${profit.max():.2f}")
    print(f"P5=${profit.quantile(.05):.2f}, P95=${profit.quantile(.95):.2f}")

    # Outlier check: IQR
    q1, q3 = profit.quantile(0.25), profit.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outliers = profit[(profit < lower) | (profit > upper)]
    print(f"IQR outliers (below {lower:.0f} or above {upper:.0f}): {len(outliers)} subjects")

    print_subsection("firstclick distribution (online only)")
    fc = df[df['online'] == 1]['firstclick'].dropna()
    print(f"N={len(fc)}, Mean={fc.mean():.1f}s, Median={fc.median():.1f}s")
    print(f"Min={fc.min():.1f}s, Max={fc.max():.1f}s")
    print(f"P5={fc.quantile(.05):.1f}s, P95={fc.quantile(.95):.1f}s")
    # Extreme values
    extreme_fc = fc[fc > fc.quantile(.99)]
    print(f"Top 1% (>{fc.quantile(.99):.0f}s): {len(extreme_fc)} obs")

    # ══════════════════════════════════════════════════════════════════════
    print_section("3. LOGICAL CONSISTENCY")
    # ══════════════════════════════════════════════════════════════════════

    # rulechoice constant within subject-axiom
    rc_check = df.groupby(['subject', 'axiomnumber'])['rulechoice'].nunique()
    print(f"rulechoice varies within subject-axiom: {(rc_check > 1).sum()} cases")

    # crulechoice constant within subject-axiom
    crc_check = df.groupby(['subject', 'axiomnumber'])['crulechoice'].nunique()
    print(f"crulechoice varies within subject-axiom: {(crc_check > 1).sum()} cases")

    # whichreconcile only when selected AND violated
    wr_ok = df['whichreconcile'].notna()
    should_wr = (df['rulechoice'] == 1) & (df['violation'] == 1)
    print(f"\nwhichreconcile present: {wr_ok.sum()}")
    print(f"  Expected (selected & violated): {should_wr.sum()}")
    print(f"  Present but shouldn't be: {(wr_ok & ~should_wr).sum()}")
    print(f"  Missing but should be: {(~wr_ok & should_wr).sum()}")

    # cwhichreconcile
    cwr_ok = df['cwhichreconcile'].notna()
    should_cwr = (df['crulechoice'] == 1) & (df['cviolation'] == 1)
    print(f"\ncwhichreconcile present: {cwr_ok.sum()}")
    print(f"  Expected (selected & violated): {should_cwr.sum()}")
    print(f"  Present but shouldn't be: {(cwr_ok & ~should_cwr).sum()}")
    print(f"  Missing but should be: {(~cwr_ok & should_cwr).sum()}")
    if (~cwr_ok & should_cwr).sum() > 0:
        missing_cwr = df[~cwr_ok & should_cwr]
        print(f"  Missing cases:")
        print(missing_cwr[['subject', 'axiomnumber', 'questionnumber', 'crulechoice', 'cviolation']].to_string())

    # rulerevise only when both selected
    rr_ok = df['rulerevise'].notna()
    both_sel = (df['rulechoice'] == 1) & (df['crulechoice'] == 1) & (df['questionnumber'] == 1)
    print(f"\nrulerevise present: {rr_ok.sum()}")
    print(f"  Both selected (q1 only): {both_sel.sum()}")
    # Note: rulerevise might also be present for non-q1 rows due to reshape
    rr_q1 = df[df['questionnumber'] == 1]
    rr_q1_ok = rr_q1['rulerevise'].notna()
    both_q1 = (rr_q1['rulechoice'] == 1) & (rr_q1['crulechoice'] == 1)
    print(f"  rulerevise present (q1): {rr_q1_ok.sum()}")
    print(f"  Both selected (q1): {both_q1.sum()}")
    print(f"  rulerevise present but no both: {(rr_q1_ok & ~both_q1).sum()}")

    # Violation should be in {-1, 0, 1}
    print(f"\nviolation values: {sorted(df['violation'].dropna().unique())}")
    print(f"cviolation values: {sorted(df['cviolation'].dropna().unique())}")

    # vacuous violations (-1) should only appear for certain axiom-question combos
    vac = df[df['violation'] == -1]
    print(f"\nVacuous violations (=-1): {len(vac)}")
    print(f"  By axiom:")
    print(vac.groupby('axiomnumber').size())

    cvac = df[df['cviolation'] == -1]
    print(f"\nc-Vacuous violations (=-1): {len(cvac)}")
    print(f"  By axiom:")
    print(cvac.groupby('axiomnumber').size())

    # ══════════════════════════════════════════════════════════════════════
    print_section("4. MISSING DATA PATTERNS")
    # ══════════════════════════════════════════════════════════════════════

    # Missing by treatment
    for label, filt in [('Lab no-cost', (df['online']==0) & (df['cost']==0)),
                         ('Lab cost', (df['online']==0) & (df['cost']==1)),
                         ('Online', df['online']==1)]:
        sub = df[filt]
        print(f"\n{label}:")
        for col in ['rulechoice', 'crulechoice', 'violation', 'cviolation',
                     'whichreconcile', 'cwhichreconcile', 'firstclick', 'wtp', 'rank']:
            nmiss = sub[col].isna().sum()
            pct = nmiss / len(sub) * 100
            if nmiss > 0:
                print(f"  {col}: {nmiss} missing ({pct:.1f}%)")

    # ══════════════════════════════════════════════════════════════════════
    print_section("5. DUPLICATES AND CODING ANOMALIES")
    # ══════════════════════════════════════════════════════════════════════

    # Exact duplicate rows
    n_dup = df.duplicated().sum()
    print(f"Exact duplicate rows: {n_dup}")

    # Subject-axiom-question uniqueness
    key = ['subject', 'axiomnumber', 'questionnumber']
    n_dup_key = df.duplicated(subset=key).sum()
    print(f"Duplicate keys (subject, axiom, question): {n_dup_key}")

    # Check subject IDs for anomalies
    subjects = df.drop_duplicates('subject')['subject'].values
    print(f"\nSubject ID range: {subjects.min():.0f} to {subjects.max():.0f}")
    print(f"Number of unique subjects: {len(subjects)}")

    # Check if any subjects appear in multiple treatment groups
    subj_treatments = df.groupby('subject').agg({'online': 'nunique', 'cost': 'nunique'})
    multi_online = (subj_treatments['online'] > 1).sum()
    multi_cost = (subj_treatments['cost'] > 1).sum()
    print(f"Subjects in multiple online groups: {multi_online}")
    print(f"Subjects in multiple cost groups: {multi_cost}")

    # ══════════════════════════════════════════════════════════════════════
    print_section("6. PANEL BALANCE")
    # ══════════════════════════════════════════════════════════════════════

    # Each subject should have exactly 6 axioms × 4 questions = 24 rows
    rps = df.groupby('subject').size()
    unbalanced = rps[rps != 24]
    print(f"Subjects with != 24 rows: {len(unbalanced)}")

    # Check question numbers per axiom
    for ax in AXIOM_ORDER:
        ax_df = df[df['axiomnumber'] == ax]
        qs = sorted(ax_df['questionnumber'].unique())
        print(f"  {axiom_label(ax)}: questions {qs}")

    # Count of valid (non-vacuous) violations per axiom
    print(f"\nNon-vacuous violation counts per axiom (violation in [0,1]):")
    for ax in AXIOM_ORDER:
        ax_df = df[df['axiomnumber'] == ax]
        valid_v = ax_df[ax_df['violation'].isin([0, 1])]
        n_valid = len(valid_v)
        n_per_subj = valid_v.groupby('subject').size().unique()
        print(f"  {axiom_label(ax)}: {n_valid} obs, questions per subject: {n_per_subj}")

    # ══════════════════════════════════════════════════════════════════════
    print_section("7. TREATMENT BALANCE CHECKS")
    # ══════════════════════════════════════════════════════════════════════

    # Compare characteristics across treatments
    subj_df = df.drop_duplicates('subject')
    for var in ['CRTscore', 'totaltestindex']:
        if var in subj_df.columns:
            for label, filt in [('Lab no-cost', (subj_df['online']==0) & (subj_df['cost']==0)),
                                 ('Lab cost', (subj_df['online']==0) & (subj_df['cost']==1)),
                                 ('Online', subj_df['online']==1)]:
                vals = subj_df[filt][var].dropna()
                if len(vals) > 0:
                    print(f"  {var} ({label}): mean={vals.mean():.2f}, std={vals.std():.2f}, n={len(vals)}")
            print()

    print_section("DATA AUDIT COMPLETE")


if __name__ == '__main__':
    main()
