"""
Phase 3: Data audit for 173341.
"""
import sys
sys.path.insert(0, '.')
from replication_173341.utils import *


def main():
    print("=" * 70)
    print("173341: Data Audit")
    print("=" * 70)

    indiv = load_individual()
    hh = load_household()
    stacked = load_stacked()
    voting = load_voting()

    # ==========================================
    # 1. Coverage and completeness
    # ==========================================
    print("\n--- 1. Coverage and Completeness ---")

    print(f"\nIndividual data: {indiv.shape[0]} obs, {indiv.shape[1]} vars")
    print(f"  Missing by key variable:")
    for v in ['treatment', 'b_clusters', 'mun_id', 'frequent_interactor',
              'c_ask_private_pol_postcoded', 'ask_pol_private_2013',
              'rainfall_2012_std', 'rainfall_2013_std']:
        if v in indiv.columns:
            n_miss = indiv[v].isna().sum()
            pct = n_miss / len(indiv) * 100
            print(f"    {v:<40} {n_miss:5d} ({pct:.1f}%)")

    print(f"\nHousehold data: {hh.shape[0]} obs, {hh.shape[1]} vars")
    for v in ['treatment', 'b_clusters', 'mun_id', 'd_Overall_index',
              'd_Happiness', 'd_Health', 'd_Child_Food_Security',
              'a_Household_Expenditure', 'rainfall_2013_std']:
        n_miss = hh[v].isna().sum()
        pct = n_miss / len(hh) * 100
        print(f"    {v:<40} {n_miss:5d} ({pct:.1f}%)")

    print(f"\nStacked data: {stacked.shape[0]} obs, {stacked.shape[1]} vars")
    for v in stacked.columns:
        n_miss = stacked[v].isna().sum()
        if n_miss > 0:
            pct = n_miss / len(stacked) * 100
            print(f"    {v:<40} {n_miss:5d} ({pct:.1f}%)")

    print(f"\nVoting data: {voting.shape[0]} obs, {voting.shape[1]} vars")
    for v in ['incumbent_votes_section', 'location_id', 'name_match',
              'eligible_in_2008', 'tot_treat_by_section_2_21', 'mun_id']:
        n_miss = voting[v].isna().sum()
        pct = n_miss / len(voting) * 100
        print(f"    {v:<40} {n_miss:5d} ({pct:.1f}%)")

    # ==========================================
    # 2. Treatment balance
    # ==========================================
    print("\n--- 2. Treatment Balance ---")

    print("\nIndividual-level treatment assignment:")
    print(f"  Treatment: {(indiv['treatment'] == 1).sum()}")
    print(f"  Control:   {(indiv['treatment'] == 0).sum()}")
    print(f"  Ratio:     {(indiv['treatment'] == 1).sum() / (indiv['treatment'] == 0).sum():.3f}")

    print("\nHousehold-level treatment assignment:")
    print(f"  Treatment: {(hh['treatment'] == 1).sum()}")
    print(f"  Control:   {(hh['treatment'] == 0).sum()}")
    print(f"  Ratio:     {(hh['treatment'] == 1).sum() / (hh['treatment'] == 0).sum():.3f}")

    # Treatment varies within municipalities?
    print("\nTreatment within municipalities (individual data):")
    for m in sorted(indiv['mun_id'].unique())[:5]:
        sub = indiv[indiv['mun_id'] == m]
        print(f"  mun_id={m}: N={len(sub)}, treat_rate={sub['treatment'].mean():.3f}")
    print(f"  ... ({indiv['mun_id'].nunique()} total municipalities)")

    # ==========================================
    # 3. Variable distributions
    # ==========================================
    print("\n--- 3. Variable Distributions ---")

    print("\nKey outcome variables (individual data):")
    outcome_vars = ['c_ask_private_pol_postcoded', 'ask_pol_private_2013',
                    'c_pol_askrec_priv_post', 'askrec_pol_private_2013',
                    'frequent_interactor']
    for v in outcome_vars:
        if v in indiv.columns:
            s = indiv[v].dropna()
            print(f"  {v}:")
            print(f"    N={len(s)}, mean={s.mean():.3f}, min={s.min():.0f}, max={s.max():.0f}")

    print("\nVulnerability indices (household data):")
    for v in ['d_Happiness', 'd_Health', 'd_Child_Food_Security', 'd_Overall_index']:
        s = hh[v].dropna()
        print(f"  {v}:")
        print(f"    N={len(s)}, mean={s.mean():.3f}, sd={s.std():.3f}, min={s.min():.3f}, max={s.max():.3f}")

    print("\nRainfall measures (individual data):")
    for v in ['rainfall_2011_std', 'rainfall_2012_std', 'rainfall_2013_std']:
        s = indiv[v].dropna()
        print(f"  {v}:")
        print(f"    N={len(s)}, mean={s.mean():.3f}, sd={s.std():.3f}, min={s.min():.3f}, max={s.max():.3f}")

    # ==========================================
    # 4. Cluster structure
    # ==========================================
    print("\n--- 4. Cluster Structure ---")

    cl_sizes = indiv.groupby('b_clusters').size()
    print(f"  Number of clusters: {len(cl_sizes)}")
    print(f"  Cluster size: min={cl_sizes.min()}, median={cl_sizes.median():.0f}, max={cl_sizes.max()}, mean={cl_sizes.mean():.1f}")

    mun_clusters = indiv.groupby('mun_id')['b_clusters'].nunique()
    print(f"  Clusters per municipality: min={mun_clusters.min()}, max={mun_clusters.max()}, mean={mun_clusters.mean():.1f}")

    # ==========================================
    # 5. Stacked data consistency
    # ==========================================
    print("\n--- 5. Stacked Data Consistency ---")

    print(f"  Total obs: {len(stacked)}")
    print(f"  2012 obs: {(stacked['year2012'] == 1).sum()}")
    print(f"  2013 obs: {(stacked['year2012'] == 0).sum()}")
    print(f"  year2012 + year2013 == 1: {(stacked['year2012'] + stacked['year2013'] == 1).all()}")

    # Check if same individuals in both waves
    # Each obs is an individual-year. Can we identify individuals?
    # stacked has b_clusters, treatment, frequent_interactor, mun_id
    # We can't identify individuals without ind_id (not in stacked)

    # ==========================================
    # 6. Voting data structure
    # ==========================================
    print("\n--- 6. Voting Data Structure ---")

    print(f"  Total voting machines: {len(voting)}")
    print(f"  Municipalities: {voting['mun_id'].nunique()}")
    print(f"  Location IDs: {voting['location_id'].nunique()}")
    print(f"  Machines per location: min={voting.groupby('location_id').size().min()}, max={voting.groupby('location_id').size().max()}")

    # Sample definitions
    mask_21 = (voting['name_match'] == 1) & (voting['eligible_in_2008'].notna())
    mask_39 = (
        ((voting['name_match'] == 1) | (voting['vp_match'] == 1) |
         (voting['pty_match'] == 1) | (voting['coalition_match'] == 1)) &
        (voting['eligible_in_2008'].notna())
    )
    print(f"\n  21-municipality sample (name_match==1): {mask_21.sum()} machines")
    print(f"  39-municipality sample (broader match): {mask_39.sum()} machines")
    print(f"  Municipalities with name_match: {voting.loc[mask_21, 'mun_id'].nunique()}")
    print(f"  Municipalities in broad sample: {voting.loc[mask_39, 'mun_id'].nunique()}")

    # ==========================================
    # 7. Logical consistency checks
    # ==========================================
    print("\n--- 7. Logical Consistency ---")

    # Ask and receive should be <= ask
    ask_2012 = indiv['c_ask_private_pol_postcoded']
    askrec_2012 = indiv['c_pol_askrec_priv_post']
    mask = ask_2012.notna() & askrec_2012.notna()
    violations = ((askrec_2012[mask] == 1) & (ask_2012[mask] == 0)).sum()
    print(f"  Receive without asking (2012): {violations} violations")

    ask_2013 = indiv['ask_pol_private_2013']
    askrec_2013 = indiv['askrec_pol_private_2013']
    mask = ask_2013.notna() & askrec_2013.notna()
    violations = ((askrec_2013[mask] == 1) & (ask_2013[mask] == 0)).sum()
    print(f"  Receive without asking (2013): {violations} violations")

    # Voting: incumbent + challenger + blank/null should sum to ~1
    v21 = voting[mask_21].copy()
    total = v21['incumbent_votes_section'] + v21['challenger_votes_section'] + v21['blank_null']
    print(f"  Votes sum (should be ~1): mean={total.mean():.4f}, min={total.min():.4f}, max={total.max():.4f}")

    # Turnout should be <= 1
    print(f"  Turnout: mean={v21['turnout'].mean():.4f}, max={v21['turnout'].max():.4f}")

    # ==========================================
    # 8. Data version discrepancy
    # ==========================================
    print("\n--- 8. Data Version Discrepancy ---")
    print("  The shipped final data has different values for 'c_ask_private_pol_postcoded'")
    print("  (2012 postcoded private goods request) compared to the published .tex outputs.")
    print(f"  Data mean: {indiv['c_ask_private_pol_postcoded'].mean():.3f} (published: 0.195)")
    print("  This propagates to ask_private_stacked (Table 3) and all Table 5 specifications")
    print("  that use ask_private_stacked as the dependent variable.")
    print("  2013 variables and non-postcoded variables match exactly.")

    # Save audit results
    audit_results = {
        'individual_n': len(indiv),
        'household_n': len(hh),
        'stacked_n': len(stacked),
        'voting_n': len(voting),
        'municipalities': indiv['mun_id'].nunique(),
        'clusters': indiv['b_clusters'].nunique(),
        'treatment_rate_indiv': indiv['treatment'].mean(),
        'treatment_rate_hh': hh['treatment'].mean(),
        'voting_21_sample': mask_21.sum(),
        'voting_39_sample': mask_39.sum(),
        'data_version_discrepancy': True,
    }
    pd.Series(audit_results).to_csv(os.path.join(OUTPUT_DIR, 'audit_summary.csv'))
    print("\n✓ Audit complete, summary saved to output/audit_summary.csv")


if __name__ == '__main__':
    main()
