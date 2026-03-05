"""
Phase 1: Load and validate all four datasets for 173341 replication.
"""
import sys
sys.path.insert(0, '.')
from replication_173341.utils import *


def main():
    print("=" * 70)
    print("173341: Vulnerability and Clientelism - Data Validation")
    print("=" * 70)

    # Load all datasets
    indiv = load_individual()
    hh = load_household()
    stacked = load_stacked()
    voting = load_voting()

    print(f"\n1. Individual data: {indiv.shape[0]} rows x {indiv.shape[1]} cols")
    print(f"   Expected: 2,990 rows")
    assert indiv.shape[0] == 2990, f"Expected 2990, got {indiv.shape[0]}"

    print(f"\n2. Household data: {hh.shape[0]} rows x {hh.shape[1]} cols")
    print(f"   Expected: 1,308 rows")
    assert hh.shape[0] == 1308, f"Expected 1308, got {hh.shape[0]}"

    print(f"\n3. Stacked data: {stacked.shape[0]} rows x {stacked.shape[1]} cols")
    print(f"   Expected: 5,334 rows")
    assert stacked.shape[0] == 5334, f"Expected 5334, got {stacked.shape[0]}"

    print(f"\n4. Voting data: {voting.shape[0]} rows x {voting.shape[1]} cols")
    print(f"   Expected: 1,794 rows")
    assert voting.shape[0] == 1794, f"Expected 1794, got {voting.shape[0]}"

    # Key variable checks
    print("\n--- Individual Data ---")
    print(f"  Treatment: {indiv['treatment'].value_counts().to_dict()}")
    print(f"  Municipalities: {indiv['mun_id'].nunique()}")
    print(f"  Clusters: {indiv['b_clusters'].nunique()}")
    print(f"  frequent_interactor mean: {indiv['frequent_interactor'].mean():.3f}")

    print("\n--- Household Data ---")
    print(f"  Treatment: {hh['treatment'].value_counts().to_dict()}")
    print(f"  Municipalities: {hh['mun_id'].nunique()}")
    print(f"  d_Overall_index non-missing: {hh['d_Overall_index'].notna().sum()}")

    print("\n--- Stacked Data ---")
    print(f"  Treatment: {stacked['treatment'].value_counts().to_dict()}")
    print(f"  year2012: {stacked['year2012'].value_counts().to_dict()}")
    yr2012 = stacked[stacked['year2012'] == 1]
    yr2013 = stacked[stacked['year2012'] == 0]
    print(f"  2012 obs: {len(yr2012)}, 2013 obs: {len(yr2013)}")
    print(f"  ask_private_stacked mean: {stacked['ask_private_stacked'].mean():.3f}")

    print("\n--- Voting Data ---")
    print(f"  Municipalities: {voting['mun_id'].nunique()}")
    print(f"  name_match==1: {(voting['name_match']==1).sum()}")
    eligible_21 = voting[(voting['name_match'] == 1) & (voting['eligible_in_2008'].notna())]
    print(f"  21-municipality sample: {len(eligible_21)}")

    broad = voting[
        ((voting['name_match'] == 1) | (voting['vp_match'] == 1) |
         (voting['pty_match'] == 1) | (voting['coalition_match'] == 1)) &
        (voting['eligible_in_2008'].notna())
    ]
    print(f"  39-municipality sample: {len(broad)}")
    print(f"  Location IDs (21-mun): {eligible_21['location_id'].nunique()}")

    # Save validated copies
    indiv.to_parquet(os.path.join(OUTPUT_DIR, 'individual.parquet'))
    hh.to_parquet(os.path.join(OUTPUT_DIR, 'household.parquet'))
    stacked.to_parquet(os.path.join(OUTPUT_DIR, 'stacked.parquet'))
    voting.to_parquet(os.path.join(OUTPUT_DIR, 'voting.parquet'))

    print("\n✓ All datasets validated and saved to output/")


if __name__ == '__main__':
    main()
