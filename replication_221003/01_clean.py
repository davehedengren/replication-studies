"""01_clean.py — Load and sanity-check data for the DICIPE replication.

Prints row counts and key sample sizes to verify against the paper.
"""
import pandas as pd
from utils import load_endline, load_baseline, OUTPUT_DIR
import os

def main():
    el = load_endline()
    bl = load_baseline()

    print("=" * 60)
    print("ENDLINE (data_EL.dta)")
    print("=" * 60)
    print(f"Total rows: {len(el)}")
    print(f"Unique HH_ID: {el['HH_ID'].nunique()}")
    print(f"Unique COM_ID (communities): {el['COM_ID'].nunique()}")
    print(f"Unique DIST_ID (districts): {el['DIST_ID'].nunique()}")

    # Target child (istarget35==1): 3,765 per paper Fig 1
    target = el[el['istarget35'] == 1]
    print(f"\nTarget children (istarget35==1): {len(target)}")
    print(target.groupby('treatment_ecd').size())

    print("\n--- Main outcome non-missing among target ---")
    for v in ['wenttopreschl', 'enroledatschool', 'schl_repetition_uncon',
              'app_grade1', 'enrolment_zi']:
        print(f"  {v}: {target[v].notna().sum()}")

    print("\n" + "=" * 60)
    print("BASELINE (data_BL.dta)")
    print("=" * 60)
    print(f"Total rows: {len(bl)}")
    print(f"Unique HH_ID: {bl['HH_ID'].nunique()}")
    print(f"Unique COM_ID: {bl['COM_ID'].nunique()}")

    # Target child condition for Table 1
    tgt = bl[(bl['target_35'] == 1) & (bl['TARGET_GROUP'] != 1)]
    print(f"\nBaseline target children (target_35==1 & TARGET_GROUP!=1): {len(tgt)}")
    print(tgt.groupby('treatment_ecd').size())

    hh = bl[(bl['relationToHead'] == 1) & (bl['TARGET_GROUP'] != 1)]
    print(f"\nHH-level obs (relationToHead==1 & TARGET_GROUP!=1): {len(hh)}")

    with open(os.path.join(OUTPUT_DIR, 'clean_summary.txt'), 'w') as f:
        f.write(f"Endline rows: {len(el)}\n")
        f.write(f"Endline target children: {len(target)}\n")
        f.write(f"Baseline target children (Table 1 sample): {len(tgt)}\n")
        f.write(f"Baseline HH-level obs: {len(hh)}\n")

    print("\nExpected (paper): 3,765 endline target children; 4,687 baseline")

if __name__ == '__main__':
    main()
