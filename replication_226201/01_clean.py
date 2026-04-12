"""Build the Table 1 regression panel from the merged IPEDS+HERD file.

Mirrors the Stata sample construction in
`empirical/codes/3 - NIH Regressions.do` (lines 170-234).
"""
from utils import build_regression_panel, OUT

wide = build_regression_panel()
print("panel shape (all unis present in 2015):", wide.shape)

needed = ["yy", "xx", "zz", "fte_wts", "sector", "geo_state"]
usable = wide.dropna(subset=needed).copy()
print("panel with non-missing yy/xx/zz/wts:", usable.shape)

# Main-spec sample: drops Dt=3 characteristics (init_*) required by spec 3.
main = usable.dropna(
    subset=["init_life_sci_uni", "init_uni_size", "init_fac_stud_ratio"]
).copy()
print("spec-3 sample (non-missing pre-period controls):", main.shape)

print("\nFlag_lib_art_col distribution (spec-3 sample):")
print(main["flag_lib_art_col"].value_counts())
print("\nSector distribution (spec-3 sample):")
print(main["sector"].value_counts())

out_file = OUT / "reg_panel.parquet"
usable.to_parquet(out_file, index=False)
print(f"\nSaved panel to {out_file}")
