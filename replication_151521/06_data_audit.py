"""Data audit for replication 151521 (Cai & Heathcote 2020)."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from utils import load_mostrecent, load_mrc_merged, load_scf_tabulated, load_figure1, OUT
import importlib
t2mod = importlib.import_module("04_table2_college")


def audit_scf():
    print("\n### SCF income tabulations")
    for year in (1989, 2016):
        tab = load_scf_tabulated(year)
        total = tab["count"].sum()
        mean_y = (tab["income"] * tab["count"]).sum() / total
        # log moments (y > 10k, matching the paper)
        mask = tab["income"] > 10_000
        t = tab[mask]
        w = t["count"].values / t["count"].sum()
        z = np.log(t["income"].values)
        mu_w = np.sum(w * z)
        v_w = np.sum(w * (z - mu_w) ** 2)
        print(f"  {year}: bins={len(tab):>5}  N_w={total:>12,.0f}  mean_y=${mean_y:>10,.0f}  "
              f"mean log y={mu_w:.4f}  var log y={v_w:.4f}")
    tab89 = load_scf_tabulated(1989); tab16 = load_scf_tabulated(2016)
    g = ((tab16["income"] * tab16["count"]).sum() / tab16["count"].sum()) / \
        ((tab89["income"] * tab89["count"]).sum() / tab89["count"].sum()) - 1
    print(f"  Mean income growth 1989->2016: {100 * g:+.1f}%  (paper: +24.7%)")


def audit_scorecard():
    print("\n### College Scorecard (mostrecent.csv)")
    ms = load_mostrecent()
    print(f"  Total rows: {len(ms):,}")
    print(f"  Unique opeid6: {ms['opeid6'].nunique():,}")
    ct = ms["control"].value_counts(dropna=False).sort_index()
    print("  control (1=public, 2=priv nonprofit, 3=priv for-profit):")
    for k, v in ct.items():
        print(f"    {k}: {v:,}")
    pd_ = ms["preddeg"].value_counts(dropna=False).sort_index()
    print("  preddeg (3=bachelor's predominant):")
    for k, v in pd_.items():
        print(f"    {k}: {v:,}")
    # After filters in main.do
    sub = ms[(ms["preddeg"] == 3) & (ms["region"] != 9) & (ms["ugds"] > 0)
             & ms["control"].isin([1, 2])]
    print(f"  After filters (preddeg=3, region!=9, ugds>0, control in {{1,2}}): {len(sub):,}")
    print(f"    missing tuitionfee_in:  {sub['tuitionfee_in'].isna().sum():,}")
    print(f"    missing npt4_pub:       {sub['npt4_pub'].isna().sum():,}")
    print(f"    missing npt4_priv:      {sub['npt4_priv'].isna().sum():,}")
    print(f"    missing costt4_a:       {sub['costt4_a'].isna().sum():,}")
    print(f"    missing SAT math mid:   {sub['satmtmid'].isna().sum():,}")
    print(f"    missing ACT comp mid:   {sub['actcmmid'].isna().sum():,}")


def audit_chetty():
    print("\n### Chetty Mobility (mrc_table2)")
    import pyreadstat
    df, _ = pyreadstat.read_dta("151521-V1/mrc_table2.dta")
    print(f"  Rows (super_opeid): {len(df):,}")
    print(f"  Rows with par_mean: {df['par_mean'].notna().sum():,}")
    print("  Tier distribution (1–8 = 4-year):")
    print(df["tier"].value_counts().sort_index().to_string())


def audit_figure1():
    print("\n### Figure 1 tuition series")
    df = load_figure1()
    print(f"  Years: {df['year'].iloc[0]} → {df['year'].iloc[-1]}  ({len(df)} rows)")
    print(df.describe().round(0).to_string())


def audit_merged_panel():
    print("\n### Merged college panel (used for Table 2 rows 10-17)")
    panel = t2mod.build_college_panel()
    need = ["t_net", "s", "i2", "eta", "ugds"]
    df = panel.dropna(subset=need)
    df = df[df["ugds"] > 0]
    print(f"  Final N colleges: {len(df):,}")
    print(f"  Weighted enrollment total: {df['ugds'].sum():,.0f}")
    print(f"  By control:")
    print(df.groupby("control").size().to_string())
    print(f"  Net tuition negative: {(df['t_net'] < 0).sum()} "
          f"({100 * (df['t_net'] < 0).mean():.1f}%)")
    print(f"  t_net stats: min=${df['t_net'].min():,.0f}, "
          f"median=${df['t_net'].median():,.0f}, max=${df['t_net'].max():,.0f}")


def main():
    print("=" * 70)
    print("06_data_audit.py — Coverage & quality checks")
    print("=" * 70)
    audit_scf()
    audit_figure1()
    audit_scorecard()
    audit_chetty()
    audit_merged_panel()
    print("\nAudit complete.")


if __name__ == "__main__":
    main()
