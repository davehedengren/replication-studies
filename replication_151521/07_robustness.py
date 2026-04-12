"""Robustness checks for replication 151521.

Because the paper's headline results come from a structural DSGE model that we
do not re-solve, robustness here tests (a) the SCF EMG fit and (b) the college
scorecard Table 2 moments along several margins.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from utils import load_mostrecent, load_mrc_merged, load_scf_tabulated, OUT, \
    weighted_mean, weighted_std, weighted_corr
import importlib
t2mod = importlib.import_module("04_table2_college")
scfmod = importlib.import_module("03_scf_params")


def robust_scf():
    print("\n### Robustness — SCF EMG fit")
    results = []
    for year in (1989, 2016):
        tab = load_scf_tabulated(year)
        # (i) baseline (paper) — untruncated MLE, drop y<=10k
        fit_base = scfmod.fit_emg(tab, y_min=10_000.0)
        # (ii) stricter floor y > 25k
        fit_25 = scfmod.fit_emg(tab, y_min=25_000.0)
        # (iii) higher floor y > 50k
        fit_50 = scfmod.fit_emg(tab, y_min=50_000.0)
        # (iv) keep all bins
        fit_all = scfmod.fit_emg(tab, y_min=0.0)
        results.append({
            "year": year,
            "sigma2_>10k": round(fit_base["sigma2"], 4),
            "alpha_>10k": round(fit_base["alpha"], 4),
            "sigma2_>25k": round(fit_25["sigma2"], 4),
            "alpha_>25k": round(fit_25["alpha"], 4),
            "sigma2_>50k": round(fit_50["sigma2"], 4),
            "alpha_>50k": round(fit_50["alpha"], 4),
            "sigma2_all": round(fit_all["sigma2"], 4),
            "alpha_all": round(fit_all["alpha"], 4),
        })
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    df.to_csv(OUT / "robustness_scf.csv", index=False)


def robust_table2():
    print("\n### Robustness — college-level moments (Table 2 rows 10-17)")
    panel = t2mod.build_college_panel()

    def stats(df, label, note=""):
        if len(df) == 0:
            return
        w = df["ugds"].values
        mn_t = weighted_mean(df["t_net"], w); sd_t = weighted_std(df["t_net"], w)
        mn_s = weighted_mean(df["s"], w);      sd_s = weighted_std(df["s"], w)
        mn_i = weighted_mean(df["i2"], w);     sd_i = weighted_std(df["i2"], w)
        mn_e = weighted_mean(df["eta"], w);    sd_e = weighted_std(df["eta"], w)
        c_st = weighted_corr(df["s"], df["t_net"], w)
        c_ti = weighted_corr(df["t_net"], df["i2"], w)
        c_te = weighted_corr(df["t_net"], df["eta"], w)
        c_ie = weighted_corr(df["i2"], df["eta"], w)
        rows.append({
            "spec": label, "N": len(df),
            "t_sd/mn": round(sd_t / mn_t, 3) if mn_t else np.nan,
            "s_sd/mn": round(sd_s / mn_s, 3),
            "i_sd/mn": round(sd_i / mn_i, 3),
            "e_sd/mn": round(sd_e / mn_e, 3),
            "corr_s_t": round(c_st, 3),
            "corr_t_i": round(c_ti, 3),
            "corr_t_e": round(c_te, 3),
            "corr_i_e": round(c_ie, 3),
            "note": note,
        })

    need = ["t_net", "s", "i2", "eta", "ugds"]
    base = panel.dropna(subset=need).copy()
    base = base[base["ugds"] > 0]
    rows = []
    stats(base, "baseline")
    stats(base[base["control"] == 1], "publics only")
    stats(base[base["control"] == 2], "private nonprofit only")
    stats(base[base["t_net"] >= 0], "drop negative net tuition")
    stats(base[base["ugds"] >= 1000], "drop small colleges (ugds<1000)")
    stats(base.assign(w=1).rename(columns={"ugds": "ugds_orig", "w": "ugds"}),
          "unweighted")
    # replace weight with simple ones — restore orig for downstream
    base2 = base.copy(); base2["ugds"] = 1
    stats(base2, "unweighted (w=1)")
    stats(base[base["s"] <= base["s"].quantile(0.99)], "trim top 1% sticker")
    stats(base[base["i2"] <= base["i2"].quantile(0.99)], "trim top 1% Chetty income")

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    df.to_csv(OUT / "robustness_table2.csv", index=False)
    print("\nPaper targets:  t_sd/mn=0.99  s_sd/mn=0.77  i_sd/mn=0.51  e_sd/mn=0.26")
    print("                corr_s_t=0.83  corr_t_i=0.60  corr_t_e=0.22  corr_i_e=0.59")


def main():
    print("=" * 70)
    print("07_robustness.py — Sensitivity checks")
    print("=" * 70)
    robust_scf()
    robust_table2()


if __name__ == "__main__":
    main()
