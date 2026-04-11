"""Validate our computed measurement against the paper's Stata intermediates.

Loads each Stata .dta intermediate, merges with our parquet outputs on keys,
and reports the max absolute deviation. Also compares aggregate medians
against the published GFC_median_distortions.xlsx (derived from Stata).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd

from utils import INTERM, OUT

OUT_I = OUT / "intermediate"
XLSX = Path("/Users/davehedengren/code/replication-studies/136342-V1/AEJ_Macro_replication_revision")


def compare(name: str, ours: pd.DataFrame, keys: list[str], col_ours: str, col_them: str,
            stata_file: str) -> None:
    theirs = pd.read_stata(INTERM / stata_file)
    theirs = theirs[keys + [col_them]].rename(columns={col_them: "__them"})
    ours2 = ours[keys + [col_ours]].rename(columns={col_ours: "__ours"})
    merged = ours2.merge(theirs, on=keys, how="outer", indicator=True)
    both = merged[merged._merge == "both"]
    only_ours = (merged._merge == "left_only").sum()
    only_them = (merged._merge == "right_only").sum()
    a = both["__ours"].to_numpy()
    b = both["__them"].to_numpy()
    finite = np.isfinite(a) & np.isfinite(b)
    a, b = a[finite], b[finite]
    max_abs = np.abs(a - b).max() if len(a) else float("nan")
    corr = np.corrcoef(a, b)[0, 1] if len(a) else float("nan")
    print(f"  {name:>18s}: n(both)={len(both):>8,}, only_ours={only_ours}, only_them={only_them}, "
          f"max|diff|={max_abs:.3e}, corr={corr:.6f}")


def main() -> None:
    print("=== cross-checking Python vs Stata intermediates ===")

    alphas = pd.read_parquet(OUT_I / "alphas.parquet")
    compare("alphas", alphas, ["year", "importer", "sec_exp"], "alphas", "alphas", "alphas.dta")

    betas = pd.read_parquet(OUT_I / "betas.parquet")
    # stata betas also has sec_exp/exporter=="TOT" from a 64-row; keep only what both have
    compare("betas", betas, ["year", "importer", "sec_imp"], "betas", "betas", "betas.dta")

    dom = pd.read_parquet(OUT_I / "dom_gammas.parquet")
    compare("dom_gammas", dom, ["year", "importer", "exporter", "sec_imp", "sec_exp"],
            "dom_gammas", "dom_gammas", "dom_gammas.dta")

    itau = pd.read_parquet(OUT_I / "int_taus_std_1995_2011.parquet")
    compare("int_taus_std", itau, ["year", "importer", "exporter", "sec_imp", "sec_exp"],
            "int_taus_std", "int_taus_std", "int_taus_std_1995_2011.dta")

    tfp = pd.read_parquet(OUT_I / "TFP_std_1995_2011.parquet")
    compare("TFP_std", tfp, ["year", "importer", "exporter", "sec_imp", "sec_exp"],
            "TFP_std", "TFP_std", "TFP_std_1995_2011.dta")

    # --- Country-level median internal distortions (GFC 2007->2009) ---
    # The paper's "GFC_median_distortions" sheet reports for each country the
    # median internal distortion in 2009 (normalized to 2007), then expresses it
    # as percent change from 1.
    d = itau.copy()
    d = d[(d.sec_imp != d.sec_exp)]  # internal (cross-sector) distortions
    d = d[d.sec_imp != 1]  # drop agriculture? paper's xlsx uses full set; try without

    # Build a 2007-normalized internal distortion series from the raw taus,
    # matching the genChangeIntDistortions_application.do style: tau2009 / tau2007.
    # int_taus_std is relative to 1995; to get 2009/2007 ratios, divide.
    base = d[d.year == 2007][["importer", "sec_imp", "sec_exp", "int_taus_std"]].rename(
        columns={"int_taus_std": "tau2007"}
    )
    later = d[d.year == 2009][["importer", "sec_imp", "sec_exp", "int_taus_std"]].rename(
        columns={"int_taus_std": "tau2009"}
    )
    m = base.merge(later, on=["importer", "sec_imp", "sec_exp"], how="inner")
    m["ratio"] = m["tau2009"] / m["tau2007"]
    my_med = m.groupby("importer")["ratio"].median().rename("my_median_distortion").reset_index()

    # Load published values
    wb = openpyxl.load_workbook(XLSX / "GFC_median_distortions.xlsx", data_only=True)
    ws = wb["GFC median dist"]
    rows = []
    for row in ws.iter_rows(values_only=True):
        if row and row[0] and isinstance(row[0], str) and len(row[0]) == 3 and row[1] is not None:
            rows.append((row[0], row[1]))
    pub = pd.DataFrame(rows, columns=["importer", "pub_median_distortion"])
    merged = my_med.merge(pub, on="importer", how="outer")
    merged["diff"] = merged["my_median_distortion"] - merged["pub_median_distortion"]
    print("\n=== Country-level median internal distortion (2009/2007) ===")
    print(merged.to_string(index=False, float_format=lambda x: f"{x: .6f}"))
    print(f"\n  max |diff| = {merged['diff'].abs().max():.6e}")
    merged.to_csv(OUT / "gfc_median_distortions_compare.csv", index=False)


if __name__ == "__main__":
    main()
