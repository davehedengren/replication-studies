"""Audit the WIOD inputs and our computed measurement series."""
from __future__ import annotations

import numpy as np
import pandas as pd

from utils import OUT, load_wiot

OUT_I = OUT / "intermediate"


def main() -> None:
    print("=== WIOD raw ===")
    wiot = load_wiot()
    print(f"rows: {len(wiot):,}")
    print(f"year range: {wiot.year.min()}-{wiot.year.max()} "
          f"({wiot.year.nunique()} years)")
    print(f"importer countries: {wiot.col_country.nunique()}")
    print(f"exporter countries: {wiot.row_country.nunique()}")
    print(f"unique col_items (incl FD): {sorted(wiot.col_item.unique())}")
    print(f"unique row_items (incl VA/GO): {sorted(wiot.row_item.unique())}")
    print(f"value range: [{wiot.value.min():.3f}, {wiot.value.max():.3e}]")
    print(f"negatives: {(wiot.value < 0).sum():,} "
          f"(inventory changes can be negative)")
    print(f"NaN values: {wiot.value.isna().sum():,}")
    print()

    # Sector by year completeness check
    comp = wiot.groupby(["year"]).apply(
        lambda g: pd.Series({
            "n_rows": len(g),
            "n_importer": g.col_country.nunique(),
            "n_exporter": g.row_country.nunique(),
            "n_col_item": g.col_item.nunique(),
            "n_row_item": g.row_item.nunique(),
        })
    )
    print(comp.to_string())

    print("\n=== alphas (shares should sum to ~1 per country-year) ===")
    alphas = pd.read_parquet(OUT_I / "alphas.parquet")
    chk = alphas.groupby(["year", "importer"])["alphas"].sum()
    print(f"  mean sum = {chk.mean():.6f}, min={chk.min():.6f}, max={chk.max():.6f}")
    print(f"  share in [0,1]: "
          f"{((alphas.alphas >= 0) & (alphas.alphas <= 1)).mean():.4f}")
    print(f"  NaN: {alphas.alphas.isna().sum()}")

    print("\n=== betas (value-added share of gross output) ===")
    betas = pd.read_parquet(OUT_I / "betas.parquet")
    print(f"  range: [{betas.betas.min():.4f}, {betas.betas.max():.4f}]")
    print(f"  share in [0,1]: "
          f"{((betas.betas >= 0) & (betas.betas <= 1)).mean():.4f}")
    print(f"  NaN: {betas.betas.isna().sum()}")
    bad = betas[(betas.betas < 0) | (betas.betas > 1)]
    print(f"  out-of-range rows: {len(bad)}")
    if len(bad):
        print(bad.head().to_string(index=False))

    print("\n=== dom_gammas (within-country input shares) ===")
    dg = pd.read_parquet(OUT_I / "dom_gammas.parquet")
    print(f"  range: [{dg.dom_gammas.min():.4f}, {dg.dom_gammas.max():.4f}]")
    # Sum by (year, importer, sec_imp) of dom_gammas: should be <= 1
    s = dg.groupby(["year", "importer", "sec_imp"])["dom_gammas"].sum()
    print(f"  sum by dest cell: mean={s.mean():.4f}, max={s.max():.4f}")
    print(f"  dom share fraction (domestic vs. total intermediates) mean = {s.mean():.3f}")

    print("\n=== int_taus_std ===")
    itau = pd.read_parquet(OUT_I / "int_taus_std_1995_2011.parquet")
    print(f"  rows: {len(itau):,}")
    print(f"  NaN: {itau.int_taus_std.isna().sum():,}")
    vals = itau.int_taus_std.dropna()
    print(f"  p01={vals.quantile(0.01):.4f} p50={vals.quantile(0.5):.4f} "
          f"p99={vals.quantile(0.99):.4f} max={vals.max():.3e}")

    print("\n=== TFP_std ===")
    tfp = pd.read_parquet(OUT_I / "TFP_std_1995_2011.parquet")
    print(f"  rows: {len(tfp):,}")
    print(f"  NaN: {tfp.TFP_std.isna().sum():,}")
    v = tfp.TFP_std.dropna()
    print(f"  p01={v.quantile(0.01):.4f} p50={v.quantile(0.5):.4f} "
          f"p99={v.quantile(0.99):.4f} max={v.max():.3e}")

    # Panel balance: do we have 17 years per (country, sec_imp) for TFP?
    panels = tfp.groupby(["importer", "sec_imp"]).year.nunique()
    print(f"  TFP panel balance: "
          f"mean years per cell={panels.mean():.1f}, "
          f"min={panels.min()}, max={panels.max()}")
    print(f"  country-sectors with full 17-year coverage: "
          f"{(panels == 17).sum()} / {len(panels)}")


if __name__ == "__main__":
    main()
