"""Translate genFrictions.do to Python.

Computes:
  - alphas: sectoral final demand share per (year, country)
  - gammas: bilateral intermediate input shares
  - dom_gammas: gammas restricted to importer==exporter
  - betas: value-added/gross-output share per (year, country, sector)
  - taus (internal): distortion on own country intermediate links, following
    tau_{ijik} = (gamma_ikik/gamma_ijik)^(1/theta) * (alpha_ij/alpha_ik)^(1/(1-sigma))
  - TFP: relative to agriculture (sector 1)
  - int_taus_std_1995_2011, TFP_std_1995_2011: normalized to 1995 levels

Writes parquet files to replication_136342/intermediate/.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from utils import (
    DROP_FINAL_ROWS,
    FINAL_DEMAND_COLS,
    INTERM,
    OUT,
    SIGMA,
    THETA,
    load_wiot,
)

OUT_I = OUT / "intermediate"
OUT_I.mkdir(exist_ok=True)


def step(msg: str, start: float) -> None:
    print(f"[{time.time() - start:6.1f}s] {msg}")


def main() -> None:
    t0 = time.time()
    step("loading wiot_full.dta ...", t0)
    wiot = load_wiot()
    wiot.rename(columns={"col_country": "importer", "row_country": "exporter"}, inplace=True)
    step(f"loaded {len(wiot):,} rows", t0)

    # ---------------- ALPHAS ----------------
    # final demand columns 37,38,39,41 (C + G + I + changes in inventories)
    fd = wiot[wiot.col_item.isin(FINAL_DEMAND_COLS) & ~wiot.row_item.isin(DROP_FINAL_ROWS)].copy()
    fd.rename(columns={"row_item": "sec_exp", "col_item": "sec_imp"}, inplace=True)
    fin_sec_demand = (
        fd.groupby(["year", "sec_exp", "importer"], as_index=False)["value"].sum()
          .rename(columns={"value": "fin_sec_demand"})
    )
    fin_demand = (
        fin_sec_demand.groupby(["year", "importer"], as_index=False)["fin_sec_demand"].sum()
                      .rename(columns={"fin_sec_demand": "fin_demand"})
    )
    alphas = fin_sec_demand.merge(fin_demand, on=["year", "importer"], how="left")
    alphas["alphas"] = alphas["fin_sec_demand"] / alphas["fin_demand"]
    alphas.to_parquet(OUT_I / "alphas.parquet")
    step(f"alphas: {len(alphas):,} rows", t0)

    # ---------------- GAMMAS ----------------
    im = wiot[(wiot.col_item >= 1) & (wiot.col_item <= 35) & (wiot.row_item < 60)].copy()
    im.rename(columns={"row_item": "sec_exp", "col_item": "sec_imp"}, inplace=True)
    total_cons = (
        im.groupby(["importer", "year", "sec_imp"], as_index=False)["value"].sum()
          .rename(columns={"value": "total_cons"})
    )
    xbilat = im.merge(total_cons, on=["importer", "year", "sec_imp"], how="left")
    xbilat["gammas"] = xbilat["value"] / xbilat["total_cons"]
    # keep a compact version of gammas
    gammas = xbilat[["year", "importer", "exporter", "sec_imp", "sec_exp", "gammas"]]
    step(f"gammas: {len(gammas):,} rows", t0)

    dom_gammas = gammas[gammas.importer == gammas.exporter].rename(columns={"gammas": "dom_gammas"}).copy()
    dom_gammas.to_parquet(OUT_I / "dom_gammas.parquet")
    step(f"dom_gammas: {len(dom_gammas):,} rows", t0)

    # ---------------- BETAS ----------------
    va = wiot[(~wiot.col_item.isin([37, 38, 39, 40, 41, 42])) & (wiot.row_item == 64)].copy()
    va.rename(columns={"row_item": "sec_exp", "col_item": "sec_imp", "value": "val_add"}, inplace=True)

    go_src = wiot[(~wiot.col_item.isin([37, 38, 39, 40, 41, 42]))
                  & (wiot.row_item.isin([60, 64]))].copy()
    gross_output = (
        go_src.groupby(["year", "importer", "col_item"], as_index=False)["value"].sum()
              .rename(columns={"col_item": "sec_imp", "value": "gross_output"})
    )
    betas = gross_output.merge(va, on=["year", "importer", "sec_imp"], how="left")
    betas["betas"] = betas["val_add"] / betas["gross_output"]
    betas.to_parquet(OUT_I / "betas.parquet")
    gross_output.to_parquet(OUT_I / "gross_output.parquet")
    step(f"betas: {len(betas):,} rows", t0)

    # ---------------- INTERNAL TAUS ----------------
    # gammas_ikik: domestic gammas with sec_exp==sec_imp (per country, sector, year)
    gammas_ikik = dom_gammas[dom_gammas.sec_exp == dom_gammas.sec_imp][
        ["year", "importer", "sec_exp", "dom_gammas"]
    ].rename(columns={"dom_gammas": "gammas_ikik"}).copy()

    # join dom_gammas with gammas_ikik on (year, sec_exp, importer) -- i.e. sec_exp acts as "k"
    gterm = dom_gammas.merge(gammas_ikik, on=["year", "sec_exp", "importer"], how="left")

    # alphas_ij: alphas indexed by sec_imp ("j")
    alphas_ij = alphas[["year", "importer", "sec_exp", "alphas"]].rename(
        columns={"sec_exp": "sec_imp", "alphas": "alphas_ij"}
    )
    # alphas_ik: alphas indexed by sec_exp ("k")
    alphas_ik = alphas[["year", "importer", "sec_exp", "alphas"]].rename(columns={"alphas": "alphas_ik"})

    taus = gterm.merge(alphas_ij, on=["year", "importer", "sec_imp"], how="left")
    taus = taus.merge(alphas_ik, on=["year", "importer", "sec_exp"], how="left")
    taus["taus"] = (
        (taus["gammas_ikik"] / taus["dom_gammas"]) ** (1 / THETA)
        * (taus["alphas_ij"] / taus["alphas_ik"]) ** (1 / (1 - SIGMA))
    )
    taus["exporter"] = taus["importer"]  # internal only
    step(f"taus: {len(taus):,} rows", t0)

    # Normalize to 1995 levels (matching int_taus_std_1995_2011)
    ini, fin_ = 1995, 2011
    t95 = taus[taus.year == ini][["importer", "exporter", "sec_imp", "sec_exp", "taus"]].rename(
        columns={"taus": "taus_ini"}
    )
    int_taus_std = taus.merge(t95, on=["importer", "exporter", "sec_imp", "sec_exp"], how="left")
    int_taus_std = int_taus_std[int_taus_std.importer != "RoW"]
    int_taus_std["int_taus_std"] = int_taus_std["taus"] / int_taus_std["taus_ini"]
    int_taus_std = int_taus_std[(int_taus_std.year >= ini) & (int_taus_std.year <= fin_)]
    int_taus_std = int_taus_std[["year", "importer", "exporter", "sec_imp", "sec_exp", "int_taus_std"]]
    int_taus_std.to_parquet(OUT_I / "int_taus_std_1995_2011.parquet")
    step(f"int_taus_std: {len(int_taus_std):,} rows", t0)

    # ---------------- TFPs ----------------
    # alphas for sec_exp==1 (agriculture)
    alphas_agr = alphas[alphas.sec_exp == 1][["year", "importer", "alphas"]].rename(
        columns={"alphas": "alphas_agr"}
    )
    # betas for sec_imp==1 (agriculture)
    betas_agr = betas[betas.sec_imp == 1][["year", "importer", "betas"]].rename(
        columns={"betas": "betas_agr"}
    )
    # dom_gammas for sec_exp==sec_imp==1
    dg_agr = dom_gammas[(dom_gammas.sec_exp == 1) & (dom_gammas.sec_imp == 1)][
        ["year", "importer", "dom_gammas"]
    ].rename(columns={"dom_gammas": "dom_gammas_agr"})

    denom = dg_agr.merge(betas_agr, on=["year", "importer"], how="left").merge(
        alphas_agr, on=["year", "importer"], how="left"
    )

    # numerator: dom_gammas for sec_imp==sec_exp, joined with betas (by sec_imp) and alphas (by sec_exp)
    num = dom_gammas[dom_gammas.sec_exp == dom_gammas.sec_imp].copy()
    num = num.merge(
        betas[["year", "importer", "sec_imp", "betas"]], on=["year", "importer", "sec_imp"], how="left"
    )
    num = num.merge(
        alphas[["year", "importer", "sec_exp", "alphas"]], on=["year", "importer", "sec_exp"], how="left"
    )
    num = num.merge(denom, on=["year", "importer"], how="left")
    num["A_ik"] = (num["alphas_agr"] ** (-num["betas_agr"] / (1 - SIGMA))) * (
        num["dom_gammas_agr"] ** (1 / THETA)
    )
    num["TFP"] = (
        (num["alphas"] ** (-num["betas"] / (1 - SIGMA))) * (num["dom_gammas"] ** (1 / THETA))
    ) / num["A_ik"]

    tfp95 = num[num.year == ini][["importer", "exporter", "sec_imp", "sec_exp", "TFP"]].rename(
        columns={"TFP": "TFP_ini"}
    )
    tfp_std = num.merge(tfp95, on=["importer", "exporter", "sec_imp", "sec_exp"], how="left")
    tfp_std["TFP_std"] = tfp_std["TFP"] / tfp_std["TFP_ini"]
    tfp_std = tfp_std[(tfp_std.year >= ini) & (tfp_std.year <= fin_)]
    tfp_std = tfp_std[["year", "importer", "exporter", "sec_imp", "sec_exp", "TFP_std"]]
    tfp_std.to_parquet(OUT_I / "TFP_std_1995_2011.parquet")
    step(f"TFP_std: {len(tfp_std):,} rows", t0)

    print("\n=== summary ===")
    for name, df in [("alphas", alphas), ("betas", betas), ("gammas", gammas),
                     ("dom_gammas", dom_gammas), ("int_taus_std", int_taus_std),
                     ("TFP_std", tfp_std)]:
        print(f"  {name:>14s}: {len(df):>10,} rows")


if __name__ == "__main__":
    main()
