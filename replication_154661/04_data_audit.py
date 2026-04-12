"""Data audit: sample coverage, missingness, weights sensitivity, source mix."""
import numpy as np
import pandas as pd

from utils import load, prep_pooled_age18to24, mark_africa


def sample_by_source(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["africa", "source"]).agg(
            countries=("country_string", "nunique"),
            rows=("country_string", "size"),
        ).reset_index()
    )


def year_range(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("africa")["year"].agg(["min", "max", "mean"]).reset_index()
    )


def missingness(df: pd.DataFrame) -> pd.DataFrame:
    vars_ = [
        "paidwk_all",
        "unpaidwk_all",
        "inwork",
        "self_paid_agri",
        "self_paid_noagri",
        "employee_paid_agri",
        "employee_paid_noagri",
        "sector_manuf",
        "sector_serv",
        "highskillwk",
        "whitecollar",
        "bluecollar",
    ]
    rows = []
    for v in vars_:
        rows.append(
            {"variable": v, "missing": int(df[v].isna().sum()), "non_missing": int(df[v].notna().sum())}
        )
    return pd.DataFrame(rows)


def country_list(df: pd.DataFrame) -> pd.DataFrame:
    return df[["country_string", "countrycode", "year", "source", "africa"]].sort_values(
        ["africa", "country_string"]
    )


def wealth_quintiles_balance() -> pd.DataFrame:
    df = load("pooled_adminXwealth")
    # count cells per quintile / coverage
    wq = []
    for i in range(1, 6):
        col = f"wealth_pca_q_{i}"
        wq.append({"quintile": i, "cells": int((df[col] == 1).sum())})
    return pd.DataFrame(wq)


if __name__ == "__main__":
    df = prep_pooled_age18to24()
    df_young = df[df["age18to24"] == 1].copy()
    print("Sample by source / Africa:")
    print(sample_by_source(df_young).to_string(index=False))
    print("\nYear range by Africa flag:")
    print(year_range(df_young).round(1).to_string(index=False))
    print("\nMissingness (pooled_age18to24 filtered, young cohort):")
    print(missingness(df_young).to_string(index=False))
    print("\nCountry sample (first 20 African):")
    print(
        country_list(df_young)[country_list(df_young)["africa"] == 1]
        .head(28)
        .to_string(index=False)
    )
    print("\nWealth-quintile province cells:")
    print(wealth_quintiles_balance().to_string(index=False))

    print("\nITU coverage merged to Figure 7 sample:")
    itu = load("ITU").rename(columns={"CountryCode": "countrycode", "Year": "year"})
    print("  ITU rows:", len(itu))
    print("  unique countries:", itu["countrycode"].nunique())
