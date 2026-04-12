"""Utilities for replicating Burlig, Jina, Sudarshan (2025) 'The Value of Clean Water'."""
from pathlib import Path
import numpy as np
import pandas as pd
import pyreadr

BASE = Path(__file__).resolve().parent.parent
PKG = BASE / "238281-V1"
CLEAN = PKG / "data" / "clean"

NO_PAYOUT_VILLAGES = [
    "Gobindapur", "Talakheta", "Kalapathar",
    "Bhaliabadi", "Retang Colony", "Mangalajodi", "Badatota",
]


def load_merged_sales():
    """Load merged_sales_full.Rdata — one row per HH-day bottle (or zero for listing-only HH)."""
    r = pyreadr.read_r(str(CLEAN / "sales_listing_merge" / "merged_sales_full.Rdata"))
    df = r["merged"].copy()
    df["date"] = pd.to_datetime(df["date"])
    return df


def build_merged_panel(df=None):
    """Reproduce merged_panel construction from paper_exhibits.qmd.

    Steps (mirroring R code lines ~122-254):
      1. drop rows with sales_data_recieved == 9999 (blank sheets / no-delivery months)
      2. fix Offer column inconsistencies against treatment_group
      3. relabel Offer=='Discount' & Sub_Offer=='0' → 'OneTime' (Ramchandi village fix)
      4. month index: (year-2022)*12 + month
      5. drop 7 ER no-payout villages
      6. retain only delivered months (sales_data_recieved == 1 or NaN)
      7. collapse to (village, scard, month): tot_consumption = sum(litres_sold), treatment = first(Offer)

    Returns DataFrame with tot_consumption, treatment, subtreatment + treatment dummies.
    """
    if df is None:
        df = load_merged_sales()

    # 1. drop 9999 rows (NaN is kept)
    df = df[(df["sales_data_recieved"] != 9999) | df["sales_data_recieved"].isna()].copy()

    # 2. fix Offer where Offer misses / disagrees with treatment_group
    # (a) if Offer=='Control' and treatment_group is NA → set treatment_group to Offer
    mask_a = (df["Offer"] == "Control") & (df["treatment_group"].isna())
    df.loc[mask_a, "treatment_group"] = df.loc[mask_a, "Offer"]
    df = df[df["treatment_group"].notna()].copy()
    # (b) if Offer != treatment_group and Offer not in {'OneTime','Control'} → Offer = treatment_group
    mask_b = (df["Offer"] != df["treatment_group"]) & (~df["Offer"].isin(["OneTime", "Control"]))
    df.loc[mask_b, "Offer"] = df.loc[mask_b, "treatment_group"]
    # (c) relabel "Free ration" → "FreeRation"
    df["Offer"] = df["Offer"].replace({"Free ration": "FreeRation"})

    # 3. Ramchandi fix: Discount+Sub=='0' → OneTime
    mask_ram = (df["Offer"] == "Discount") & (df["Sub_Offer"].astype(str) == "0")
    df.loc[mask_ram, "Offer"] = "OneTime"

    # 4. month index
    df["month_idx"] = (df["date"].dt.year - 2022) * 12 + df["date"].dt.month

    # 5. drop no-payout villages
    df = df[~df["proper_village"].isin(NO_PAYOUT_VILLAGES)].copy()

    # 6. retain delivered months only
    df = df[(df["sales_data_recieved"] == 1) | df["sales_data_recieved"].isna()].copy()

    # 7. collapse
    df["litres_sold"] = df["litres_sold"].fillna(0)
    panel = (
        df.groupby(["proper_village", "only_digits_scard_id", "month_idx"], dropna=False)
        .agg(
            tot_consumption=("litres_sold", "sum"),
            treatment=("Offer", "first"),
            subtreatment=("Sub_Offer", "first"),
        )
        .reset_index()
        .rename(columns={"proper_village": "village"})
    )
    panel["bought"] = (panel["tot_consumption"] > 0).astype(float)

    # treatment dummies
    for trt in ["Control", "Discount", "Exchangeable", "FreeRation", "OneTime"]:
        panel[trt] = (panel["treatment"] == trt).astype(float)

    # subtreatment dummies (e.g. Discount0.1, Exchangeable0.5, ...)
    panel["sub_str"] = panel["subtreatment"].astype(str)
    panel["fulltreatment"] = panel["treatment"].astype(str) + panel["sub_str"]
    for trt in ["Discount", "Exchangeable"]:
        for sub in ["0.1", "0.5", "0.9", "1"]:
            col = f"{trt}{sub}"
            panel[col] = (panel["fulltreatment"] == col).astype(float)

    return panel


def twoway_fe_cluster(y, X, village, month):
    """Two-way fixed-effects OLS with cluster-robust SE (clustered on village).

    Uses the within transformation (demean by village & month) iteratively.
    Returns dict with coef, se, n.
    """
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    village = np.asarray(village)
    month = np.asarray(month)

    # iterative demeaning
    def demean(arr):
        arr = arr.astype(float).copy()
        if arr.ndim == 1:
            arr = arr[:, None]
        for _ in range(30):
            prev = arr.copy()
            # demean by village
            dfg = pd.DataFrame(arr)
            dfg["_v"] = village
            arr = arr - dfg.groupby("_v").transform("mean").values
            # demean by month
            dfg = pd.DataFrame(arr)
            dfg["_m"] = month
            arr = arr - dfg.groupby("_m").transform("mean").values
            if np.max(np.abs(arr - prev)) < 1e-10:
                break
        return arr

    yd = demean(y).ravel()
    Xd = demean(X)
    beta, *_ = np.linalg.lstsq(Xd, yd, rcond=None)
    resid = yd - Xd @ beta

    # degrees of freedom
    nvill = len(np.unique(village))
    nmonth = len(np.unique(month))
    k = Xd.shape[1]
    n = len(yd)
    dof = n - k - nvill - nmonth + 1  # two-way FE dof adjustment

    # CR1 cluster-robust: sum_g (X_g' e_g)(e_g' X_g) sandwich
    XtX_inv = np.linalg.inv(Xd.T @ Xd)
    G = nvill
    meat = np.zeros((k, k))
    for v in np.unique(village):
        m = (village == v)
        Xg = Xd[m]
        eg = resid[m]
        u = Xg.T @ eg
        meat += np.outer(u, u)
    # fixest cluster adjustment: G/(G-1) * (N-1)/(N-k)  — we'll use the version that matches fixest "cluster" default
    adj = (G / (G - 1)) * ((n - 1) / max(dof, 1))
    V = adj * XtX_inv @ meat @ XtX_inv
    se = np.sqrt(np.diag(V))
    return {"coef": beta, "se": se, "n": n, "vcov": V}
