"""Robustness checks for the headline ITT estimates (Table 1, col 3 — orders in litres).

8 specifications. The headline coefficients we focus on are:
  - Discount  (Prices arm pooled) — paper 95.93
  - Exchangeable (Exchangeable arm pooled) — paper 290.79
  - FreeRation — paper 269.93
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import twoway_fe_cluster, build_merged_panel, load_merged_sales

PANEL = Path(__file__).resolve().parent / "panel.parquet"
panel0 = pd.read_parquet(PANEL)

POOLED = ["Discount", "Exchangeable", "FreeRation", "OneTime"]


def fit(panel, y_col="tot_consumption"):
    # drop columns that are all zero (e.g. when an arm is excluded)
    cols = [c for c in POOLED if panel[c].sum() > 0]
    res = twoway_fe_cluster(
        panel[y_col].values,
        panel[cols].values,
        panel["village"].values,
        panel["month_idx"].values,
    )
    res["cols"] = cols
    return res


def row(label, res):
    cmap = dict(zip(res["cols"], res["coef"]))
    smap = dict(zip(res["cols"], res["se"]))
    return {
        "spec": label,
        "n": res["n"],
        "Discount":     cmap.get("Discount",     np.nan),
        "se_Discount":  smap.get("Discount",     np.nan),
        "Exchange":     cmap.get("Exchangeable", np.nan),
        "se_Exchange":  smap.get("Exchangeable", np.nan),
        "FreeRation":   cmap.get("FreeRation",   np.nan),
        "se_FreeRation": smap.get("FreeRation",  np.nan),
        "OneTime":      cmap.get("OneTime",      np.nan),
        "se_OneTime":   smap.get("OneTime",      np.nan),
    }


rows = []

# (1) baseline (paper Table 1 col 3)
rows.append(row("R1: baseline (paper)", fit(panel0)))

# (2) winsorise tot_consumption at 99%
p = panel0.copy()
cap = p["tot_consumption"].quantile(0.99)
p["tot_consumption"] = np.minimum(p["tot_consumption"], cap)
rows.append(row(f"R2: winsorise @ p99 ({cap:.0f}L)", fit(p)))

# (3) drop top 1% HH-months
p = panel0.copy()
cap = p["tot_consumption"].quantile(0.99)
p = p[p["tot_consumption"] <= cap]
rows.append(row("R3: drop top 1%", fit(p)))

# (4) drop OneTime arm
p = panel0[panel0["treatment"] != "OneTime"].copy()
rows.append(row("R4: drop OneTime arm", fit(p)))

# (5) drop type-B (20L) villages — focus on the 1.4 INR/L villages only
raw = load_merged_sales()
type_a = sorted(raw.loc[raw["unit_price"] == 14.0, "proper_village"].unique())
p = panel0[panel0["village"].isin(type_a)].copy()
rows.append(row(f"R5: 10L villages only (N_v={p['village'].nunique()})", fit(p)))

# (6) restrict to 'stable' months 2..6 of treatment per HH (drop ramp-up + tail)
# we don't have month-of-treatment, so use calendar months 9..15 (2022-09 .. 2023-03)
p = panel0[(panel0["month_idx"] >= 9) & (panel0["month_idx"] <= 15)].copy()
rows.append(row(f"R6: months 9..15 only", fit(p)))

# (7) drop the smallest villages (<200 HH)
hh_per_v = panel0.groupby("village")["only_digits_scard_id"].nunique()
keep = hh_per_v[hh_per_v >= 200].index
p = panel0[panel0["village"].isin(keep)].copy()
rows.append(row(f"R7: drop villages with <200 HH ({len(keep)} v)", fit(p)))

# (8) only households with >=3 months observed
moh = panel0.groupby(["village", "only_digits_scard_id"])["month_idx"].transform("nunique")
p = panel0[moh >= 3].copy()
rows.append(row("R8: HH observed ≥3 months", fit(p)))

# (9) "any orders" outcome instead of litres
rows.append(row("R9: y = any-orders binary", fit(panel0, y_col="bought")))

# (10) village FE only (no month FE)
res = twoway_fe_cluster(
    panel0["tot_consumption"].values,
    panel0[POOLED].values,
    panel0["village"].values,
    np.zeros(len(panel0)),
)
res["cols"] = POOLED
rows.append(row("R10: village FE only", res))

# (11) alt: treat tot_consumption as log(1+x)
p = panel0.copy()
p["log_tot"] = np.log1p(p["tot_consumption"])
rows.append(row("R11: y = log(1+litres)", fit(p, y_col="log_tot")))

df = pd.DataFrame(rows)
df.to_csv(Path(__file__).resolve().parent / "robustness.csv", index=False)
with pd.option_context("display.max_columns", None, "display.width", 200,
                       "display.float_format", "{:8.3f}".format):
    print(df.to_string(index=False))
