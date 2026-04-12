"""Compute the WTP figure (paper's headline INR 132 / month) via the trapezoid demand curve.

Reproduces the calc behind Section 5.2: area under the demand curve fit to the
prices arm in the 'type-A' (10-litre, 1.4 INR) villages, with the demand at
zero price taken from the free-ration arm.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import build_merged_panel, load_merged_sales

# Type-A villages = 10L bottles at unit_price=1.4 INR/litre. Identify them from raw.
raw = load_merged_sales()
type_a = sorted(raw.loc[raw["unit_price"] == 14.0, "proper_village"].unique())

panel = pd.read_parquet(Path(__file__).resolve().parent / "panel.parquet")
panel = panel[panel["village"].isin(type_a)].copy()


def mean_litres(mask):
    sub = panel[mask]
    return sub["tot_consumption"].mean()


# Mean orders by sub-treatment in price arm (control = full price 1.4)
def lit_at(price_label, treatment, sub):
    if treatment == "Control":
        m = panel["Control"] == 1
    else:
        m = panel["fulltreatment"] == f"{treatment}{sub}"
    return mean_litres(m)


# Demand curve points (price, quantity)
points = [
    (1.40, lit_at("1.40", "Control",  None)),       # control = full price
    (1.26, lit_at("1.26", "Discount", "0.1")),
    (0.70, lit_at("0.70", "Discount", "0.5")),
    (0.14, lit_at("0.14", "Discount", "0.9")),
    (0.00, mean_litres(panel["FreeRation"] == 1)),
]
print("Demand curve points (price → litres):")
for p, q in points:
    print(f"  {p:0.2f}  {q:8.3f}")


# Replicate the R wtpcalc(): linear extrapolation to choke price using the
# slope between the first two demand points (1.4 and 1.26), then trapezoid
# integration.
prices = np.array([p for p, _ in points])
qs = np.array([q for _, q in points])

# Slope from feols 0+control+discount0.1+...+free_ration is just the means.
# In R: data1$marginal_price = c(1.4, 1.26, 0.7, 0.14, 0)
# slope = (x[2]-x[1])/(y[2]-y[1])  -> note R 1-indexed, so this is (1.26-1.4)/(d10-control)
slope = (prices[1] - prices[0]) / (qs[1] - qs[0])
x_max = 1.4 + slope * (0 - qs[0])
x_choke = np.concatenate([[x_max], prices])
y_choke = np.concatenate([[0],     qs])

# Trapezoid area, ordered by quantity (the integral is over quantity but R's
# bayestestR::area_under_curve(method='trapezoid') treats x as the integration
# variable). The R code passes x = price, y = quantity — so the area is in
# units of (price * quantity) = INR per month.
order = np.argsort(x_choke)
x_s = x_choke[order]
y_s = y_choke[order]
area = float(np.trapezoid(y_s, x_s))
print(f"\nWTP (trapezoid, INR / month): {area:.2f}")
print("Paper headline: INR 132 / month")
