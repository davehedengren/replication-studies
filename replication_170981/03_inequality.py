"""Reproduce the inequality/dispersion measures underpinning Figure 3 and
Appendix Table A2 (std dev, p90-p10, p75-p25 by year, plus diff-in-diff)."""
import numpy as np
import pandas as pd
from utils import build_analysis_frame

df = build_analysis_frame()

years = [2000, 2002, 2004, 2006, 2008, 2010, 2012]
sectors = {
    "restaurants": [f"lat{y}" for y in years],
    "retail": [f"lat{y}_d" for y in years],
    "food_retail": [f"lat{y}_a" for y in years],
}


def stata_percentile(x, p):
    """Stata `summarize, detail` uses a specific percentile definition;
    numpy's default 'linear' differs from Stata's. Stata's method is closer to
    `linear` in newer versions, so we stick with numpy default for this audit."""
    return np.percentile(x, p)


def dispersion_stats(x):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    return {
        "mean": x.mean(),
        "sd": x.std(ddof=1),
        "p90p10": np.percentile(x, 90) - np.percentile(x, 10),
        "p75p25": np.percentile(x, 75) - np.percentile(x, 25),
        "N": len(x),
    }


print("Figure 3 / Appendix Table A2 — dispersion by year\n")
for label, cols in sectors.items():
    print(f"=== {label} ===")
    rows = []
    for y, c in zip(years, cols):
        s = dispersion_stats(df[c])
        s["year"] = y
        rows.append(s)
    out = pd.DataFrame(rows).set_index("year")[["mean", "sd", "p90p10", "p75p25", "N"]]
    print(out.round(3))
    d_sd_0004 = out.loc[2004, "sd"] - out.loc[2000, "sd"]
    d_sd_0412 = out.loc[2012, "sd"] - out.loc[2004, "sd"]
    d_90_0004 = out.loc[2004, "p90p10"] - out.loc[2000, "p90p10"]
    d_90_0412 = out.loc[2012, "p90p10"] - out.loc[2004, "p90p10"]
    d_75_0004 = out.loc[2004, "p75p25"] - out.loc[2000, "p75p25"]
    d_75_0412 = out.loc[2012, "p75p25"] - out.loc[2004, "p75p25"]
    print(
        f"  ΔSD:    2000→2004 = {d_sd_0004:+.3f}   2004→2012 = {d_sd_0412:+.3f}   "
        f"DiD = {d_sd_0412 - d_sd_0004:+.3f}"
    )
    print(
        f"  Δp90p10: 2000→2004 = {d_90_0004:+.3f}   2004→2012 = {d_90_0412:+.3f}   "
        f"DiD = {d_90_0412 - d_90_0004:+.3f}"
    )
    print(
        f"  Δp75p25: 2000→2004 = {d_75_0004:+.3f}   2004→2012 = {d_75_0412:+.3f}   "
        f"DiD = {d_75_0412 - d_75_0004:+.3f}"
    )
    print()
