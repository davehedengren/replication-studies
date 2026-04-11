"""Shared paths and helpers for the Abramitzky et al. (2023) replication.

Paper: "The Effect of Immigration Restrictions on Local Labor Markets:
Lessons from the 1920s Border Closure", AEJ: Applied 15(1), 164-191.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS

REPO = Path(__file__).resolve().parents[1]
PKG = REPO / "143182-V1" / "Replication_Package"
DATA = PKG / "Data" / "Build" / "SEA_Level"
OUT = REPO / "replication_143182"

SEA_PANEL = DATA / "sea_panel_data.dta"

SAMPLES = {"urban": 1, "mining": 2, "rural": 3}

TABLE1_OUTCOMES = ["qrtot", "nqrtot", "wtot", "nwtot", "fbold", "fbnqr"]
TABLE1_LABELS = {
    "qrtot": "Euro imm, recent arrivals",
    "nqrtot": "All unrestricted pop",
    "wtot": "Native-born white",
    "nwtot": "Native-born non-white",
    "fbold": "Euro imm, 10+ years in US",
    "fbnqr": "Imm from Western Hemisphere",
}


def load_panel(drop_outlier=True):
    df = pd.read_stata(SEA_PANEL, convert_categoricals=False)
    if drop_outlier:
        df = df[df["sea"] != 311].copy()
    return df


def build_did_panel(df, outcomes, include_1920=False):
    """Replicate the Stata prep from Table_1.do:

    * keep census years
    * declare panel (sea, year)
    * for each outcome, d<var> = (y_t - y_{t-10}) / totpop_{t-10}
    * keep year in {1910, 1930} (i.e., the 1900-1910 and 1920-1930 decades)
    * build post, QE2Xpost, QE1Xpost, FB1900Xpost, region_k*post dummies
    """
    # Stata keeps 1900/1910/1920/1930 for the lag, then restricts to
    # 1910 & 1930 after the decadal diff is computed.
    years = [1900, 1910, 1920, 1930]
    d = df[df["year"].isin(years)].copy()
    d = d.sort_values(["sea", "year"]).reset_index(drop=True)
    d["year"] = d["year"].astype(int)

    d["l_totpop"] = d.groupby("sea")["totpop"].shift(1)
    for v in outcomes:
        d[f"l_{v}"] = d.groupby("sea")[v].shift(1)
        d[f"d{v}"] = (d[v] - d[f"l_{v}"]) / d["l_totpop"]

    d["post"] = (d["year"] == 1930).astype(float)
    d["QE2Xpost"] = d["QE2"] * d["post"]
    d["QE1Xpost"] = d["QE1"] * d["post"]
    d["FB1900Xpost"] = d["FB1900"] * d["post"]

    # Stata: egen r = group(region); forval r gen region_r*post
    d["r"] = d["region"].astype("category").cat.codes + 1
    for r in sorted(d["r"].unique()):
        d[f"region_{int(r)}Xpost"] = ((d["r"] == r).astype(float)) * d["post"]

    d = d[d["year"].isin([1910, 1930])].copy()
    return d


def xtreg_fe(panel, y, treat="QE2Xpost", fb_control=True, extra_controls=None):
    """Stata equivalent of `xtreg y treat region_*Xpost FB1900Xpost, fe cluster(sea)`.

    Returns a dict with coef, se, n, groups. Uses linearmodels PanelOLS,
    which applies Stata xtreg fe small-sample corrections and cluster SEs
    at the panel (entity) level.
    """
    region_cols = sorted(
        [c for c in panel.columns if c.startswith("region_") and c.endswith("Xpost")],
        key=lambda s: int(s.split("_")[1].replace("Xpost", "")),
    )
    controls = list(region_cols)
    if fb_control:
        controls.append("FB1900Xpost")
    if extra_controls:
        controls = controls + list(extra_controls)

    keep = [y, treat, "sea", "year"] + controls
    sub = panel[keep].dropna().copy()
    sub = sub.set_index(["sea", "year"])

    # Stata xtreg, fe absorbs both sea and (implicitly) the time fixed effect
    # through post, but here we keep the regressors exactly as in Stata:
    # entity effects + [treat, region*post, FB1900*post] and NO time FE
    # (post is colinear with time, and its variance is carried by the
    # region*post and FB1900*post and treat interactions). This matches
    # the .do file, which uses xtreg ... fe with only those RHS vars.
    X = sub[[treat] + controls]
    yv = sub[y]

    # Drop regressors that are identically zero (e.g., a region dummy for
    # a region that's been filtered out).
    X = X.loc[:, (X != 0).any()]
    mod = PanelOLS(yv, X, entity_effects=True, drop_absorbed=True)
    res = mod.fit(cov_type="clustered", cluster_entity=True)

    coef = float(res.params[treat])
    se = float(res.std_errors[treat])
    n = int(res.nobs)
    # Number of entities (SEAs) actually used:
    groups = int(sub.reset_index()["sea"].nunique())
    return {"coef": coef, "se": se, "n": n, "groups": groups}


def fmt_coef(c, se, width=8, prec=4):
    return f"{c:{width}.{prec}f} ({se:.{prec}f})"
