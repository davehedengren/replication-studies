"""Shared helpers for replication of Saccardo & Serra-Garcia (AER 2023)."""
from pathlib import Path
import pandas as pd
import numpy as np
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parent
PKG = ROOT.parent / "180741-V1"
DATA = PKG / "Data" / "Clean Data"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)


def load(name):
    return pd.read_stata(DATA / name, convert_categoricals=False)


def hc3(y, X):
    X = sm.add_constant(X)
    return sm.OLS(y.astype(float), X.astype(float)).fit(cov_type="HC3")


def two_prop_z(x1, n1, x2, n2):
    p1, p2 = x1 / n1, x2 / n2
    p = (x1 + x2) / (n1 + n2)
    se = np.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se
    from scipy.stats import norm
    pval = 2 * (1 - norm.cdf(abs(z)))
    return p1, p2, z, pval


COVARS = [
    "professionalsfree", "seeincentivecostly", "seequalitycostly",
    "wave2", "wave3", "professionalscloudresearch",
    "incentiveshigh", "incentiveleft", "age", "female",
]


def prep_choice():
    df = load("choice_experiments.dta")
    df = df[~((df["study"] != 1) & df["alphavaluefinal"].isna())].copy()
    main = df[(df["Highx10"] == 0) & (df["Highx100"] == 0)].copy()
    main["incentiveshigh_incentiveleft"] = main["incentiveshigh"] * main["incentiveleft"]
    return main
