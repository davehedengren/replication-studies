from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PKG = ROOT / "215441-V1"
CLEAN = PKG / "input" / "clean"
RAW = PKG / "input" / "raw"
OUT = ROOT / "replication_215441"

BMA_COVARS = [
    "SE", "DIDorRKD", "RDD", "MacroTreatment", "PBD",
    "RRxBaselinePBD", "RRxBaselineRR",
    "PBDxBaselinePBD", "PBDxBaselineRR",
    "Admin", "NonemploymentAsOutcome", "HazardModel",
    "YearsTo2023", "RelativeUnemp", "USA", "TaxWedge", "ImpactFactorZ",
]


def load_long(drop_hunt=False):
    """Cleaned long-format review estimates.

    drop_hunt=False mirrors 04_parametric_ak.R (Hunt kept for the baseline).
    drop_hunt=True mirrors 02_model_averaging.R, dropping Hunt's -3.3 outlier.
    """
    df = pd.read_csv(CLEAN / "clean_review_estimates_long.csv")
    if drop_hunt:
        df = df[~((df["elasticity"] < -2) & (df["authors"].str.contains("Hunt", na=False)))]
    return df.reset_index(drop=True)


def collapse_one_per_margin(df):
    """Inverse-variance weighting within (paper, margin, ue_measure).
    Mirrors 02_model_averaging.R lines 27-41."""
    df = df.copy()
    df["e_denom"] = df.groupby(["paper_id", "pbd_vs_rr", "ue_measure"])["se"].transform(
        lambda s: (1 / s**2).sum()
    )
    df["_num"] = df.groupby(["paper_id", "pbd_vs_rr", "ue_measure"]).apply(
        lambda g: pd.Series((g["elasticity"] / g["se"]**2).sum(), index=g.index)
    ).reset_index(level=[0, 1, 2], drop=True)
    df["elasticity"] = df["_num"] / df["e_denom"]
    df["se"] = np.sqrt(1 / df["e_denom"])
    # one row per group, pick max sample_year as tiebreak, then first row
    df = df.sort_values(["paper_id", "pbd_vs_rr", "ue_measure", "sample_year"],
                        ascending=[True, True, True, False])
    df = df.drop_duplicates(subset=["paper_id", "pbd_vs_rr", "ue_measure"], keep="first")
    return df.sort_values("Unnamed: 0" if "Unnamed: 0" in df.columns else df.columns[0]).reset_index(drop=True)


def build_bma_covariates(df):
    """Construct the design matrix used by 02_model_averaging.R lines 52-102."""
    df = df.copy()
    df["YearsTo2023"] = 2023 - df["year"]
    df["design_RKD"] = (df["research_design"] == "RKD").astype(int)
    df["RR"] = (df["pbd_vs_rr"] != "PBD").astype(int)
    df["DIDorRKD"] = df["design_DID"] + df["design_RKD"]
    df["NonemploymentAsOutcome"] = pd.to_numeric(df["ue_measure_nonemp"], errors="coerce")
    rename = {
        "se": "SE", "PBD_indicator": "PBD", "macro_treatment": "MacroTreatment",
        "design_RDD_RKD": "RDD", "admin": "Admin", "hazard": "HazardModel",
        "us_country": "USA", "country_tax_wedge": "TaxWedge",
        "ue_deviation": "RelativeUnemp", "impact_factor_z": "ImpactFactorZ",
        "mean_pbd": "BaselinePBD", "mean_rr": "BaselineRR",
    }
    df = df.rename(columns=rename)
    df["PBDxBaselinePBD"] = df["PBD"] * df["BaselinePBD"]
    df["RRxBaselinePBD"] = df["RR"] * df["BaselinePBD"]
    df["PBDxBaselineRR"] = df["PBD"] * df["BaselineRR"]
    df["RRxBaselineRR"] = df["RR"] * df["BaselineRR"]
    keep = ["elasticity"] + BMA_COVARS
    return df[keep].dropna().reset_index(drop=True)
