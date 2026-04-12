"""Shared paths and helpers for replication of Harrell et al. (2023)."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
DATA_PATH = REPO / "183121-V1" / "DtaReplication.dta"
OUTPUT_DIR = ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

CONTROLS = ["Transornb", "Black", "Hispanic", "Anxiety", "Depression"]
OUTCOME = "Posoutcome"
CLUSTER = "EmailPairID"
ABSORB = "State"

SPEC_CASES = ["stdSTCasesT", "stdSTDeathsT"]
SPEC_EXCESS = ["stdSTExLowEstT"]

SPEC_CASES_7D = ["stdSTCasesT7daysEar", "stdSTDeathsT7daysEar"]
SPEC_EXCESS_7D = ["stdSTExLowEstT7daysEar"]
SPEC_CASES_14D = ["stdSTCasesT14daysEar", "stdSTDeathsT14daysEar"]
SPEC_EXCESS_14D = ["stdSTExLowEstT14daysEar"]


def load_data() -> pd.DataFrame:
    return pd.read_stata(DATA_PATH, convert_categoricals=False)


def run_areg(df: pd.DataFrame, covid_vars: list[str]):
    """Replicate Stata `areg y covid X i.Week i.Day, absorb(State) cluster(EmailPairID)`.

    Uses linearmodels.AbsorbingLS so State is absorbed and SEs are clustered at EmailPairID.
    """
    from linearmodels.iv.absorbing import AbsorbingLS
    import numpy as np

    data = df.copy()
    rhs = covid_vars + CONTROLS
    week_d = pd.get_dummies(data["Weeksent"], prefix="week", drop_first=True).astype(float)
    day_d = pd.get_dummies(data["Daysent"], prefix="day", drop_first=True).astype(float)
    X = pd.concat(
        [data[rhs].astype(float).reset_index(drop=True),
         week_d.reset_index(drop=True),
         day_d.reset_index(drop=True)],
        axis=1,
    )
    X.insert(0, "const", 1.0)
    y = data[OUTCOME].astype(float).reset_index(drop=True)
    absorb = data[[ABSORB]].reset_index(drop=True).astype("category")
    clusters = data[[CLUSTER]].reset_index(drop=True).astype(np.int64)

    mod = AbsorbingLS(y, X, absorb=absorb, drop_absorbed=True)
    res = mod.fit(cov_type="clustered", clusters=clusters, debiased=True)
    return res
