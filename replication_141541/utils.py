"""Shared helpers for the 141541 replication (Echenique, Gonzalez, Wilson, Yariv,
"Top of the Batch: Interviews and the Match")."""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PKG = ROOT.parent / "141541-V1"
DATA = PKG / "data"
TABLES = PKG / "tables"
OUT = ROOT / "out"
OUT.mkdir(exist_ok=True)

# The six balanced values of N reported in the paper's appendix.
N_VEC = [50, 100, 200, 500, 1000, 1700]
# The four interview-capacity values at N=500.
K_VEC = [2, 5, 10, 20]
# The full grid of common-weight values.
LAMBDA_FULL = [0.05, 0.25, 0.5, 0.75, 0.95]
# The grid displayed in the paper tables.
LAMBDA_D_TABLE = [0.25, 0.5, 0.75]
LAMBDA_H_TABLE = [0.25, 0.75]

# 2020 NRMP Table 13 sub-market sizes, hard-coded from the Mathematica notebook
# (./analysis/MakeTablesAndFigures.nb). Sum should be 35,704.
MARKET_SIZES = [
    1884, 193, 538, 2665, 4662, 9127, 1911, 30, 390, 23, 156, 232, 946, 1443,
    849, 23, 350, 603, 2956, 22, 22, 480, 180, 1858, 192, 1146, 1536, 1174, 38,
    75,
]
MARKET_SUM = sum(MARKET_SIZES)
assert MARKET_SUM == 35704, MARKET_SUM


def load_balanced() -> pd.DataFrame:
    df = pd.read_csv(DATA / "outSimBalanced.csv", index_col=0)
    return df.rename(columns={"Group.1": "N", "Group.2": "lambdaD", "Group.3": "lambdaH"})


def load_k() -> pd.DataFrame:
    df = pd.read_csv(DATA / "outSimBalanced500k.csv", index_col=0)
    return df.rename(columns={"Group.1": "k", "Group.2": "lambdaD", "Group.3": "lambdaH"})


def load_unbalanced() -> pd.DataFrame:
    df = pd.read_csv(DATA / "outSimUnbalanced.csv", index_col=0)
    return df.rename(columns={"Group.1": "nD", "Group.2": "lambdaD", "Group.3": "lambdaH"})


def load_sigs() -> pd.DataFrame:
    df = pd.read_csv(DATA / "outSimSIGS500.csv", index_col=0)
    return df.rename(columns={"Group.1": "N", "Group.2": "lambdaD", "Group.3": "lambdaH"})


def load_rank_diffs() -> pd.DataFrame:
    df = pd.read_csv(DATA / "rankDiffsx.csv", index_col=0)
    return df


# Derivations from the Mathematica notebook (./141541-V1/analysis/
# MakeTablesAndFigures.nb). Matching-outcome variables are raw rates over all
# participants; Panel B variables are conditional on being matched / unmatched.
# Blocking-pair counts are first normalised by N (the market size on each
# side) and then conditioned on matched/unmatched share.
VAR_FNS: dict = {
    "UnmatchedDA":     lambda row, N: 100 * row["unmatched.gs"],
    "UnmatchedIntDA":  lambda row, N: 100 * row["unmatched.tags"],
    "UnmatchedTrDA":   lambda row, N: 100 * row["unmatched.trgs"],
    "TopDA":           lambda row, N: 100 * row["top.gs"],
    "TopIntDA":        lambda row, N: 100 * row["top.tags"],
    "TopTrDA":         lambda row, N: 100 * row["top.trgs"],
    "Top3DA":          lambda row, N: 100 * row["top3.gs"],
    "Top3IntDA":       lambda row, N: 100 * row["top3.tags"],
    "Top3TrDA":        lambda row, N: 100 * row["top3.trgs"],
    "SamePartnerDA":   lambda row, N: 100 * row["samepartner.gs"] / (1 - row["unmatched.gs"]),
    "SamePartnerIntDA":lambda row, N: 100 * row["samepartner.tags"] / (1 - row["unmatched.tags"]),
    "IdenticalIntDA":  lambda row, N: 100 * row["gs.tags.identical"] / (1 - row["unmatched.tags"]),
    "BPMatched":       lambda row, N: 100 * (row["bp.match.tags"] / N) / (1 - row["unmatched.tags"]),
    "BPUnmatched":     lambda row, N: 100 * (row["bp.unmatch.tags"] / N) / row["unmatched.tags"],
}


def pct(df: pd.DataFrame, stem: str, lamD: float, lamH: float, key: str = "N", n=None) -> float:
    """Return a paper-variable as a percentage for the requested cell.

    ``stem`` is a key from :data:`VAR_FNS`. ``key`` names the column that
    parametrises the table (N for balanced, k for interview capacity, nD for
    unbalanced). For the blocking-pair variables, ``N`` (the denominator) is
    inferred from the balanced-table key; unbalanced tables pass ``n_for_bp``
    via the helper below.
    """
    row = df[(df[key] == n) & (np.isclose(df["lambdaD"], lamD)) & (np.isclose(df["lambdaH"], lamH))]
    if row.empty:
        raise KeyError(f"no row for {key}={n} lambdaD={lamD} lambdaH={lamH}")
    r = row.iloc[0]
    # For the balanced and interview-capacity tables N is just the key value,
    # except for interview-capacity (where k varies and N is fixed at 500) and
    # unbalanced (where the doctor-side count is the relevant normalizer).
    if key == "N":
        N_val = n
    elif key == "k":
        N_val = 500
    elif key == "nD":
        N_val = n  # doctor-side count (600 here), consistent with Mathematica
    else:
        N_val = n
    return float(VAR_FNS[stem](r, N_val))


def fmt_pct(x: float, digits: int = 1) -> str:
    return f"{x:.{digits}f}%"


def parse_tex_table(path: Path) -> dict:
    """Parse published .tex tables from ./141541-V1/tables into a flat dict
    keyed by ``(row_label, lamD, lamH) -> percent (float)`` for comparison.

    Row labels are the literal strings used in the tex file (DA / Int-DA /
    Tr-DA / Matched / Unmatched) scoped by the current panel block.
    """
    text = path.read_text()
    block = {}
    # Walk line by line. The six numeric columns correspond to
    # (lamH=0.25, lamD=0.25/0.5/0.75) and (lamH=0.75, lamD=0.25/0.5/0.75).
    lamD_cols = [(0.25, 0.25), (0.5, 0.25), (0.75, 0.25),
                 (0.25, 0.75), (0.5, 0.75), (0.75, 0.75)]
    current_panel = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        # Panel headers are long \multicolumn lines with emph labels.
        lower = line.lower()
        if "unmatched" in lower and "multicolumn" in lower and "emph" in lower:
            current_panel = "unmatched"
            continue
        if "first-ranked" in lower:
            current_panel = "top"
            continue
        if "top-three" in lower:
            current_panel = "top3"
            continue
        if "same partner" in lower:
            current_panel = "samepartner"
            continue
        if "identical partner" in lower:
            current_panel = "identical"
            continue
        if "blocking programs" in lower:
            current_panel = "bp"
            continue
        # Data rows look like "Int-DA & 5.7\% & 6.1\% ..." or "Matched & ...".
        if "\\%" in line and "&" in line and "multicolumn" not in line:
            parts = [p.strip() for p in line.split("&")]
            # Drop a trailing "\\" on the last cell.
            parts = [p.replace("\\\\", "").strip() for p in parts]
            label = parts[0]
            vals = []
            for p in parts[1:]:
                p = p.replace("\\%", "").strip()
                if not p:
                    continue
                try:
                    vals.append(float(p))
                except ValueError:
                    pass
            if len(vals) == 6 and current_panel is not None:
                for (lamD, lamH), v in zip(lamD_cols, vals):
                    block[(current_panel, label, lamD, lamH)] = v
    return block
