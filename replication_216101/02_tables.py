"""Replicate Tables 1, 2, 3, 4 (NLSY97) and Table 5 (HSLS:09)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

DATA = Path(__file__).resolve().parent / "data"
OUT = Path(__file__).resolve().parent / "output"
OUT.mkdir(exist_ok=True)


def mean_se(x: pd.Series) -> tuple[float, float, int]:
    x = x.dropna()
    n = len(x)
    if n == 0:
        return np.nan, np.nan, 0
    m = x.mean()
    se = x.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
    return m, se, n


def fmt(m, se, digits=2):
    if np.isnan(m):
        return "-"
    return f"{m:,.{digits}f} ({se:,.{digits}f})"


def table1(df: pd.DataFrame) -> str:
    """Table 1 Panels A (HS grads) and B (BA enrollees) by belief bin."""
    rows = []
    for panel, mask in [
        ("A: HS grads by 30", df["flag_beliefs_sample"] == 1),
        ("B: BA enrollees by 30", (df["flag_beliefs_sample"] == 1) & (df["EnrBA_by_30"] == 1)),
    ]:
        sub = df[mask]
        cols = [("All", sub)]
        for b, label in [(1, "[0,32]"), (2, "[33,66]"), (3, "[67,100]")]:
            cols.append((label, sub[sub["belief_bin"] == b]))
        out_rows = [f"\n{panel}  (N={len(sub):,})"]
        for var, lab, digits in [
            ("EnrBA_by_30_pct", "Enrolled BA by 30", 2),
            ("pct_chance_BA", "Expected Pr. BA by 30", 2),
            ("BA_by_30_pct", "BA by 30", 2),
            ("asvab", "Skill (ASVAB) *100", 2),
            ("sex_female", "Female", 2),
            ("age_youth_beliefs", "Age beliefs response", 2),
            ("age_1997", "Age in 1997", 2),
            ("r_hh_gross_Y_parents", "Real parental income 1997", 0),
            ("flag_highedrespar", "At least one parent BA+", 2),
        ]:
            series_row = [f"  {lab:40s}"]
            for _, sd in cols:
                x = sd[var]
                if var == "asvab":
                    x = x * 100  # ASVAB stored as 0-1 after dividing by 1000
                m, se, _ = mean_se(x)
                series_row.append(fmt(m, se, digits))
            out_rows.append("  ".join(series_row))
        row_obs = [f"  {'Obs':40s}"]
        row_share = [f"  {'Sample share':40s}"]
        total = len(sub)
        for _, sd in cols:
            row_obs.append(f"{len(sd):>12,}")
            row_share.append(f"{len(sd)/total:>12.2f}" if total else "-")
        out_rows.append("  ".join(row_obs))
        out_rows.append("  ".join(row_share))
        rows.append("\n".join(out_rows))
    return "\n".join(rows)


def table2(df: pd.DataFrame) -> str:
    """Table 2: probit AMEs for BA enrollment on belief + controls."""
    s = df[df["flag_beliefs_sample"] == 1].copy()
    s["decimal_pct_chance_BA"] = s["pct_chance_BA"] / 100.0
    s["flag_EnrBA_by_30"] = s["EnrBA_by_30"]
    out_lines = ["\nTable 2 — Probit AMEs (BA enrollment)"]

    def run(cols):
        d = s.dropna(subset=cols + ["flag_EnrBA_by_30"]).copy()
        X = sm.add_constant(d[cols])
        y = d["flag_EnrBA_by_30"].astype(float)
        model = sm.Probit(y, X).fit(disp=False)
        ame = model.get_margeff(at="overall", method="dydx")
        return model, ame, len(d)

    # Model 1: belief, skill, female, age_1997
    cols1 = ["decimal_pct_chance_BA", "asvab", "sex_female", "age_1997"]
    m1, ame1, n1 = run(cols1)

    # Model 2: add log income + parent ed
    cols2 = cols1 + ["r_log_hhinc", "flag_highedrespar"]
    m2, ame2, n2 = run(cols2)

    def add(lines, label, a):
        ame_df = pd.DataFrame({"ame": a.margeff, "se": a.margeff_se}, index=a.margeff_names if hasattr(a, "margeff_names") else cols2)
        # statsmodels returns margeff in the order of the X columns
        return ame_df

    out_lines.append(f"  Model 1 (N={n1:,})")
    for i, v in enumerate(cols1):
        out_lines.append(f"    {v:30s}  AME={ame1.margeff[i]:.4f}  ({ame1.margeff_se[i]:.4f})")
    out_lines.append(f"  Model 2 (N={n2:,})")
    for i, v in enumerate(cols2):
        out_lines.append(f"    {v:30s}  AME={ame2.margeff[i]:.4f}  ({ame2.margeff_se[i]:.4f})")
    return "\n".join(out_lines)


def _optimism_by_group(d: pd.DataFrame, group_col: str):
    rows = []
    for g in sorted(d[group_col].dropna().unique()):
        sub = d[d[group_col] == g]
        m_b, se_b, n = mean_se(sub["pct_chance_BA"])
        m_r, se_r, _ = mean_se(sub["BA_by_30_pct"])
        diff = m_b - m_r
        # Unpaired difference SE via delta method
        se_diff = np.sqrt(se_b ** 2 + se_r ** 2) if not (np.isnan(se_b) or np.isnan(se_r)) else np.nan
        rows.append((int(g), n, m_b, se_b, m_r, se_r, diff, se_diff))
    return rows


def table3(df: pd.DataFrame) -> str:
    s = df[df["flag_beliefs_sample"] == 1].copy()

    out = ["\nTable 3 — Optimism by skill tercile"]
    # Panel A: students
    enr = s[s["EnrBA_by_30"] == 1]
    rows = _optimism_by_group(enr, "q_tabulations")
    out.append(f"  Panel A (student beliefs, enrollees) N={len(enr):,}")
    out.append("    Skill  Obs    ExpPr    (se)    Real    (se)    Diff    (se)")
    for g, n, mb, seb, mr, ser, d, sd in rows:
        out.append(f"    {g}     {n:4d}  {mb:6.2f} ({seb:4.2f})  {mr:6.2f} ({ser:4.2f})  {d:6.2f} ({sd:4.2f})")

    # Panel B: student beliefs before/after enrollment (learning sample)
    # We approximate: compare beliefs from 1997 vs 2001 for those who
    # answered both and enrolled in BA. Our NLSY clean doesn't store the
    # learning vars separately; skip this panel.
    out.append("  Panel B (learning sample): not replicated here; requires "
               "both 1997 and 2001 beliefs for same respondent.")

    # Panel C: parent beliefs
    enr_p = enr.dropna(subset=["parent_pct_chanceBA"])
    rows_p = []
    for g in sorted(enr_p["q_tabulations"].dropna().unique()):
        sub = enr_p[enr_p["q_tabulations"] == g]
        m_b, se_b, n = mean_se(sub["parent_pct_chanceBA"])
        m_r, se_r, _ = mean_se(sub["BA_by_30_pct"])
        se_diff = np.sqrt(se_b ** 2 + se_r ** 2)
        rows_p.append((int(g), n, m_b, se_b, m_r, se_r, m_b - m_r, se_diff))
    out.append(f"  Panel C (parent beliefs, enrollees) N={len(enr_p):,}")
    out.append("    Skill  Obs    ExpPr    (se)    Real    (se)    Diff    (se)")
    for g, n, mb, seb, mr, ser, d, sd in rows_p:
        out.append(f"    {g}     {n:4d}  {mb:6.2f} ({seb:4.2f})  {mr:6.2f} ({ser:4.2f})  {d:6.2f} ({sd:4.2f})")
    return "\n".join(out)


def table4(df: pd.DataFrame) -> str:
    """Non-enrollees expected vs realized grad (imputed from enrollees of same tercile)."""
    s = df[df["flag_beliefs_sample"] == 1].copy()
    enr = s[s["EnrBA_by_30"] == 1]
    non = s[s["EnrBA_by_30"] == 0]
    # realized grad rate imputed = enrollees' BA_by_30 in same tercile (Panel A)
    enr_grad = enr.groupby("q_tabulations")["BA_by_30_pct"].agg(["mean", "sem"]).to_dict()
    out = ["\nTable 4 — Non-enrollees", "    Skill  Obs    ExpPr    (se)    Real_imp (se)    Diff    (se)"]
    for g in sorted(non["q_tabulations"].dropna().unique()):
        sub = non[non["q_tabulations"] == g]
        m_b, se_b, n = mean_se(sub["pct_chance_BA"])
        m_r = enr_grad["mean"][g]
        se_r = enr_grad["sem"][g]
        diff = m_b - m_r
        se_d = np.sqrt(se_b ** 2 + se_r ** 2)
        out.append(f"    {int(g)}     {n:4d}  {m_b:6.2f} ({se_b:4.2f})  {m_r:6.2f} ({se_r:4.2f})  {diff:6.2f} ({se_d:4.2f})")
    return "\n".join(out)


def table5(df: pd.DataFrame) -> str:
    """HSLS:09 Table 5 — federal student loan incidence by persistence status."""
    s = df[(df["flag_sample_college"] == 1) & (df["W5W1W2W3W4PSRECORDS"] > 0)].copy()
    w = s["W5W1W2W3W4PSRECORDS"]
    n_total = int((w > 0).sum())  # count of 2356

    out = ["\nTable 5 — Federal student loans by persistence status (HSLS:09)"]
    out.append(f"  Obs: {n_total:,} (target 2,356)")

    # Per-group statistics, weighted by W5W1W2W3W4PSRECORDS
    total_fed_dollars = (s["fed_SLcum"] * w).sum()
    groups = []
    for name, mask in [("Did not persist", s["flag_persisted"] == 0),
                       ("Persisted", s["flag_persisted"] == 1)]:
        sub = s[mask]
        ww = sub["W5W1W2W3W4PSRECORDS"]
        share_enrollees = ww.sum() / w.sum() * 100
        share_fed_dollars = (sub["fed_SLcum"] * ww).sum() / total_fed_dollars * 100
        pct_with_fed = (sub["anyfed"] * ww).sum() / ww.sum() * 100
        fed_cond = sub[sub["fed_SLcum"] > 0]
        ww_cond = fed_cond["W5W1W2W3W4PSRECORDS"]
        avg_fed = (fed_cond["fed_SLcum"] * ww_cond).sum() / ww_cond.sum()
        # Weighted median via expanding then numpy
        order = fed_cond.sort_values("fed_SLcum")
        ww_sorted = order["W5W1W2W3W4PSRECORDS"].cumsum()
        cutoff = ww_sorted.iloc[-1] / 2
        med = order.loc[ww_sorted >= cutoff, "fed_SLcum"].iloc[0]
        groups.append((name, share_enrollees, share_fed_dollars, pct_with_fed, avg_fed, med))
    out.append("    Status             % enrollees  % of SL $  % with SL  Avg $     Median $")
    for name, se, sd, pw, avg, med in groups:
        out.append(f"    {name:18s} {se:11.1f}  {sd:9.1f}  {pw:9.1f}  {avg:8,.0f}  {med:8,.0f}")
    return "\n".join(out)


def main():
    nlsy = pd.read_parquet(DATA / "nlsy97_clean.parquet")
    hsls = pd.read_parquet(DATA / "hsls_clean.parquet")

    parts = [
        table1(nlsy),
        table2(nlsy),
        table3(nlsy),
        table4(nlsy),
        table5(hsls),
    ]
    report = "\n".join(parts)
    print(report)
    (OUT / "tables_report.txt").write_text(report)


if __name__ == "__main__":
    main()
