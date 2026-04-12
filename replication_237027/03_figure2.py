"""Figure 2: interaction of admin capacity (4 components) with state legitimacy.

Reproduces Jensen & Weigel (2025, JEL) Figure 2 panels A-D. Each panel runs an
OLS regression of tax_ex_sc on z(admin component) interacted with the standardized
legitimacy index (z(constraints on exec) + z(rigorous impartial admin)),
predicts xb, and plots fitted values stratified by above/below median legitimacy.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

from utils import OUT, load_final, zscore


def make_indices(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["z_exec"] = zscore(df["vdem_executive"])
    df["z_impartial"] = zscore(df["vdem_rigorous_impartial"])
    df["index_legitimate"] = df[["z_exec", "z_impartial"]].mean(axis=1)

    df["z_fte"] = zscore(df["fte_per_laborforce"])
    df["z_merito"] = zscore(df["vdem_meritocratic_recruitment"])
    # staff z-score is restricted to staff_index <= 2 (matches Stata's gen ... if staff_index<=2)
    mask = df["staff_index"] <= 2
    mu = df.loc[mask, "staff_index"].mean()
    sd = df.loc[mask, "staff_index"].std(ddof=1)
    df["z_staff"] = (df["staff_index"] - mu) / sd
    df.loc[~mask, "z_staff"] = np.nan

    df["z_electronic"] = zscore(df["electronic_payments_pct_number"])

    # Median split on legitimacy index — Stata xtile to 2 quantiles (then -1 → {0,1})
    med = df["index_legitimate"].median()
    df["indexlegit_median"] = (df["index_legitimate"] > med).astype(int)
    return df


def reg_predict(df: pd.DataFrame, x: str, restrict_staff: bool = False):
    sub = df.copy()
    if restrict_staff:
        sub = sub[sub["staff_index"] <= 2]
    sub = sub.dropna(subset=["tax_ex_sc", x, "index_legitimate"]).copy()
    sub["interaction"] = sub[x] * sub["index_legitimate"]
    X = sm.add_constant(sub[[x, "index_legitimate", "interaction"]])
    res = sm.OLS(sub["tax_ex_sc"], X).fit()
    sub["xb"] = res.predict(X)
    return res, sub


def main():
    df = load_final()
    df = make_indices(df)

    panels = [
        ("A", "z_fte",        "Tax staff per labor force (index)",            False),
        ("B", "z_merito",     "Meritocratic recruitment (index)",             False),
        ("C", "z_staff",      "Staff qualifications and tenure (index)",      True),
        ("D", "z_electronic", "Share of payments received electronically",    False),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    rows = []
    for ax, (panel, xvar, xlab, restrict) in zip(axes.ravel(), panels):
        res, sub = reg_predict(df, xvar, restrict_staff=restrict)
        rows.append({
            "panel": panel, "xvar": xvar, "n": int(res.nobs),
            "b_x": res.params[xvar],
            "b_legit": res.params["index_legitimate"],
            "b_int": res.params["interaction"],
            "se_int": res.bse["interaction"],
            "p_int": res.pvalues["interaction"],
            "r2": res.rsquared,
        })
        for grp, color, label in [(0, "lightblue", "Low legitimacy"),
                                  (1, "gray", "High legitimacy")]:
            sg = sub[sub["indexlegit_median"] == grp]
            if len(sg) < 3:
                continue
            xs = np.linspace(sg[xvar].min(), sg[xvar].max(), 30)
            # local linear fit of xb on the x var (matches Stata's lfitci on the predicted xb)
            slope, intercept = np.polyfit(sg[xvar], sg["xb"], 1)
            ax.plot(xs, intercept + slope * xs, color=color if grp == 0 else "dimgray",
                    lw=2, label=label)
        ax.set_title(f"Figure 2{panel}")
        ax.set_xlabel(xlab)
        ax.set_ylabel("Tax revenue excl. soc sec (% of GDP)")
        ax.legend()
    plt.tight_layout()
    fig.savefig(OUT / "figure2.png", dpi=140)
    plt.close(fig)

    summary = pd.DataFrame(rows)
    print("\nFigure 2 — interaction regressions tax_ex_sc = a + b1*z + b2*legit + b3*z*legit:\n")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    summary.to_csv(OUT / "figure2_fits.csv", index=False)


if __name__ == "__main__":
    main()
