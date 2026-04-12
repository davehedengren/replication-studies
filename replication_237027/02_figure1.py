"""Figure 1: six bivariate scatter+linear-fit plots of tax revenue vs admin/governance variables.

Reproduces Jensen & Weigel (2025, JEL) Figure 1 panels A-F. Reports the implied
OLS slope, intercept, t-stat, and N for each panel so the 'pattern' (positive
slope) can be compared to the published figure even though the original paper
does not print the regression numbers.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import OUT, load_final, ols_fit


PANELS = [
    ("A", "fte_per_laborforce",            "Tax administration staff per labor force",        None),
    ("B", "vdem_meritocratic_recruitment", "Meritocratic recruitment score",                  None),
    ("C", "staff_index",                   "Index of qualifications and tenure of staff",     "staff_index <= 2"),
    ("D", "electronic_payments_pct_number","Share of payments received electronically",       None),
    ("E", "vdem_executive",                "Constraints on the executive",                    None),
    ("F", "vdem_rigorous_impartial",       "Impartial and rigorous public administration",    None),
]


def main():
    df = load_final()
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    summary = []
    for ax, (panel, xvar, xlab, restrict) in zip(axes.ravel(), PANELS):
        sub = df.copy()
        if restrict == "staff_index <= 2":
            sub = sub[sub["staff_index"] <= 2]
        y = sub["tax_ex_sc"]
        x = sub[xvar]
        fit = ols_fit(y, x)
        ax.scatter(x, y, color="gray", s=22, alpha=0.85)
        if fit:
            xs = np.linspace(np.nanmin(x), np.nanmax(x), 50)
            ys = fit["intercept"] + fit["slope"] * xs
            ax.plot(xs, ys, color="navy", lw=2)
            # 95% CI for the mean prediction
            mask = (~y.isna()) & (~x.isna())
            xv = x[mask].to_numpy()
            yv = y[mask].to_numpy()
            n = fit["n"]
            sigma2 = ((yv - (fit["intercept"] + fit["slope"] * xv)) ** 2).sum() / (n - 2)
            xbar = xv.mean()
            sxx = ((xv - xbar) ** 2).sum()
            se_mean = np.sqrt(sigma2 * (1/n + (xs - xbar) ** 2 / sxx))
            from scipy import stats
            tcrit = stats.t.ppf(0.975, df=n - 2)
            ax.fill_between(xs, ys - tcrit * se_mean, ys + tcrit * se_mean, color="lightblue", alpha=0.5)
        ax.set_title(f"Figure 1{panel}")
        ax.set_xlabel(xlab)
        ax.set_ylabel("Tax revenue excl. soc sec (% of GDP)")
        ax.set_ylim(0, 40)
        if fit:
            summary.append({
                "panel": panel, "xvar": xvar, "n": fit["n"],
                "slope": fit["slope"], "se": fit["se_slope"],
                "t": fit["t_slope"], "p": fit["p_slope"], "r2": fit["r2"],
            })
    plt.tight_layout()
    fig.savefig(OUT / "figure1.png", dpi=140)
    plt.close(fig)

    sdf = pd.DataFrame(summary)
    print("\nFigure 1 — bivariate OLS of tax_ex_sc on each predictor:\n")
    print(sdf.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    sdf.to_csv(OUT / "figure1_fits.csv", index=False)


if __name__ == "__main__":
    main()
