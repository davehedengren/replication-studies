"""Reproduce Figures 1, 3, 4 from List, Petrie, Samek (2021 JEL w28825).

Figure 1: Growth in economic experiments with children (count of papers by year).
Figure 3 Panel A: Histogram of proportion delayed choices (time preference).
Figure 3 Panel B: Histogram of proportion risky choices (risk preference).
Figure 4 Panel A: Histogram of proportion shared (dictator, wave 2012).
Figure 4 Panel B: Mean share & prop-sharing-half by child age (wave 2012), +/- 1 SE.
"""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import load_papers, load_experimental, first_nonmissing, egen_cut, cut_bins, OUT


def fig1():
    papers = load_papers()
    df = papers[papers["year"] != 1966].copy()
    counts = df.groupby("year").size().sort_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(counts.index, counts.values, "-o", color="blue")
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of publications")
    ax.set_title("Figure 1: Growth in Economic Experiments with Children")
    ax.set_xticks(range(2000, 2021))
    ax.tick_params(axis="x", rotation=45)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT / "Fig1_growth.png", dpi=150)
    plt.close(fig)

    counts.to_csv(OUT / "fig1_counts.csv")
    print("Fig 1: total obs after drop 1966 =", len(df))
    print(counts.to_string())
    return {"total_papers_used": int(len(df)), "counts": counts.to_dict()}


def fig3_and_fig4a():
    exp = load_experimental()
    time = first_nonmissing(exp, ["child_2010_all", "child_2012_all", "child_2013_all"])
    risk = first_nonmissing(
        exp, ["child_2010_risk_all", "child_2012_risk_all", "child_2013_risk_all"]
    )
    d2012 = exp["dictator_giving2012"] / 6

    bins = cut_bins()
    time_cat = egen_cut(time, bins)
    risk_cat = egen_cut(risk, bins)
    dict_cat = egen_cut(d2012, bins)

    def _hist(cat_series, title, xlabel, fname):
        s = pd.to_numeric(cat_series, errors="coerce").dropna()
        n = len(s)
        counts = s.value_counts().sort_index()
        pct = counts / n * 100
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.bar(pct.index, pct.values, width=0.08, color="none", edgecolor="black")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Percent")
        ax.set_title(title)
        ax.set_xticks(np.round(np.arange(0, 1.01, 0.1), 2))
        fig.tight_layout()
        fig.savefig(OUT / fname, dpi=150)
        plt.close(fig)
        return n, pct

    n_time, pct_time = _hist(
        time_cat,
        "Figure 3 Panel A: Heterogeneity in Time Preferences",
        "Proportion Delayed Choices",
        "Fig3_PanelA_time.png",
    )
    n_risk, pct_risk = _hist(
        risk_cat,
        "Figure 3 Panel B: Heterogeneity in Risk Preferences",
        "Proportion Risky Choices",
        "Fig3_PanelB_risk.png",
    )
    n_dict, pct_dict = _hist(
        dict_cat,
        "Figure 4 Panel A: Heterogeneity in Social Preferences",
        "Proportion Shared",
        "Fig4_PanelA_dictator.png",
    )
    print(f"Fig3A  time N={n_time}")
    print(f"Fig3B  risk N={n_risk}")
    print(f"Fig4A  dictator2012 N={n_dict}")
    print("Time distribution (%):")
    print(pct_time.round(2).to_string())
    print("Risk distribution (%):")
    print(pct_risk.round(2).to_string())
    print("Dictator distribution (%):")
    print(pct_dict.round(2).to_string())
    return {
        "n_time": int(n_time),
        "n_risk": int(n_risk),
        "n_dict": int(n_dict),
        "time_dist_pct": pct_time.round(4).to_dict(),
        "risk_dist_pct": pct_risk.round(4).to_dict(),
        "dict_dist_pct": pct_dict.round(4).to_dict(),
    }


def fig4b():
    exp = load_experimental()
    d2012 = exp["dictator_giving2012"] / 6
    age2012 = exp["age_in_month_at_exp_2012"] / 12
    dicthalf = (d2012 == 0.5).astype(float)
    dicthalf[d2012.isna()] = np.nan

    age_bins = [4, 4.5, 5, 5.5, 6, 6.5, 10]
    age_cat = egen_cut(age2012, age_bins)

    tmp = pd.DataFrame({"d": d2012, "h": dicthalf, "age": age_cat})
    tmp = tmp.dropna(subset=["age"])

    grp = tmp.groupby("age")
    agg = grp.agg(
        meanprop=("d", "mean"),
        sdprop=("d", "std"),
        nprop=("d", "count"),
        meanhalf=("h", "mean"),
        sdhalf=("h", "std"),
        nhalf=("h", "count"),
    ).reset_index()
    agg["seprop"] = agg["sdprop"] / np.sqrt(agg["nprop"])
    agg["sehalf"] = agg["sdhalf"] / np.sqrt(agg["nhalf"])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(
        agg["age"], agg["meanhalf"],
        yerr=agg["sehalf"], fmt="-o", color="blue", ecolor="red",
        capsize=4, label="Likelihood of donating half",
    )
    ax.errorbar(
        agg["age"], agg["meanprop"],
        yerr=agg["seprop"], fmt="-o", color="black", ecolor="red",
        capsize=4, label="Proportion shared",
    )
    ax.set_xlabel("Child's Age")
    ax.set_ylabel("Proportion")
    ax.set_xticks([4, 4.5, 5, 5.5, 6, 6.5])
    ax.set_yticks(np.arange(0, 1.01, 0.25))
    ax.set_ylim(0, 1)
    ax.legend()
    ax.set_title("Figure 4 Panel B: Social Preferences by Age")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT / "Fig4_PanelB_socialbyage.png", dpi=150)
    plt.close(fig)

    print("Fig 4B by age category:")
    print(agg.to_string(index=False))
    agg.to_csv(OUT / "fig4b_byage.csv", index=False)
    return agg


def main():
    f1 = fig1()
    f34 = fig3_and_fig4a()
    f4b = fig4b()
    with open(OUT / "figure_numbers.json", "w") as fh:
        json.dump({"fig1": f1, "fig3_fig4a": f34}, fh, indent=2, default=str)
    print("\nFigures saved to", OUT)


if __name__ == "__main__":
    main()
