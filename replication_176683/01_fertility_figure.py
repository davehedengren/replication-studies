"""Reproduces Figure 1: Total Fertility Rate by region, 1950-2015.

Original: FertilityData.m (MATLAB)
"""
import matplotlib.pyplot as plt
from utils import FERTILITY_DATA, FERTILITY_COLS, FERTILITY_YEARS, OUT


def main():
    yrs = FERTILITY_YEARS
    world = FERTILITY_DATA[:, 0]
    hic = FERTILITY_DATA[:, 1]
    usa = FERTILITY_DATA[:, 2]
    chn = FERTILITY_DATA[:, 3]
    ind = FERTILITY_DATA[:, 4]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.axhline(2, linestyle="--", color="black", linewidth=1)
    ax.plot(yrs, chn, color="#d62728", linewidth=2, label="China")
    ax.plot(yrs, hic, color="#9467bd", linewidth=2, label="High income")
    ax.plot(yrs, ind, color="#2ca02c", linewidth=2, label="India")
    ax.plot(yrs, world, color="black", linewidth=2, label="World")
    ax.plot(yrs, usa, color="#1f77b4", linewidth=2, label="U.S.")
    ax.set_ylim(1.0, 6.5)
    ax.set_ylabel("Live births per woman")
    ax.set_title("Figure 1: The Total Fertility Rate (Live Births per Woman)")
    ax.text(2013, 2.7, "World")
    ax.text(1990, 4.2, "India")
    ax.text(2002, 1.4, "China")
    ax.text(1998, 2.3, "U.S.")
    ax.text(2013, 1.6, "High income\ncountries")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "figure1_fertility.png", dpi=150)
    plt.close(fig)

    # Verify the most recent (2015-2020) values against paper text (p.1):
    # "1.8 for the United States, 1.7 for China and for High Income Countries on average,
    #  1.6 for Germany, 1.4 for Japan, and 1.3 for Italy and Spain"
    last = FERTILITY_DATA[-1]
    paper = {"USA": 1.8, "China": 1.7, "HighInc": 1.7, "Germany": 1.6,
             "Japan": 1.4, "Italy": 1.3, "Spain": 1.3}
    print("\nFigure 1 — 2015-2020 TFR, data vs paper text:")
    print(f"{'Region':12s} {'Data':>6s} {'Paper':>6s} {'Match':>6s}")
    for name, val in paper.items():
        i = FERTILITY_COLS.index(name)
        d = round(last[i], 1)
        mark = "OK" if abs(d - val) < 0.05 else "MISS"
        print(f"{name:12s} {d:6.2f} {val:6.2f} {mark:>6s}")

    print(f"\nSaved: {OUT/'figure1_fertility.png'}")


if __name__ == "__main__":
    main()
