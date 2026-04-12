"""Load and sanity-check all inputs for replication 151521."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils import DATA, load_mostrecent, load_mrc_merged, load_figure1, load_scf_tabulated


def main():
    print("=" * 70)
    print("01_clean.py — Loading inputs for replication 151521")
    print("=" * 70)

    # Figure 1 tuition series
    fig1 = load_figure1()
    print(f"\nFigure 1 tuition series: {len(fig1)} years, {fig1['year'].iloc[0]}–{fig1['year'].iloc[-1]}")
    print(f"  1990–91 public sticker = ${int(fig1['public_sticker'].iloc[0]):,}")
    print(f"  2016–17 private sticker = ${int(fig1['private_sticker'].iloc[-1]):,}")
    assert int(fig1['public_sticker'].iloc[0]) == 3520
    assert int(fig1['private_sticker'].iloc[-1]) == 33480

    # College Scorecard
    ms = load_mostrecent()
    print(f"\nCollege Scorecard (mostrecent.dta): {ms.shape}")

    # Chetty mobility
    mrc = load_mrc_merged()
    print(f"Chetty mrc_merged.dta: {mrc.shape}")
    print(f"  has opeid6: {'opeid6' in mrc.columns}")
    print(f"  has par_mean: {'par_mean' in mrc.columns}")

    # SCF tabulated income (for Pareto-lognormal fit)
    for yr in (1989, 2016):
        tab = load_scf_tabulated(yr)
        total = tab['count'].sum()
        mean_y = (tab['income'] * tab['count']).sum() / total
        print(f"\nSCF {yr} tabulation: {len(tab)} bins, N_w = {total:,.0f}, mean income = ${mean_y:,.0f}")

    print("\nAll inputs loaded OK.")


if __name__ == "__main__":
    main()
