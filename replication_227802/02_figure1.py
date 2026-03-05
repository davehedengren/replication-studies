"""
02_figure1.py — Figure 1: Childcare Law Adoption Trends

Panel A: Cumulative number of countries with childcare laws by region
Panel B: Cumulative count by law features (Availability, Affordability, Quality)
"""

import sys, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

sys.path.insert(0, os.path.dirname(__file__))
from utils import RAW_DATA_DIR, OUTPUT_DIR


def figure1a():
    """Panel A: By region (stacked bar chart)."""
    wbl = pd.read_csv(os.path.join(RAW_DATA_DIR, 'WBL_childcare_2024.csv'))

    # Create cumulative counts by region and year
    enacted = wbl[wbl['enactment'].notna()].copy()
    years = range(1950, 2024)
    regions = sorted(enacted['region'].unique())

    cum_data = {}
    for region in regions:
        region_df = enacted[enacted['region'] == region]
        counts = []
        for yr in years:
            counts.append((region_df['enactment'] <= yr).sum())
        cum_data[region] = counts

    fig, ax = plt.subplots(figsize=(8, 5))
    bottom = np.zeros(len(list(years)))
    colors = plt.cm.tab10(np.linspace(0, 1, len(regions)))

    for i, region in enumerate(regions):
        values = np.array(cum_data[region])
        ax.bar(list(years), values - bottom, bottom=bottom, label=region,
               color=colors[i], width=1.0)
        bottom = values.astype(float)

    ax.set_xlim(1955, 2023)
    ax.set_xlabel('Year')
    ax.set_ylabel('Number of countries that have enacted a childcare law')
    ax.set_title('Panel A. By region')
    ax.legend(fontsize=7, loc='upper left')
    ax.set_yticks(range(0, 151, 25))

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'figure1a.png'), dpi=150)
    plt.savefig(os.path.join(OUTPUT_DIR, 'figure1a.pdf'))
    plt.close()


def figure1b():
    """Panel B: By law dimension (line chart)."""
    wbl = pd.read_csv(os.path.join(RAW_DATA_DIR, 'WBL_childcare_2024.csv'))
    enacted = wbl[wbl['enactment'].notna()].copy()

    years_range = range(1950, 2024)
    dims = {'Availability': 'availability', 'Affordability': 'affordability',
            'Quality': 'quality'}

    fig, ax = plt.subplots(figsize=(8, 5))
    for label, col in dims.items():
        cum_counts = []
        for yr in years_range:
            subset = enacted[(enacted['enactment'] <= yr) & (enacted[col] == 1)]
            cum_counts.append(len(subset))
        ax.plot(list(years_range), cum_counts, label=label, linewidth=2)

    ax.set_xlim(1955, 2023)
    ax.set_xlabel('Year')
    ax.set_ylabel('Number of countries that have enacted a childcare law')
    ax.set_title('Panel B. By issues addressed by childcare laws')
    ax.legend(fontsize=8, loc='upper left')
    ax.set_yticks(range(0, 151, 25))

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'figure1b.png'), dpi=150)
    plt.savefig(os.path.join(OUTPUT_DIR, 'figure1b.pdf'))
    plt.close()


def main():
    print("=" * 60)
    print("02_figure1.py: Childcare Law Adoption Trends")
    print("=" * 60)

    figure1a()
    print("Figure 1a saved")

    figure1b()
    print("Figure 1b saved")

    # Print summary stats matching paper
    wbl = pd.read_csv(os.path.join(RAW_DATA_DIR, 'WBL_childcare_2024.csv'))
    n_with_law = wbl['has_childcare_law'].sum()
    n_without = len(wbl) - n_with_law
    print(f"\nCountries with childcare law: {int(n_with_law)}")
    print(f"Countries without: {int(n_without)}")
    print(f"Enacted by 1991: {(wbl['enactment'] <= 1991).sum()}")
    print(f"Enacted by 2022: {(wbl['enactment'] <= 2022).sum()}")
    print(f"Availability: {int(wbl['availability'].sum())}")
    print(f"Affordability: {int(wbl['affordability'].sum())}")
    print(f"Quality: {int(wbl['quality'].sum())}")


if __name__ == '__main__':
    main()
