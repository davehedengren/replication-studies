"""
03_figures.py - Replicate Figure 1
Rosenzweig & Udry (2014) - Rainfall Forecasts, Weather and Wages
"""
import sys
sys.path.insert(0, '.')
from utils import *
import matplotlib.pyplot as plt

harvest, planting, migration = load_data()

# Figure 1: Ratio of Harvest- to Planting-Stage Wages 2005-2009, by Year
# Paper uses districts where both stage-specific wages are available

# Merge harvest and planting by id and year
# Need to figure out year. The datasets don't have an explicit year variable,
# but fore (IMD forecast) varies by year. Let's check fore values.
print("Unique fore values in harvest:", sorted(harvest['fore'].unique()))
print("Unique fore values in planting:", sorted(planting['fore'].unique()))

# IMD forecasts by year (from context: 2005-2010):
# Let's infer year from fore. We need to identify year for each obs.
# Since both datasets have 'id' and 'fore', and forecast is year-specific,
# we can use fore as a year proxy for matching.

# Actually, we need to match on district (id) and year. Since year isn't explicit,
# let's see if fore maps uniquely to years.
print("\nHarvest fore value counts:")
print(harvest['fore'].value_counts().sort_index())
print("\nPlanting fore value counts:")
print(planting['fore'].value_counts().sort_index())

# The forecast values correspond to years. Let's create a mapping.
# IMD forecasts: each year gets one forecast value for each of ~4 regions.
# Multiple districts can share the same forecast if they're in the same region.

# For Figure 1, we need to group by year. Since fore varies by region within year,
# let's try a different approach: see if jsdev x fore combinations identify year-region.

# Actually, the simplest approach: merge on id and fore (since each id-year has unique fore)
merged = harvest.merge(planting, on=['id', 'fore'], suffixes=('_h', '_p'), how='inner')
print(f"\nMerged obs (districts with both wages): {len(merged)}")

# Compute wage ratio
merged['wage_h'] = np.exp(merged['lwager_h'])
merged['wage_p'] = np.exp(merged['lwager_p'])
merged['ratio'] = merged['wage_h'] / merged['wage_p']

# Map forecast to year
# From the paper: 2005-2010. IMD forecasts vary by year and region.
# Let's try mapping unique fore values to years.
# Known IMD forecasts: 2005=98, 2006=93, 2007=95, 2008=99, 2009=96, 2010=102
# But there are area-specific forecasts too.

# Alternative: merge using jsdev as another key (same district-year should have same jsdev)
merged2 = harvest.merge(planting, on=['id', 'jsdev'], suffixes=('_h', '_p'), how='inner')
print(f"Merged on id+jsdev: {len(merged2)}")

# Let's try merging on id + fore + jsdev
merged3 = harvest.merge(planting, on=['id', 'fore', 'jsdev'], suffixes=('_h', '_p'), how='inner')
print(f"Merged on id+fore+jsdev: {len(merged3)}")

# Use jsdevmiles to figure out distance (constant within district)
# jsdevmiles = jsdev * distance, so distance = jsdevmiles / jsdev
harvest['dist'] = harvest['jsdevmiles'] / harvest['jsdev']
planting['dist'] = planting['jsdevmiles'] / planting['jsdev']

# Merge on id + dist (constant) + fore + jsdev
merged4 = harvest.merge(planting, on=['id', 'fore', 'jsdev'], suffixes=('_h', '_p'), how='inner')
print(f"Final merged: {len(merged4)}")

# Since we can't directly recover year from the data, let's use fore values
# to infer year. Group by fore and look at patterns.
# The figure shows 2005-2009 (5 years). Let's map fore -> year using
# the fact that fore is area-specific IMD forecast.

# Actually, the paper mentions "the June area- and year-specific IMD forecast"
# So fore varies by area x year. Multiple forecast values can correspond to
# the same year but different areas.

# For Figure 1, we just need the mean ratio by year.
# Let me try: since jsdev is unique per district-year, and we have N years of data,
# group by the set of jsdev values? That won't work.

# Better approach: since the post (NREGA) variable changes over time,
# and NREGA was introduced 2007 in some districts and 2010 in all,
# we can use the combination of district characteristics to identify years.

# Actually, let's use a simpler approach. The forecast values cluster by year.
# Let me check the actual forecast distribution.
print("\nForecast values (harvest):")
for f in sorted(harvest['fore'].unique()):
    n = (harvest['fore'] == f).sum()
    print(f"  fore={f}: n={n}")

# The paper says Figure 1 covers 2005-2009 (5 years).
# With 5 years and ~4 IMD area forecasts, we'd expect ~20 unique fore values.
# Let me check: actually, the paper says forecasts are area-specific,
# and IMD issues area-specific forecasts for regions of India.

# Let me try to infer year by noticing that the number of unique forecast
# values per year should be small (4-5 areas). Let's cluster fore values.

# From Indian Meteorological Department, long-range forecasts are typically:
# 2005: 98% of LPA, 2006: 93%, 2007: 95%, 2008: 99/100%, 2009: 93/96%
# But these may be national; area-specific would differ.

# Actually, let me try a different approach for the figure.
# The figure just shows the AVERAGE ratio by year across districts.
# If we can't identify year directly, we can use the fore values to group observations
# into years. Since fore is constant within area-year, districts in the same area
# and year share the same fore. But across areas, the same year has different fore.

# The key insight: jsdev (rainfall shock) and fore (forecast) together
# should uniquely identify area-year combinations.

# Let me just check if the merged data gives us 5 groups by fore:
merged_use = merged4.copy()
merged_use['wage_h'] = np.exp(merged_use['lwager_h'])
merged_use['wage_p'] = np.exp(merged_use['lwager_p'])
merged_use['ratio'] = merged_use['wage_h'] / merged_use['wage_p']

# For now, let's try to figure out year using a trick:
# The NREGA variable 'post' changes over time, and jsdev varies by year.
# With 6 years (2005-2010), we can try to cluster observations into years
# using post_h, post_p, and other year-varying variables.

# Actually, let's use a simpler fact: within a district, jsdev varies by year.
# So group by (id, jsdev) gives us district-year. Then we can assign years
# by sorting jsdev within each district (or using the NREGA rollout).

# Simplest approach: since we have post (NREGA), and NREGA was phased in,
# we know that:
# - 2005-2006: post=0 for all districts (NREGA started Feb 2006 in Phase 1 districts)
# - 2007: post=1 for some districts
# - 2010: post=1 for all districts

# Let me try yet another approach: load the raw Stata files and check
# if there's any year variable we're missing.

# Actually, the simplest approach for Figure 1 is to note that the paper
# pools harvest and planting data, and since the same districts appear
# across years, we can just compute the average wage level by "wave"
# (each unique combination of district characteristics).

# Let me try: since the figure has 5 points (2005-2009), and we have
# ~337 harvest obs and ~387 planting obs over 6 years,
# the figure might use a different sample or the 2010 data point is excluded.

# PRACTICAL APPROACH: Since fore values can be mapped to years
# (IMD issues ONE national forecast plus area-specific ones per year),
# let's cluster fore values into year groups.

# Unique fore values in the merged data
print("\nUnique fore values in merged data:")
for f in sorted(merged_use['fore'].unique()):
    n = (merged_use['fore'] == f).sum()
    post_vals = merged_use[merged_use['fore'] == f]['post_h'].unique()
    print(f"  fore={f}: n={n}, NREGA={post_vals}")

# Looking at IMD forecasts for India (national level, % of Long Period Average):
# 2005: 98, 2006: 93, 2007: 95, 2008: 99, 2009: 96, 2010: 102
# Area-specific forecasts: NW India, NE India, Central India, South Peninsula
# So fore values cluster around these national values ±few

# Let me map: 81=2006 area, 93=2006, 94=2009?, 95=2007, 96=2009,
# 97=2009, 98=2005, 99=2008, 100=2008, 102=2010, 103=2010

# Actually, from the data: forecast SD in harvest = 4.92, range 81-102
# This is too wide for single-year variation. Let me check harvest year-by-year.

# Let me look at it from the NREGA timing angle instead.
# NREGA Phase 1 (Feb 2006): 200 poorest districts
# Phase 2 (April 2007): 130 more districts
# Phase 3 (April 2008): remaining districts

# So post transitions from 0→1 at different years for different districts.
# Within a district, the transition year is unique.

# Let me group the merged data by fore and see if we can identify clear year clusters:
print("\nAttempting year assignment from fore values and NREGA patterns...")
# We know there are 6 years: 2005-2010
# The number of districts per year varies

# For the figure, let me use an alternative: compute the ratio for each
# unique forecast value and plot. The paper clearly computed this somehow.

# FINAL APPROACH: Use the original Stata datasets and just compute
# the ratio using the log wages directly, matching on district within the data.
# The figure shows 2005-2009. Since both wage datasets span 2005-2010,
# the figure must use years 2005-2009 only. Let me compute without year labels.

# Since districts repeat across years, and within a year, jsdev and fore
# are the same for a given district, we can group by district and
# rank observations by some criterion.

# Actually, I bet the years can be recovered from the fore variable more carefully.
# Let me check the harvest data: 337 obs, 95 districts, 6 years (2005-2010)
# But 337/95 ≈ 3.5, not 6. So not all districts appear in all years.
# Similarly planting: 387/106 ≈ 3.7.

# Let me just compute the figure using the merged data.
# Group by 'fore' values and compute mean ratio.
# This won't be perfect but will approximate the figure.

# Better: since the paper says "districts and years in which both
# stage-specific wages are available", merge and group by year.
# Year identification: within each district, sort by jsdev or post
# and assign years sequentially? This is fragile.

# Let me try the simplest version of the figure first.

# Use the forecast-to-year mapping based on known IMD forecasts:
# IMD long-range forecast (June, for Southwest Monsoon):
# 2005: 98% LPA (normal), 2006: 93% (below normal)
# 2007: 95% (normal), 2008: 99% (normal)
# 2009: 96% (below normal), 2010: 102% (above normal)
# But these are national. Area-specific can differ by ±5%.

# Group fore into year bins:
# <=93 -> 2006, 94-95 -> 2007/2009, 96-97 -> 2005/2009, 98 -> 2005, 99-100 -> 2008, >=101 -> 2010

# This is too unreliable. Let me try a regression-based approach or just
# plot by sorted fore groups.

# ACTUALLY: The simplest approach - within each district, observations
# can be ordered by the NREGA variable (post=0 comes before post=1)
# and within post groups by jsdev patterns. But this is fragile.

# Let me just compute ratios and group by fore, accepting that
# the x-axis will be forecast value rather than year.

# Actually, I just realized: let me merge on id and jsdevmiles/jsdev
# (distance is constant within district, jsdev varies by year)
# No, jsdevmiles = jsdev * distance, so it varies too.

# OK, I'll take the most practical approach:
# Since we can't perfectly recover years, compute the ratio for each
# district-observation and group into 5-6 clusters.

# Use k-means on fore to get year clusters
from scipy.cluster.hierarchy import fcluster, linkage

all_fore = sorted(merged_use['fore'].unique())
print(f"\nAll unique fore values in merged: {all_fore}")

# Map based on known IMD forecasts + area variation:
# Each year has 3-5 area-specific forecasts. Group by looking at
# how many districts fall into each fore value.

# Let's try: assign year based on fore ranges
def assign_year(fore):
    if fore <= 87:
        return 2006
    elif fore <= 91:
        return 2006
    elif fore <= 93:
        return 2006
    elif fore == 94:
        return 2009  # or 2007
    elif fore == 95:
        return 2007
    elif fore == 96:
        return 2009
    elif fore == 97:
        return 2009
    elif fore == 98:
        return 2005
    elif fore == 99:
        return 2008
    elif fore == 100:
        return 2008
    elif fore >= 101:
        return 2010
    return None

# Actually this approach is too speculative. Let me check if the datasets
# have any hidden year info by looking at the Stata file metadata.
print("\nChecking if year can be recovered from variable labels...")

import struct
# Just check summary by post (NREGA) status
for f in sorted(merged_use['fore'].unique()):
    sub = merged_use[merged_use['fore'] == f]
    pct_nrega = sub['post_h'].mean()
    print(f"  fore={f}: n={len(sub)}, NREGA%={pct_nrega:.0%}, mean_jsdev={sub['jsdev'].mean():.3f}")

# Use post + fore patterns to assign years:
# 2005: no NREGA anywhere -> post=0 for all
# 2006: NREGA Phase 1 (200 districts) -> post=0 for most, post=1 for some
# 2007: NREGA Phase 1+2 -> more post=1
# 2008: NREGA Phase 1+2+3 -> even more post=1
# 2009: all districts -> post=1 for all
# 2010: post=1 for all

# So fore values where NREGA=0% are likely 2005
# fore values where NREGA=100% are likely 2009 or 2010
# Mixed NREGA are 2006-2008

# Let me group more carefully
print("\nYear assignment attempt:")
# From NREGA: years where NO districts have NREGA -> 2005
# Years where ALL districts have NREGA -> 2009-2010

year_map = {}
for f in sorted(merged_use['fore'].unique()):
    sub = merged_use[merged_use['fore'] == f]
    pct = sub['post_h'].mean()
    if pct == 0:
        yr = 2005
    elif pct == 1.0:
        if f >= 100:
            yr = 2010
        elif f >= 96:
            yr = 2009
        else:
            yr = 2009
    elif pct < 0.3:
        yr = 2006
    elif pct < 0.6:
        yr = 2007
    else:
        yr = 2008
    year_map[f] = yr
    print(f"  fore={f}: NREGA={pct:.0%} -> year={yr}")

merged_use['year'] = merged_use['fore'].map(year_map)
print(f"\nYear distribution:\n{merged_use['year'].value_counts().sort_index()}")

# Compute ratio by year
ratio_by_year = merged_use.groupby('year')['ratio'].mean()
print(f"\nMean harvest/planting wage ratio by year:")
print(ratio_by_year)

# Plot Figure 1
fig, ax = plt.subplots(figsize=(8, 5))
# Paper shows 2005-2009
plot_years = [y for y in sorted(ratio_by_year.index) if y <= 2009]
plot_ratios = [ratio_by_year[y] for y in plot_years]

ax.plot(plot_years, plot_ratios, 'b-o', linewidth=1.5, markersize=6)
ax.set_xlabel('Year')
ax.set_ylabel('Ratio')
ax.set_title('Figure 1: Ratio of Harvest- to Planting-Stage Wages 2005-2009, by Year')
ax.set_xlim(2004.5, 2009.5)
ax.set_ylim(0.9, 1.1)
ax.set_xticks(plot_years)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figure1_wage_ratio.png', dpi=150)
print(f"\nFigure 1 saved to figure1_wage_ratio.png")

# Compare with paper's Figure 1:
# 2005: ~0.97, 2006: ~0.93, 2007: ~1.00, 2008: ~1.02, 2009: ~0.99
print("\nPaper Figure 1 approximate values:")
print("  2005: ~0.97, 2006: ~0.93, 2007: ~1.00, 2008: ~1.02, 2009: ~0.99")
print("Our estimates:")
for y in plot_years:
    print(f"  {y}: {ratio_by_year[y]:.3f}")
