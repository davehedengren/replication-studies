"""
02_figures.py — Reproduce all figures from Gobbi, Hannusch & Rossi (2026).

Generates: Figure1a-d, Figure2a-d, Figure4a-d (PNG format)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import statsmodels.api as sm
from utils import OUTPUT_DIR, ANALYSIS_DATA_PATH, ANALYSIS_DATA_FIG4_PATH
import os

print("=" * 60)
print("02_figures.py — Reproducing all figures")
print("=" * 60)

# ── Load data ──────────────────────────────────────────────────────────

df = pd.read_parquet(ANALYSIS_DATA_PATH)
df_diff = pd.read_parquet(ANALYSIS_DATA_FIG4_PATH)
df_cross = pd.read_parquet(os.path.join(OUTPUT_DIR, 'cross_section.parquet'))

# ── Color scheme (matching R code's viridis/plasma) ────────────────────

REGION_COLORS = {
    'East Asia & Pacific': '#d95f02',
    'Europe & Central Asia': '#FDE725',
    'Latin America & Caribbean': '#21908C',
    'Middle East & North Africa': '#e7298a',
    'North America': '#440154',
    'South Asia': '#7570B3',
    'Sub-Saharan Africa': '#0D0887',
}

REGION_MARKERS = {
    'East Asia & Pacific': 's',       # square (open)
    'Europe & Central Asia': '^',     # triangle up
    'Latin America & Caribbean': 's', # square (filled)
    'Middle East & North Africa': 'o',# circle (open)
    'North America': '^',            # triangle (open)
    'South Asia': '*',               # star
    'Sub-Saharan Africa': 'o',       # circle (filled)
}

REGION_FILL = {
    'East Asia & Pacific': 'none',
    'Europe & Central Asia': '#FDE725',
    'Latin America & Caribbean': '#21908C',
    'Middle East & North Africa': 'none',
    'North America': 'none',
    'South Asia': '#7570B3',
    'Sub-Saharan Africa': '#0D0887',
}

CUSTOM_ORDER = [
    'East Asia & Pacific', 'Europe & Central Asia',
    'Latin America & Caribbean', 'Middle East & North Africa',
    'North America', 'South Asia', 'Sub-Saharan Africa'
]


def add_ols_line(ax, x, y, color='black', linestyle='--', linewidth=1.5,
                 show_ci=True, alpha=0.15):
    """Add OLS regression line with optional confidence band."""
    mask = np.isfinite(x) & np.isfinite(y)
    x_clean, y_clean = x[mask], y[mask]
    X = sm.add_constant(x_clean)
    res = sm.OLS(y_clean, X).fit()

    x_pred = np.linspace(x_clean.min(), x_clean.max(), 100)
    X_pred = sm.add_constant(x_pred)
    y_pred = res.predict(X_pred)

    ax.plot(x_pred, y_pred, color=color, linestyle=linestyle,
            linewidth=linewidth)

    if show_ci:
        pred = res.get_prediction(X_pred)
        ci = pred.conf_int(alpha=0.05)
        ax.fill_between(x_pred, ci[:, 0], ci[:, 1], alpha=alpha, color=color)

    return res


# ══════════════════════════════════════════════════════════════════════
# FIGURE 1: Cross-sectional scatter plots (TFR vs covariates)
# ══════════════════════════════════════════════════════════════════════

def make_figure1(x_var, y_var, x_label, years, fig_name, xlim=None, ylim=(-0.5, 9)):
    """Create Figure 1-style scatter plot for two years."""
    fig, ax = plt.subplots(figsize=(11.5, 7.5))

    colors = {'early': '#0D0887', 'late': '#B12A90'}
    markers = {'early': 'o', 'late': '^'}
    fills = {'early': '#0D0887', 'late': 'none'}

    r2_texts = []

    for i, (yr, label) in enumerate(zip(years, ['early', 'late'])):
        sub = df[(df['Year'] == yr) & df[x_var].notna() & df[y_var].notna()]

        facecolor = colors[label] if label == 'early' else 'none'
        ax.scatter(sub[x_var], sub[y_var], c=colors[label],
                   marker=markers[label], s=50, alpha=0.8,
                   facecolors=facecolor if label == 'early' else 'none',
                   edgecolors=colors[label], linewidths=1.5,
                   label=str(yr), zorder=3)

        # OLS fit
        res = add_ols_line(ax, sub[x_var].values, sub[y_var].values,
                           color=colors[label], linewidth=2.5)
        r2_texts.append(f"R² ({yr}) = {res.rsquared_adj:.3f}")

    if xlim:
        ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel(x_label, fontsize=18)
    ax.set_ylabel('Total Fertility Rate', fontsize=18)
    ax.tick_params(labelsize=16)
    ax.legend(title='Year', fontsize=16, title_fontsize=16, loc='best')

    # R² annotations
    y_pos = [0.75, -0.25] if ylim[0] < 0 else [ylim[0] + 1, ylim[0] + 0.5]
    x_pos = xlim[0] + 0.05 * (xlim[1] - xlim[0]) if xlim else ax.get_xlim()[0]
    for j, txt in enumerate(r2_texts):
        ax.text(x_pos, y_pos[j], txt, fontsize=14, fontweight='bold')

    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, fig_name), dpi=150)
    plt.close()
    print(f"  {fig_name}: {', '.join(r2_texts)}")


# Figure 1a: TFR vs Human Capital Index
make_figure1('hc', 'TFR', 'Human Capital Index', [1960, 2010],
             'Figure1a.png', xlim=(0.8, 3.8))

# Figure 1b: TFR vs Log GDP pc
make_figure1('log_gdppc', 'TFR', 'Log of Real per-capita GDP', [1960, 2010],
             'Figure1b.png', xlim=(3.5, 12))

# Figure 1c: TFR vs Child Mortality
make_figure1('chmortality', 'TFR', 'Child Mortality', [1960, 2010],
             'Figure1c.png', xlim=(0, 400))

# Figure 1d: TFR vs Maternal Mortality (1985, 2010)
make_figure1('matmortality', 'TFR', 'Maternal Mortality', [1985, 2010],
             'Figure1d.png', xlim=(0, 2000))


# ══════════════════════════════════════════════════════════════════════
# FIGURE 2: First-difference scatter plots (ΔX vs ΔTFR)
# ══════════════════════════════════════════════════════════════════════

def make_figure2(x_var, y_var, x_label, y_label, fig_name, xlim=None):
    """Create Figure 2-style scatter plot colored by WB region."""
    fig, ax = plt.subplots(figsize=(7, 5))

    sub = df_cross[df_cross[x_var].notna() & df_cross[y_var].notna() &
                   df_cross['region_wb'].notna()].copy()

    for region in CUSTOM_ORDER:
        rsub = sub[sub['region_wb'] == region]
        if len(rsub) == 0:
            continue
        fc = REGION_FILL.get(region, REGION_COLORS[region])
        ax.scatter(rsub[x_var], rsub[y_var],
                   c=REGION_COLORS[region], marker=REGION_MARKERS[region],
                   s=40, facecolors=fc, edgecolors=REGION_COLORS[region],
                   linewidths=1, label=region, zorder=3)

    # Overall OLS
    res = add_ols_line(ax, sub[x_var].values, sub[y_var].values,
                       color='black', alpha=0.2)
    r2_text = f"R² = {res.rsquared:.3f}"

    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_ylim(-7, 1)
    if xlim:
        ax.set_xlim(xlim)
    ax.tick_params(labelsize=10)

    # R² annotation
    x_pos = xlim[0] if xlim else ax.get_xlim()[0]
    ax.text(x_pos + 0.02 * abs(ax.get_xlim()[1] - ax.get_xlim()[0]),
            -6.5, r2_text, fontsize=14, fontweight='bold')

    ax.legend(fontsize=7, loc='upper right', ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, fig_name), dpi=150)
    plt.close()
    print(f"  {fig_name}: {r2_text} (N={len(sub)})")


# Figure 2a
make_figure2('hc_increase', 'tfr_change',
             'Difference in Human Capital (2010\u20131960)',
             'Difference in TFR (2010\u20131960)', 'Figure2a.png')

# Figure 2b
make_figure2('loggdp_change', 'tfr_change',
             'Difference in Log GDP pc (2010\u20131960)',
             'Difference in TFR (2010\u20131960)', 'Figure2b.png')

# Figure 2c
make_figure2('chmort_change', 'tfr_change',
             'Difference in Child Mortality (2010\u20131960)',
             'Difference in TFR (2010\u20131960)', 'Figure2c.png')

# Figure 2d
make_figure2('matmort_change', 'tfr_change_1985',
             'Difference in Maternal Mortality (2010\u20131985)',
             'Difference in TFR (2010\u20131985)', 'Figure2d.png',
             xlim=(-1500, 500))


# ══════════════════════════════════════════════════════════════════════
# FIGURE 4: Predicted vs Actual TFR change (by family institutions)
# ══════════════════════════════════════════════════════════════════════

PRED_VARS = ['dlgdppc', 'dsec', 'dchildmort', 'dmatmort']


def make_figure4(sub, regions_to_show, fig_name, pred_vars=PRED_VARS):
    """Create Figure 4-style actual vs predicted TFR change plot."""
    fig, ax = plt.subplots(figsize=(7, 5))

    # Drop NaN in regression variables
    dep = 'dtfr'
    all_vars = [dep] + pred_vars
    sub_clean = sub.dropna(subset=all_vars).copy()
    sub_clean = sub_clean[sub_clean['region_wb'].isin(regions_to_show)]

    if len(sub_clean) < len(pred_vars) + 2:
        print(f"  {fig_name}: SKIPPED (only {len(sub_clean)} obs)")
        plt.close()
        return None

    # Fit OLS
    X = sm.add_constant(sub_clean[pred_vars])
    res = sm.OLS(sub_clean[dep], X).fit()
    sub_clean['dtfr_hat'] = res.predict(X)

    r2_text = f"R² = {res.rsquared:.3f}"

    # Plot by region
    for region in CUSTOM_ORDER:
        if region not in regions_to_show:
            continue
        rsub = sub_clean[sub_clean['region_wb'] == region]
        if len(rsub) == 0:
            continue
        fc = REGION_FILL.get(region, REGION_COLORS[region])
        ax.scatter(rsub[dep], rsub['dtfr_hat'],
                   c=REGION_COLORS[region], marker=REGION_MARKERS[region],
                   s=60, facecolors=fc, edgecolors=REGION_COLORS[region],
                   linewidths=1.2, label=region, zorder=3)

    # 45-degree line
    ax.plot([-4, 1], [-4, 1], '--', color='gray', linewidth=1)

    ax.set_xlim(-4, 1)
    ax.set_ylim(-4, 1)
    ax.set_xlabel('Change in TFR', fontsize=14)
    ax.set_ylabel('Predicted Change in TFR', fontsize=14)
    ax.tick_params(labelsize=13)

    ax.text(-0.5, -3.8, r2_text, fontsize=14, fontweight='bold')

    ax.legend(fontsize=9, loc='upper left', ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, fig_name), dpi=150)
    plt.close()

    print(f"  {fig_name}: {r2_text} (N={len(sub_clean)})")
    return res


# Figure 4a: Monogamous countries (good_institution_mono == 1)
mono = df_diff[df_diff['good_institution_mono'] == 1].copy()
all_regions = [r for r in CUSTOM_ORDER]
res4a = make_figure4(mono, all_regions, 'Figure4a.png')

# Figure 4b: Non-monogamous countries (good_institution_mono == 0)
non_mono = df_diff[df_diff['good_institution_mono'] == 0].copy()
res4b = make_figure4(non_mono, all_regions, 'Figure4b.png')

# Figure 4c: SSA with partible inheritance (good_institution_part == 1)
part_ssa = df_diff[(df_diff['good_institution_part'] == 1)].copy()
res4c = make_figure4(part_ssa, ['Sub-Saharan Africa'], 'Figure4c.png')

# Figure 4d: SSA without partible inheritance (good_institution_part == 0)
nopart_ssa = df_diff[(df_diff['good_institution_part'] == 0)].copy()
res4d = make_figure4(nopart_ssa, ['Sub-Saharan Africa'], 'Figure4d.png')


# ══════════════════════════════════════════════════════════════════════
# APPENDIX FIGURES A1 and A2
# ══════════════════════════════════════════════════════════════════════

# A1a-f: Split by good_institution_mono, different predictor sets
for suffix, sub_df, regions in [
    ('A1a', mono, all_regions),
    ('A1b', non_mono, all_regions),
]:
    make_figure4(sub_df, regions, f'Figure{suffix}.png', PRED_VARS)

# A1c-d: GDP + enrollment only
for suffix, sub_df, regions in [
    ('A1c', mono, all_regions),
    ('A1d', non_mono, all_regions),
]:
    make_figure4(sub_df, regions, f'Figure{suffix}.png', ['dlgdppc', 'dsec'])

# A1e-f: Child + maternal mortality only
for suffix, sub_df, regions in [
    ('A1e', mono, all_regions),
    ('A1f', non_mono, all_regions),
]:
    make_figure4(sub_df, regions, f'Figure{suffix}.png',
                 ['dchildmort', 'dmatmort'])

# A2a-f: Split by good_institution_part, SSA only
for suffix, sub_df in [
    ('A2a', part_ssa), ('A2b', nopart_ssa),
]:
    make_figure4(sub_df, ['Sub-Saharan Africa'], f'Figure{suffix}.png',
                 PRED_VARS)

for suffix, sub_df in [
    ('A2c', part_ssa), ('A2d', nopart_ssa),
]:
    make_figure4(sub_df, ['Sub-Saharan Africa'], f'Figure{suffix}.png',
                 ['dlgdppc', 'dsec'])

for suffix, sub_df in [
    ('A2e', part_ssa), ('A2f', nopart_ssa),
]:
    make_figure4(sub_df, ['Sub-Saharan Africa'], f'Figure{suffix}.png',
                 ['dchildmort', 'dmatmort'])


print("\n" + "=" * 60)
print("02_figures.py — DONE")
print("=" * 60)
