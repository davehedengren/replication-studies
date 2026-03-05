"""
03_figures.py -- Reproduce Figures 2-6 from Federle et al. (2025)
"The Price of War" (openICPSR 238484-V1)

Part 1 (Figures 2-3): Created from sites data alone (no panel build required).
  - Figure 2: Bar plot of casualties as % of local population by war
  - Figure 3a: Histogram of log10(casualty distributions) for interstate vs. other

Part 2 (Figures 4-6): Local projection impulse response functions.
  - Figure 4: Casroles specification (war site / belligerent / third party)
  - Figure 5: Trade channel specification
  - Figure 6: Proximity specification
  These require build_panel() which loads trade_gravity.dta (~80 MB).
  If panel construction fails, errors are caught and reported.

The Stata source files are:
  src/02_export/figures/barplots.R (Figures 2-3)
  src/02_export/figures/lp.do      (Figures 4-6)
"""

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ---------------------------------------------------------------------------
# Imports from project utils
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (
    load_sites,
    OUTPUT_DIR,
)

# ---------------------------------------------------------------------------
# Local output directory (NOT the 238484-V1 directory)
# ---------------------------------------------------------------------------
REPL_OUTPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
os.makedirs(REPL_OUTPUT, exist_ok=True)


def section(title: str) -> None:
    """Print a section header."""
    print("\n" + "=" * 72)
    print(f"  {title}")
    print("=" * 72)


# ===================================================================
# PART 1: FIGURES FROM SITES DATA
# ===================================================================

# Load sites data
print("Loading sites data...")
sites_interstate = load_sites("interstate")
sites_intrastate = load_sites("intrastate")
print(f"  sites_interstate: {sites_interstate.shape[0]:>5,} rows")
print(f"  sites_intrastate: {sites_intrastate.shape[0]:>5,} rows")


# -------------------------------------------------------------------
# FIGURE 2: Casualty bar plots (interstate wars)
# -------------------------------------------------------------------
section("FIGURE 2: CASUALTY BAR PLOTS (INTERSTATE WARS)")

# Compute mean shock_caspop_home by war, sorted descending
# The R script thresholds at 0.1 (10%) for readability
cas_by_war = (
    sites_interstate
    .groupby("warname")["shock_caspop_home"]
    .mean()
    .dropna()
    .sort_values(ascending=True)
)

# Convert to percentage
cas_by_war_pct = cas_by_war * 100

# Apply threshold at 10% for readability (matching R script)
THRESHOLD_PCT = 10.0

print(f"  Total wars with casualty data: {len(cas_by_war_pct)}")
print(f"  Wars exceeding {THRESHOLD_PCT}% threshold: "
      f"{(cas_by_war_pct > THRESHOLD_PCT).sum()}")
print(f"  Range: {cas_by_war_pct.min():.4f}% to {cas_by_war_pct.max():.4f}%")

# Create figure
fig, ax = plt.subplots(figsize=(12, max(8, len(cas_by_war_pct) * 0.22)))

# Clip for readability (matching R script threshold)
cas_display = cas_by_war_pct.clip(upper=THRESHOLD_PCT)

colors = ['#c0392b' if v >= THRESHOLD_PCT else '#2c3e50'
          for v in cas_by_war_pct.values]

bars = ax.barh(range(len(cas_display)), cas_display.values, color=colors,
               edgecolor='white', linewidth=0.3)

ax.set_yticks(range(len(cas_display)))
ax.set_yticklabels(cas_display.index, fontsize=7)
ax.set_xlabel("Casualties as % of Local Population", fontsize=11)
ax.set_title("Figure 2: Casualties as Share of Local Population\n(Interstate Wars)",
             fontsize=13, fontweight='bold')

# Add note about threshold
n_clipped = (cas_by_war_pct > THRESHOLD_PCT).sum()
if n_clipped > 0:
    ax.text(0.98, 0.02,
            f"Note: {n_clipped} war(s) clipped at {THRESHOLD_PCT}% for readability.\n"
            f"Red bars indicate values exceeding threshold.",
            transform=ax.transAxes, fontsize=8, ha='right', va='bottom',
            style='italic', color='gray',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow',
                      edgecolor='gray', alpha=0.8))

ax.set_xlim(0, THRESHOLD_PCT * 1.05)
ax.xaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
ax.grid(axis='x', alpha=0.3, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
fig2_path = os.path.join(REPL_OUTPUT, "figure_2_casualty_barplot.png")
fig.savefig(fig2_path, dpi=200, bbox_inches='tight')
plt.close(fig)
print(f"  Saved: {fig2_path}")


# -------------------------------------------------------------------
# FIGURE 3a: Histogram of casualty distributions
# -------------------------------------------------------------------
section("FIGURE 3a: CASUALTY DISTRIBUTION HISTOGRAMS")

# Combine interstate and other (intrastate) sites
sites_interstate_labeled = sites_interstate.copy()
sites_interstate_labeled["war_type"] = "Interstate"

sites_intrastate_labeled = sites_intrastate.copy()
sites_intrastate_labeled["war_type"] = "Other (intrastate/extrastate)"

combined = pd.concat([sites_interstate_labeled, sites_intrastate_labeled],
                     ignore_index=True)

# Compute log10(shock_caspop_home), dropping zeros and NaN
combined_valid = combined[combined["shock_caspop_home"] > 0].copy()
combined_valid["log10_caspop"] = np.log10(combined_valid["shock_caspop_home"])

inter_vals = combined_valid.loc[
    combined_valid["war_type"] == "Interstate", "log10_caspop"
].values
other_vals = combined_valid.loc[
    combined_valid["war_type"] == "Other (intrastate/extrastate)", "log10_caspop"
].values

print(f"  Interstate observations (positive caspop): {len(inter_vals)}")
print(f"  Other observations (positive caspop): {len(other_vals)}")
print(f"  Interstate log10(caspop): mean={np.mean(inter_vals):.3f}, "
      f"sd={np.std(inter_vals):.3f}")
print(f"  Other log10(caspop):      mean={np.mean(other_vals):.3f}, "
      f"sd={np.std(other_vals):.3f}")

# Kolmogorov-Smirnov test
from scipy.stats import ks_2samp

ks_stat, ks_pval = ks_2samp(inter_vals, other_vals)
print(f"\n  Kolmogorov-Smirnov test:")
print(f"    KS statistic: {ks_stat:.4f}")
print(f"    p-value:      {ks_pval:.4f}")
print(f"    Result:       {'Reject H0 (different distributions) at 5%' if ks_pval < 0.05 else 'Cannot reject H0 (similar distributions) at 5%'}")

# Create histogram figure
fig, ax = plt.subplots(figsize=(10, 6))

bins = np.linspace(
    min(np.min(inter_vals), np.min(other_vals)),
    max(np.max(inter_vals), np.max(other_vals)),
    30,
)

ax.hist(inter_vals, bins=bins, alpha=0.6, label="Interstate",
        color='#2980b9', edgecolor='white', linewidth=0.5, density=True)
ax.hist(other_vals, bins=bins, alpha=0.6, label="Other (intrastate/extrastate)",
        color='#e74c3c', edgecolor='white', linewidth=0.5, density=True)

ax.set_xlabel("log$_{10}$(Casualties / Local Population)", fontsize=11)
ax.set_ylabel("Density", fontsize=11)
ax.set_title("Figure 3a: Distribution of Casualty Intensity\n"
             "(Interstate vs. Other Wars)",
             fontsize=13, fontweight='bold')
ax.legend(fontsize=10, framealpha=0.9)

# Add KS test annotation
ax.text(0.98, 0.95,
        f"KS test: D = {ks_stat:.3f}, p = {ks_pval:.3f}",
        transform=ax.transAxes, fontsize=9, ha='right', va='top',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow',
                  edgecolor='gray', alpha=0.8))

ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
fig3_path = os.path.join(REPL_OUTPUT, "figure_3a_casualty_histograms.png")
fig.savefig(fig3_path, dpi=200, bbox_inches='tight')
plt.close(fig)
print(f"  Saved: {fig3_path}")


# ===================================================================
# PART 2: LOCAL PROJECTION IMPULSE RESPONSE FUNCTIONS (Figures 4-6)
# ===================================================================
section("PART 2: LOCAL PROJECTION IRFs (FIGURES 4-6)")

print("  Attempting to build panel and run local projections...")
print("  This requires loading trade_gravity.dta (~80 MB).")
print("  If this fails, the error will be caught and reported.")

# Try to import the LP-related functions
try:
    from utils import build_panel, run_local_projection, compute_irf
    from utils import spec_casroles, spec_castrd, spec_casprox
    lp_imports_ok = True
    print("  LP utility functions imported successfully.")
except ImportError as e:
    lp_imports_ok = False
    print(f"  [ERROR] Could not import LP functions: {e}")


def plot_irf_panel(
    irf_dict: dict,
    depvar_label: str,
    title: str,
    filename: str,
    h_max: int = 8,
) -> None:
    """
    Plot impulse response functions for multiple lines (e.g., war site,
    belligerent, third party) on a single axes.

    Parameters
    ----------
    irf_dict : dict
        Keys are line labels (str), values are DataFrames from compute_irf()
        with columns: horizon, irf, se, ci_lower, ci_upper.
    depvar_label : str
        Y-axis label (e.g., 'log GDP x 100', 'log CPI x 100').
    title : str
        Figure title.
    filename : str
        Filename (without path) to save to REPL_OUTPUT.
    h_max : int
        Maximum horizon to plot.
    """
    colors = {
        0: '#c0392b',  # red
        1: '#2980b9',  # blue
        2: '#27ae60',  # green
        3: '#8e44ad',  # purple
        4: '#e67e22',  # orange
    }
    linestyles = ['-', '--', '-.', ':', '-']
    markers = ['o', 's', 'D', '^', 'v']

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, (label, irf_df) in enumerate(irf_dict.items()):
        color = colors.get(i, f'C{i}')
        ls = linestyles[i % len(linestyles)]
        marker = markers[i % len(markers)]

        h = irf_df["horizon"].values
        irf = irf_df["irf"].values
        ci_lo = irf_df["ci_lower"].values
        ci_hi = irf_df["ci_upper"].values

        # Plot confidence band (90% CI)
        ax.fill_between(h, ci_lo, ci_hi, alpha=0.15, color=color)
        # Plot IRF line
        ax.plot(h, irf, color=color, linestyle=ls, marker=marker,
                markersize=5, linewidth=1.8, label=label)

    # Zero line
    ax.axhline(y=0, color='black', linewidth=0.8, linestyle='-')

    ax.set_xlabel("Horizon (years after war onset)", fontsize=11)
    ax.set_ylabel(depvar_label, fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xticks(range(h_max + 1))
    ax.legend(fontsize=10, framealpha=0.9)
    ax.grid(alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    out_path = os.path.join(REPL_OUTPUT, filename)
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path}")


# -------------------------------------------------------------------
# Build the panel
# -------------------------------------------------------------------
panel = None
if lp_imports_ok:
    try:
        print("\n  Building panel (wars='interstate')...")
        panel = build_panel(wars="interstate")
        print(f"  Panel built successfully: {panel.shape[0]:,} rows x "
              f"{panel.shape[1]} cols")
    except Exception as e:
        print(f"  [ERROR] build_panel() failed: {type(e).__name__}: {e}")
        print("  Figures 4-6 will be skipped.")


# -------------------------------------------------------------------
# FIGURE 4: Casroles specification
# -------------------------------------------------------------------
if panel is not None:
    section("FIGURE 4: CASROLES IRFs (WAR SITE / BELLIGERENT / THIRD PARTY)")

    try:
        xvars, lincom_specs = spec_casroles(cas=0.02, integration=0.03)
        print(f"  xvars: {xvars}")
        print(f"  IRF lines: {list(lincom_specs.keys())}")

        for depvar, depvar_label in [("lgdp", "log GDP x 100"),
                                     ("lcpi", "log CPI x 100")]:
            print(f"\n  Running LP for depvar = {depvar}...")

            lp_results = run_local_projection(
                panel=panel,
                depvar=depvar,
                xvars=xvars,
                lags=4,
                h_max=8,
                confidence=0.90,
            )

            print(f"  LP completed. Horizons estimated: {len(lp_results)}")
            print(f"  Obs range: {lp_results['N'].min()} - {lp_results['N'].max()}")

            # Compute IRFs for each line
            irf_dict = {}
            for label, lincom_expr in lincom_specs.items():
                irf_df = compute_irf(lp_results, lincom_expr, confidence=0.90)
                irf_dict[label] = irf_df
                # Print summary
                h0_val = irf_df.loc[irf_df["horizon"] == 0, "irf"].values
                h8_val = irf_df.loc[irf_df["horizon"] == 8, "irf"].values
                print(f"    {label:15s}: h=0 = {h0_val[0]:>8.4f}, "
                      f"h=8 = {h8_val[0]:>8.4f}")

            plot_irf_panel(
                irf_dict=irf_dict,
                depvar_label=depvar_label,
                title=f"Figure 4: Casroles IRF -- {depvar.upper()}\n"
                      f"(War site / Belligerent / Third party, 90% CI)",
                filename=f"figure_4_casroles_{depvar}.png",
                h_max=8,
            )

    except Exception as e:
        print(f"  [ERROR] Figure 4 failed: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()


# -------------------------------------------------------------------
# FIGURE 5: Trade channel specification
# -------------------------------------------------------------------
if panel is not None:
    section("FIGURE 5: TRADE CHANNEL IRFs")

    try:
        xvars_trd, lincom_trd = spec_castrd(cas=0.02, integration=0.03)
        print(f"  xvars: {xvars_trd}")
        print(f"  IRF lines: {list(lincom_trd.keys())}")

        for depvar, depvar_label in [("lgdp", "log GDP x 100"),
                                     ("lcpi", "log CPI x 100")]:
            print(f"\n  Running LP for depvar = {depvar}...")

            lp_results = run_local_projection(
                panel=panel,
                depvar=depvar,
                xvars=xvars_trd,
                lags=4,
                h_max=8,
                confidence=0.90,
            )

            print(f"  LP completed. Horizons estimated: {len(lp_results)}")

            irf_dict = {}
            for label, lincom_expr in lincom_trd.items():
                irf_df = compute_irf(lp_results, lincom_expr, confidence=0.90)
                irf_dict[label] = irf_df
                h0_val = irf_df.loc[irf_df["horizon"] == 0, "irf"].values
                h8_val = irf_df.loc[irf_df["horizon"] == 8, "irf"].values
                print(f"    {label:30s}: h=0 = {h0_val[0]:>8.4f}, "
                      f"h=8 = {h8_val[0]:>8.4f}")

            plot_irf_panel(
                irf_dict=irf_dict,
                depvar_label=depvar_label,
                title=f"Figure 5: Trade Channel IRF -- {depvar.upper()}\n"
                      f"(Third party: 0% vs. 3% trade exposure, 90% CI)",
                filename=f"figure_5_castrd_{depvar}.png",
                h_max=8,
            )

    except Exception as e:
        print(f"  [ERROR] Figure 5 failed: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()


# -------------------------------------------------------------------
# FIGURE 6: Proximity specification
# -------------------------------------------------------------------
if panel is not None:
    section("FIGURE 6: PROXIMITY CHANNEL IRFs")

    try:
        xvars_prox, lincom_prox = spec_casprox(cas=0.02)
        print(f"  xvars: {xvars_prox}")
        print(f"  IRF lines: {list(lincom_prox.keys())}")

        for depvar, depvar_label in [("lgdp", "log GDP x 100"),
                                     ("lcpi", "log CPI x 100")]:
            print(f"\n  Running LP for depvar = {depvar}...")

            lp_results = run_local_projection(
                panel=panel,
                depvar=depvar,
                xvars=xvars_prox,
                lags=4,
                h_max=8,
                confidence=0.90,
            )

            print(f"  LP completed. Horizons estimated: {len(lp_results)}")

            irf_dict = {}
            for label, lincom_expr in lincom_prox.items():
                irf_df = compute_irf(lp_results, lincom_expr, confidence=0.90)
                irf_dict[label] = irf_df
                h0_val = irf_df.loc[irf_df["horizon"] == 0, "irf"].values
                h8_val = irf_df.loc[irf_df["horizon"] == 8, "irf"].values
                print(f"    {label:25s}: h=0 = {h0_val[0]:>8.4f}, "
                      f"h=8 = {h8_val[0]:>8.4f}")

            plot_irf_panel(
                irf_dict=irf_dict,
                depvar_label=depvar_label,
                title=f"Figure 6: Proximity Channel IRF -- {depvar.upper()}\n"
                      f"(Third party: non-neighbors vs. neighbors, 90% CI)",
                filename=f"figure_6_casprox_{depvar}.png",
                h_max=8,
            )

    except Exception as e:
        print(f"  [ERROR] Figure 6 failed: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()


# ===================================================================
# SUMMARY
# ===================================================================
section("SUMMARY")

# List all files written
output_files = sorted(
    f for f in os.listdir(REPL_OUTPUT) if f.endswith(".png")
)
print(f"  Output directory: {REPL_OUTPUT}")
print(f"  Files generated ({len(output_files)}):")
for f in output_files:
    fpath = os.path.join(REPL_OUTPUT, f)
    size_kb = os.path.getsize(fpath) / 1024
    print(f"    {f:50s}  ({size_kb:.0f} KB)")

if panel is None:
    print("\n  Note: Figures 4-6 (LP IRFs) were NOT generated because")
    print("  build_panel() failed or could not be imported.")
    print("  This typically happens because trade_gravity.dta (~80 MB)")
    print("  could not be loaded. To generate these figures, ensure")
    print("  the full replication data package is available at:")
    print(f"    {OUTPUT_DIR.parent.parent}")

print()
