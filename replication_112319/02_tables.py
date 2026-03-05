"""
02_tables.py - Reproduce Tables 1-8 from Charness & Levin (2005)
"""
import sys
sys.path.insert(0, '.')
from utils import *
from scipy import stats
import statsmodels.api as sm

df = load_data_fast()

def fmt(errors, total):
    pct = errors / total * 100 if total > 0 else 0
    return f"{int(errors)}/{total} ({pct:.1f}%)"


# ══════════════════════════════════════════════════════════════════════
# TABLE 1: Treatment 1 Switching-Error Rates
# ══════════════════════════════════════════════════════════════════════
print("=" * 80)
print("TABLE 1 - Treatment 1 Switching-Error Rates")
print("=" * 80)

t1 = df[df['treatment'] == 1]

print("\nAfter a Right draw:")
print(f"{'Draw':<10} {'Phase I':<20} {'Phase II':<20} {'Phase III':<20} {'Aggregated':<20}")
for draw_name, black_val in [('Black', 1), ('White', 0)]:
    results = []
    for phase in ['I', 'II', 'III']:
        mask = (t1['phase'] == phase) & (t1['Left1'] == 0) & (t1['first_draw_black'] == black_val)
        sub = t1[mask]
        results.append(fmt(sub['switching_error'].sum(), len(sub)))
    # Aggregated
    mask = (t1['Left1'] == 0) & (t1['first_draw_black'] == black_val)
    sub = t1[mask]
    agg = fmt(sub['switching_error'].sum(), len(sub))
    print(f"{draw_name:<10} {results[0]:<20} {results[1]:<20} {results[2]:<20} {agg:<20}")

# Combined
results = []
for phase in ['I', 'II', 'III']:
    mask = (t1['phase'] == phase) & (t1['Left1'] == 0)
    sub = t1[mask]
    results.append(fmt(sub['switching_error'].sum(), len(sub)))
mask = (t1['Left1'] == 0)
sub = t1[mask]
agg = fmt(sub['switching_error'].sum(), len(sub))
print(f"{'Combined':<10} {results[0]:<20} {results[1]:<20} {results[2]:<20} {agg:<20}")

print("\nAfter a Left draw:")
print(f"{'Draw':<10} {'Phase I':<20} {'Phase II':<20} {'Agg I+II':<20} {'Phase III':<20}")
for draw_name, black_val in [('Black', 1), ('White', 0)]:
    results = []
    for phase in ['I', 'II']:
        mask = (t1['phase'] == phase) & (t1['Left1'] == 1) & (t1['first_draw_black'] == black_val)
        sub = t1[mask]
        results.append(fmt(sub['switching_error'].sum(), len(sub)))
    # Aggregated I+II
    mask = t1['phase'].isin(['I', 'II']) & (t1['Left1'] == 1) & (t1['first_draw_black'] == black_val)
    sub = t1[mask]
    agg12 = fmt(sub['switching_error'].sum(), len(sub))
    # Phase III
    mask = (t1['phase'] == 'III') & (t1['Left1'] == 1) & (t1['first_draw_black'] == black_val)
    sub = t1[mask]
    p3 = fmt(sub['switching_error'].sum(), len(sub))
    print(f"{draw_name:<10} {results[0]:<20} {results[1]:<20} {agg12:<20} {p3:<20}")

# Combined Left
results = []
for phase in ['I', 'II']:
    mask = (t1['phase'] == phase) & (t1['Left1'] == 1)
    sub = t1[mask]
    results.append(fmt(sub['switching_error'].sum(), len(sub)))
mask = t1['phase'].isin(['I', 'II']) & (t1['Left1'] == 1)
sub = t1[mask]
agg12 = fmt(sub['switching_error'].sum(), len(sub))
mask = (t1['phase'] == 'III') & (t1['Left1'] == 1)
sub = t1[mask]
p3 = fmt(sub['switching_error'].sum(), len(sub))
print(f"{'Combined':<10} {results[0]:<20} {results[1]:<20} {agg12:<20} {p3:<20}")

print("\nPublished Table 1 comparison (key cells):")
pub_t1 = {
    ('Right', 'Black', 'I'): (12, 286), ('Right', 'Black', 'II'): (20, 596),
    ('Right', 'Black', 'III'): (20, 161), ('Right', 'White', 'I'): (15, 304),
    ('Right', 'White', 'II'): (23, 683), ('Right', 'White', 'III'): (7, 160),
    ('Right', 'Combined', 'I'): (27, 590), ('Right', 'Combined', 'II'): (43, 1279),
    ('Right', 'Combined', 'III'): (27, 321),
    ('Left', 'Black', 'I'): (179, 332), ('Left', 'Black', 'II'): (180, 272),
    ('Left', 'White', 'I'): (98, 258), ('Left', 'White', 'II'): (63, 219),
    ('Left', 'Black', 'I+II'): (359, 604), ('Left', 'White', 'I+II'): (161, 477),
    ('Left', 'Combined', 'I'): (277, 590), ('Left', 'Combined', 'II'): (243, 491),
    ('Left', 'Combined', 'I+II'): (520, 1081),
    ('Left', 'Black', 'III'): (31, 157), ('Left', 'White', 'III'): (17, 112),
    ('Left', 'Combined', 'III'): (48, 269),
}
for (side, draw, phase), (pub_err, pub_n) in sorted(pub_t1.items()):
    if draw == 'Combined':
        if phase in ('I', 'II', 'III'):
            mask = (t1['phase'] == phase) & (t1['Left1'] == (1 if side == 'Left' else 0))
        else:  # I+II
            mask = t1['phase'].isin(['I', 'II']) & (t1['Left1'] == (1 if side == 'Left' else 0))
    else:
        bv = 1 if draw == 'Black' else 0
        if phase in ('I', 'II', 'III'):
            mask = (t1['phase'] == phase) & (t1['Left1'] == (1 if side == 'Left' else 0)) & (t1['first_draw_black'] == bv)
        else:
            mask = t1['phase'].isin(['I', 'II']) & (t1['Left1'] == (1 if side == 'Left' else 0)) & (t1['first_draw_black'] == bv)
    sub = t1[mask]
    my_err = int(sub['switching_error'].sum())
    my_n = len(sub)
    match_str = "✓" if my_err == pub_err and my_n == pub_n else "✗"
    if my_err != pub_err or my_n != pub_n:
        print(f"  {match_str} {side} {draw} Ph{phase}: pub={pub_err}/{pub_n}, mine={my_err}/{my_n}")


# ══════════════════════════════════════════════════════════════════════
# TABLE 2: Treatment 2 Switching-Error Rates
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("TABLE 2 - Treatment 2 Switching-Error Rates")
print("=" * 80)

t2 = df[df['treatment'] == 2]

print("\nAfter a Right draw:")
print(f"{'Draw':<10} {'Phase I':<20} {'Phase II':<20} {'Phase III':<20} {'Aggregated':<20}")
for draw_name, black_val in [('Black', 1), ('White', 0)]:
    results = []
    for phase in ['I', 'II', 'III']:
        mask = (t2['phase'] == phase) & (t2['Left1'] == 0) & (t2['first_draw_black'] == black_val)
        sub = t2[mask]
        results.append(fmt(sub['switching_error'].sum(), len(sub)))
    mask = (t2['Left1'] == 0) & (t2['first_draw_black'] == black_val)
    sub = t2[mask]
    agg = fmt(sub['switching_error'].sum(), len(sub))
    print(f"{draw_name:<10} {results[0]:<20} {results[1]:<20} {results[2]:<20} {agg:<20}")

# Combined Right
results = []
for phase in ['I', 'II', 'III']:
    mask = (t2['phase'] == phase) & (t2['Left1'] == 0)
    sub = t2[mask]
    results.append(fmt(sub['switching_error'].sum(), len(sub)))
mask = (t2['Left1'] == 0)
sub = t2[mask]
agg = fmt(sub['switching_error'].sum(), len(sub))
print(f"{'Combined':<10} {results[0]:<20} {results[1]:<20} {results[2]:<20} {agg:<20}")

print("\nAfter a Left draw:")
print(f"{'Draw':<10} {'Phase I':<20} {'Phase II':<20} {'Agg I+II':<20} {'Phase III':<20}")
for draw_name, black_val in [('Black', 1), ('White', 0)]:
    results = []
    for phase in ['I', 'II']:
        mask = (t2['phase'] == phase) & (t2['Left1'] == 1) & (t2['first_draw_black'] == black_val)
        sub = t2[mask]
        results.append(fmt(sub['switching_error'].sum(), len(sub)))
    mask = t2['phase'].isin(['I', 'II']) & (t2['Left1'] == 1) & (t2['first_draw_black'] == black_val)
    sub = t2[mask]
    agg12 = fmt(sub['switching_error'].sum(), len(sub))
    mask = (t2['phase'] == 'III') & (t2['Left1'] == 1) & (t2['first_draw_black'] == black_val)
    sub = t2[mask]
    p3 = fmt(sub['switching_error'].sum(), len(sub))
    print(f"{draw_name:<10} {results[0]:<20} {results[1]:<20} {agg12:<20} {p3:<20}")

# Combined Left
results = []
for phase in ['I', 'II']:
    mask = (t2['phase'] == phase) & (t2['Left1'] == 1)
    sub = t2[mask]
    results.append(fmt(sub['switching_error'].sum(), len(sub)))
mask = t2['phase'].isin(['I', 'II']) & (t2['Left1'] == 1)
sub = t2[mask]
agg12 = fmt(sub['switching_error'].sum(), len(sub))
mask = (t2['phase'] == 'III') & (t2['Left1'] == 1)
sub = t2[mask]
p3 = fmt(sub['switching_error'].sum(), len(sub))
print(f"{'Combined':<10} {results[0]:<20} {results[1]:<20} {agg12:<20} {p3:<20}")

print("\nPublished Table 2 comparison (discrepancies):")
pub_t2 = {
    ('Right', 'Black', 'I'): (32, 255), ('Right', 'Black', 'II'): (31, 642),
    ('Right', 'Black', 'III'): (20, 178), ('Right', 'White', 'I'): (12, 285),
    ('Right', 'White', 'II'): (16, 627), ('Right', 'White', 'III'): (3, 161),
    ('Right', 'Combined', 'I'): (44, 540), ('Right', 'Combined', 'II'): (47, 1269),
    ('Right', 'Combined', 'III'): (23, 339),
    ('Left', 'Black', 'I'): (95, 258), ('Left', 'Black', 'II'): (75, 155),
    ('Left', 'White', 'I'): (157, 282), ('Left', 'White', 'II'): (102, 196),
    ('Left', 'Black', 'I+II'): (170, 413), ('Left', 'White', 'I+II'): (259, 478),
    ('Left', 'Combined', 'I'): (252, 540), ('Left', 'Combined', 'II'): (177, 351),
    ('Left', 'Combined', 'I+II'): (429, 891),
    ('Left', 'Black', 'III'): (69, 99), ('Left', 'White', 'III'): (30, 102),
    ('Left', 'Combined', 'III'): (99, 201),
}
for (side, draw, phase), (pub_err, pub_n) in sorted(pub_t2.items()):
    if draw == 'Combined':
        if phase in ('I', 'II', 'III'):
            mask = (t2['phase'] == phase) & (t2['Left1'] == (1 if side == 'Left' else 0))
        else:
            mask = t2['phase'].isin(['I', 'II']) & (t2['Left1'] == (1 if side == 'Left' else 0))
    else:
        bv = 1 if draw == 'Black' else 0
        if phase in ('I', 'II', 'III'):
            mask = (t2['phase'] == phase) & (t2['Left1'] == (1 if side == 'Left' else 0)) & (t2['first_draw_black'] == bv)
        else:
            mask = t2['phase'].isin(['I', 'II']) & (t2['Left1'] == (1 if side == 'Left' else 0)) & (t2['first_draw_black'] == bv)
    sub = t2[mask]
    my_err = int(sub['switching_error'].sum())
    my_n = len(sub)
    if my_err != pub_err or my_n != pub_n:
        print(f"  ✗ {side} {draw} Ph{phase}: pub={pub_err}/{pub_n}, mine={my_err}/{my_n}")


# ══════════════════════════════════════════════════════════════════════
# TABLE 3: Treatment 3 Switching-Error Rates
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("TABLE 3 - Treatment 3 Switching-Error Rates")
print("=" * 80)

t3 = df[df['treatment'] == 3]

print(f"\n{'Draw':<15} {'Phase I':<20} {'Phase II':<20} {'Combined':<20}")
for draw_name, fav_val in [('Favorable', 1), ('Unfavorable', 0)]:
    results = []
    for phase in ['I', 'II']:
        mask = (t3['phase'] == phase) & (t3['first_draw_favorable'] == fav_val)
        sub = t3[mask]
        results.append(fmt(sub['switching_error'].sum(), len(sub)))
    mask = (t3['first_draw_favorable'] == fav_val)
    sub = t3[mask]
    comb = fmt(sub['switching_error'].sum(), len(sub))
    print(f"{draw_name:<15} {results[0]:<20} {results[1]:<20} {comb:<20}")

# Combined
results = []
for phase in ['I', 'II']:
    mask = (t3['phase'] == phase)
    sub = t3[mask]
    results.append(fmt(sub['switching_error'].sum(), len(sub)))
mask_all = t3.index
comb = fmt(t3['switching_error'].sum(), len(t3))
print(f"{'Combined':<15} {results[0]:<20} {results[1]:<20} {comb:<20}")

print("\nPublished Table 3 comparison:")
pub_t3 = {
    ('Favorable', 'I'): (245, 1811), ('Favorable', 'II'): (78, 259),
    ('Favorable', 'Combined'): (323, 2070),
    ('Unfavorable', 'I'): (776, 1829), ('Unfavorable', 'II'): (96, 261),
    ('Unfavorable', 'Combined'): (872, 2090),
    ('Combined', 'I'): (1021, 3640), ('Combined', 'II'): (174, 520),
    ('Combined', 'Combined'): (1195, 4160),
}
for (draw, phase), (pub_err, pub_n) in sorted(pub_t3.items()):
    if draw == 'Combined':
        if phase == 'Combined':
            sub = t3
        else:
            sub = t3[t3['phase'] == phase]
    else:
        fv = 1 if draw == 'Favorable' else 0
        if phase == 'Combined':
            sub = t3[t3['first_draw_favorable'] == fv]
        else:
            sub = t3[(t3['phase'] == phase) & (t3['first_draw_favorable'] == fv)]
    my_err = int(sub['switching_error'].sum())
    my_n = len(sub)
    if my_err != pub_err or my_n != pub_n:
        print(f"  ✗ {draw} Ph{phase}: pub={pub_err}/{pub_n}, mine={my_err}/{my_n}")
    else:
        print(f"  ✓ {draw} Ph{phase}: {my_err}/{my_n}")


# ══════════════════════════════════════════════════════════════════════
# TABLE 4: Starting-Error Rates
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("TABLE 4 - Starting-Error Rates")
print("=" * 80)

print(f"\n{'Treatment':<12} {'Phase':<8} {'BEU-start':<12} {'Error Rate':<20}")
pub_t4 = [(1, 'II', 'R', 491, 1770), (2, 'II', 'R', 351, 1620),
           (1, 'III', 'L', 321, 590), (2, 'III', 'R', 201, 540)]
for t, phase, beu, pub_err, pub_n in pub_t4:
    sub = df[(df['treatment'] == t) & (df['phase'] == phase)]
    my_err = int(sub['starting_error'].sum())
    my_n = len(sub)
    match = "✓" if my_err == pub_err and my_n == pub_n else "✗"
    print(f"  {t:<12} {phase:<8} {beu:<12} mine={my_err}/{my_n}, pub={pub_err}/{pub_n} {match}")


# ══════════════════════════════════════════════════════════════════════
# TABLE 5: Spearman's rho for Measures in Treatments 1 and 2
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("TABLE 5 - Spearman's rho for Measures in Treatments 1 and 2")
print("=" * 80)

t12 = df[df['treatment'].isin([1, 2])]

# Compute individual-level measures for Phase II
# Need to compute per-individual switching rates and cross-period measures

# For each individual, compute:
# LBL = switching error rate after Left Black in Phase II (within-period)
# LWR = switching error rate after Left White in Phase II (within-period)
# R rate = switching error rate after Right starts (voluntary) in Phase II

# Cross-period measures:
# L diff = difference in P(start Left next period | Left Black this period) - P(start Left | Left White)
# Diff 20 = rate of switching starting side after 0-1 black draws vs after 2 black draws (Phase II)
# Diff 210 = similar measure

individuals = sorted(t12['Individual'].unique())
measures = []

for ind in individuals:
    sub = t12[t12['Individual'] == ind]
    phase2 = sub[sub['phase'] == 'II']

    # LBL: error rate after Left Black in Phase II
    lb = phase2[(phase2['Left1'] == 1) & (phase2['first_draw_black'] == 1)]
    lbl_rate = lb['switching_error'].mean() if len(lb) > 0 else np.nan

    # LWR: error rate after Left White in Phase II
    lw = phase2[(phase2['Left1'] == 1) & (phase2['first_draw_black'] == 0)]
    lwr_rate = lw['switching_error'].mean() if len(lw) > 0 else np.nan

    # R rate: error rate after voluntary Right draws in Phase II
    r_vol = phase2[phase2['Left1'] == 0]
    r_rate = r_vol['switching_error'].mean() if len(r_vol) > 0 else np.nan

    # Cross-period: L diff
    # P(start Left period t+1 | started Left and drew Black period t)
    # vs P(start Left period t+1 | started Left and drew White period t)
    # Only for Phase II voluntary starts
    phase2_sorted = phase2.sort_values('Round')
    rounds = phase2_sorted['Round'].values
    left1_vals = phase2_sorted['Left1'].values
    black_vals = phase2_sorted['first_draw_black'].values

    after_lb_starts = []
    after_lw_starts = []
    for i in range(len(rounds) - 1):
        if left1_vals[i] == 1 and black_vals[i] == 1:
            after_lb_starts.append(left1_vals[i + 1])
        elif left1_vals[i] == 1 and black_vals[i] == 0:
            after_lw_starts.append(left1_vals[i + 1])

    p_left_after_lb = np.mean(after_lb_starts) if len(after_lb_starts) > 0 else np.nan
    p_left_after_lw = np.mean(after_lw_starts) if len(after_lw_starts) > 0 else np.nan
    l_diff = p_left_after_lb - p_left_after_lw if not (np.isnan(p_left_after_lb) or np.isnan(p_left_after_lw)) else np.nan

    # Diff 20: switch rate after 0-1 black draws vs 2 black draws
    # Count black draws per period: Cyan1 + Cyan2 (for T1/T2 where Cyan_Pays=1)
    phase2_sorted = phase2_sorted.copy()
    phase2_sorted['n_black'] = phase2_sorted['Cyan1'] + phase2_sorted['Cyan2']

    switch_after = {}
    for i in range(len(phase2_sorted) - 1):
        n_b = phase2_sorted.iloc[i]['n_black']
        next_left = phase2_sorted.iloc[i + 1]['Left1']
        curr_left = phase2_sorted.iloc[i]['Left1']
        switched_start = int(next_left != curr_left)
        if n_b not in switch_after:
            switch_after[n_b] = []
        switch_after[n_b].append(switched_start)

    rate_01 = np.mean(switch_after.get(0, []) + switch_after.get(1, [])) if (0 in switch_after or 1 in switch_after) else np.nan
    rate_2 = np.mean(switch_after.get(2, [])) if 2 in switch_after else np.nan
    diff_20 = rate_01 - rate_2 if not (np.isnan(rate_01) or np.isnan(rate_2)) else np.nan

    # Diff 210 uses all three levels
    rate_0 = np.mean(switch_after.get(0, [])) if 0 in switch_after else np.nan
    rate_1_val = np.mean(switch_after.get(1, [])) if 1 in switch_after else np.nan
    # Diff210 = weakly monotonic decline: rate_0 >= rate_1 >= rate_2
    diff_210 = rate_01 - rate_2 if not (np.isnan(rate_01) or np.isnan(rate_2)) else np.nan

    measures.append({
        'Individual': ind,
        'LBL': lbl_rate,
        'LWR': lwr_rate,
        'R_rate': r_rate,
        'L_diff': l_diff,
        'Diff_20': diff_20,
        'Diff_210': diff_210,
    })

mdf = pd.DataFrame(measures)

# Compute Spearman correlations
vars_order = ['L_diff', 'Diff_20', 'LBL', 'LWR', 'Diff_210', 'R_rate']
var_labels = ['L diff', 'Diff 20', 'LBL', 'LWR', 'Diff 210', 'R rate']

print(f"\n{'':>12}", end='')
for v in var_labels:
    print(f"{v:>12}", end='')
print()

for i, vi in enumerate(vars_order):
    print(f"{var_labels[i]:>12}", end='')
    for j, vj in enumerate(vars_order):
        if j <= i:
            print(f"{'':>12}", end='')
        else:
            valid = mdf[[vi, vj]].dropna()
            if len(valid) >= 3:
                rho, pval = stats.spearmanr(valid[vi], valid[vj])
                stars = '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.10 else ''
                print(f"{rho:>8.3f}{stars:<4}", end='')
            else:
                print(f"{'N/A':>12}", end='')
    print()

print("\nPublished Table 5 key values:")
print("  Diff20-Ldiff: .417***, LBL-Ldiff: .152, LWR-Ldiff: .229***")
print("  LBL-Diff20: .072, LWR-Diff20: .254**, R_rate-Diff20: .294***")


# ══════════════════════════════════════════════════════════════════════
# TABLE 6: Spearman's rho for Measures in Treatment 3
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("TABLE 6 - Spearman's rho for Measures in Treatment 3")
print("=" * 80)

t3 = df[df['treatment'] == 3]
t3_measures = []

for ind in sorted(t3['Individual'].unique()):
    sub = t3[t3['Individual'] == ind]

    # LFL: switching error rate after favorable Left draw
    # LUR: switching error rate after unfavorable Left draw
    lfl_data = sub[sub['first_draw_favorable'] == 1]
    lfl_rate = lfl_data['switching_error'].mean() if len(lfl_data) > 0 else np.nan

    lur_data = sub[sub['first_draw_favorable'] == 0]
    lur_rate = lur_data['switching_error'].mean() if len(lur_data) > 0 else np.nan

    # L2 diff: cross-period measure - P(Left2 next | favorable this) - P(Left2 next | unfavorable this)
    sub_sorted = sub.sort_values('Round')
    favorable_vals = sub_sorted['first_draw_favorable'].values
    left2_vals = sub_sorted['Left2'].values

    after_fav = []
    after_unfav = []
    for i in range(len(sub_sorted) - 1):
        if favorable_vals[i] == 1:
            after_fav.append(left2_vals[i + 1])
        else:
            after_unfav.append(left2_vals[i + 1])

    p_left2_after_fav = np.mean(after_fav) if len(after_fav) > 0 else np.nan
    p_left2_after_unfav = np.mean(after_unfav) if len(after_unfav) > 0 else np.nan
    l2_diff = p_left2_after_fav - p_left2_after_unfav if not (np.isnan(p_left2_after_fav) or np.isnan(p_left2_after_unfav)) else np.nan

    # Diff 10: switching rate based on outcomes
    n_success = sub_sorted['first_draw_favorable'].values
    left2_this = sub_sorted['Left2'].values

    switch_after = {}
    for i in range(len(sub_sorted) - 1):
        ns = int(n_success[i])
        next_left2 = left2_this[i + 1]
        curr_left2 = left2_this[i]
        switched = int(next_left2 != curr_left2)
        if ns not in switch_after:
            switch_after[ns] = []
        switch_after[ns].append(switched)

    rate_0 = np.mean(switch_after.get(0, [])) if 0 in switch_after else np.nan
    rate_1 = np.mean(switch_after.get(1, [])) if 1 in switch_after else np.nan
    diff_10 = rate_0 - rate_1 if not (np.isnan(rate_0) or np.isnan(rate_1)) else np.nan

    t3_measures.append({
        'Individual': ind,
        'LFL': lfl_rate,
        'LUR': lur_rate,
        'L2_diff': l2_diff,
        'Diff_10': diff_10,
    })

t3_mdf = pd.DataFrame(t3_measures)

vars_order3 = ['L2_diff', 'Diff_10', 'LFL', 'LUR']
var_labels3 = ['L2 diff', 'Diff 10', 'LFL', 'LUR']

print(f"\n{'':>12}", end='')
for v in var_labels3:
    print(f"{v:>12}", end='')
print()

for i, vi in enumerate(vars_order3):
    print(f"{var_labels3[i]:>12}", end='')
    for j, vj in enumerate(vars_order3):
        if j <= i:
            print(f"{'':>12}", end='')
        else:
            valid = t3_mdf[[vi, vj]].dropna()
            if len(valid) >= 3:
                rho, pval = stats.spearmanr(valid[vi], valid[vj])
                stars = '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.10 else ''
                print(f"{rho:>8.3f}{stars:<4}", end='')
            else:
                print(f"{'N/A':>12}", end='')
    print()

print("\nPublished Table 6:")
print("  Diff10-L2diff: .362***, LFL-L2diff: .172, LUR-L2diff: .013")
print("  LFL-Diff10: .118, LUR-Diff10: -.050, LUR-LFL: .531***")


# ══════════════════════════════════════════════════════════════════════
# TABLE 7: Switching-Error Rates over Time Segments
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("TABLE 7 - Switching-Error Rates over Time Segments")
print("=" * 80)

print("\nTreatment 1 Switching-error Rates (Phase II segments):")
print(f"{'Init Draw':<12} {'Periods 21-30':<16} {'Periods 31-40':<16} {'Periods 41-50':<16}")
t1_p2 = t1[t1['phase'] == 'II']
for draw_name, left_val, black_val in [('RB', 0, 1), ('RW', 0, 0), ('LB', 1, 1), ('LW', 1, 0)]:
    results = []
    for r_min, r_max in [(21, 30), (31, 40), (41, 50)]:
        mask = (t1_p2['Round'] >= r_min) & (t1_p2['Round'] <= r_max) & \
               (t1_p2['Left1'] == left_val) & (t1_p2['first_draw_black'] == black_val)
        sub = t1_p2[mask]
        rate = sub['switching_error'].mean() * 100 if len(sub) > 0 else 0
        results.append(f"{rate:.1f}%")
    print(f"{draw_name:<12} {results[0]:<16} {results[1]:<16} {results[2]:<16}")

print("\nPublished T1: RB: 5.0/2.5/2.6, RW: 4.4/3.3/2.3, LB: 60.4/61.4/77.8, LW: 27.8/26.7/33.8")

print("\nTreatment 2 Switching-error Rates (Phase II segments):")
print(f"{'Init Draw':<12} {'Periods 21-30':<16} {'Periods 31-40':<16} {'Periods 41-50':<16}")
t2_p2 = t2[t2['phase'] == 'II']
for draw_name, left_val, black_val in [('RB', 0, 1), ('RW', 0, 0), ('LB', 1, 1), ('LW', 1, 0)]:
    results = []
    for r_min, r_max in [(21, 30), (31, 40), (41, 50)]:
        mask = (t2_p2['Round'] >= r_min) & (t2_p2['Round'] <= r_max) & \
               (t2_p2['Left1'] == left_val) & (t2_p2['first_draw_black'] == black_val)
        sub = t2_p2[mask]
        rate = sub['switching_error'].mean() * 100 if len(sub) > 0 else 0
        results.append(f"{rate:.1f}%")
    print(f"{draw_name:<12} {results[0]:<16} {results[1]:<16} {results[2]:<16}")

print("\nPublished T2: RB: 3.9/5.4/5.4, RW: 0.5/2.3/4.6, LB: 49.1/52.0/58.0, LW: 49.2/54.3/52.2")

print("\nTreatment 3 Switching-error Rates:")
print(f"{'Init Draw':<12} {'Periods 1-20':<16} {'Periods 21-50':<16} {'Periods 51-70':<16}")
for draw_name, fav_val in [("L'B'", 1), ("L'W'", 0)]:
    results = []
    for r_min, r_max in [(1, 20), (21, 50), (51, 70)]:
        mask = (t3['phase'] == 'I') & (t3['Round'] >= r_min) & (t3['Round'] <= r_max) & \
               (t3['first_draw_favorable'] == fav_val)
        sub = t3[mask]
        rate = sub['switching_error'].mean() * 100 if len(sub) > 0 else 0
        results.append(f"{rate:.1f}%")
    print(f"{draw_name:<12} {results[0]:<16} {results[1]:<16} {results[2]:<16}")

print("\nPublished T3: L'B': 13.4/13.9/13.1, L'W': 46.2/41.0/41.5")


# ══════════════════════════════════════════════════════════════════════
# TABLE 8: Regressions for Frequency of Error in Decisions
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("TABLE 8 - Regressions for Frequency of Error in Decisions")
print("=" * 80)

# Specs (1) and (2): OLS on population error rates vs cost
# Each observation = one of 24 possible decision errors
# Need to compute: for each error type, its frequency and expected cost

# Define all possible error types
# T1/T2 have 8 types each: {Phase I-II, III} × {Left, Right} × {Black, White} for switching
# But paper says 24 possible errors total
# Actually: for each treatment/phase/side/color combo, there's one switching error type

# Let's compute the 24 error types
error_types = []

# T1 Phases I-II: 4 switching errors (RB stay->should switch, RW switch->should stay, etc)
# Wait, the paper says there are 24 "possible errors" across all treatments
# Let me think about this differently...

# For T1:
#   Phase I-II Right: Black(stay Right, error=switch Left), White(switch Left, error=stay Right)
#   Phase I-II Left: Black(switch Right, error=stay Left), White(stay Left, error=switch Right)
#   Phase III Right: same as I-II for Right
#   Phase III Left: Black(stay Left, error=switch Right), White(stay Left, error=switch Right)
# That's 8 for T1

# For T2:
#   Phase I-II: same 4 as T1
#   Phase III: 4 types (RB, RW, LB, LW)
# That's 8 for T2

# For T3:
#   Phase I: 2 types (Favorable, Unfavorable)
#   Phase II: 2 types (Favorable, Unfavorable)
# That's 4 for T3

# Total: 8 + 8 + 4 = 20 switching errors
# Plus 4 starting errors (T1 Ph II, T2 Ph II, T1 Ph III, T2 Ph III) = 24

# For each error type, we need:
# - Population frequency (error rate)
# - Expected cost of that error (difference in EU between optimal and actual choice)

# Let me compute expected costs using the Bayesian EU calculations
# I'll focus on switching errors first

def compute_eu_second_draw(treatment, state, urn, phase):
    """Expected utility of second draw from given urn, given state."""
    if treatment in (1, 2):
        urns = T1_URNS if treatment == 1 else T2_URNS
        b, w = urns[(state, urn)]
        p_black = b / (b + w) if (b + w) > 0 else 0
        if phase in ('I', 'II'):
            if urn == 'Left':
                return p_black * PAYOFF_LEFT_BLACK
            else:
                return p_black * PAYOFF_RIGHT_BLACK
        else:  # Phase III reversed
            if urn == 'Left':
                return p_black * PAYOFF_RIGHT_BLACK  # Left pays 7/6
            else:
                return p_black * PAYOFF_LEFT_BLACK    # Right pays 1
    return 0

def cost_of_switching_error(treatment, phase, first_left, first_black):
    """Expected cost of making the wrong second-draw choice."""
    # Bayesian posterior over states given first draw
    if treatment in (1, 2):
        urns = T1_URNS if treatment == 1 else T2_URNS
        first_urn = 'Left' if first_left else 'Right'

        # P(Up|first draw) using Bayes rule
        p_up_prior = 0.5
        b_up, w_up = urns[('Up', first_urn)]
        b_down, w_down = urns[('Down', first_urn)]
        p_black_up = b_up / (b_up + w_up) if (b_up + w_up) > 0 else 0
        p_black_down = b_down / (b_down + w_down) if (b_down + w_down) > 0 else 0

        if first_black:
            p_draw_up = p_black_up
            p_draw_down = p_black_down
        else:
            p_draw_up = 1 - p_black_up
            p_draw_down = 1 - p_black_down

        denom = p_up_prior * p_draw_up + (1 - p_up_prior) * p_draw_down
        if denom == 0:
            return 0
        p_up_post = p_up_prior * p_draw_up / denom

        # EU of second draw from Left vs Right
        eu_left = p_up_post * compute_eu_second_draw(treatment, 'Up', 'Left', phase) + \
                  (1 - p_up_post) * compute_eu_second_draw(treatment, 'Down', 'Left', phase)
        eu_right = p_up_post * compute_eu_second_draw(treatment, 'Up', 'Right', phase) + \
                   (1 - p_up_post) * compute_eu_second_draw(treatment, 'Down', 'Right', phase)

        return abs(eu_left - eu_right)
    return 0

# Build error type dataset
error_data = []

for treatment in [1, 2]:
    sub = df[df['treatment'] == treatment]
    for phase in ['I', 'II', 'III']:
        # Right start errors
        for black_val, draw_name in [(1, 'Black'), (0, 'White')]:
            mask = (sub['phase'] == phase) & (sub['Left1'] == 0) & (sub['first_draw_black'] == black_val)
            obs = sub[mask]
            if len(obs) > 0:
                freq = obs['switching_error'].mean()
                cost = cost_of_switching_error(treatment, phase, False, bool(black_val))
                error_data.append({
                    'treatment': treatment, 'phase': phase,
                    'side': 'Right', 'draw': draw_name,
                    'frequency': freq, 'cost': cost,
                    'left': 0, 'affect': 1,
                    'n': len(obs)
                })

        # Left start errors
        mask_left = (sub['phase'] == phase) & (sub['Left1'] == 1)
        for black_val, draw_name in [(1, 'Black'), (0, 'White')]:
            mask = mask_left & (sub['first_draw_black'] == black_val)
            obs = sub[mask]
            if len(obs) > 0:
                freq = obs['switching_error'].mean()
                cost = cost_of_switching_error(treatment, phase, True, bool(black_val))
                error_data.append({
                    'treatment': treatment, 'phase': phase,
                    'side': 'Left', 'draw': draw_name,
                    'frequency': freq, 'cost': cost,
                    'left': 1, 'affect': 1,
                    'n': len(obs)
                })

# T3 errors
t3_sub = df[df['treatment'] == 3]
for phase in ['I', 'II']:
    for fav_val, draw_name in [(1, 'Favorable'), (0, 'Unfavorable')]:
        mask = (t3_sub['phase'] == phase) & (t3_sub['first_draw_favorable'] == fav_val)
        obs = t3_sub[mask]
        if len(obs) > 0:
            freq = obs['switching_error'].mean()
            # T3 cost: same urn compositions as T2, but no affect
            cost = cost_of_switching_error(2, phase if phase == 'I' else 'III', True, bool(fav_val))
            error_data.append({
                'treatment': 3, 'phase': phase,
                'side': 'Left', 'draw': draw_name,
                'frequency': freq, 'cost': cost,
                'left': 1, 'affect': 0,
                'n': len(obs)
            })

edf = pd.DataFrame(error_data)
print(f"\nNumber of error types: {len(edf)}")
print(edf[['treatment', 'phase', 'side', 'draw', 'frequency', 'cost', 'n']].to_string(index=False))

# Spec (1): OLS: frequency ~ cost
print("\n--- Specification (1): OLS frequency ~ cost ---")
X = sm.add_constant(edf['cost'])
y = edf['frequency']
model1 = sm.OLS(y, X).fit()
print(f"  Constant: {model1.params['const']:.3f} ({model1.bse['const']:.3f})")
print(f"  Cost: {model1.params['cost']:.3f} ({model1.bse['cost']:.3f})")
print(f"  N: {len(edf)}, Adj R²: {model1.rsquared_adj:.3f}")
print(f"  Published: Constant .469*** (.042), Cost -.928*** (.153), Adj R²=.609")

# Spec (2): OLS: frequency ~ cost + cost_squared
print("\n--- Specification (2): OLS frequency ~ cost + cost² ---")
edf['cost_sq'] = edf['cost'] ** 2
X2 = sm.add_constant(edf[['cost', 'cost_sq']])
model2 = sm.OLS(y, X2).fit()
print(f"  Constant: {model2.params['const']:.3f} ({model2.bse['const']:.3f})")
print(f"  Cost: {model2.params['cost']:.3f} ({model2.bse['cost']:.3f})")
print(f"  Cost²: {model2.params['cost_sq']:.2f} ({model2.bse['cost_sq']:.3f})")
print(f"  N: {len(edf)}, Adj R²: {model2.rsquared_adj:.3f}")
print(f"  Published: Constant .547*** (.066), Cost -1.88*** (.498), Cost² 1.76** (.861), Adj R²=.656")

# Spec (3): OLS on switching errors only: frequency ~ Left
switching = edf[edf['side'].isin(['Left', 'Right'])]
# For proper comparison, limit to switching errors (not starting)
# Paper says specs 3-4 use only the 20 switching errors
print("\n--- Specification (3): OLS frequency ~ Left (switching errors only) ---")
switch_only = edf.copy()  # All are switching errors in our data
X3 = sm.add_constant(switch_only['left'])
model3 = sm.OLS(switch_only['frequency'], X3).fit()
print(f"  Constant: {model3.params['const']:.3f} ({model3.bse['const']:.3f})")
print(f"  Left: {model3.params['left']:.3f} ({model3.bse['left']:.3f})")
print(f"  N: {len(switch_only)}, Adj R²: {model3.rsquared_adj:.3f}")
print(f"  Published: Constant .061 (.049), Left .310*** (.063), N=20, Adj R²=.547")

# Spec (4): OLS frequency ~ cost + cost² + Left + Affect
print("\n--- Specification (4): OLS frequency ~ cost + cost² + Left + Affect ---")
X4 = sm.add_constant(switch_only[['cost', 'cost_sq', 'left', 'affect']])
model4 = sm.OLS(switch_only['frequency'], X4).fit()
print(f"  Constant: {model4.params['const']:.3f} ({model4.bse['const']:.3f})")
print(f"  Cost: {model4.params['cost']:.3f} ({model4.bse['cost']:.3f})")
print(f"  Cost²: {model4.params['cost_sq']:.2f} ({model4.bse['cost_sq']:.3f})")
print(f"  Left: {model4.params['left']:.3f} ({model4.bse['left']:.3f})")
print(f"  Affect: {model4.params['affect']:.3f} ({model4.bse['affect']:.3f})")
print(f"  N: {len(switch_only)}, Adj R²: {model4.rsquared_adj:.3f}")
print(f"  Published: Constant .305** (.129), Cost -1.84*** (.566), Cost² 2.14** (.889)")
print(f"             Left .172** (.085), Affect .114* (.066), N=20, Adj R²=.723")

# Specs (5) and (6): Random effects at individual level
# Each individual has 8 (T1/T2) or 4 (T3) error rates
# frequency = individual error rate for each error type
print("\n--- Specifications (5) and (6): Individual-level RE regressions ---")

# Build individual-level dataset
ind_errors = []
for ind in df['Individual'].unique():
    sub = df[df['Individual'] == ind]
    t = sub['treatment'].iloc[0]
    female = sub['Female'].iloc[0]

    if t in (1, 2):
        for phase in ['I', 'II', 'III']:
            for left_val in [0, 1]:
                for black_val in [0, 1]:
                    mask = (sub['phase'] == phase) & (sub['Left1'] == left_val) & (sub['first_draw_black'] == black_val)
                    obs = sub[mask]
                    if len(obs) > 0:
                        freq = obs['switching_error'].mean()
                        cost = cost_of_switching_error(t, phase, bool(left_val), bool(black_val))
                        ind_errors.append({
                            'Individual': ind,
                            'treatment': t,
                            'frequency': freq,
                            'cost': cost,
                            'left': left_val,
                            'affect': 1,
                            'female': female,
                        })
    else:  # T3
        for phase in ['I', 'II']:
            for fav_val in [0, 1]:
                mask = (sub['phase'] == phase) & (sub['first_draw_favorable'] == fav_val)
                obs = sub[mask]
                if len(obs) > 0:
                    freq = obs['switching_error'].mean()
                    cost = cost_of_switching_error(2, phase if phase == 'I' else 'III', True, bool(fav_val))
                    ind_errors.append({
                        'Individual': ind,
                        'treatment': 3,
                        'frequency': freq,
                        'cost': cost,
                        'left': 1,
                        'affect': 0,
                        'female': female,
                    })

iedf = pd.DataFrame(ind_errors)
iedf['cost_sq'] = iedf['cost'] ** 2

print(f"\n  Individual-level observations: {len(iedf)}")

# Spec (5): RE: frequency ~ cost + left + affect + female
X5 = sm.add_constant(iedf[['cost', 'left', 'affect', 'female']])
# Using pooled OLS as approximation (statsmodels RE requires panel setup)
model5 = sm.OLS(iedf['frequency'], X5).fit(cov_type='cluster',
                 cov_kwds={'groups': iedf['Individual']})
print(f"\n  Spec (5) - OLS clustered by individual:")
print(f"  Constant: {model5.params['const']:.3f} ({model5.bse['const']:.3f})")
print(f"  Cost: {model5.params['cost']:.3f} ({model5.bse['cost']:.3f})")
print(f"  Left: {model5.params['left']:.3f} ({model5.bse['left']:.3f})")
print(f"  Affect: {model5.params['affect']:.3f} ({model5.bse['affect']:.3f})")
print(f"  Female: {model5.params['female']:.3f} ({model5.bse['female']:.3f})")
print(f"  N: {len(iedf)}, R²: {model5.rsquared:.3f}")
print(f"  Published: Constant .100** (.051), Cost -.483*** (.090), Left .226*** (.033)")
print(f"             Affect .118*** (.031), Female .079*** (.025), N=999, Adj R²=.252")

# Spec (6): with cost_squared
X6 = sm.add_constant(iedf[['cost', 'cost_sq', 'left', 'affect', 'female']])
model6 = sm.OLS(iedf['frequency'], X6).fit(cov_type='cluster',
                 cov_kwds={'groups': iedf['Individual']})
print(f"\n  Spec (6) - OLS clustered by individual (with cost²):")
print(f"  Constant: {model6.params['const']:.3f} ({model6.bse['const']:.3f})")
print(f"  Cost: {model6.params['cost']:.3f} ({model6.bse['cost']:.3f})")
print(f"  Cost²: {model6.params['cost_sq']:.2f} ({model6.bse['cost_sq']:.3f})")
print(f"  Left: {model6.params['left']:.3f} ({model6.bse['left']:.3f})")
print(f"  Affect: {model6.params['affect']:.3f} ({model6.bse['affect']:.3f})")
print(f"  Female: {model6.params['female']:.3f} ({model6.bse['female']:.3f})")
print(f"  N: {len(iedf)}, R²: {model6.rsquared:.3f}")
print(f"  Published: Constant .175*** (.052), Cost -1.48*** (.215), Cost² 1.75*** (.344)")
print(f"             Left .232*** (.032), Affect .127*** (.031), Female .079*** (.019)")
print(f"             N=999, Adj R²=.268")
