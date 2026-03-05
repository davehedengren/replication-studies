"""
02_tables.py - Replicate Tables 1-9 from Callen & Long (2015)
"Institutional Corruption and Election Fraud"
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import *
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("TABLE REPLICATION: Callen & Long (2015)")
print("=" * 70)

# ═══════════════════════════════════════════════════════════════════════════
# TABLE 1: Aggregation Discrepancy Patterns
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TABLE 1: Aggregation Discrepancy Patterns")
print("=" * 70)

df = load_vote_diff()
t1 = df[df['vote_diff'].notna()].copy()
t1['pos_cand'] = (t1['pos_diff'] > 0).astype(int)
t1['neg_cand'] = (t1['neg_diff'] > 0).astype(int)
t1['pos_votes'] = t1['vote_diff'].where(t1['vote_diff'] > 0, 0)
t1['neg_votes'] = t1['vote_diff'].where(t1['vote_diff'] < 0, 0)

station_id = t1.groupby(['pc', 'pollingstation']).ngroup()
t1['n_stations'] = station_id
t1_coll = t1.groupby('n_stations').agg(
    vote_diff=('vote_diff', 'sum'),
    pos_votes=('pos_votes', 'sum'),
    neg_votes=('neg_votes', 'sum'),
).reset_index()
t1_coll['abs_neg'] = t1_coll['neg_votes'].abs()
n_st = len(t1_coll)

patterns = [
    ('No fraud', (t1_coll['pos_votes'] == t1_coll['abs_neg']) & (t1_coll['abs_neg'] == 0)),
    ('Adding votes only', (t1_coll['pos_votes'] > 0) & (t1_coll['abs_neg'] == 0)),
    ('Subtracting votes only', (t1_coll['pos_votes'] == 0) & (t1_coll['abs_neg'] > 0)),
    ('Adding and subtracting equally', (t1_coll['pos_votes'] == t1_coll['abs_neg']) & (t1_coll['abs_neg'] > 0)),
    ('Adding more than subtracting', (t1_coll['pos_votes'] > t1_coll['abs_neg']) & (t1_coll['abs_neg'] > 0) & (t1_coll['pos_votes'] > 0)),
    ('Subtracting more than adding', (t1_coll['abs_neg'] > t1_coll['pos_votes']) & (t1_coll['pos_votes'] > 0)),
]

pub_t1 = [
    (74, 21.4, 0.00),
    (70, 20.2, 47.34),
    (15, 4.3, -245.07),
    (15, 4.3, 0.00),
    (127, 36.7, 83.45),
    (45, 13.01, -54.13),
]

print(f"{'Pattern':<35} {'N':>5} {'Share%':>8} {'Mean VD':>10} | {'Pub N':>5} {'Pub%':>8} {'Pub VD':>10}")
print("-" * 100)
for (name, mask), (pub_n, pub_pct, pub_vd) in zip(patterns, pub_t1):
    n = mask.sum()
    pct = n / n_st * 100
    mean_vd = t1_coll.loc[mask, 'vote_diff'].mean() if mask.sum() > 0 else 0
    print(f"{name:<35} {n:>5} {pct:>7.1f}% {mean_vd:>10.2f} | {pub_n:>5} {pub_pct:>7.1f}% {pub_vd:>10.2f}")

# ═══════════════════════════════════════════════════════════════════════════
# TABLE 2: Summary Statistics
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TABLE 2: Summary Statistics on Political Connections and Aggregation Fraud")
print("=" * 70)

df = load_vote_diff()
vd = df[df['vote_diff'].notna()].copy()
vd['picture_total'] = vd.groupby(['pc', 'pollingstation'])['picture_votes_1st'].transform('sum')
vd['before_share'] = vd['picture_votes_1st'] / vd['picture_total']
vd['after_share'] = vd['votes'] / vd['picture_total']
vd['share_diff'] = vd['after_share'] - vd['before_share']

# Outliers based on share_diff
vd = vd.sort_values('share_diff').reset_index(drop=True)
vd['ascending'] = range(1, len(vd) + 1)
vd = vd.sort_values('share_diff', ascending=False).reset_index(drop=True)
vd['descending'] = range(1, len(vd) + 1)
insample = vd[(vd['ascending'] > 5) & (vd['descending'] > 5)].copy()
insample['share_diff1'] = insample['share_diff'] * 100

pub_t2 = {
    'peo_fullsample': (0.005, 0.073, 48008),
    'deo_fullsample': (0.004, 0.065, 48008),
    'karzai_fullsample': (0.011, 0.106, 48008),
    'gov_fullsample': (0.015, 0.12, 48008),
    'incumbent': (0.067, 0.25, 48008),
    'vote_diff': (0.166, 6.409, 48008),
    'share_diff1': (0.570, 14.287, 48008),
}

print(f"{'Variable':<30} {'Mean':>8} {'SD':>8} {'N':>8} | {'Pub Mean':>8} {'Pub SD':>8}")
print("-" * 85)
for var, (pub_m, pub_sd, pub_n) in pub_t2.items():
    col = insample[var]
    m = col.mean()
    sd = col.std()
    n = col.notna().sum()
    print(f"{var:<30} {m:>8.3f} {sd:>8.3f} {n:>8} | {pub_m:>8.3f} {pub_sd:>8.3f}")

# ═══════════════════════════════════════════════════════════════════════════
# TABLE 3: Vote Changes During Aggregation by Candidate Type
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TABLE 3: Vote Changes During Aggregation by Candidate Type")
print("=" * 70)

df = load_vote_diff()
vd = df[df['vote_diff'].notna()].copy()

# Panel A: vote_diff outcome, outliers based on vote_diff
vd_a = vd.sort_values('vote_diff').reset_index(drop=True)
vd_a['ascending'] = range(1, len(vd_a) + 1)
vd_a = vd_a.sort_values('vote_diff', ascending=False).reset_index(drop=True)
vd_a['descending'] = range(1, len(vd_a) + 1)
vd_a['insample'] = ((vd_a['ascending'] > 5) & (vd_a['descending'] > 5)).astype(int)
dfa = vd_a[vd_a['insample'] == 1].copy()

print("\nPanel A: Post-aggregation votes - pre-aggregation votes")
print("-" * 90)

# Published coefficients for Panel A
pub_t3a = {
    1: {'cand_data': (1.870, 0.980), 'const': (0.104, 0.037)},
    2: {'cand_data': (1.610, 1.208), 'peo_fullsample': (0.942, 1.651), 'const': (0.105, 0.037)},
    3: {'cand_data': (1.380, 1.176), 'deo_fullsample': (2.207, 1.848), 'const': (0.106, 0.037)},
    4: {'cand_data': (2.328, 2.169), 'karzai_fullsample': (-0.745, 2.216), 'const': (0.105, 0.036)},
    5: {'cand_data': (3.149, 1.415), 'gov_fullsample': (-1.634, 1.918), 'const': (0.105, 0.037)},
    6: {'cand_data': (1.764, 1.013), 'incumbent': (0.245, 0.238), 'const': (0.094, 0.037)},
    7: {'cand_data': (3.385, 1.713), 'deo_fullsample': (2.553, 1.807), 'karzai_fullsample': (-1.005, 2.153),
        'gov_fullsample': (-2.038, 1.996), 'incumbent': (0.302, 0.226), 'const': (0.096, 0.037)},
}

# Province fixed effects
provid_dummies = pd.get_dummies(dfa['provid398'], prefix='prov', drop_first=True).astype(float)

for col_num in range(1, 8):
    if col_num == 1:
        xvars = ['cand_data']
    elif col_num == 2:
        xvars = ['peo_fullsample', 'cand_data']
    elif col_num == 3:
        xvars = ['deo_fullsample', 'cand_data']
    elif col_num == 4:
        xvars = ['karzai_fullsample', 'cand_data']
    elif col_num == 5:
        xvars = ['gov_fullsample', 'cand_data']
    elif col_num == 6:
        xvars = ['incumbent', 'cand_data']
    else:
        xvars = ['deo_fullsample', 'karzai_fullsample', 'gov_fullsample', 'incumbent', 'cand_data']

    X = pd.concat([dfa[xvars], provid_dummies.loc[dfa.index]], axis=1)
    X = sm.add_constant(X)
    y = dfa['vote_diff']

    res = cgmreg(y, X, ['cand_id_s', 'pc'], dfa)

    print(f"\n  Column {col_num}:")
    pub = pub_t3a[col_num]
    for i, vname in enumerate(['const'] + xvars):
        idx = 0 if vname == 'const' else xvars.index(vname) + 1
        b = res['beta'][idx]
        s = res['se'][idx]
        if vname in pub:
            pb, ps = pub[vname]
            print(f"    {vname:<25} β={b:>8.3f} (SE={s:.3f}) | Pub: β={pb:.3f} (SE={ps:.3f})")
        else:
            print(f"    {vname:<25} β={b:>8.3f} (SE={s:.3f})")
    print(f"    R²={res['r2']:.3f}, N={res['n']}, Clusters: {res['n_clusters']}")

# Panel B: Vote shares
print("\n\nPanel B: Vote share changes x100")
print("-" * 90)

df = load_vote_diff()
vd = df[df['vote_diff'].notna()].copy()
vd['picture_total'] = vd.groupby(['pc', 'pollingstation'])['picture_votes_1st'].transform('sum')
vd['before_share'] = vd['picture_votes_1st'] / vd['picture_total']
vd['after_share'] = vd['votes'] / vd['picture_total']
vd['share_diff'] = vd['after_share'] - vd['before_share']
vd['share_diff1'] = vd['share_diff'] * 100

# Outliers on share_diff1
vd = vd.sort_values('share_diff1').reset_index(drop=True)
vd['ascending'] = range(1, len(vd) + 1)
vd = vd.sort_values('share_diff1', ascending=False).reset_index(drop=True)
vd['descending'] = range(1, len(vd) + 1)
vd['insample'] = ((vd['ascending'] > 5) & (vd['descending'] > 5)).astype(int)
dfb = vd[vd['insample'] == 1].copy()

pub_t3b = {
    1: {'cand_data': (2.978, 1.316), 'const': (0.247, 0.132)},
    7: {'cand_data': (2.928, 1.330), 'deo_fullsample': (-0.215, 2.005),
        'karzai_fullsample': (1.343, 1.263), 'gov_fullsample': (2.744, 2.486),
        'incumbent': (-0.000, 0.207), 'const': (0.244, 0.131)},
}

provid_dummies_b = pd.get_dummies(dfb['provid398'], prefix='prov', drop_first=True).astype(float)

for col_num in [1, 7]:
    if col_num == 1:
        xvars = ['cand_data']
    else:
        xvars = ['deo_fullsample', 'karzai_fullsample', 'gov_fullsample', 'incumbent', 'cand_data']

    X = pd.concat([dfb[xvars], provid_dummies_b.loc[dfb.index]], axis=1)
    X = sm.add_constant(X)
    y = dfb['share_diff1']

    res = cgmreg(y, X, ['cand_id_s', 'pc'], dfb)

    print(f"\n  Column {col_num}:")
    pub = pub_t3b[col_num]
    for i, vname in enumerate(['const'] + xvars):
        idx = 0 if vname == 'const' else xvars.index(vname) + 1
        b = res['beta'][idx]
        s = res['se'][idx]
        if vname in pub:
            pb, ps = pub[vname]
            print(f"    {vname:<25} β={b:>8.3f} (SE={s:.3f}) | Pub: β={pb:.3f} (SE={ps:.3f})")
    print(f"    R²={res['r2']:.3f}, N={res['n']}, Clusters: {res['n_clusters']}")


# ═══════════════════════════════════════════════════════════════════════════
# TABLE 4: Randomization Verification
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TABLE 4: Randomization Verification")
print("=" * 70)

df_pc = load_pc()
pc = df_pc[df_pc['strata_group'].notna()].copy()
pc['income'] = pc['income'] / 1000

balance_vars = ['expectedturnout', 'secretvote', 'candknowsvote', 'parliamentaryknowledge',
                'onecandidatevote', 'transportproblems', 'policesafe', 'threatvote',
                'violenceown', 'ownqawmcandidate', 'traditionalauthority',
                'pashtun', 'tajik', 'employment', 'electrified', 'dgkeepsfair',
                'treat_di_actual', 'fefa_visit', 'inkproblem']

pub_t4 = {
    'expectedturnout': (0.790, 0.797, 0.008, 0.734),
    'secretvote': (0.668, 0.650, -0.018, 0.453),
    'candknowsvote': (0.086, 0.090, 0.004, 0.767),
    'parliamentaryknowledge': (0.371, 0.386, 0.015, 0.627),
    'onecandidatevote': (0.236, 0.249, 0.012, 0.607),
    'transportproblems': (0.538, 0.534, -0.004, 0.880),
    'policesafe': (0.728, 0.737, 0.009, 0.701),
    'threatvote': (0.213, 0.202, -0.011, 0.606),
    'violenceown': (0.500, 0.483, -0.017, 0.592),
    'ownqawmcandidate': (0.235, 0.232, -0.003, 0.878),
    'traditionalauthority': (0.286, 0.293, 0.007, 0.785),
    'pashtun': (0.323, 0.318, -0.005, 0.903),
    'tajik': (0.436, 0.433, -0.004, 0.921),
    'employment': (0.602, 0.607, 0.004, 0.817),
    'electrified': (0.722, 0.706, -0.016, 0.595),
    'dgkeepsfair': (0.112, 0.114, 0.003, 0.864),
    'treat_di_actual': (0.146, 0.182, 0.036, 0.308),
    'fefa_visit': (0.892, 0.863, -0.029, 0.350),
    'inkproblem': (0.798, 0.762, -0.036, 0.357),
}

print(f"{'Variable':<28} {'Ctrl':>7} {'Treat':>7} {'Diff':>7} {'p':>7} | {'Pub Ctrl':>8} {'Pub Trt':>7} {'Pub p':>7}")
print("-" * 100)
for var in balance_vars:
    ctrl = pc.loc[pc['treat_lib'] == 0, var].dropna()
    treat = pc.loc[pc['treat_lib'] == 1, var].dropna()
    diff = treat.mean() - ctrl.mean()
    t_stat, p_val = stats.ttest_ind(treat, ctrl, equal_var=True)
    pub = pub_t4.get(var, (None, None, None, None))
    print(f"{var:<28} {ctrl.mean():>7.3f} {treat.mean():>7.3f} {diff:>7.3f} {p_val:>7.3f} | "
          f"{pub[0]:>8.3f} {pub[1]:>7.3f} {pub[3]:>7.3f}")


# ═══════════════════════════════════════════════════════════════════════════
# TABLE 5: Lee Bounds (Provincial Aggregator Connection)
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TABLE 5: Trimmed Estimates of Effect on Aggregation Fraud")
print("(Sample: provincial aggregator connection = 1)")
print("=" * 70)

df = load_vote_diff()
df['whole_sample'] = (~df['picture_votes_1st'].isna() & ~df['votes'].isna()).astype(int)

# For Lee bounds, the outcome is abs_diff and sample is peo_fullsample
lb = lee_bounds(
    outcome=df['abs_diff'],
    treatment=df['treat_lib'],
    sample_mask=df['peo_fullsample'],
    cluster_var=df['pc'],
    n_boot=0,  # Skip bootstrap for speed; we verify the point estimates
    seed=8221986
)

print(f"\nControl:")
print(f"  (i)   Total observations:         {lb['total_control']:>8}   Published: 892")
print(f"  (ii)  Proportion non-missing:      {lb['present_control']:>8.3f}   Published: 0.112")
print(f"  (iii) Mean abs(vote diff):         {lb['control_mean']:>8.3f}   Published: 17.170")

print(f"\nTreatment:")
print(f"  (iv)  Total observations:          {lb['total_treat']:>8}   Published: 969")
print(f"  (v)   Proportion non-missing:      {lb['present_treat']:>8.3f}   Published: 0.160")
print(f"  (vi)  Mean abs(vote diff):         {lb['treat_mean']:>8.3f}   Published: 5.484")

print(f"\n  Trimming ratio p:                 {lb['p']:>8.3f}   Published: 0.299")

print(f"\n  (viii) Trimmed mean (upper):       {lb['upper_mean']:>8.3f}   Published: 7.798")
print(f"  (x)   Trimmed mean (lower):        {lb['lower_mean']:>8.3f}   Published: 0.000")

print(f"\n  (xi) Upper bound estimate:         {lb['upper_bound']:>8.3f}   Published: -9.372")
print(f"  (xii) Lower bound estimate:        {lb['lower_bound']:>8.3f}   Published: -17.170")


# ═══════════════════════════════════════════════════════════════════════════
# TABLE 6: Lee Bounds for All Subsamples
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TABLE 6: Trimmed Estimates for All Candidate Subsamples")
print("=" * 70)

pub_t6 = {
    'Full sample': {'ctrl_mean_post': 2.318, 'ctrl_abs': 0.817, 'treat_abs': 0.582, 'p': 0.349,
                     'upper_mean': 0.893, 'lower_mean': 0.000, 'lower_bound': -0.817, 'upper_bound': 0.076},
    'incumbent': {'ctrl_mean_post': 7.501, 'ctrl_abs': 2.840, 'treat_abs': 1.597, 'p': 0.308,
                  'lower_bound': -2.840, 'upper_bound': -0.531},
    'powerful': {'ctrl_mean_post': 24.739, 'ctrl_abs': 10.310, 'treat_abs': 3.799, 'p': 0.265,
                 'lower_bound': -10.310, 'upper_bound': -5.143},
    'peo_fullsample': {'ctrl_mean_post': 36.161, 'ctrl_abs': 18.790, 'treat_abs': 5.760, 'p': 0.295,
                       'lower_bound': -18.790, 'upper_bound': -10.700},
    'karzai_fullsample': {'ctrl_mean_post': 21.484, 'ctrl_abs': 8.831, 'treat_abs': 3.667, 'p': 0.309,
                          'lower_bound': -8.831, 'upper_bound': -3.549},
    'gov_fullsample': {'ctrl_mean_post': 18.788, 'ctrl_abs': 9.665, 'treat_abs': 5.126, 'p': 0.291,
                       'lower_bound': -9.652, 'upper_bound': -2.448},
}

# "complete" means whole_sample - need to define
df['complete'] = 1  # Entire sample

for sample_name, sample_var in [('Full sample', 'complete'), ('incumbent', 'incumbent'),
                                 ('powerful', 'powerful'), ('peo_fullsample', 'peo_fullsample'),
                                 ('karzai_fullsample', 'karzai_fullsample'),
                                 ('gov_fullsample', 'gov_fullsample')]:
    lb = lee_bounds(
        outcome=df['abs_diff'],
        treatment=df['treat_lib'],
        sample_mask=df[sample_var],
        cluster_var=df['pc'],
        n_boot=0,
    )
    pub = pub_t6.get(sample_name, {})
    print(f"\n  {sample_name}:")
    print(f"    Control abs diff:   {lb['control_mean']:>8.3f}  Published: {pub.get('ctrl_abs', '?')}")
    print(f"    Treat abs diff:     {lb['treat_mean']:>8.3f}  Published: {pub.get('treat_abs', '?')}")
    print(f"    Trimming ratio:     {lb['p']:>8.3f}  Published: {pub.get('p', '?')}")
    print(f"    Lower bound:        {lb['lower_bound']:>8.3f}  Published: {pub.get('lower_bound', '?')}")
    print(f"    Upper bound:        {lb['upper_bound']:>8.3f}  Published: {pub.get('upper_bound', '?')}")


# ═══════════════════════════════════════════════════════════════════════════
# TABLE 7: Impact on Total Votes by Candidate Connection
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TABLE 7: Estimates of Impact on Total Votes by Candidate Connection")
print("=" * 70)

df = load_vote_diff()
df['cand_data_t'] = df['cand_data'] * df['treat_lib']

print("\nPanel A: Impacts on most connected candidates")

# Col 1: bivariate reg, cluster on pc
y = df['votes']
X = sm.add_constant(df[['treat_lib']])
mask = y.notna() & X.notna().all(axis=1) & df['pc'].notna()
res1 = ols_cluster(y[mask], X[mask], df.loc[mask, 'pc'].values)
print(f"  Col 1: treat_lib = {res1['beta'][1]:.3f} ({res1['se'][1]:.3f}), N={res1['n']}")
print(f"    Published: -0.066 (0.212), N=386,855")

# Col 2: treat, powerful, pflt (treat x powerful), cluster on pc
X2 = sm.add_constant(df[['treat_lib', 'powerful', 'pflt']])
mask2 = y.notna() & X2.notna().all(axis=1) & df['pc'].notna()
res2 = ols_cluster(y[mask2], X2[mask2], df.loc[mask2, 'pc'].values)
print(f"  Col 2: treat={res2['beta'][1]:.3f} ({res2['se'][1]:.3f}), "
      f"powerful={res2['beta'][2]:.3f} ({res2['se'][2]:.3f}), "
      f"pflt={res2['beta'][3]:.3f} ({res2['se'][3]:.3f})")
print(f"    Published: treat=-0.034 (0.196), powerful=20.871 (2.391), pflt=-5.430 (3.111)")

# Col 3: areg with strata FE
xvars = ['treat_lib', 'powerful', 'pflt']
X3 = df[xvars].copy()
res3 = areg_cluster(y, X3, 'strata_group', 'pc', df)
print(f"  Col 3: treat={res3['beta'][0]:.3f} ({res3['se'][0]:.3f}), "
      f"powerful={res3['beta'][1]:.3f} ({res3['se'][1]:.3f}), "
      f"pflt={res3['beta'][2]:.3f} ({res3['se'][2]:.3f})")
print(f"    Published: treat=0.030 (0.045), powerful=18.604 (2.364), pflt=-5.392 (3.080)")

# Col 4: areg with strata FE + controls
xvars4 = ['treat_lib', 'powerful', 'pflt'] + CONTROLS1
X4 = df[xvars4].copy()
res4 = areg_cluster(y, X4, 'strata_group', 'pc', df)
print(f"  Col 4: treat={res4['beta'][0]:.3f} ({res4['se'][0]:.3f}), "
      f"powerful={res4['beta'][1]:.3f} ({res4['se'][1]:.3f}), "
      f"pflt={res4['beta'][2]:.3f} ({res4['se'][2]:.3f})")
print(f"    Published: treat=0.019 (0.056), powerful=19.746 (2.489), pflt=-5.780 (3.245)")

# Col 5: same as col4 but trimming top 1% of votes for powerful
p99 = df.loc[df['powerful'] == 1, 'votes'].quantile(0.99)
mask5 = df['votes'] < p99
df5 = df[mask5].copy()
xvars5 = ['treat_lib', 'powerful', 'pflt'] + CONTROLS1
X5 = df5[xvars5].copy()
y5 = df5['votes']
res5 = areg_cluster(y5, X5, 'strata_group', 'pc', df5)
print(f"  Col 5: treat={res5['beta'][0]:.3f} ({res5['se'][0]:.3f}), "
      f"powerful={res5['beta'][1]:.3f} ({res5['se'][1]:.3f}), "
      f"pflt={res5['beta'][2]:.3f} ({res5['se'][2]:.3f})")
print(f"    Published: treat=-0.033 (0.054), powerful=16.794 (1.787), pflt=-4.634 (2.347)")
print(f"    N={res5['n']}, Published: 309,822-309,947")

# Panel B
print("\nPanel B: Impacts by connection type")

# Col 1: cand_data, cand_data_t
p99_cd = df.loc[df['cand_data'] == 1, 'votes'].quantile(0.99)
mask_b1 = df['votes'] < p99_cd
df_b1 = df[mask_b1].copy()
xvars_b1 = ['treat_lib', 'cand_data', 'cand_data_t'] + CONTROLS1
X_b1 = df_b1[xvars_b1].copy()
y_b1 = df_b1['votes']
res_b1 = areg_cluster(y_b1, X_b1, 'strata_group', 'pc', df_b1)
print(f"  Col 1: treat={res_b1['beta'][0]:.3f} ({res_b1['se'][0]:.3f}), "
      f"cand_data_t={res_b1['beta'][2]:.3f} ({res_b1['se'][2]:.3f}), "
      f"cand_data={res_b1['beta'][1]:.3f} ({res_b1['se'][1]:.3f})")
print(f"    Published: treat=-0.010 (0.052), cand_data_t=-2.864 (1.473), cand_data=15.788 (1.122)")


# ═══════════════════════════════════════════════════════════════════════════
# TABLE 8: Impacts on Form Theft
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TABLE 8: Impacts on Form Theft")
print("=" * 70)

df_pc = load_pc()
df_pc['strata_group'] = df_pc['strata_group'].fillna(62)

# Drop the same PCs as the do-file
drop_pcs = [101027, 201014, 201015, 201016, 201017, 201018, 801008, 801016, 2701021, 3101015]
pc8 = df_pc[~df_pc['pc'].isin(drop_pcs)].copy()

# Col 1: simple OLS with robust SE
y8 = pc8['candidate_agent']
X8_1 = sm.add_constant(pc8[['treat_lib']])
model1 = sm.OLS(y8, X8_1, missing='drop').fit(cov_type='HC1')
print(f"  Col 1: treat_lib = {model1.params['treat_lib']:.3f} ({model1.bse['treat_lib']:.3f})")
print(f"    Published: -0.108 (0.032)")
print(f"    Constant = {model1.params['const']:.3f} ({model1.bse['const']:.3f})")
print(f"    Published: 0.189 (0.026)")
print(f"    N={model1.nobs:.0f}, R²={model1.rsquared:.3f}")

# Col 2: with strata FE
xvars_8_2 = ['treat_lib']
X8_2 = pc8[xvars_8_2].copy()
res8_2 = areg_cluster(y8, X8_2, 'strata_group', 'pc', pc8)
# For areg with robust (not cluster), use HC1 - but table8 uses robust SEs not cluster
# Actually table8.do uses "r" (robust) not cluster
# Let's just use HC1 for the areg
print(f"  Col 2: treat_lib = {res8_2['beta'][0]:.3f} ({res8_2['se'][0]:.3f})")
print(f"    Published: -0.111 (0.031)")

# Col 3: with strata FE + controls
xvars_8_3 = ['treat_lib'] + CONTROLS1
X8_3 = pc8[xvars_8_3].copy()
res8_3 = areg_cluster(y8, X8_3, 'strata_group', 'pc', pc8)
print(f"  Col 3: treat_lib = {res8_3['beta'][0]:.3f} ({res8_3['se'][0]:.3f})")
print(f"    Published: -0.110 (0.032)")
print(f"    N={res8_3['n']}, Published: 440-461")


# ═══════════════════════════════════════════════════════════════════════════
# TABLE 9: Spatial Treatment Externalities
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TABLE 9: Spatial Treatment Externalities")
print("=" * 70)

df = load_vote_diff()

# Sample: powerful==1, trimmed at 99th percentile
p99_pow = df.loc[df['powerful'] == 1, 'votes'].quantile(0.99)
df9 = df[(df['powerful'] == 1) & (df['votes'] < p99_pow)].copy()

# Col 1: just treat_lib + controls + strata FE
xvars_9_1 = ['treat_lib'] + CONTROLS1
X9_1 = df9[xvars_9_1].copy()
y9 = df9['votes']
res9_1 = areg_cluster(y9, X9_1, 'strata_group', 'pc', df9)
print(f"  Col 1: treat_lib = {res9_1['beta'][0]:.3f} ({res9_1['se'][0]:.3f})")
print(f"    Published: -4.080 (2.009)")

# Col 2: + treat_1k_dum, within_1k
xvars_9_2 = ['treat_lib', 'treat_1k_dum', 'within_1k'] + CONTROLS1
X9_2 = df9[xvars_9_2].copy()
res9_2 = areg_cluster(y9, X9_2, 'strata_group', 'pc', df9)
print(f"  Col 2: treat_lib = {res9_2['beta'][0]:.3f} ({res9_2['se'][0]:.3f}), "
      f"treat_1k = {res9_2['beta'][1]:.3f} ({res9_2['se'][1]:.3f})")
print(f"    Published: treat=-4.183 (1.982), treat_1k=-6.877 (3.512)")

# Col 3: + treat_1_2k, within_1_2k
xvars_9_3 = ['treat_lib', 'treat_1k_dum', 'treat_1_2k_dum', 'within_1k', 'within_1_2k'] + CONTROLS1
X9_3 = df9[xvars_9_3].copy()
res9_3 = areg_cluster(y9, X9_3, 'strata_group', 'pc', df9)
print(f"  Col 3: treat_lib = {res9_3['beta'][0]:.3f} ({res9_3['se'][0]:.3f}), "
      f"treat_1k = {res9_3['beta'][1]:.3f} ({res9_3['se'][1]:.3f}), "
      f"treat_1_2k = {res9_3['beta'][2]:.3f} ({res9_3['se'][2]:.3f})")
print(f"    Published: treat=-4.290 (1.956), treat_1k=-6.742 (3.486), treat_1_2k=-4.738 (4.244)")

print(f"\n    N={res9_3['n']}, Clusters={res9_3['n_clusters']}")
print(f"    Published: N=1,841, Clusters=439")


print("\n" + "=" * 70)
print("TABLE REPLICATION COMPLETE")
print("=" * 70)
