"""
05_robustness.py – Robustness checks
Nielsen & Rehbeck (2022) "When Choices Are Mistakes"
"""

import sys, os, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings('ignore')
from utils import *


def compute_reconcile_stats(d, var='whichreconcile'):
    """Compute revision rate and favor-axiom rate from reconciliation data."""
    wr = d[var].replace({4: 3})
    valid = wr.dropna()
    if len(valid) == 0:
        return {'n': 0, 'revision_rate': np.nan, 'favor_rate': np.nan}
    revised = (valid != 0).sum()
    revision_rate = revised / len(valid) * 100
    change_lot = (valid == 2).sum()
    unselect = (valid == 1).sum()
    favor_rate = change_lot / (change_lot + unselect) * 100 if (change_lot + unselect) > 0 else np.nan
    return {'n': int(len(valid)), 'revision_rate': revision_rate, 'favor_rate': favor_rate}


def main():
    df = load_data()
    lab_nc = filter_lab_nocost(df)
    onl = filter_online(df)

    # ══════════════════════════════════════════════════════════════════════
    print_section("CHECK 1: Alternative Reconciliation Coding (keep 4 separate)")
    # ══════════════════════════════════════════════════════════════════════
    # Without recoding 4->3
    wr = lab_nc['whichreconcile'].dropna()
    print(f"Total violations: {len(wr)}")
    print(f"Distribution (original coding):")
    print(wr.value_counts().sort_index())
    print(f"\n0=Keep: {(wr==0).sum()} ({(wr==0).mean()*100:.1f}%)")
    print(f"1=Unselect: {(wr==1).sum()} ({(wr==1).mean()*100:.1f}%)")
    print(f"2=Change Lotteries: {(wr==2).sum()} ({(wr==2).mean()*100:.1f}%)")
    print(f"3=Change Both: {(wr==3).sum()} ({(wr==3).mean()*100:.1f}%)")
    print(f"4=Change Still Inconsistent: {(wr==4).sum()} ({(wr==4).mean()*100:.1f}%)")
    # Favor rate excluding category 4
    change_lot = (wr == 2).sum()
    unselect = (wr == 1).sum()
    print(f"\nFavor axiom (2 vs 1): {change_lot}/{change_lot+unselect} = {change_lot/(change_lot+unselect)*100:.0f}%")
    print(f"Published (with recode 4->3): 79%")
    print(f"Without recode: still {change_lot/(change_lot+unselect)*100:.0f}% (category 4 is rare: {(wr==4).sum()} obs)")

    # ══════════════════════════════════════════════════════════════════════
    print_section("CHECK 2: Leave-One-Axiom-Out Sensitivity")
    # ══════════════════════════════════════════════════════════════════════
    baseline = compute_reconcile_stats(lab_nc)
    print(f"Baseline: n={baseline['n']}, revision={baseline['revision_rate']:.1f}%, favor={baseline['favor_rate']:.1f}%")
    print()
    for ax in AXIOM_ORDER:
        sub = lab_nc[lab_nc['axiomnumber'] != ax]
        stats = compute_reconcile_stats(sub)
        print(f"  Drop {axiom_label(ax)}: n={stats['n']}, revision={stats['revision_rate']:.1f}%, favor={stats['favor_rate']:.1f}%")

    # ══════════════════════════════════════════════════════════════════════
    print_section("CHECK 3: Drop FOSD (most violations, lowest revision rate)")
    # ══════════════════════════════════════════════════════════════════════
    no_fosd = lab_nc[lab_nc['axiomnumber'] != 2]
    stats = compute_reconcile_stats(no_fosd)
    print(f"Without FOSD: n={stats['n']}, revision={stats['revision_rate']:.1f}%, favor={stats['favor_rate']:.1f}%")
    print(f"With FOSD:    n={baseline['n']}, revision={baseline['revision_rate']:.1f}%, favor={baseline['favor_rate']:.1f}%")
    print(f"\nFOSD pulls DOWN both revision rate and favor rate.")
    fosd_stats = compute_reconcile_stats(lab_nc[lab_nc['axiomnumber'] == 2])
    print(f"FOSD alone:   n={fosd_stats['n']}, revision={fosd_stats['revision_rate']:.1f}%, favor={fosd_stats['favor_rate']:.1f}%")

    # ══════════════════════════════════════════════════════════════════════
    print_section("CHECK 4: Restrict to Subjects Who Selected ALL 6 Axioms")
    # ══════════════════════════════════════════════════════════════════════
    q1 = lab_nc[lab_nc['questionnumber'] == 1]
    numgood = q1.groupby('subject')['rulechoice'].sum()
    all6 = numgood[numgood == 6].index
    not_all6 = numgood[numgood < 6].index

    for label, subjs in [('All 6 axioms', all6), ('< 6 axioms', not_all6)]:
        sub = lab_nc[lab_nc['subject'].isin(subjs)]
        stats = compute_reconcile_stats(sub)
        print(f"  {label} (n_subj={len(subjs)}): n_viol={stats['n']}, revision={stats['revision_rate']:.1f}%, favor={stats['favor_rate']:.1f}%")

    # ══════════════════════════════════════════════════════════════════════
    print_section("CHECK 5: Axiom vs c-Axiom Selection Gap (proportion test)")
    # ══════════════════════════════════════════════════════════════════════
    q1 = lab_nc[lab_nc['questionnumber'] == 1]
    from scipy.stats import chi2_contingency, norm as scipy_norm
    for ax in AXIOM_ORDER:
        sub = q1[q1['axiomnumber'] == ax]
        n_ax = sub['rulechoice'].sum()
        n_cax = sub['crulechoice'].sum()
        n = len(sub)
        # Two-proportion z-test
        p1 = n_ax / n
        p2 = n_cax / n
        p_pool = (n_ax + n_cax) / (2 * n)
        se = np.sqrt(2 * p_pool * (1 - p_pool) / n) if p_pool > 0 and p_pool < 1 else 0.001
        z = (p1 - p2) / se if se > 0 else 0
        pval = 2 * (1 - scipy_norm.cdf(abs(z)))
        print(f"  {axiom_label(ax)}: axiom={p1*100:.0f}%, c-axiom={p2*100:.0f}%, gap={p1*100-p2*100:.0f}pp, z={z:.2f}, p={pval:.4f}")

    # ══════════════════════════════════════════════════════════════════════
    print_section("CHECK 6: Heterogeneity by Number of Violations")
    # ══════════════════════════════════════════════════════════════════════
    # Count violations per subject
    viol_count = lab_nc[lab_nc['whichreconcile'].notna()].groupby('subject').size()
    median_v = viol_count.median()
    print(f"Median violations per subject: {median_v}")

    few_subjs = viol_count[viol_count <= median_v].index
    many_subjs = viol_count[viol_count > median_v].index

    for label, subjs in [('Few violations', few_subjs), ('Many violations', many_subjs)]:
        sub = lab_nc[lab_nc['subject'].isin(subjs)]
        stats_out = compute_reconcile_stats(sub)
        print(f"  {label} (n_subj={len(subjs)}): n_viol={stats_out['n']}, revision={stats_out['revision_rate']:.1f}%, favor={stats_out['favor_rate']:.1f}%")

    # ══════════════════════════════════════════════════════════════════════
    print_section("CHECK 7: Order Effects Within Axiom (first vs later questions)")
    # ══════════════════════════════════════════════════════════════════════
    multi_q_axioms = [1, 2, 3, 4, 6]  # IIA(4q), FOSD(4q), TRANS(3q), IND(3q), CONS(2q)
    for ax in multi_q_axioms:
        sub = lab_nc[lab_nc['axiomnumber'] == ax]
        valid = sub[sub['whichreconcile'].notna()]
        if len(valid) == 0:
            continue
        min_q = valid['questionnumber'].min()
        max_q = valid['questionnumber'].max()
        first = valid[valid['questionnumber'] == min_q]
        last = valid[valid['questionnumber'] == max_q]
        first_stats = compute_reconcile_stats(first)
        last_stats = compute_reconcile_stats(last)
        print(f"  {axiom_label(ax)}: first q (n={first_stats['n']}) revision={first_stats['revision_rate']:.0f}%, "
              f"last q (n={last_stats['n']}) revision={last_stats['revision_rate']:.0f}%")

    # ══════════════════════════════════════════════════════════════════════
    print_section("CHECK 8: Bootstrap CIs for Key Statistics")
    # ══════════════════════════════════════════════════════════════════════
    np.random.seed(42)
    n_boot = 10000

    # Bootstrap favor-axiom rate
    wr = lab_nc['whichreconcile'].replace({4: 3}).dropna()
    revised = wr[wr != 0]
    favor_obs = ((revised == 2).sum()) / ((revised == 1).sum() + (revised == 2).sum()) * 100

    boot_favor = []
    for _ in range(n_boot):
        sample = revised.sample(len(revised), replace=True)
        cl = (sample == 2).sum()
        us = (sample == 1).sum()
        if cl + us > 0:
            boot_favor.append(cl / (cl + us) * 100)
    ci_lo, ci_hi = np.percentile(boot_favor, [2.5, 97.5])
    print(f"Axiom favor rate: {favor_obs:.1f}% [95% CI: {ci_lo:.1f}%, {ci_hi:.1f}%]")

    # Bootstrap c-axiom favor rate
    cwr = lab_nc['cwhichreconcile'].replace({4: 3}).dropna()
    c_revised = cwr[cwr != 0]
    c_favor_obs = ((c_revised == 2).sum()) / ((c_revised == 1).sum() + (c_revised == 2).sum()) * 100

    boot_cfavor = []
    for _ in range(n_boot):
        sample = c_revised.sample(len(c_revised), replace=True)
        cl = (sample == 2).sum()
        us = (sample == 1).sum()
        if cl + us > 0:
            boot_cfavor.append(cl / (cl + us) * 100)
    ci_lo_c, ci_hi_c = np.percentile(boot_cfavor, [2.5, 97.5])
    print(f"c-Axiom favor rate: {c_favor_obs:.1f}% [95% CI: {ci_lo_c:.1f}%, {ci_hi_c:.1f}%]")
    print(f"\nCI overlap: {'YES' if ci_hi_c > ci_lo else 'NO'}")

    # Bootstrap revision rate
    boot_rev = []
    for _ in range(n_boot):
        sample = wr.sample(len(wr), replace=True)
        boot_rev.append((sample != 0).mean() * 100)
    ci_lo_r, ci_hi_r = np.percentile(boot_rev, [2.5, 97.5])
    print(f"Revision rate: {(wr!=0).mean()*100:.1f}% [95% CI: {ci_lo_r:.1f}%, {ci_hi_r:.1f}%]")

    # ══════════════════════════════════════════════════════════════════════
    print_section("CHECK 9: MNLogit Without Cluster SEs (Table III)")
    # ══════════════════════════════════════════════════════════════════════
    import statsmodels.api as sm
    onl_ind = onl[onl['axiomnumber'] == 4].copy()
    onl_ind = onl_ind.dropna(subset=['whichreconcile', 'CRTscore', 'totaltestindex'])
    X = onl_ind[['CRTscore', 'totaltestindex']].astype(float)
    X = sm.add_constant(X)
    y = onl_ind['whichreconcile'].astype(int)

    try:
        # Without cluster
        mod_noclust = sm.MNLogit(y, X).fit(disp=0)
        print("MNLogit WITHOUT cluster SEs:")
        print(mod_noclust.summary2().tables[1])
    except Exception as e:
        print(f"Failed: {e}")

    # Without interaction
    try:
        mod_clust = sm.MNLogit(y, X).fit(disp=0, cov_type='cluster',
                                          cov_kwds={'groups': onl_ind['subject'].values})
        print("\nMNLogit WITH cluster SEs (same as paper):")
        print(mod_clust.summary2().tables[1])
    except Exception as e:
        print(f"Failed: {e}")

    # ══════════════════════════════════════════════════════════════════════
    print_section("CHECK 10: Permutation Test (Axiom vs c-Axiom Reconciliation)")
    # ══════════════════════════════════════════════════════════════════════
    np.random.seed(42)

    # Observed: among revisions, what fraction change lotteries for axiom vs c-axiom?
    wr_ax = lab_nc['whichreconcile'].replace({4: 3}).dropna()
    wr_cax = lab_nc['cwhichreconcile'].replace({4: 3}).dropna()

    ax_revised = wr_ax[wr_ax != 0]
    cax_revised = wr_cax[wr_cax != 0]

    ax_favor = (ax_revised == 2).mean() * 100
    cax_favor = (cax_revised == 2).mean() * 100
    obs_diff = ax_favor - cax_favor
    print(f"Observed: axiom favor={ax_favor:.1f}%, c-axiom favor={cax_favor:.1f}%, diff={obs_diff:.1f}pp")

    # Permutation: pool all revised observations, randomly assign to axiom/c-axiom
    pooled = pd.concat([ax_revised, cax_revised], ignore_index=True)
    n_ax = len(ax_revised)
    n_cax = len(cax_revised)
    n_perm = 10000
    perm_diffs = []
    for _ in range(n_perm):
        perm = np.random.permutation(pooled.values)
        perm_ax = perm[:n_ax]
        perm_cax = perm[n_ax:]
        perm_ax_favor = (perm_ax == 2).mean() * 100
        perm_cax_favor = (perm_cax == 2).mean() * 100
        perm_diffs.append(perm_ax_favor - perm_cax_favor)

    p_perm = np.mean(np.array(perm_diffs) >= obs_diff)
    print(f"Permutation p-value (one-sided): {p_perm:.4f}")
    print(f"The {obs_diff:.1f}pp gap is {'significant' if p_perm < 0.05 else 'NOT significant'} at 5%")

    # ══════════════════════════════════════════════════════════════════════
    print_section("CHECK 11: CRT Threshold Sensitivity (Online, IND)")
    # ══════════════════════════════════════════════════════════════════════
    onl_ind = onl[onl['axiomnumber'] == 4].copy()
    for threshold in [3, 4, 5, 6, 7]:
        below = onl_ind[onl_ind['CRTscore'] < threshold]
        above = onl_ind[onl_ind['CRTscore'] >= threshold]
        below_stats = compute_reconcile_stats(below)
        above_stats = compute_reconcile_stats(above)
        print(f"  CRT<{threshold}: n={below_stats['n']}, revision={below_stats['revision_rate']:.0f}%, favor={below_stats['favor_rate']:.0f}%")
        print(f"  CRT>={threshold}: n={above_stats['n']}, revision={above_stats['revision_rate']:.0f}%, favor={above_stats['favor_rate']:.0f}%")
        print()

    # ══════════════════════════════════════════════════════════════════════
    print_section("CHECK 12: Session Fixed Effects")
    # ══════════════════════════════════════════════════════════════════════
    lab_nc2 = lab_nc.copy()
    lab_nc2['revised'] = np.where(lab_nc2['whichreconcile'].notna() & (lab_nc2['whichreconcile'] != 0), 1,
                         np.where(lab_nc2['whichreconcile'] == 0, 0, np.nan))
    valid = lab_nc2[lab_nc2['revised'].notna()]

    print("Revision rate by session:")
    for ses in sorted(valid['sesnumber'].unique()):
        sub = valid[valid['sesnumber'] == ses]
        rate = sub['revised'].mean() * 100
        n = len(sub)
        n_subj = sub['subject'].nunique()
        print(f"  Session {int(ses)}: {rate:.0f}% revised (n={n}, {n_subj} subjects)")

    # Chi-squared test across sessions
    ct = pd.crosstab(valid['revised'], valid['sesnumber'])
    from scipy.stats import chi2_contingency
    chi2, pval, dof, expected = chi2_contingency(ct)
    print(f"\nChi-squared test: chi2={chi2:.2f}, dof={dof}, p={pval:.4f}")
    print(f"Session effects are {'significant' if pval < 0.05 else 'NOT significant'} at 5%")

    print_section("ROBUSTNESS CHECKS COMPLETE")


if __name__ == '__main__':
    main()
