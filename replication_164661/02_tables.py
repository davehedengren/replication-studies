"""
02_tables.py – Replicate all tables and in-text statistics
Nielsen & Rehbeck (2022) "When Choices Are Mistakes"
"""

import sys, os, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings('ignore')
from utils import *
import statsmodels.api as sm


def reconcile_tab(df, var, axiom_var='axiomnumber', label=''):
    """
    Produce Table-I-style reconciliation table.
    Recode 4->3 first, then row-percentage cross-tab by axiom.
    """
    d = df.copy()
    d[var] = d[var].replace({4: 3})
    valid = d[d[var].notna()].copy()

    print(f"\n{label}")
    print(f"{'Axiom':<16} {'n':>5}  {'%Keep':>6}  {'%Unsel':>6}  {'%ChgLot':>7}  {'%ChgBoth':>8}")
    print("-" * 55)

    total_n = 0
    rows = []
    for ax in AXIOM_ORDER:
        sub = valid[valid[axiom_var] == ax]
        n = len(sub)
        if n == 0:
            continue
        total_n += n
        pcts = sub[var].value_counts(normalize=True).reindex([0, 1, 2, 3], fill_value=0) * 100
        rows.append((axiom_label(ax), n, pcts[0], pcts[1], pcts[2], pcts[3]))
        print(f"{axiom_label(ax)+f' (n={n})':<16} {n:>5}  {pcts[0]:>5.0f}%  {pcts[1]:>5.0f}%  {pcts[2]:>6.0f}%  {pcts[3]:>7.0f}%")

    # Total
    pcts = valid[var].value_counts(normalize=True).reindex([0, 1, 2, 3], fill_value=0) * 100
    print("-" * 55)
    print(f"{'Total (n='+str(total_n)+')':<16} {total_n:>5}  {pcts[0]:>5.0f}%  {pcts[1]:>5.0f}%  {pcts[2]:>6.0f}%  {pcts[3]:>7.0f}%")
    return rows


def table_ii_style(df, var, group_var='online', axiom_num=4, label=''):
    """Table II style: reconciliation by online/lab for a specific axiom."""
    d = df.copy()
    d[var] = d[var].replace({4: 3})
    valid = d[(d[var].notna()) & (d['axiomnumber'] == axiom_num)].copy()

    print(f"\n{label}")
    print(f"{'Group':<16} {'n':>5}  {'%Keep':>6}  {'%Unsel':>6}  {'%ChgLot':>7}  {'%ChgBoth':>8}")
    print("-" * 55)

    for gval, glabel in [(0, 'Lab'), (1, 'Online')]:
        sub = valid[valid[group_var] == gval]
        n = len(sub)
        if n == 0:
            continue
        pcts = sub[var].value_counts(normalize=True).reindex([0, 1, 2, 3], fill_value=0) * 100
        print(f"{glabel+f' (n={n})':<16} {n:>5}  {pcts[0]:>5.0f}%  {pcts[1]:>5.0f}%  {pcts[2]:>6.0f}%  {pcts[3]:>7.0f}%")


def main():
    df = load_data()
    lab_nc = filter_lab_nocost(df)
    lab_c = filter_lab_cost(df)
    onl = filter_online(df)
    nocost = filter_nocost_all(df)
    lab_all = filter_lab_all(df)

    # ══════════════════════════════════════════════════════════════════════
    # TABLE I: Reconciliation (Lab No-Cost)
    # ══════════════════════════════════════════════════════════════════════
    print_section("TABLE I: Axiom Reconciliation (Lab No-Cost)")
    reconcile_tab(lab_nc, 'whichreconcile', label='Panel A: Axioms')
    print("\nPublished: Total n=468, 37% Keep, 13% Unselect, 47% Change Lot, 3% Both")

    print_section("TABLE I (c-Axioms): c-Axiom Reconciliation (Lab No-Cost)")
    reconcile_tab(lab_nc, 'cwhichreconcile', label='Panel B: c-Axioms')
    print("\nPublished: Total n=124, 33% Keep, 35% Unselect, 20% Change Lot, 11% Both")

    # ══════════════════════════════════════════════════════════════════════
    # TABLE II: IND Reconciliation (Lab vs Online)
    # ══════════════════════════════════════════════════════════════════════
    print_section("TABLE II: IND Reconciliation (Lab vs Online)")
    table_ii_style(nocost, 'whichreconcile', axiom_num=4, label='Panel A: IND')
    print("\nPublished: Lab n=96 (47/16/34/3), Online n=471 (40/24/31/5)")

    table_ii_style(nocost, 'cwhichreconcile', axiom_num=4, label='Panel B: c-IND')
    print("\nPublished: Lab n=29 (38/28/24/10), Online n=216 (41/22/31/6)")

    # ══════════════════════════════════════════════════════════════════════
    # TABLE IV: Individual-level Rule Selection (Lab No-Cost)
    # ══════════════════════════════════════════════════════════════════════
    print_section("TABLE IV: Individual-level Rule Selection (Lab No-Cost)")
    q1 = lab_nc[lab_nc['questionnumber'] == 1].copy()
    numgood = q1.groupby('subject')['rulechoice'].sum().astype(int)
    numbad = q1.groupby('subject')['crulechoice'].sum().astype(int)

    ct = pd.crosstab(numgood, numbad, margins=True, normalize='all') * 100
    print(ct.round(1))
    print(f"\nPublished: 60% selected all 6 axioms, 64.5% selected 0 c-axioms")
    print(f"Replicated: {(numgood==6).mean()*100:.1f}% selected all 6 axioms, "
          f"{(numbad==0).mean()*100:.1f}% selected 0 c-axioms")

    # ══════════════════════════════════════════════════════════════════════
    # IN-TEXT: Subject counts
    # ══════════════════════════════════════════════════════════════════════
    print_section("IN-TEXT STATISTICS")

    print_subsection("Subject Counts")
    print(f"Lab no-cost: {n_subjects(lab_nc)} (paper: 110)")
    q1_nc = lab_nc[lab_nc['questionnumber'] == 1]
    print(f"Lab no-cost, axiom 1, q1: {len(q1_nc[q1_nc['axiomnumber']==1])}")
    print(f"Online: {n_subjects(onl)}")
    print(f"Online, axiom 1, q1: {len(onl[(onl['axiomnumber']==1) & (onl['questionnumber']==1)])}")

    # ── Number of reconciliation opportunities ───────────────────────────
    print_subsection("Number of Reconciliation Opportunities (Lab No-Cost)")
    is_revision = ((lab_nc['rulechoice'] == 1) & (lab_nc['violation'] == 1)).astype(int)
    is_c_revision = ((lab_nc['crulechoice'] == 1) & (lab_nc['cviolation'] == 1)).astype(int)
    is_rule_revision = ((lab_nc['questionnumber'] == 1) & lab_nc['rulerevise'].notna()).astype(int)

    lab_nc_copy = lab_nc.copy()
    lab_nc_copy['is_revision'] = is_revision.values + is_c_revision.values
    lab_nc_copy['is_rule_revision'] = is_rule_revision.values

    num_rev = lab_nc_copy.groupby('subject')['is_revision'].sum()
    num_rule_rev = lab_nc_copy.groupby('subject')['is_rule_revision'].sum()
    total_rev = num_rev + num_rule_rev
    print(f"Total revisions: mean={total_rev.mean():.1f}, min={total_rev.min()}, max={total_rev.max()}")

    # ── Final profit ──────────────────────────────────────────────────────
    print_subsection("Final Profit (Lab No-Cost)")
    profit = lab_nc.drop_duplicates('subject')['finalprofit']
    print(f"Mean: ${profit.mean():.2f} (paper: ~$14)")

    # ── Result 1: Axiom selection rates ──────────────────────────────────
    print_subsection("Result 1: Axiom Selection Rates (Lab No-Cost)")
    q1_nc = lab_nc[lab_nc['questionnumber'] == 1]
    for ax in AXIOM_ORDER:
        sub = q1_nc[q1_nc['axiomnumber'] == ax]
        rule_pct = sub['rulechoice'].mean() * 100
        crule_pct = sub['crulechoice'].mean() * 100
        print(f"  {axiom_label(ax)}: axiom={rule_pct:.0f}%, c-axiom={crule_pct:.0f}%")

    overall_rule = q1_nc['rulechoice'].mean() * 100
    overall_crule = q1_nc['crulechoice'].mean() * 100
    print(f"  Overall: axiom={overall_rule:.0f}%, c-axiom={overall_crule:.0f}%")
    print(f"  Paper: ~85% axioms, ~10% c-axioms")

    # ── Violation rates ──────────────────────────────────────────────────
    print_subsection("Violation Rates (Lab No-Cost, among those who selected axiom)")
    for ax in AXIOM_ORDER:
        sub = lab_nc[(lab_nc['axiomnumber'] == ax) & (lab_nc['rulechoice'] == 1)]
        subj_ax = sub.groupby('subject')['violation'].max()
        ever_viol = (subj_ax == 1).mean() * 100
        print(f"  {axiom_label(ax)}: {ever_viol:.0f}% ever violated")
    print("  Paper: FOSD 85%, IND 75%, Consistency 46%, Transitivity 43%, IIA 38%, Branch 24%")

    # ── Violation rates: selected vs not ─────────────────────────────────
    print_subsection("Violation by Selection Status (Lab No-Cost)")
    viol_sel = lab_nc[lab_nc['rulechoice'] == 1]['violation']
    viol_nosel = lab_nc[lab_nc['rulechoice'] == 0]['violation']
    ct_sel = pd.crosstab(lab_nc['violation'], lab_nc['rulechoice'], margins=True)
    print(ct_sel)
    # Fisher exact
    sel_viol = ((lab_nc['rulechoice'] == 1) & (lab_nc['violation'] == 1)).sum()
    sel_noviol = ((lab_nc['rulechoice'] == 1) & (lab_nc['violation'] == 0)).sum()
    nosel_viol = ((lab_nc['rulechoice'] == 0) & (lab_nc['violation'] == 1)).sum()
    nosel_noviol = ((lab_nc['rulechoice'] == 0) & (lab_nc['violation'] == 0)).sum()
    table_2x2 = [[sel_viol, sel_noviol], [nosel_viol, nosel_noviol]]
    oddsratio, pval = fisher_exact_2x2(table_2x2)
    print(f"Fisher exact p={pval:.4f}")

    # ── Reconciliation direction: axiom vs c-axiom ───────────────────────
    print_subsection("Reconciliation Direction: Axiom Favor Rate")
    # Among revisions of axiom violations, what % change lotteries (favor axiom)?
    wr = lab_nc['whichreconcile'].copy()
    wr = wr.replace({4: 3})
    revised = wr[wr.notna() & (wr != 0)]  # those who revised (not keep inconsistent)
    favor_axiom = (revised == 2).sum()
    unselect = (revised == 1).sum()
    change_both = (revised == 3).sum()
    total_revised = len(revised)
    favor_pct = favor_axiom / (favor_axiom + unselect) * 100 if (favor_axiom + unselect) > 0 else 0
    print(f"Among revisions: {favor_axiom} change lotteries, {unselect} unselect, {change_both} change both")
    print(f"Favor axiom: {favor_pct:.0f}% (paper: 79%)")

    # t-test / ranksum: favor axiom vs 50-50
    good_favor = lab_nc.copy()
    good_favor['goodrulefavoraxiom'] = np.where(good_favor['whichreconcile'] == 2, 1,
                                        np.where(good_favor['whichreconcile'] == 1, 0, np.nan))
    good_favor['badrulefavoraxiom'] = np.where(good_favor['cwhichreconcile'] == 2, 1,
                                       np.where(good_favor['cwhichreconcile'] == 1, 0, np.nan))
    gf = good_favor['goodrulefavoraxiom'].dropna().values
    bf = good_favor['badrulefavoraxiom'].dropna().values

    stat_t, pval_t = ttest_ind(gf, bf)
    stat_rs, pval_rs = wilcoxon_ranksum(gf, bf)
    print(f"t-test axiom vs c-axiom favor rate: t={stat_t:.4f}, p={pval_t:.4f}")
    print(f"Ranksum axiom vs c-axiom favor rate: z={stat_rs:.4f}, p={pval_rs:.4f}")
    print(f"Paper: Ranksum p=0.0000")

    # ── Reconciliation: c-axiom direction (signrank vs 50-50) ────────────
    print_subsection("c-Axiom Reconciliation Direction")
    cwr = lab_nc['cwhichreconcile'].copy()
    cwr = cwr.replace({4: 3})
    c_revised = cwr[cwr.notna() & (cwr != 0)]
    c_favor = (c_revised == 2).sum()
    c_unselect = (c_revised == 1).sum()
    c_favor_pct = c_favor / (c_favor + c_unselect) * 100 if (c_favor + c_unselect) > 0 else 0
    print(f"c-Axiom favor rate: {c_favor_pct:.0f}% (paper: ~40%)")
    # signrank test
    try:
        stat_sr, pval_sr = signrank_vs_constant(bf, 0.5)
        print(f"Signrank c-axiom favor vs 0.5: p={pval_sr:.3f} (paper: p=0.101)")
    except Exception as e:
        print(f"Signrank failed: {e}")

    # ── Likelihood of revising: axiom vs c-axiom ─────────────────────────
    print_subsection("Likelihood of Revising (axiom vs c-axiom)")
    ax_revised_rate = lab_nc['whichreconcile'].notna() & (lab_nc['whichreconcile'] != 0)
    ax_total = lab_nc['whichreconcile'].notna()
    cax_revised_rate = lab_nc['cwhichreconcile'].notna() & (lab_nc['cwhichreconcile'] != 0)
    cax_total = lab_nc['cwhichreconcile'].notna()
    print(f"Axiom: {ax_revised_rate.sum()}/{ax_total.sum()} = {ax_revised_rate.sum()/ax_total.sum()*100:.0f}% revised")
    print(f"c-Axiom: {cax_revised_rate.sum()}/{cax_total.sum()} = {cax_revised_rate.sum()/cax_total.sum()*100:.0f}% revised")
    # Fisher exact
    a = ax_revised_rate.sum()
    b = ax_total.sum() - a
    c = cax_revised_rate.sum()
    d_val = cax_total.sum() - c
    _, pval_f = fisher_exact_2x2([[a, b], [c, d_val]])
    print(f"Fisher exact p={pval_f:.2f} (paper: p=0.59)")

    # ── Both rules selected: reconciliation ──────────────────────────────
    print_subsection("Both Rules Selected: Reconciliation")
    both = lab_nc[(lab_nc['rulechoice'] == 1) & (lab_nc['crulechoice'] == 1)]
    wr_both = both['whichreconcile'].dropna()
    cwr_both = both['cwhichreconcile'].dropna()
    print(f"Axiom revisions when both selected (n={len(wr_both)}):")
    print(wr_both.value_counts().sort_index())
    print(f"\nc-Axiom revisions when both selected (n={len(cwr_both)}):")
    print(cwr_both.value_counts().sort_index())

    # ── Rule reconciliation ──────────────────────────────────────────────
    print_subsection("Rule Reconciliation (Lab No-Cost)")
    rr = lab_nc[lab_nc['questionnumber'] == 1]['rulerevise'].dropna()
    print(f"Total instances: {len(rr)}")
    print(rr.value_counts().sort_index())
    # Among those who unselect only one
    unselect_one = rr[(rr == 2) | (rr == 3)]
    print(f"Of those unselecting one rule (n={len(unselect_one)}):")
    print(f"  Unselect c-axiom: {(unselect_one==2).sum()} ({(unselect_one==2).mean()*100:.0f}%)")
    print(f"  Unselect axiom: {(unselect_one==3).sum()} ({(unselect_one==3).mean()*100:.0f}%)")
    print(f"Paper: >89% unselect c-axiom")

    # ── First vs Last revision ───────────────────────────────────────────
    print_subsection("Revision Order: First vs Last (Lab)")
    for cost_val, cost_label in [(0, 'No-Cost'), (1, 'Cost')]:
        d = lab_all[(lab_all['cost'] == cost_val)].copy()
        d['firstinstance'] = np.nan
        # First instances
        d.loc[(d['axiomnumber']==1) & (d['questionnumber']==1), 'firstinstance'] = 1
        d.loc[(d['axiomnumber']==2) & (d['questionnumber']==2), 'firstinstance'] = 1
        d.loc[(d['axiomnumber']==3) & (d['questionnumber']==1), 'firstinstance'] = 1
        d.loc[(d['axiomnumber']==4) & (d['questionnumber']==1), 'firstinstance'] = 1
        d.loc[(d['axiomnumber']==6) & (d['questionnumber']==1), 'firstinstance'] = 1
        # Last instances
        d.loc[(d['axiomnumber']==1) & (d['questionnumber']==4), 'firstinstance'] = 0
        d.loc[(d['axiomnumber']==2) & (d['questionnumber']==3), 'firstinstance'] = 0
        d.loc[(d['axiomnumber']==3) & (d['questionnumber']==2), 'firstinstance'] = 0
        d.loc[(d['axiomnumber']==4) & (d['questionnumber']==3), 'firstinstance'] = 0
        d.loc[(d['axiomnumber']==6) & (d['questionnumber']==2), 'firstinstance'] = 0

        d['keepinconsistent'] = np.where(d['whichreconcile'] == 0, 1,
                                np.where(d['whichreconcile'].notna() & (d['whichreconcile'] != 0), 0, np.nan))
        valid = d[d['firstinstance'].notna() & d['keepinconsistent'].notna()]
        first = valid[valid['firstinstance'] == 1]
        last = valid[valid['firstinstance'] == 0]
        first_keep = first['keepinconsistent'].mean() * 100
        last_keep = last['keepinconsistent'].mean() * 100
        # Fisher exact
        a = int((first['keepinconsistent'] == 1).sum())
        b = int((first['keepinconsistent'] == 0).sum())
        c = int((last['keepinconsistent'] == 1).sum())
        d_val = int((last['keepinconsistent'] == 0).sum())
        _, pval = fisher_exact_2x2([[a, b], [c, d_val]])
        print(f"  {cost_label}: First {first_keep:.0f}% keep inconsistent, Last {last_keep:.0f}%, Fisher p={pval:.3f}")
    print("  Paper: No-cost 31% vs 40% p=0.124; Cost 33% vs 65% p=0.000")

    # ══════════════════════════════════════════════════════════════════════
    # ONLINE RESULTS
    # ══════════════════════════════════════════════════════════════════════
    print_section("ONLINE RESULTS")

    print_subsection("Online Subject Count")
    print(f"N = {n_subjects(onl)}")

    print_subsection("IND Selection Rates (Online)")
    onl_ind_q1 = onl[(onl['axiomnumber'] == 4) & (onl['questionnumber'] == 1)]
    ind_rate = onl_ind_q1['rulechoice'].mean() * 100
    cind_rate = onl_ind_q1['crulechoice'].mean() * 100
    print(f"IND: {ind_rate:.1f}%")
    print(f"c-IND: {cind_rate:.1f}%")

    # signrank: rulechoice vs crulechoice
    try:
        stat, pval = wilcoxon_signrank(onl_ind_q1['rulechoice'].values,
                                        onl_ind_q1['crulechoice'].values)
        print(f"Signrank rulechoice vs crulechoice: stat={stat:.1f}, p={pval:.4f}")
    except Exception as e:
        print(f"Signrank failed: {e}")

    # Lab vs Online IND selection
    print_subsection("Lab vs Online IND Selection (No-Cost)")
    nc_ind_q1 = nocost[(nocost['axiomnumber'] == 4) & (nocost['questionnumber'] == 1)]
    ct_rule = pd.crosstab(nc_ind_q1['rulechoice'], nc_ind_q1['online'])
    print("rulechoice by online:")
    print(ct_rule)
    _, pval_rule = fisher_exact_2x2(ct_rule.values)
    print(f"Fisher exact p={pval_rule:.4f}")

    ct_crule = pd.crosstab(nc_ind_q1['crulechoice'], nc_ind_q1['online'])
    print("\ncrulechoice by online:")
    print(ct_crule)
    _, pval_crule = fisher_exact_2x2(ct_crule.values)
    print(f"Fisher exact p={pval_crule:.4f}")

    # Lab vs Online IND violation rates
    print_subsection("Lab vs Online IND Violation Rates (No-Cost, axiom selected)")
    nc_ind_sel = nocost[(nocost['axiomnumber'] == 4) & (nocost['rulechoice'] == 1)]
    ct_viol = pd.crosstab(nc_ind_sel['violation'], nc_ind_sel['online'])
    print(ct_viol)
    # Only use violation in {0,1}
    viol_01 = nc_ind_sel[nc_ind_sel['violation'].isin([0, 1])]
    ct_v01 = pd.crosstab(viol_01['violation'], viol_01['online'])
    if ct_v01.shape == (2, 2):
        _, pval_v = fisher_exact_2x2(ct_v01.values)
        print(f"Fisher exact p={pval_v:.4f}")

    # Online selected vs not: violation rates
    print_subsection("Online: Selected vs Not IND Violation")
    onl_ind = onl[onl['axiomnumber'] == 4]
    ct_sv = pd.crosstab(onl_ind['violation'], onl_ind['rulechoice'])
    print(ct_sv)
    sv_01 = onl_ind[onl_ind['violation'].isin([0, 1])]
    ct_sv01 = pd.crosstab(sv_01['violation'], sv_01['rulechoice'])
    if ct_sv01.shape == (2, 2):
        _, pval_sv = fisher_exact_2x2(ct_sv01.values)
        print(f"Fisher exact p={pval_sv:.4f}")

    # ── Online IND vs c-IND reconciliation ───────────────────────────────
    print_subsection("Online IND vs c-IND Reconciliation")
    onl2 = onl.copy()
    onl2['whichreconcile'] = onl2['whichreconcile'].replace({4: 3})
    onl2['cwhichreconcile'] = onl2['cwhichreconcile'].replace({4: 3})
    onl_ind2 = onl2[onl2['axiomnumber'] == 4]

    wr_ind = onl_ind2['whichreconcile'].dropna()
    cwr_ind = onl_ind2['cwhichreconcile'].dropna()
    print(f"IND reconciliation (n={len(wr_ind)}):")
    print((wr_ind.value_counts(normalize=True).sort_index() * 100).round(0))
    print(f"\nc-IND reconciliation (n={len(cwr_ind)}):")
    print((cwr_ind.value_counts(normalize=True).sort_index() * 100).round(0))

    # Fisher exact comparing distributions
    # Stack and test
    wr_long = pd.DataFrame({'revise': pd.concat([wr_ind, cwr_ind], ignore_index=True),
                            'goodrule': [1]*len(wr_ind) + [2]*len(cwr_ind)})
    ct_wr = pd.crosstab(wr_long['revise'], wr_long['goodrule'])
    print(f"\nCross-tab revise × goodrule:")
    print(ct_wr)

    # ── Online: Unselect vs Change signrank tests ────────────────────────
    print_subsection("Online: Unselect IND vs Change Lotteries (signrank vs 0.5)")
    onl_ind3 = onl[onl['axiomnumber'] == 4].copy()
    unsel_ind = np.where(onl_ind3['whichreconcile'] == 1, 1,
                np.where(onl_ind3['whichreconcile'] == 2, 0, np.nan))
    unsel_ind = unsel_ind[~np.isnan(unsel_ind)]
    try:
        stat, pval = signrank_vs_constant(unsel_ind, 0.5)
        print(f"IND unselect vs 0.5: stat={stat:.1f}, p={pval:.4f}")
    except:
        print("Signrank IND failed")

    unsel_cind = np.where(onl_ind3['cwhichreconcile'] == 1, 1,
                 np.where(onl_ind3['cwhichreconcile'] == 2, 0, np.nan))
    unsel_cind = unsel_cind[~np.isnan(unsel_cind)]
    try:
        stat, pval = signrank_vs_constant(unsel_cind, 0.5)
        print(f"c-IND unselect vs 0.5: stat={stat:.1f}, p={pval:.4f}")
    except:
        print("Signrank c-IND failed")

    # ── CRT correlations ─────────────────────────────────────────────────
    print_subsection("CRT Correlations (Online, IND, q1)")
    onl_ind_q1 = onl[(onl['axiomnumber'] == 4) & (onl['questionnumber'] == 1)]
    rho, pval = spearman_corr(onl_ind_q1['CRTscore'].values, onl_ind_q1['rulechoice'].values)
    print(f"Spearman CRT vs rulechoice: rho={rho:.4f}, p={pval:.4f}")
    rho, pval = spearman_corr(onl_ind_q1['CRTscore'].values, onl_ind_q1['crulechoice'].values)
    print(f"Spearman CRT vs crulechoice: rho={rho:.4f}, p={pval:.4f}")
    rho, pval = spearman_corr(onl_ind_q1['CRTscore'].values, onl_ind_q1['totaltestindex'].values)
    print(f"Spearman CRT vs totaltestindex: rho={rho:.4f}, p={pval:.4f}")

    # ══════════════════════════════════════════════════════════════════════
    # TABLE V: Probit (Online, IND selection)
    # ══════════════════════════════════════════════════════════════════════
    print_section("TABLE V: Probit (Online, IND selection on CRT×testindex)")
    onl_ind_q1 = onl[(onl['axiomnumber'] == 4) & (onl['questionnumber'] == 1)].copy()
    onl_ind_q1 = onl_ind_q1.dropna(subset=['CRTscore', 'totaltestindex', 'rulechoice', 'crulechoice'])
    X = onl_ind_q1[['CRTscore', 'totaltestindex']].astype(float)
    X['CRT_x_test'] = X['CRTscore'] * X['totaltestindex']
    X = sm.add_constant(X)

    for yvar, ylabel in [('rulechoice', 'IND'), ('crulechoice', 'c-IND')]:
        y = onl_ind_q1[yvar].astype(float)
        try:
            mod = sm.Probit(y, X).fit(disp=0, cov_type='cluster',
                                       cov_kwds={'groups': onl_ind_q1['subject'].values})
            print(f"\n{ylabel} Probit:")
            print(mod.summary2().tables[1])
        except Exception as e:
            print(f"\n{ylabel} Probit failed: {e}")

    # ══════════════════════════════════════════════════════════════════════
    # TABLE III: Multinomial Logit (Online, IND reconciliation)
    # ══════════════════════════════════════════════════════════════════════
    print_section("TABLE III: MNLogit (Online, IND reconciliation)")
    onl_ind = onl[onl['axiomnumber'] == 4].copy()
    onl_ind = onl_ind.dropna(subset=['whichreconcile', 'CRTscore', 'totaltestindex'])
    X_mn = onl_ind[['CRTscore', 'totaltestindex']].astype(float)
    X_mn = sm.add_constant(X_mn)
    y_mn = onl_ind['whichreconcile'].astype(int)

    try:
        mod = sm.MNLogit(y_mn, X_mn).fit(disp=0, cov_type='cluster',
                                          cov_kwds={'groups': onl_ind['subject'].values})
        print(f"Base outcome: {mod.model.endog.min()}")
        print(mod.summary2().tables[1])
    except Exception as e:
        print(f"MNLogit failed: {e}")
        # Try without cluster
        try:
            mod = sm.MNLogit(y_mn, X_mn).fit(disp=0)
            print("(Without cluster SEs)")
            print(mod.summary2().tables[1])
        except Exception as e2:
            print(f"MNLogit also failed without cluster: {e2}")

    # ══════════════════════════════════════════════════════════════════════
    # TABLE VI: MNLogit (Online, c-IND reconciliation)
    # ══════════════════════════════════════════════════════════════════════
    print_section("TABLE VI: MNLogit (Online, c-IND reconciliation)")
    onl_ind = onl[onl['axiomnumber'] == 4].copy()
    onl_ind = onl_ind.dropna(subset=['cwhichreconcile', 'CRTscore', 'totaltestindex'])
    X_mn = onl_ind[['CRTscore', 'totaltestindex']].astype(float)
    X_mn = sm.add_constant(X_mn)
    y_mn = onl_ind['cwhichreconcile'].astype(int)

    try:
        mod = sm.MNLogit(y_mn, X_mn).fit(disp=0, cov_type='cluster',
                                          cov_kwds={'groups': onl_ind['subject'].values})
        print(mod.summary2().tables[1])
    except Exception as e:
        print(f"MNLogit failed: {e}")
        try:
            mod = sm.MNLogit(y_mn, X_mn).fit(disp=0)
            print("(Without cluster SEs)")
            print(mod.summary2().tables[1])
        except Exception as e2:
            print(f"Also failed: {e2}")

    # ── CRT subgroup analysis ────────────────────────────────────────────
    print_section("CRT SUBGROUP ANALYSIS (Online, IND)")
    onl_ind = onl[onl['axiomnumber'] == 4].copy()
    for crt_label, crt_cond in [('Below avg (CRT<5)', onl_ind['CRTscore'] < 5),
                                 ('Above avg (CRT>=5)', onl_ind['CRTscore'] >= 5)]:
        sub = onl_ind[crt_cond]
        wr = sub['whichreconcile'].dropna()
        print(f"\n{crt_label} (n={len(wr)}):")
        print(wr.value_counts().sort_index())
        # signrank: unselect vs change lotteries
        unsel = np.where(sub['whichreconcile'] == 1, 1,
                np.where(sub['whichreconcile'] == 2, 0, np.nan))
        unsel = unsel[~np.isnan(unsel)]
        if len(unsel) > 0:
            try:
                stat, pval = signrank_vs_constant(unsel, 0.5)
                print(f"  Signrank unselect vs 0.5: stat={stat:.1f}, p={pval:.4f}")
            except:
                print("  Signrank failed")

    # ── Response time ────────────────────────────────────────────────────
    print_section("RESPONSE TIME (Online, IND)")
    onl_ind = onl[onl['axiomnumber'] == 4].copy()
    onl_ind['unselectIND'] = np.where(onl_ind['whichreconcile'] == 1, 1,
                             np.where(onl_ind['whichreconcile'] == 2, 0,
                             np.where(onl_ind['whichreconcile'] == 0, 2, np.nan)))

    # Unselect vs Change lotteries
    unsel_fc = onl_ind[onl_ind['unselectIND'] == 1]['firstclick'].dropna()
    change_fc = onl_ind[onl_ind['unselectIND'] == 0]['firstclick'].dropna()
    keep_fc = onl_ind[onl_ind['unselectIND'] == 2]['firstclick'].dropna()

    if len(unsel_fc) > 0 and len(change_fc) > 0:
        stat_t, pval_t = ttest_ind(unsel_fc, change_fc)
        stat_rs, pval_rs = wilcoxon_ranksum(unsel_fc, change_fc)
        print(f"Unselect vs Change: ttest p={pval_t:.4f}, ranksum p={pval_rs:.4f}")
        print(f"  Unselect mean={unsel_fc.mean():.1f}, Change mean={change_fc.mean():.1f}")

    if len(change_fc) > 0 and len(keep_fc) > 0:
        stat_t, pval_t = ttest_ind(keep_fc, change_fc)
        stat_rs, pval_rs = wilcoxon_ranksum(keep_fc, change_fc)
        print(f"Keep vs Change: ttest p={pval_t:.4f}, ranksum p={pval_rs:.4f}")

    # ── WTP (Cost treatment) ─────────────────────────────────────────────
    print_section("WTP (Cost Treatment)")
    for ax in AXIOM_ORDER:
        sub = lab_c[(lab_c['axiomnumber'] == ax)]
        wtp = sub['wtp'].dropna()
        if len(wtp) > 0:
            print(f"  {axiom_label(ax)}: mean={wtp.mean():.3f}, n={len(wtp)}")
    wtp_all = lab_c['wtp'].dropna()
    print(f"  Overall: mean={wtp_all.mean():.3f}, n={len(wtp_all)}")

    # ══════════════════════════════════════════════════════════════════════
    # APPENDIX TABLES XIV-XVII
    # ══════════════════════════════════════════════════════════════════════
    print_section("APPENDIX TABLE XIV: Detailed Reconciliation (Lab No-Cost, Axioms)")
    d = lab_nc.copy()
    d['whichreconcile'] = d['whichreconcile'].replace({4: 3})
    d['axiomquestion'] = d['axiomnumber'] * 10 + d['questionnumber']
    valid = d[d['whichreconcile'].notna()]
    ct = pd.crosstab(valid['axiomquestion'], valid['whichreconcile'])
    print(ct)

    print_section("APPENDIX TABLE XV: Detailed Reconciliation (Lab No-Cost, c-Axioms)")
    d = lab_nc.copy()
    d['cwhichreconcile'] = d['cwhichreconcile'].replace({4: 3})
    d['axiomquestion'] = d['axiomnumber'] * 10 + d['questionnumber']
    valid = d[d['cwhichreconcile'].notna()]
    ct = pd.crosstab(valid['axiomquestion'], valid['cwhichreconcile'])
    print(ct)

    print_section("APPENDIX TABLE XVI: Detailed Reconciliation (Lab Cost, Axioms)")
    d = lab_c.copy()
    d['whichreconcile'] = d['whichreconcile'].replace({4: 3})
    d['axiomquestion'] = d['axiomnumber'] * 10 + d['questionnumber']
    valid = d[d['whichreconcile'].notna()]
    ct = pd.crosstab(valid['axiomquestion'], valid['whichreconcile'])
    print(ct)

    print_section("APPENDIX TABLE XVII: Detailed Reconciliation (Lab Cost, c-Axioms)")
    d = lab_c.copy()
    d['cwhichreconcile'] = d['cwhichreconcile'].replace({4: 3})
    d['axiomquestion'] = d['axiomnumber'] * 10 + d['questionnumber']
    valid = d[d['cwhichreconcile'].notna()]
    ct = pd.crosstab(valid['axiomquestion'], valid['cwhichreconcile'])
    print(ct)

    # ── Appendix D: Cost treatment reconciliation ────────────────────────
    print_section("APPENDIX D: Cost Treatment Reconciliation Rates")
    d = lab_all.copy()
    d['whichreconcile'] = d['whichreconcile'].replace({4: 3})
    d['revised'] = np.where(d['whichreconcile'] == 0, 0,
                   np.where(d['whichreconcile'].notna() & (d['whichreconcile'] > 0), 1, np.nan))
    valid = d[d['revised'].notna()]
    ct = pd.crosstab(valid['revised'], valid['cost'])
    print("Revised vs not by cost:")
    print(ct)
    ct_pct = pd.crosstab(valid['revised'], valid['cost'], normalize='columns') * 100
    print(ct_pct.round(0))
    if ct.shape == (2, 2):
        _, pval = fisher_exact_2x2(ct.values)
        print(f"Fisher exact p={pval:.4f}")

    # among those who revised (excluding change both): unselect vs change lotteries
    revised_sub = d[(d['whichreconcile'].isin([1, 2]))].copy()
    ct2 = pd.crosstab(revised_sub['whichreconcile'], revised_sub['cost'])
    print("\nUnselect(1) vs Change Lotteries(2) by cost:")
    print(ct2)
    ct2_pct = pd.crosstab(revised_sub['whichreconcile'], revised_sub['cost'], normalize='columns') * 100
    print(ct2_pct.round(0))
    if ct2.shape == (2, 2):
        _, pval = fisher_exact_2x2(ct2.values)
        print(f"Fisher exact p={pval:.4f}")

    # Same for c-axioms
    d['cwhichreconcile'] = d['cwhichreconcile'].replace({4: 3})
    d['crevised'] = np.where(d['cwhichreconcile'] == 0, 0,
                    np.where(d['cwhichreconcile'].notna() & (d['cwhichreconcile'] > 0), 1, np.nan))
    valid_c = d[d['crevised'].notna()]
    ct_c = pd.crosstab(valid_c['crevised'], valid_c['cost'])
    print("\nc-Axiom revised vs not by cost:")
    print(ct_c)
    ct_c_pct = pd.crosstab(valid_c['crevised'], valid_c['cost'], normalize='columns') * 100
    print(ct_c_pct.round(0))

    # ── Appendix E: WTP summary ──────────────────────────────────────────
    print_section("APPENDIX E: WTP Summary (Cost Treatment)")
    for ax in AXIOM_ORDER:
        sub = lab_c[(lab_c['axiomnumber'] == ax)]
        wtp_vals = sub.drop_duplicates('subject')['wtp'].dropna()
        if len(wtp_vals) > 0:
            print(f"  {axiom_label(ax)}: mean={wtp_vals.mean():.3f}, n={len(wtp_vals)}")
    wtp_all = lab_c.drop_duplicates('subject')['wtp']
    # Actually wtp varies by axiom, so use all rows
    wtp_all2 = lab_c[lab_c['questionnumber']==1]['wtp'].dropna()
    print(f"  Overall (q1): mean={wtp_all2.mean():.3f}, n={len(wtp_all2)}")


if __name__ == '__main__':
    main()
