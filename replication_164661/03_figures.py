"""
03_figures.py – Replicate Figures IV-VII
Nielsen & Rehbeck (2022) "When Choices Are Mistakes"
"""

import sys, os, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings('ignore')
from utils import *
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


def figure_iv():
    """Figure IV: Rule Choices (Lab No-Cost) - stacked bar by axiom."""
    df = load_data()
    d = filter_lab_nocost(df)
    q1 = d[d['questionnumber'] == 1].copy()

    # bothrules: 1=axiom only, 2=both, 3=c-axiom only, 4=neither
    q1['bothrules'] = np.nan
    q1.loc[(q1['rulechoice']==1) & (q1['crulechoice']==0), 'bothrules'] = 1
    q1.loc[(q1['rulechoice']==1) & (q1['crulechoice']==1), 'bothrules'] = 2
    q1.loc[(q1['rulechoice']==0) & (q1['crulechoice']==1), 'bothrules'] = 3
    q1.loc[(q1['rulechoice']==0) & (q1['crulechoice']==0), 'bothrules'] = 4

    cats = [1, 2, 3, 4]
    cat_labels = ['Axiom Only', 'Axiom and c-Axiom', 'c-Axiom Only', 'Neither']
    colors = ['black', '#444444', '#AAAAAA', '#DDDDDD']

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(6)
    width = 0.6

    bottoms = np.zeros(6)
    for cat, clabel, color in zip(cats, cat_labels, colors):
        vals = []
        for ax_num in AXIOM_ORDER:
            sub = q1[q1['axiomnumber'] == ax_num]
            pct = (sub['bothrules'] == cat).mean() * 100
            vals.append(pct)
        ax.bar(x, vals, width, bottom=bottoms, label=clabel, color=color)
        bottoms += vals

    ax.set_xticks(x)
    ax.set_xticklabels([AXIOM_LABELS[a] for a in AXIOM_ORDER])
    ax.set_ylabel('Percentage of Subjects')
    ax.set_title('Axiom Choices')
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=4, fontsize=9)
    ax.set_ylim(0, 105)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'FigureIV.pdf'))
    plt.close()
    print("Figure IV saved.")


def figure_v():
    """Figure V: Axiom Choice Revisions by axiom×question (Lab No-Cost)."""
    df = load_data()
    d = filter_lab_nocost(df)
    d['whichreconcile'] = d['whichreconcile'].replace({4: 3})
    valid = d[d['whichreconcile'].notna()].copy()

    # Build labels: axiom-question pairs
    aq_list = []
    aq_labels = []
    for ax in AXIOM_ORDER:
        nq = len(valid[valid['axiomnumber'] == ax]['questionnumber'].unique())
        for q in sorted(valid[valid['axiomnumber'] == ax]['questionnumber'].unique()):
            sub = valid[(valid['axiomnumber'] == ax) & (valid['questionnumber'] == q)]
            if len(sub) > 0:
                aq_list.append((ax, q))
                aq_labels.append(f"{AXIOM_LABELS[ax]}{q}")

    cats = [2, 1, 3, 0]  # Change Lotteries, Unselect Rule, Change Both, Keep Inconsistent
    cat_labels = ['Change Lotteries', 'Unselect Rule', 'Change Both', 'Keep Inconsistent']
    colors = ['black', '#555555', '#AAAAAA', '#DDDDDD']

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(aq_list))
    width = 0.7

    bottoms = np.zeros(len(aq_list))
    for cat, clabel, color in zip(cats, cat_labels, colors):
        vals = []
        for ax_num, q in aq_list:
            sub = valid[(valid['axiomnumber'] == ax_num) & (valid['questionnumber'] == q)]
            pct = (sub['whichreconcile'] == cat).mean() * 100 if len(sub) > 0 else 0
            vals.append(pct)
        ax.bar(x, vals, width, bottom=bottoms, label=clabel, color=color)
        bottoms += np.array(vals)

    ax.set_xticks(x)
    ax.set_xticklabels(aq_labels, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Percentage of Subjects')
    ax.set_title('Axiom Choice Revisions')
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25), ncol=4, fontsize=9)
    ax.set_ylim(0, 105)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'FigureV.pdf'))
    plt.close()
    print("Figure V saved.")


def figure_vi():
    """Figure VI: c-Axiom Choice Revisions by axiom×question (Lab No-Cost)."""
    df = load_data()
    d = filter_lab_nocost(df)
    d['cwhichreconcile'] = d['cwhichreconcile'].replace({4: 3})
    valid = d[d['cwhichreconcile'].notna()].copy()

    aq_list = []
    aq_labels = []
    for ax in AXIOM_ORDER:
        for q in sorted(valid[valid['axiomnumber'] == ax]['questionnumber'].unique()):
            sub = valid[(valid['axiomnumber'] == ax) & (valid['questionnumber'] == q)]
            if len(sub) > 0:
                aq_list.append((ax, q))
                aq_labels.append(f"c-{AXIOM_LABELS[ax]}{q}")

    cats = [2, 1, 3, 0]
    cat_labels = ['Change Lotteries', 'Unselect Rule', 'Change Both', 'Keep Inconsistent']
    colors = ['black', '#555555', '#AAAAAA', '#DDDDDD']

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(aq_list))
    width = 0.7

    bottoms = np.zeros(len(aq_list))
    for cat, clabel, color in zip(cats, cat_labels, colors):
        vals = []
        for ax_num, q in aq_list:
            sub = valid[(valid['axiomnumber'] == ax_num) & (valid['questionnumber'] == q)]
            pct = (sub['cwhichreconcile'] == cat).mean() * 100 if len(sub) > 0 else 0
            vals.append(pct)
        ax.bar(x, vals, width, bottom=bottoms, label=clabel, color=color)
        bottoms += np.array(vals)

    ax.set_xticks(x)
    ax.set_xticklabels(aq_labels, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Percentage of Subjects')
    ax.set_title('Control Axiom Choice Revisions')
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25), ncol=4, fontsize=9)
    ax.set_ylim(0, 105)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'FigureVI.pdf'))
    plt.close()
    print("Figure VI saved.")


def figure_vii():
    """Figure VII: Conflicting Rule Revisions (Lab No-Cost) - stacked bar by axiom."""
    df = load_data()
    d = filter_lab_nocost(df)
    q1 = d[d['questionnumber'] == 1].copy()
    valid = q1[q1['rulerevise'].notna()].copy()

    # Recode: 2->1 (Unselect c-Axiom), 3->2 (Unselect Axiom), 1->3 (Keep Both), 4 stays
    remap = {2: 1, 3: 2, 1: 3, 4: 4}
    valid['rr_recoded'] = valid['rulerevise'].map(remap)

    cats = [1, 2, 3, 4]
    cat_labels = ['Unselect c-Axiom', 'Unselect Axiom', 'Keep Both', 'Unselect Both']
    colors = ['black', '#555555', '#AAAAAA', '#DDDDDD']

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(6)
    width = 0.6

    bottoms = np.zeros(6)
    for cat, clabel, color in zip(cats, cat_labels, colors):
        vals = []
        for ax_num in AXIOM_ORDER:
            sub = valid[valid['axiomnumber'] == ax_num]
            pct = (sub['rr_recoded'] == cat).mean() * 100 if len(sub) > 0 else 0
            vals.append(pct)
        ax.bar(x, vals, width, bottom=bottoms, label=clabel, color=color)
        bottoms += np.array(vals)

    ax.set_xticks(x)
    ax.set_xticklabels([AXIOM_LABELS[a] for a in AXIOM_ORDER])
    ax.set_ylabel('Percentage of Subjects')
    ax.set_title('Conflicting Rule Revisions')
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=4, fontsize=9)
    ax.set_ylim(0, 105)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'FigureVII.pdf'))
    plt.close()
    print("Figure VII saved.")


def figure_app_vi():
    """Appendix Figure VI: Rule Choices by Cost (Lab, all)."""
    df = load_data()
    d = filter_lab_all(df)
    q1 = d[d['questionnumber'] == 1].copy()

    q1['bothrules'] = np.nan
    q1.loc[(q1['rulechoice']==1) & (q1['crulechoice']==0), 'bothrules'] = 1
    q1.loc[(q1['rulechoice']==1) & (q1['crulechoice']==1), 'bothrules'] = 2
    q1.loc[(q1['rulechoice']==0) & (q1['crulechoice']==1), 'bothrules'] = 3
    q1.loc[(q1['rulechoice']==0) & (q1['crulechoice']==0), 'bothrules'] = 4

    cats = [1, 2, 3, 4]
    cat_labels = ['Axiom Only', 'Axiom and c-Axiom', 'c-Axiom Only', 'Neither']
    colors = ['black', '#444444', '#AAAAAA', '#DDDDDD']

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    for cost_val, cost_label, ax in zip([0, 1], ['$0', '$1'], axes):
        sub = q1[q1['cost'] == cost_val]
        x = np.arange(6)
        width = 0.6
        bottoms = np.zeros(6)

        for cat, clabel, color in zip(cats, cat_labels, colors):
            vals = []
            for ax_num in AXIOM_ORDER:
                s = sub[sub['axiomnumber'] == ax_num]
                pct = (s['bothrules'] == cat).mean() * 100 if len(s) > 0 else 0
                vals.append(pct)
            ax.bar(x, vals, width, bottom=bottoms, label=clabel, color=color)
            bottoms += np.array(vals)

        ax.set_xticks(x)
        ax.set_xticklabels([AXIOM_LABELS[a] for a in AXIOM_ORDER])
        ax.set_title(f'Cost = {cost_label}')
        ax.set_ylim(0, 105)

    axes[0].set_ylabel('Percentage of Subjects')
    axes[0].legend(loc='lower center', bbox_to_anchor=(1.1, -0.15), ncol=4, fontsize=9)
    fig.suptitle('Axiom and c-Axiom Choices', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'FigureVI_Appendix.pdf'))
    plt.close()
    print("Figure VI (Appendix) saved.")


def figure_app_vii():
    """Appendix Figure VII: Axiom Ranking (Cost Treatment)."""
    df = load_data()
    d = filter_lab_cost(df)
    q1 = d[d['questionnumber'] == 1].copy()

    # Spread rankings across axioms for each subject, then add outside option
    rankings = {}
    for ax in AXIOM_ORDER:
        sub = q1[q1['axiomnumber'] == ax]
        rankings[AXIOM_LABELS[ax]] = sub['rank'].dropna().values

    # Outside option rank
    oor = q1[q1['axiomnumber'] == 1]['outsideoptionrank'].dropna().values
    rankings['$1'] = oor

    labels = [AXIOM_LABELS[a] for a in AXIOM_ORDER] + ['$1']
    means = [np.nanmean(rankings.get(l, [])) for l in labels]

    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.arange(len(labels))
    ax.bar(x, means, color='#666666', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Average Ranking Out of Seven')
    ax.set_title('Axiom Ranking')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'FigureVII_Appendix.pdf'))
    plt.close()
    print("Figure VII (Appendix, ranking) saved.")


def main():
    figure_iv()
    figure_v()
    figure_vi()
    figure_vii()
    figure_app_vi()
    figure_app_vii()
    print(f"\nAll figures saved to {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
