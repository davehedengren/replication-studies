"""
03_figure.py: Reproduce Figure 1 for replication of
Bryan, Karlan & Osman (AER, 2024) "Big Loans to Small Businesses"
Figure 1: Outstanding ABA debt over time
"""

import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from utils import OUTPUT_DIR, RAW_DATA_DIR, winsor


def main():
    print("=" * 60)
    print("03_figure.py: Figure 1 — Outstanding Debt Trajectory")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load installment-level admin data
    admin = pd.read_stata(os.path.join(RAW_DATA_DIR, '13-ABA-admin.dta'),
                          convert_categoricals=False)

    # Drop problem loans
    bad_loans = [63568220150514, 3676720160727, 32773920141012, 611443320141209]
    admin = admin[~admin['loan_id'].isin(bad_loans)]

    # Find installment columns
    mature_cols = [c for c in admin.columns if c.startswith('mature') and c[6:].isdigit()]
    max_inst = max(int(c[6:]) for c in mature_cols) if mature_cols else 0

    # Reshape to installment level
    rows = []
    for _, loan in admin.iterrows():
        for i in range(1, max_inst + 1):
            mat_col = f'mature{i}'
            pay_col = f'payment{i}'
            amt_col = f'amount{i}'
            if mat_col in admin.columns and pd.notna(loan.get(mat_col)):
                payment = loan.get(pay_col)
                if pd.isna(payment):
                    continue
                rows.append({
                    'client_key': loan['client_key'],
                    'loan_id': loan.get('loan_id', np.nan),
                    'installment': i,
                    'mature': loan[mat_col],
                    'payment': payment,
                    'amount': loan.get(amt_col, 0),
                    'loansize': loan.get('loansize', 0),
                    'ninstal': loan.get('ninstal', 0),
                    'duration': loan.get('duration', 0),
                    'treatment': loan.get('treatment', np.nan),
                    'edm': loan.get('edm', np.nan),
                })

    inst = pd.DataFrame(rows)
    inst['payment'] = pd.to_datetime(inst['payment'], errors='coerce')
    inst = inst[inst['payment'] >= pd.Timestamp('2015-05-01')]

    # Compute cumulative amount paid
    inst = inst.sort_values(['loan_id', 'installment'])
    inst['amountpaid'] = inst.groupby('loan_id')['amount'].cumsum()

    # Total due per loan
    inst['mduration'] = np.floor(inst['duration'] / 30)
    inst.loc[inst['mduration'] < 0, 'mduration'] = inst['ninstal']
    inst['interest'] = (1.42 * inst['mduration']) / 100
    inst['totdue'] = inst['loansize'] * (1 + inst['interest'])

    # Monthly
    inst['payment_m'] = inst['payment'].dt.to_period('M')
    inst = inst.sort_values(['loan_id', 'payment_m', 'installment'])
    inst = inst.drop_duplicates(subset=['loan_id', 'payment_m'], keep='last')

    # Outstanding
    inst['outstanding_m'] = inst['totdue'] - inst['amountpaid']

    # Months since EDM
    inst['edm'] = pd.to_datetime(inst['edm'], errors='coerce')
    inst['edm_period'] = inst['edm'].dt.to_period('M')
    inst['msinceedm'] = (inst['payment_m'] - inst['edm_period']).apply(
        lambda x: x.n if pd.notna(x) else np.nan)

    # Aggregate: mean outstanding by treatment × months since EDM
    # Drop outlier months
    inst = inst[(inst['msinceedm'] >= -5) & (inst['msinceedm'] <= 40)]

    # Collapse by client-month
    client_month = inst.groupby(['client_key', 'msinceedm', 'treatment']).agg(
        outstanding=('outstanding_m', 'sum')
    ).reset_index()

    # Mean by treatment × month
    summary = client_month.groupby(['msinceedm', 'treatment']).agg(
        mean_outstanding=('outstanding', 'mean'),
        se_outstanding=('outstanding', 'sem'),
        n=('outstanding', 'count'),
    ).reset_index()

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    for treat, label, color in [(0, 'Control', 'blue'), (1, 'Treatment', 'red')]:
        d = summary[summary['treatment'] == treat].sort_values('msinceedm')
        ax.plot(d['msinceedm'], d['mean_outstanding'], label=label, color=color, linewidth=2)
        ax.fill_between(d['msinceedm'],
                        d['mean_outstanding'] - 1.96 * d['se_outstanding'],
                        d['mean_outstanding'] + 1.96 * d['se_outstanding'],
                        alpha=0.2, color=color)

    ax.set_xlabel('Months since EDM')
    ax.set_ylabel('Average Outstanding ABA Debt (EGP)')
    ax.set_title('Figure 1: Outstanding ABA Debt Over Time')
    ax.legend()
    ax.axhline(0, color='black', linewidth=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'figure1_outstanding_debt.png'), dpi=150)
    print(f"Saved figure to {OUTPUT_DIR}/figure1_outstanding_debt.png")


if __name__ == '__main__':
    main()
