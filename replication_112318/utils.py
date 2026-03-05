"""
Shared utilities for replication of Urquiola (2005)
"Does School Choice Lead to Sorting? Evidence from Tiebout Variation"
AER, 95(4), 1310-1326.
"""
import os
import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '112318-V1')
OUT_DIR = os.path.dirname(__file__)

# Variable orders for each .raw file (from SAS put statements and Stata infile)

ARC1E9S_COLS = [
    'ma9dn', 'ca', 'rgns9d', 'rgnw9d', 'rgnm9d', 'rgnn9d',
    'g19d', 'g29d', 'g39d', 'e9d', 'prm9d', 'sec9d',
    'n9d', 'n9f', 'n19f', 'n29f', 'n39f', 'nm9f',
    'hrc1e9s', 'yrc1e9s', 'hrc9f', 'yrc9f', 'hrc2n9d',
    'hrc1ep9s', 'hrc1es9s', 'yrc1ep9s', 'yrc1es9s',
    'riin9c', 'rcldn9c', 'rcth1n9c', 'rppn9c', 'mdy9c',
    'rwwh9c', 'rooh9c', 'n9c', 'n9c2', 'n9c3'
]

AEDE9S_COLS = [
    'ma9dn', 'ca', 'rgns9d', 'rgnw9d', 'rgnm9d', 'rgnn9d',
    'g19d', 'g29d', 'g39d', 'e9d', 'prm9d', 'sec9d',
    'n9d', 'n9f', 'n19f', 'n29f', 'n39f', 'nm9f',
    'hede9s', 'yede9s', 'burr1', 'burr2', 'hedn9d', 'yedn9d',
    'burr6', 'burr7', 'burr8',
    'riin9c', 'rcldn9c', 'rcth1n9c', 'rppn9c', 'mdy9c',
    'rwwh9c', 'rooh9c', 'n9c', 'n9c2', 'n9c3'
]

ASRC1E9S_COLS = [
    'ma9dn', 'ca', 'zsec', 'n9d', 'rgns9d', 'rgnw9d', 'rgnm9d', 'rgnn9d',
    'nlvl', 'nlvlmf', 'e9d',
    'hrc1e9s', 'yrc1e9s', 'hrc9f', 'yrc9f',
    'rb1n9c', 'rh1n9c', 'riin9c', 'rcldn9c', 'rcth1n9c', 'rppn9c',
    'mdy9c', 'rwwh9c', 'rooh9c', 'n9c', 'n9c2', 'n9c3'
]

ASEDE9S_COLS = [
    'ma9dn', 'ca', 'zsec', 'n9d', 'rgns9d', 'rgnw9d', 'rgnm9d', 'rgnn9d',
    'nlvl', 'nlvlmf', 'e9d',
    'hede9s', 'yede9s', 'hrc9f', 'yrc9f',
    'rb1n9c', 'rh1n9c', 'riin9c', 'rcldn9c', 'rcth1n9c', 'rppn9c',
    'mdy9c', 'rwwh9c', 'rooh9c', 'n9c', 'n9c2', 'n9c3'
]

SCH_COLS = [
    'anid9d', 'ma9dn', 'ca', 'rgns9d', 'rgnw9d', 'rgnm9d', 'rgnn9d',
    'n9d', 'sch9d', 'g19f', 'g29f', 'g39f',
    'n9f', 'n19f', 'n29f', 'n39f', 'nm19f', 'nm29f', 'nm39f', 'nm9f',
    'prmm9f', 'secm9f', 'spd9df',
    'e9f', 'hsrc9f', 'ysrc9f', 'vsrc9f',
    'rb1n9c', 'rh1n9c', 'riin9c', 'rcldn9c', 'rcth1n9c', 'rppn9c',
    'mdy9c', 'rwwh9c', 'rooh9c', 'n9c', 'n9c2', 'n9c3'
]

SSCH_COLS = [
    'ma9dn', 'ca', 'nlvl', 'nlvlmf', 'zsec', 'n9d',
    'rgns9d', 'rgnw9d', 'rgnm9d', 'rgnn9d',
    'hsrc9f', 'ysrc9f', 'vsrc9f',
    'rb1n9c', 'rh1n9c', 'rcldn9c', 'rppn9c', 'riin9c', 'rcth1n9c',
    'rooh9c', 'rwwh9c', 'mdy9c', 'n9c', 'n9c2', 'n9c3'
]

MA9_COLS = [
    'ma9dn', 'ca', 'rN', 'rS', 'rM', 'rW',
    'n19d', 'n29d', 'n39d', 'n9d', 'prm9d', 'sec9d',
    'nm19f', 'nm29f', 'nm39f', 'nm9f', 'prmm9f', 'secm9f',
    'e9d', 'sch9d', 'e9f', 'rve9s', 'rvep9s', 'rves9s',
    'hmrc1e9s', 'hmede9s', 'hmrc2n9d', 'hmrc9f',
    'rb1n9c', 'rh1n9c', 'riin9c', 'rcldn9c', 'rcth1n9c', 'rppn9c',
    'mdy9c', 'rwwh9c', 'rooh9c', 'n9c', 'n9c2', 'n9c3', 'dnty9c',
    'ms9f', 'ms19f', 'ms29f'
]

REFEREE_COLS = [
    'ma9dn', 'ma9g', 'n9g', 'rblkn9g', 'ru18n9g', 'rcthn9g',
    'h9g', 'rowch9g', 'riih9g', 'rwwh9g', 'rooh9g',
    'rhspn9g', 'rimn9g', 'rcldn9g', 'e9g', 'mdny9g', 'pcy9g'
]

MA9S_COLS = [
    'ma9dn', 'ca',
    'nlvl', 'nlvlf', 'prm9d', 'sec9d', 'zsec', 'e9d', 'rve9s',
    'rb1n9c', 'rh1n9c', 'riin9c', 'rcldn9c', 'rcth1n9c', 'rppn9c',
    'mdy9c', 'rwwh9c', 'rooh9c', 'n9c', 'n9c2', 'n9c3'
]


def read_raw(filename, columns):
    """Read a whitespace-delimited .raw file with given column names.
    Handles multi-line records (Stata infile reads tokens sequentially)."""
    path = os.path.join(DATA_DIR, filename)
    with open(path, 'r') as f:
        text = f.read()
    tokens = text.split()
    ncols = len(columns)
    nrows = len(tokens) // ncols
    if len(tokens) % ncols != 0:
        print(f"WARNING: {filename} has {len(tokens)} tokens, not evenly divisible by {ncols} columns")
    data = []
    for i in range(nrows):
        row = tokens[i * ncols:(i + 1) * ncols]
        data.append(row)
    df = pd.DataFrame(data, columns=columns)
    for col in df.columns:
        df[col] = df[col].replace('.', np.nan)
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def load_district_race():
    """Load district-level race sorting data (arc1e9s.raw) for Table 3 cols 1-2."""
    df = read_raw('arc1e9s.raw', ARC1E9S_COLS)
    df['ln9d'] = np.log(df['n9d'])
    df['lnm9f'] = np.log(df['nm9f'])
    df['hrc1e9s'] = df['hrc1e9s'] * 100
    df['yrc1e9s'] = df['yrc1e9s'] * 100
    df['hrc2n9d'] = df['hrc2n9d'] * 100
    df['hrc9f'] = df['hrc9f'] * 100
    df['yrc9f'] = df['yrc9f'] * 100
    df['prm9f'] = df['n19f'] + df['n39f']
    df['sec9f'] = df['n39f'] + df['n29f']
    return df


def load_district_education():
    """Load district-level education sorting data (aede9s.raw) for Table 3 cols 1-2."""
    df = read_raw('aede9s.raw', AEDE9S_COLS)
    df['ln9d'] = np.log(df['n9d'])
    df['lnprm9d'] = np.log(df['prm9d'])
    df['lnsec9d'] = np.log(df['sec9d'])
    df['lnm9f'] = np.log(df['nm9f'])
    df['hede9s'] = df['hede9s'] * 100
    df['hedn9d'] = df['hedn9d'] * 100
    df['yede9s'] = df['yede9s'] * 100
    return df


def load_district_race_stacked():
    """Load stacked district race data (asrc1e9s.raw) for Table 3 cols 3-5."""
    df = read_raw('asrc1e9s.raw', ASRC1E9S_COLS)
    df['lnlvl'] = np.log(df['nlvl'])
    df['lnlvlmf'] = np.log(df['nlvlmf'])
    df['hrc1e9s'] = df['hrc1e9s'] * 100
    df['yrc1e9s'] = df['yrc1e9s'] * 100
    return df


def load_district_edu_stacked():
    """Load stacked district education data (asede9s.raw) for Table 3 cols 3-5."""
    df = read_raw('asede9s.raw', ASEDE9S_COLS)
    df['lnlvl'] = np.log(df['nlvl'])
    df['lnlvlmf'] = np.log(df['nlvlmf'])
    df['hede9s'] = df['hede9s'] * 100
    df['yede9s'] = df['yede9s'] * 100
    return df


def load_schools():
    """Load school-level data (sch.raw) for Table 4 cols 1-3."""
    df = read_raw('sch.raw', SCH_COLS)
    df['ln9d'] = np.log(df['n9d'])
    df['ln9f'] = np.log(df['n9f'])
    df['lnm9f'] = np.log(df['nm9f'])
    df['pf'] = np.where((df['g19f'] == 1) | (df['g39f'] == 1), 1, np.nan)
    df['sf'] = np.where((df['g29f'] == 1) | (df['g39f'] == 1), 1, np.nan)
    df['hsrc9f'] = df['hsrc9f'] * 100
    df['ysrc9f'] = df['ysrc9f'] * 100
    df['vsrc9f'] = df['vsrc9f'] * 100
    return df


def load_schools_stacked():
    """Load stacked school data (ssch.raw) for Table 4 cols 4-6."""
    df = read_raw('ssch.raw', SSCH_COLS)
    df['lnlvl'] = np.log(df['nlvl'])
    df['lnlvlmf'] = np.log(df['nlvlmf'])
    df['hsrc9f'] = df['hsrc9f'] * 100
    df['ysrc9f'] = df['ysrc9f'] * 100
    return df


def load_ma():
    """Load MA-level data (ma9.raw + referee.raw) for Tables 2 and 5."""
    ma = read_raw('ma9.raw', MA9_COLS)
    ref = read_raw('referee.raw', REFEREE_COLS)
    df = ma.merge(ref, on='ma9dn', how='left')

    df['ln9d'] = np.log(df['n9d'])
    df['lprm9d'] = np.log(df['prm9d'])
    df['lsec9d'] = np.log(df['sec9d'])
    df['lnnm9f'] = np.log(df['nm9f'])
    df['diff'] = df['prm9d'] - df['sec9d']
    df['rdiff'] = ((df['prm9d'] - df['sec9d']) / df['prm9d']) * 100
    df['rdiff3'] = ((df['prm9d'] - df['sec9d']) / df['n9d']) * 100
    df['diff2'] = df['prm9d'] / df['sec9d']
    df['spd9df'] = df['nm9f'] / df['n9d']
    df['e9d2'] = df['e9d'] * df['e9d']
    df['hmrc1e9s'] = df['hmrc1e9s'] * 100
    df['hmrc9f'] = df['hmrc9f'] * 100
    df['hmede9s'] = df['hmede9s'] * 100
    df['rms9f'] = df['ms19f'] / df['ms9f']
    df['rel1'] = df['prm9d'] / df['sec9d']
    df['rel2'] = df['prmm9f'] / df['secm9f']

    # Fix region assignments for specific MAs (from Stata code)
    for ma_id in [2440, 3400, 4520, 6020, 8080, 9160]:
        df.loc[df['ma9dn'] == ma_id, 'rS'] = 0

    df['dumm'] = np.where(df['diff2'] == 1, 0, 1)

    return df


def load_ma_private():
    """Load stacked MA data for private enrollment (ma9s.raw) for Table 5 cols 3-6."""
    df = read_raw('ma9s.raw', MA9S_COLS)
    df['lnlvl'] = np.log(df['nlvl'])
    df['lnlvlf'] = np.log(df['nlvlf'])
    return df


def run_ols(df, y_col, x_cols, cluster_col=None, robust=True):
    """Run OLS regression with robust or clustered SEs. Returns statsmodels result."""
    import statsmodels.api as sm

    data = df[list(dict.fromkeys([y_col] + x_cols + ([cluster_col] if cluster_col else [])))].dropna()
    Y = data[y_col]
    X = sm.add_constant(data[x_cols])

    if cluster_col:
        groups = data[cluster_col]
        model = sm.OLS(Y, X).fit(
            cov_type='cluster',
            cov_kwds={'groups': groups}
        )
    elif robust:
        model = sm.OLS(Y, X).fit(cov_type='HC1')
    else:
        model = sm.OLS(Y, X).fit()

    return model


def run_areg(df, y_col, x_cols, absorb_col, cluster_col=None):
    """Run OLS with absorbed fixed effects (areg equivalent)."""
    import statsmodels.api as sm

    all_cols = list(dict.fromkeys([y_col] + x_cols + [absorb_col] + ([cluster_col] if cluster_col else [])))
    data = df[all_cols].dropna().copy()

    # Demean by absorb group
    group_means = data.groupby(absorb_col).transform('mean')
    for col in [y_col] + x_cols:
        data[col] = data[col].astype(float) - group_means[col].astype(float)

    Y = data[y_col]
    X = data[x_cols]

    if cluster_col:
        groups = data[cluster_col]
        # Get number of absorbed groups for DOF correction
        n_groups = df[absorb_col].dropna().nunique()
        model = sm.OLS(Y, X).fit(
            cov_type='cluster',
            cov_kwds={'groups': groups}
        )
    else:
        model = sm.OLS(Y, X).fit(cov_type='HC1')

    return model


def print_reg_result(model, label=""):
    """Print key regression results."""
    if label:
        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"{'='*60}")
    print(f"  N = {int(model.nobs)}")
    print(f"  R² = {model.rsquared:.4f}")
    for var in model.params.index:
        coef = model.params[var]
        se = model.bse[var]
        pval = model.pvalues[var]
        stars = '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.10 else ''
        if var != 'const':
            print(f"  {var:20s}: {coef:10.4f} ({se:.4f}){stars}")
