"""
Replication utilities for paper 225841
Greenwald & Guren - Do Credit Conditions Move House Prices?
"""
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', '225841-V1', 'Empirical', 'data', 'combined')


def load_dta(filename):
    """Load a Stata .dta file from the data directory."""
    return pd.read_stata(os.path.join(DATA_DIR, filename), convert_categoricals=False)


def run_local_projection(df, yvar, xvar, controls, fe_vars, cluster_var, weight_var=None,
                         condition=None, max_h=5):
    """
    Run local projection IRF: for h=0,...,max_h, regress F(h).y on x + controls
    with fixed effects (absorbed via demeaning) and cluster-robust SEs.

    Returns dict with arrays: b, se, b_hi, b_lo for each horizon h.
    """
    data = df.copy()
    if condition is not None:
        data = data[condition].copy()

    # Sort by panel and time for lead generation
    data = data.sort_values(fe_vars).reset_index(drop=True)

    results = {'h': [], 'b': [], 'se': [], 'b_hi': [], 'b_lo': [], 'n': []}

    for h in range(max_h + 1):
        # Create forward lead of y
        lead_col = f'F{h}_{yvar}'
        data[lead_col] = data.groupby(fe_vars[1])[yvar].shift(-h)  # fe_vars[1] = panelvar

        all_vars = [lead_col, xvar] + controls + fe_vars
        if weight_var:
            all_vars.append(weight_var)
        all_vars.append(cluster_var)
        all_vars = list(dict.fromkeys(all_vars))

        sub = data.dropna(subset=all_vars).copy()
        if len(sub) < 10:
            continue

        y = sub[lead_col].astype(float)
        X_vars = [xvar] + controls

        # Demean by each FE
        y_dm = y.copy()
        X_dm = sub[X_vars].astype(float).copy()
        for fe in fe_vars:
            groups = sub[fe]
            y_dm = y_dm - y_dm.groupby(groups).transform('mean')
            for col in X_vars:
                X_dm[col] = X_dm[col] - X_dm[col].groupby(groups).transform('mean')

        # Weighted regression
        if weight_var and weight_var in sub.columns:
            w = sub[weight_var].astype(float).values
            w = w / w.mean()  # normalize
            sqw = np.sqrt(w)
            y_w = y_dm.values * sqw
            X_w = X_dm.values * sqw[:, None]
        else:
            y_w = y_dm.values
            X_w = X_dm.values

        n = len(y_w)
        k = X_w.shape[1]
        n_fe = sum(sub[fe].nunique() for fe in fe_vars)

        try:
            XtX = X_w.T @ X_w
            Xty = X_w.T @ y_w
            beta = np.linalg.solve(XtX, Xty)
        except np.linalg.LinAlgError:
            continue

        resid = y_w - X_w @ beta

        # Cluster-robust SEs
        cluster_arr = sub[cluster_var].values
        n_clusters = len(np.unique(cluster_arr))
        dof_resid = n - k - n_fe
        if dof_resid <= 0:
            dof_resid = n - k
        correction = (n_clusters / (n_clusters - 1)) * ((n - 1) / dof_resid)

        bread = np.linalg.inv(XtX)
        meat = np.zeros((k, k))

        # Need to use unweighted X for cluster meat when using WLS
        X_dm_arr = X_dm.values
        if weight_var and weight_var in sub.columns:
            resid_uw = y_dm.values - X_dm_arr @ beta
            for g in np.unique(cluster_arr):
                mask = cluster_arr == g
                Xg = X_dm_arr[mask] * np.sqrt(w[mask])[:, None]
                eg = resid[mask]
                score_g = Xg.T @ eg
                meat += np.outer(score_g, score_g)
        else:
            for g in np.unique(cluster_arr):
                mask = cluster_arr == g
                Xg = X_dm_arr[mask]
                eg = resid[mask]
                score_g = Xg.T @ eg
                meat += np.outer(score_g, score_g)

        var_cluster = correction * bread @ meat @ bread
        se_vals = np.sqrt(np.diag(var_cluster))

        b_x = beta[0]  # xvar is first
        se_x = se_vals[0]

        results['h'].append(h)
        results['b'].append(b_x)
        results['se'].append(se_x)
        results['b_hi'].append(b_x + 1.96 * se_x)
        results['b_lo'].append(b_x - 1.96 * se_x)
        results['n'].append(n)

    return results


def print_irf(results, title=""):
    """Print IRF results."""
    if title:
        print(f"\n{'='*65}")
        print(title)
        print(f"{'='*65}")
    print(f"{'h':<5} {'Coef':>10} {'SE':>10} {'95% CI':>25} {'N':>8}")
    print("-" * 60)
    for i in range(len(results['h'])):
        h = results['h'][i]
        b = results['b'][i]
        se = results['se'][i]
        lo = results['b_lo'][i]
        hi = results['b_hi'][i]
        n = results['n'][i]
        print(f"{h:<5} {b:>10.4f} {se:>10.4f}   [{lo:>8.4f}, {hi:>8.4f}] {n:>8}")
