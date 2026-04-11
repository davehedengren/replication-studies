"""Phase 2: Reproduce Tables 4 and 5 — fit and out-of-sample log-likelihoods
for logit-QRE and mu-equilibrium on the four 3x3 games."""

import numpy as np
from scipy.optimize import minimize, minimize_scalar

from utils import (
    GAMES_3x3,
    GAME_NAMES,
    load_3x3,
    load_mat,
    loglik_3x3_qre,
    loglik_3x3_mu_indiv,
)


def load_all_3x3():
    choices = {
        1: load_3x3("3x3_DS1")[0],
        2: load_3x3("3x3_DS2")[0],
        3: load_3x3("3x3_Nologit")[0],
        4: load_3x3("3x3_KM")[0],  # n=240 for KM
    }
    ids_all = load_mat("id_data")["ID"]  # 480 x 4
    ids = {i + 1: ids_all[:, i] for i in range(4)}
    return choices, ids


def fit_qre(game_idx, choices, x0=0.5):
    """Local minimizer with x0=0.5 to match MATLAB's fmincon starting point.
    The QRE likelihood is bimodal (very flat near lambda=0, then a cliff
    to -infinity as lambda becomes large), so a local start is essential."""
    # Grid-warmstart to avoid landing in the bad basin (lambda>=0.3 collapses
    # to near-pure strategy and gives huge negative log-likelihood).
    grid = [1e-6, 0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.15, 0.2]
    best = (np.inf, None)
    for lam0 in grid:
        val = -loglik_3x3_qre(lam0, GAMES_3x3[game_idx], choices)
        if val < best[0]:
            best = (val, lam0)
    lam_start = best[1]
    res = minimize(
        lambda lam: -loglik_3x3_qre(max(lam[0], 0.0), GAMES_3x3[game_idx], choices),
        x0=[lam_start],
        method="L-BFGS-B",
        bounds=[(0.0, 5.0)],
        options={"ftol": 1e-12, "gtol": 1e-10},
    )
    lam_hat = max(float(res.x[0]), 0.0)
    return lam_hat, loglik_3x3_qre(lam_hat, GAMES_3x3[game_idx], choices)


def fit_mu(game_idx, choices, ids, x0=0.3):
    res = minimize_scalar(
        lambda eps: -loglik_3x3_mu_indiv(eps, game_idx, choices, ids),
        bounds=(0.0, 1.0),
        method="bounded",
        options={"xatol": 1e-8},
    )
    return res.x, -res.fun


def table_4_and_5():
    choices, ids = load_all_3x3()

    # KM has 240 observations
    obs = {1: 480, 2: 480, 3: 480, 4: 240}

    # Fit per-game
    lam = {}
    logL_qre = {}
    eps = {}
    logL_mu = {}
    for g in [1, 2, 3, 4]:
        ch = choices[g]
        idv = ids[g][:len(ch)] if g == 4 else ids[g]
        lam[g], logL_qre[g] = fit_qre(g, ch)
        eps[g], logL_mu[g] = fit_mu(g, ch, idv)

    # Out of sample
    oos_qre = np.full((5, 4), np.nan)
    oos_mu = np.full((5, 4), np.nan)
    for g in [1, 2, 3, 4]:
        for j in [1, 2, 3, 4]:
            if g == j:
                continue
            ch_j = choices[j]
            id_j = ids[j][:len(ch_j)] if j == 4 else ids[j]
            oos_qre[g - 1, j - 1] = loglik_3x3_qre(lam[g], GAMES_3x3[j], ch_j)
            oos_mu[g - 1, j - 1] = loglik_3x3_mu_indiv(eps[g], j, ch_j, id_j)

    # Pooled
    def pooled_qre_neg(lam_val):
        if np.ndim(lam_val) > 0:
            lam_val = lam_val[0]
        return -sum(
            loglik_3x3_qre(lam_val, GAMES_3x3[g], choices[g])
            for g in [1, 2, 3, 4]
        )

    def pooled_mu_neg(eps_val):
        total = 0.0
        for g in [1, 2, 3, 4]:
            ch = choices[g]
            idv = ids[g][:len(ch)] if g == 4 else ids[g]
            total += loglik_3x3_mu_indiv(eps_val, g, ch, idv)
        return -total

    # Pooled QRE — grid warmstart then Nelder-Mead
    grid = [0.001, 0.005, 0.01, 0.0125, 0.02, 0.05, 0.08]
    best = (np.inf, None)
    for lam0 in grid:
        v = pooled_qre_neg(lam0)
        if v < best[0]:
            best = (v, lam0)
    res_q = minimize(pooled_qre_neg, x0=[best[1]], method="Nelder-Mead",
                     options={"xatol": 1e-9, "fatol": 1e-10})
    lam_pooled = float(res_q.x[0])
    logL_qre_pooled = -float(res_q.fun)

    # Pooled mu — grid warmstart
    grid = np.linspace(0.05, 0.95, 19)
    best = (np.inf, None)
    for e0 in grid:
        v = pooled_mu_neg(float(e0))
        if v < best[0]:
            best = (v, float(e0))
    res_m = minimize(
        lambda e: pooled_mu_neg(e[0] if np.ndim(e) > 0 else e),
        x0=[best[1]],
        method="Nelder-Mead",
        options={"xatol": 1e-9, "fatol": 1e-10},
    )
    eps_pooled = float(res_m.x[0])
    logL_mu_pooled = -float(res_m.fun)

    for j in [1, 2, 3, 4]:
        ch_j = choices[j]
        id_j = ids[j][:len(ch_j)] if j == 4 else ids[j]
        oos_qre[4, j - 1] = loglik_3x3_qre(lam_pooled, GAMES_3x3[j], ch_j)
        oos_mu[4, j - 1] = loglik_3x3_mu_indiv(eps_pooled, j, ch_j, id_j)

    # Published values for comparison
    pub_qre = {
        "lam": {1: 0.0376, 2: 0.0000, 3: 0.0078, 4: 0.0696, "P": 0.0125},
        "logL": {1: -470.0, 2: -527.3, 3: -496.2, 4: -183.3, "P": -1784.3},
        "oos": np.array([
            [np.nan, -616.0, -543.5, -203.2],
            [-527.3, np.nan, -527.3, -263.7],
            [-501.4, -540.5, np.nan, -250.8],
            [-480.3, -678.2, -582.9, np.nan],
            [-490.4, -550.8, -499.9, -243.1],
        ]),
    }
    pub_mu = {
        "eps": {1: 0.382, 2: 0.378, 3: 0.522, 4: 0.290, "P": 0.384},
        "logL": {1: -429.8, 2: -449.8, 3: -468.3, 4: -175.3, "P": -1529.9},
        "oos": np.array([
            [np.nan, -450.0, -479.9, -178.7],
            [-429.9, np.nan, -480.6, -178.4],
            [-468.5, -640.0, np.nan, -192.4],
            [-445.5, -465.3, -506.0, np.nan],
            [-429.8, -450.1, -479.5, -178.8],
        ]),
    }

    # Print Table 4
    print("=" * 88)
    print("Table 4: logit-QRE fit & out-of-sample log-likelihoods")
    print("=" * 88)
    print("Replication")
    header = f"{'Game':<8}{'Obs':>6}{'lambda':>10}{'logL':>12}{'OoS-DS1':>12}{'OoS-DS2':>12}{'OoS-NL':>12}{'OoS-KM':>12}"
    print(header)
    for g in [1, 2, 3, 4]:
        name = GAME_NAMES[g]
        row = f"{name:<8}{obs[g]:>6}{lam[g]:>10.4f}{logL_qre[g]:>12.1f}"
        for j in [1, 2, 3, 4]:
            row += f"{oos_qre[g - 1, j - 1]:>12.1f}" if not np.isnan(oos_qre[g - 1, j - 1]) else f"{'':>12}"
        print(row)
    row = f"{'Pooled':<8}{1680:>6}{lam_pooled:>10.4f}{logL_qre_pooled:>12.1f}"
    for j in [1, 2, 3, 4]:
        row += f"{oos_qre[4, j - 1]:>12.1f}"
    print(row)

    print()
    print("Published")
    for g, name in GAME_NAMES.items():
        row = f"{name:<8}{obs[g]:>6}{pub_qre['lam'][g]:>10.4f}{pub_qre['logL'][g]:>12.1f}"
        for j in [1, 2, 3, 4]:
            v = pub_qre["oos"][g - 1, j - 1]
            row += f"{v:>12.1f}" if not np.isnan(v) else f"{'':>12}"
        print(row)
    row = f"{'Pooled':<8}{1680:>6}{pub_qre['lam']['P']:>10.4f}{pub_qre['logL']['P']:>12.1f}"
    for j in [1, 2, 3, 4]:
        row += f"{pub_qre['oos'][4, j - 1]:>12.1f}"
    print(row)

    # Print Table 5
    print()
    print("=" * 88)
    print("Table 5: mu-equilibrium fit & out-of-sample log-likelihoods")
    print("=" * 88)
    print("Replication")
    header = f"{'Game':<8}{'Obs':>6}{'eps':>10}{'logL':>12}{'OoS-DS1':>12}{'OoS-DS2':>12}{'OoS-NL':>12}{'OoS-KM':>12}"
    print(header)
    for g in [1, 2, 3, 4]:
        name = GAME_NAMES[g]
        row = f"{name:<8}{obs[g]:>6}{eps[g]:>10.4f}{logL_mu[g]:>12.1f}"
        for j in [1, 2, 3, 4]:
            row += f"{oos_mu[g - 1, j - 1]:>12.1f}" if not np.isnan(oos_mu[g - 1, j - 1]) else f"{'':>12}"
        print(row)
    row = f"{'Pooled':<8}{1680:>6}{eps_pooled:>10.4f}{logL_mu_pooled:>12.1f}"
    for j in [1, 2, 3, 4]:
        row += f"{oos_mu[4, j - 1]:>12.1f}"
    print(row)

    print()
    print("Published")
    for g, name in GAME_NAMES.items():
        row = f"{name:<8}{obs[g]:>6}{pub_mu['eps'][g]:>10.4f}{pub_mu['logL'][g]:>12.1f}"
        for j in [1, 2, 3, 4]:
            v = pub_mu["oos"][g - 1, j - 1]
            row += f"{v:>12.1f}" if not np.isnan(v) else f"{'':>12}"
        print(row)
    row = f"{'Pooled':<8}{1680:>6}{pub_mu['eps']['P']:>10.4f}{pub_mu['logL']['P']:>12.1f}"
    for j in [1, 2, 3, 4]:
        row += f"{pub_mu['oos'][4, j - 1]:>12.1f}"
    print(row)

    return {
        "lam": lam,
        "logL_qre": logL_qre,
        "eps": eps,
        "logL_mu": logL_mu,
        "oos_qre": oos_qre,
        "oos_mu": oos_mu,
        "lam_pooled": lam_pooled,
        "logL_qre_pooled": logL_qre_pooled,
        "eps_pooled": eps_pooled,
        "logL_mu_pooled": logL_mu_pooled,
    }


if __name__ == "__main__":
    table_4_and_5()
