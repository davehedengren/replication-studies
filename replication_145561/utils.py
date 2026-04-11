"""Shared utilities for M Equilibrium replication (Goeree & Louis, 2021)."""

import os
import numpy as np
import scipy.io as sio

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "145561-V1", "Data")
OUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUT_DIR, exist_ok=True)


def load_mat(name):
    """Load a .mat file and return a dict of numpy arrays (squeezed)."""
    d = sio.loadmat(os.path.join(DATA_DIR, f"{name}.mat"))
    return {k: np.squeeze(v) for k, v in d.items() if not k.startswith("__")}


def load_2x2_amp():
    """Load and relabel 2x2 AMP data so that equivalent game pairs are aligned.

    In the experiment, games 1&6, 2&7, 3&8, 4&9, 5&10 are strategically equivalent
    versions. The relabelings below mirror main_replication.m lines 36-55.
    """
    d = load_mat("2x2_AMP")
    choice = d["choice"].astype(float).copy()
    belief = d["belief"].astype(float).copy()
    game = d["game"]
    row = d["row"]
    Row = d["Row"]  # 2x2x10
    Col = d["Col"]

    for i in range(len(game)):
        g = int(game[i])
        if g in (2, 6, 10):
            if row[i] == 1:
                choice[i] = 1 - choice[i]
            else:
                belief[i] = 100 - belief[i]
        if g in (3, 7):
            choice[i] = 1 - choice[i]
            belief[i] = 100 - belief[i]
        if g in (4, 8):
            if row[i] == 0:
                choice[i] = 1 - choice[i]
            else:
                belief[i] = 100 - belief[i]

    return {
        "Subject": d["Subject"],
        "session": d["session"],
        "Period": d["Period"],
        "round1": d["round1"],
        "Group": d["Group"],
        "game": game,
        "Row": Row,
        "Col": Col,
        "choice": choice,
        "belief": belief,
        "row": row,
    }


def best_response_2x2(pay_2x2, belief_prob_A):
    """Given a 2x2 payoff matrix (row player) and belief that opponent plays A,
    return the best response (1 = first action A, 0 = second action B)."""
    p = belief_prob_A
    payoff = pay_2x2 @ np.array([p, 1.0 - p])
    return int(np.argmax(payoff) == 0)


def load_3x3(game_name):
    """Load a 3x3 game file. Returns (choices_nx3, beliefs_nx3 as fractions, payoff_3x3)."""
    d = load_mat(game_name)
    choices = np.column_stack([d["choiceA"], d["choiceB"], d["choiceC"]]).astype(float)
    beliefs = np.column_stack([d["beliefA"], d["beliefB"], d["beliefC"]]).astype(float)
    payoff = d["payoff"].astype(float)
    return choices, beliefs, payoff


def hotelling_two_sample(x, y):
    """Two-sample Hotelling T^2 test with diagonal covariance (matches
    hotellingttest.m: the paper uses only the diagonal of S, not the full
    covariance matrix). Returns p-value from F distribution."""
    from scipy.stats import f as fdist

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = x.shape[0]  # assumes equal sample sizes (matches MATLAB code)
    p = x.shape[1]
    m_x = x.mean(axis=0)
    m_y = y.mean(axis=0)
    s2_x = x.var(axis=0, ddof=1)
    s2_y = y.var(axis=0, ddof=1)
    Sigma = np.diag((n - 1) * (s2_x + s2_y) / (2 * n - 2))
    diff = m_x - m_y
    T2 = (n * n) / (2 * n) * diff @ np.linalg.inv(Sigma) @ diff
    Fstat = (2 * n - p - 1) / ((2 * n - 2) * p) * T2
    return fdist.sf(Fstat, p, 2 * n - 1 - p)


def hotelling_one_sample(x, mu):
    """One-sample Hotelling T^2 test with diagonal covariance (matches
    hotelling1sample.m)."""
    from scipy.stats import f as fdist

    x = np.asarray(x, dtype=float)
    mu = np.asarray(mu, dtype=float)
    n = x.shape[0]
    p = x.shape[1]
    m_x = x.mean(axis=0)
    s2_x = x.var(axis=0, ddof=1)
    Sigma = np.diag(s2_x) / n
    diff = m_x - mu
    T2 = (n * n) / (2 * n) * diff @ np.linalg.inv(Sigma) @ diff
    Fstat = (n - p) / ((n - 1) * p) * T2
    return fdist.sf(Fstat, p, n - p)


# -----------------------------------------------------------------------------
# Logit-QRE for symmetric 3x3 games
# -----------------------------------------------------------------------------


def qre_residuals(p, game, lam):
    """Fixed-point residual: p - logit(lam * game @ p)."""
    p = np.clip(p, 0, 1)
    ex_pay = game @ p
    exp_u = np.exp(lam * (ex_pay - ex_pay.max()))  # numerical stability
    logR = exp_u / exp_u.sum()
    return p - logR


def calculate_qre_3x3(lam, game):
    """Solve for symmetric logit-QRE in a 3x3 game using least-squares
    with bounds, matching calculate_qre3x3.m (lsqnonlin with 0,1 bounds)."""
    from scipy.optimize import least_squares

    if game[0, 0] == 120:  # KM
        x0 = np.array([0.5, 0.3, 0.2])
    else:
        x0 = np.array([0.01, 0.02, 0.97])
    res = least_squares(
        qre_residuals, x0, args=(game, lam), bounds=([0, 0, 0], [1, 1, 1])
    )
    return res.x


def loglik_3x3_qre(lam, game, choices):
    """Log-likelihood of choice data under logit-QRE with parameter lambda."""
    ns = choices.sum(axis=0)
    qre_p = calculate_qre_3x3(lam, game)
    qre_p = np.clip(qre_p, 1e-12, 1.0)
    return float(ns[0] * np.log(qre_p[0]) + ns[1] * np.log(qre_p[1]) + ns[2] * np.log(qre_p[2]))


# -----------------------------------------------------------------------------
# Mu-equilibrium closed form for the four 3x3 games (Table 3)
# -----------------------------------------------------------------------------


def calculate_mu_3x3(epsilon, game_idx):
    """Return mu-equilibrium profiles for a given game (1=DS1, 2=DS2, 3=NL, 4=KM).

    Matches calculate_mu3x3_closed.m exactly. Returns an (n_eq, 3) array of
    probability profiles over actions (A, B, C)."""
    e = float(epsilon)
    num = 1 + e + e ** 2

    if game_idx in (1, 2):
        if 0 <= e < 0.125:
            sigma = np.array([[e, e ** 2, 1]]) / num
        elif 0.125 <= e < 0.146:
            sigma = np.array([[1 / 8, -1 / 8 + e + e ** 2, 1]]) / num
        elif 0.146 <= e < 1 / 3:
            sigma = (
                np.array(
                    [
                        [1 / 8, -1 / 8 + e + e ** 2, 1],
                        [num / 8, 1, (-1 + 7 * e + 7 * e ** 2) / 8],
                        [e, 1, e ** 2],
                    ]
                )
                / num
            )
        elif 1 / 3 <= e < 0.354:
            sigma = (
                np.array(
                    [
                        [1 / 8, -1 / 8 + e + e ** 2, 1],
                        [num / 8, 1, (-1 + 7 * e + 7 * e ** 2) / 8],
                        [e, 1, e ** 2],
                        [e ** 2, 2 + 2 * e - 15 * e ** 2, -1 - e + 15 * e ** 2],
                        [e ** 2, 1, e],
                    ]
                )
                / num
            )
        elif 0.354 <= e < 0.4:
            sigma = (
                np.array(
                    [
                        [e ** 2, e, 1],
                        [num / 8, 1, (-1 + 7 * e + 7 * e ** 2) / 8],
                        [e, 1, e ** 2],
                        [e ** 2, 2 + 2 * e - 15 * e ** 2, -1 - e + 15 * e ** 2],
                        [e ** 2, 1, e],
                    ]
                )
                / num
            )
        elif 0.4 <= e < 0.456:
            sigma = (
                np.array(
                    [
                        [num / 8, 1, (-1 + 7 * e + 7 * e ** 2) / 8],
                        [e, 1, e ** 2],
                        [e ** 2, 1, e],
                    ]
                )
                / num
            )
        else:
            sigma = np.array([[e, 1, e ** 2]]) / num

    elif game_idx == 3:
        if 0 <= e < 0.024:
            sigma = np.array([[e, e ** 2, 1]]) / num
        elif 0.024 <= e < 0.053:
            sigma = np.array([[(1 + 2 * e + 2 * e ** 2) / 45, (-1 + 43 * e + 43 * e ** 2) / 45, 1]]) / num
        elif 0.053 <= e < 0.178:
            sigma = (
                np.array(
                    [
                        [(1 + 2 * e + 2 * e ** 2) / 45, (-1 + 43 * e + 43 * e ** 2) / 45, 1],
                        [e, 1, e ** 2],
                        [1 / 40 + e / 2 + (e ** 2) / 2, 1, -1 / 40 + e / 2 + (e ** 2) / 2],
                    ]
                )
                / num
            )
        elif 0.178 <= e < 0.252:
            sigma = (
                np.array(
                    [
                        [e ** 2, e, 1],
                        [e, 1, e ** 2],
                        [1 / 40 + e / 2 + (e ** 2) / 2, 1, -1 / 40 + e / 2 + (e ** 2) / 2],
                    ]
                )
                / num
            )
        elif 0.252 <= e < 0.283:
            sigma = (
                np.array(
                    [
                        [e ** 2, e, 1],
                        [e, 1, e ** 2],
                        [1 / 40 + e / 2 + (e ** 2) / 2, 1, -1 / 40 + e / 2 + (e ** 2) / 2],
                        [e ** 2, 1, e],
                        [e ** 2, (10 + 10 * e - 150 * e ** 2) / 3, (-7 - 7 * e + 150 * e ** 2) / 3],
                    ]
                )
                / num
            )
        else:
            sigma = (
                np.array(
                    [
                        [e, 1, e ** 2],
                        [1 / 40 + e / 2 + (e ** 2) / 2, 1, -1 / 40 + e / 2 + (e ** 2) / 2],
                        [e ** 2, 1, e],
                    ]
                )
                / num
            )
    elif game_idx == 4:
        sigma = np.array([[1, e ** 2, e]]) / num
    else:
        raise ValueError(f"Unknown game_idx {game_idx}")

    return sigma


def loglik_3x3_mu_indiv(epsilon, game_idx, choices, ids):
    """Log-likelihood under mu-equilibrium: each individual's choices are
    assigned to the mu-profile that best explains them (max across profiles).
    Mirrors likelihood_3x3_mu_indiv.m."""
    ids = np.asarray(ids).ravel()
    unique_ids = np.unique(ids)
    if game_idx == 4 and unique_ids[0] == 0:
        unique_ids = unique_ids[1:]
    mu_p = calculate_mu_3x3(epsilon, game_idx)
    mu_p = np.clip(mu_p, 1e-12, 1.0)
    log_mu = np.log(mu_p)  # n_eq x 3

    total = 0.0
    for uid in unique_ids:
        mask = ids == uid
        subj_choices = choices[mask]  # n_obs x 3
        ns = subj_choices.sum(axis=0)  # 3-vector
        lls = log_mu @ ns  # n_eq
        total += float(lls.max())
    return total


GAMES_3x3 = {
    1: np.array([[80, 30, 20], [160, 30, 10], [10, 40, 30]], dtype=float),  # DS1
    2: np.array([[75, 5, 190], [155, 5, 180], [5, 15, 200]], dtype=float),  # DS2
    3: np.array([[70, 60, 10], [500, 40, 0], [50, 61, 30]], dtype=float),  # NL
    4: np.array([[120, 90, 60], [60, 90, 60], [120, 90, 30]], dtype=float),  # KM
}

GAME_NAMES = {1: "DS1", 2: "DS2", 3: "NL", 4: "KM"}
