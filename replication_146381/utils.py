"""
Python port of the synthdid R package (Arkhangelsky et al. 2021).

Implements Algorithm 1 from the paper for the block-treatment case:
  1. Compute regularization parameter zeta from pre-period first differences.
  2. Compute unit weights omega via regularized synthetic control on
     pre-treatment outcomes (Frank-Wolfe with intercept).
  3. Compute time weights lambda analogously (near-zero regularization).
  4. tau_hat = [-omega, 1/N1]' (Y - X.beta) [-lambda, 1/T1]

Ports R/solver.R, R/synthdid.R, R/vcov.R (placebo SE).
Matches the R package default sparsify_function: after a first Frank-Wolfe
pass, zero out weights below max/4, renormalize, then re-fit.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DATA_CSV = REPO / "146381-V1" / "synthdid-sdid-paper" / "data" / "california_prop99.csv"


# ---------------------------------------------------------------------------
# Panel matrix construction
# ---------------------------------------------------------------------------

def panel_matrices(df: pd.DataFrame, unit="State", time="Year",
                   outcome="PacksPerCapita", treatment="treated"):
    """Convert a balanced long panel to Y, W matrices with control units first
    and treated units last. Returns dict with Y, N0, T0, W, units, times."""
    df = df[[unit, time, outcome, treatment]].copy()
    if df.isna().any().any():
        raise ValueError("Missing values in panel")
    df = df.sort_values([unit, time])
    units = sorted(df[unit].unique())
    times = sorted(df[time].unique())
    N, T = len(units), len(times)
    # Reshape
    Y = df.pivot(index=unit, columns=time, values=outcome).loc[units, times].values
    W = df.pivot(index=unit, columns=time, values=treatment).loc[units, times].values
    # identify treated units (treated at any time)
    treated_unit = W.any(axis=1)
    # first period anyone is treated
    any_treated_t = W.any(axis=0)
    if not any_treated_t.any():
        raise ValueError("No treated observations")
    T0 = int(np.argmax(any_treated_t))  # index of first treated period
    N0 = int((~treated_unit).sum())
    # Validate block structure
    assert (W[~treated_unit] == 0).all()
    assert (W[:, :T0] == 0).all()
    assert (W[treated_unit][:, T0:] == 1).all(), "Treatment must be block"
    # Reorder: controls first (by name), treated last (by name)
    ctrl_order = [i for i in range(N) if not treated_unit[i]]
    trt_order = [i for i in range(N) if treated_unit[i]]
    order = ctrl_order + trt_order
    Y = Y[order]
    W = W[order]
    units = [units[i] for i in order]
    return dict(Y=Y, W=W, N0=N0, T0=T0, units=units, times=times)


# ---------------------------------------------------------------------------
# Frank-Wolfe solver (port of R/solver.R)
# ---------------------------------------------------------------------------

def _fw_step(A, x, b, eta):
    """Frank-Wolfe step minimizing ||Ax - b||^2 + eta ||x||^2 over the simplex."""
    Ax = A @ x
    half_grad = (Ax - b) @ A + eta * x
    i = int(np.argmin(half_grad))
    d_x = -x.copy()
    d_x[i] += 1.0
    if not np.any(d_x):
        return x
    d_err = A[:, i] - Ax
    denom = float(d_err @ d_err) + eta * float(d_x @ d_x)
    if denom == 0:
        return x
    step = -float(half_grad @ d_x) / denom
    step = min(1.0, max(0.0, step))
    return x + step * d_x


def sc_weight_fw(Y, zeta, intercept=True, lam=None,
                 min_decrease=1e-3, max_iter=1000):
    """Solve for weights matching Y[:, :T0] @ lam ≈ Y[:, T0] on the simplex.

    Port of R sc.weight.fw. Here the local `T0` inside the function is
    ncol(Y)-1 and `N0` is nrow(Y) (just follow the R names so the port is
    easy to audit)."""
    Y = np.asarray(Y, dtype=float).copy()
    N0, ncol = Y.shape
    T0 = ncol - 1
    if lam is None:
        lam = np.full(T0, 1.0 / T0)
    else:
        lam = np.asarray(lam, dtype=float).copy()
    if intercept:
        Y = Y - Y.mean(axis=0, keepdims=True)
    A = Y[:, :T0]
    b = Y[:, T0]
    eta = N0 * float(np.real(zeta) ** 2)
    vals = []
    t = 0
    while t < max_iter:
        lam_new = _fw_step(A, lam, b, eta)
        lam = lam_new
        err = Y @ np.concatenate([lam, [-1.0]])
        v = (zeta ** 2) * float(lam @ lam) + float(err @ err) / N0
        vals.append(v)
        t += 1
        if t >= 2 and vals[-2] - vals[-1] <= min_decrease ** 2:
            break
    return dict(lam=lam, vals=np.array(vals))


def collapsed_form(Y, N0, T0):
    """Collapse Y to an (N0+1) x (T0+1) matrix by averaging the treated block
    and the post-period block — matches R collapsed.form."""
    Y = np.asarray(Y, dtype=float)
    top_left = Y[:N0, :T0]
    top_right = Y[:N0, T0:].mean(axis=1, keepdims=True)
    bot_left = Y[N0:, :T0].mean(axis=0, keepdims=True)
    bot_right = np.array([[Y[N0:, T0:].mean()]])
    top = np.hstack([top_left, top_right])
    bot = np.hstack([bot_left, bot_right])
    return np.vstack([top, bot])


def _sparsify(v):
    v = v.copy()
    thresh = v.max() / 4.0
    v[v <= thresh] = 0.0
    s = v.sum()
    if s > 0:
        v = v / s
    return v


# ---------------------------------------------------------------------------
# SDID estimator (port of R/synthdid.R synthdid_estimate)
# ---------------------------------------------------------------------------

def synthdid_estimate(Y, N0, T0, noise_level=None,
                      eta_omega=None, eta_lambda=1e-6,
                      omega_intercept=True, lambda_intercept=True,
                      weights=None, update_omega=None, update_lambda=None,
                      min_decrease=None, max_iter=10000,
                      max_iter_pre_sparsify=100,
                      sparsify=True):
    """Returns dict with estimate, omega, lambda, and setup."""
    Y = np.asarray(Y, dtype=float)
    N, T = Y.shape
    N1 = N - N0
    T1 = T - T0
    if noise_level is None:
        diffs = np.diff(Y[:N0, :T0], axis=1)
        noise_level = float(np.std(diffs.flatten(), ddof=1))
    if eta_omega is None:
        eta_omega = (N1 * T1) ** 0.25
    zeta_omega = eta_omega * noise_level
    zeta_lambda = eta_lambda * noise_level
    if min_decrease is None:
        min_decrease = 1e-5 * noise_level

    if weights is None:
        weights = {"omega": None, "lam": None}
    if update_omega is None:
        update_omega = weights.get("omega") is None
    if update_lambda is None:
        update_lambda = weights.get("lam") is None

    Yc = collapsed_form(Y, N0, T0)

    if update_lambda:
        # Fit lambda: match control-only pre-periods to control post average
        res = sc_weight_fw(Yc[:N0, :], zeta=zeta_lambda,
                           intercept=lambda_intercept,
                           lam=weights.get("lam"),
                           min_decrease=min_decrease,
                           max_iter=max_iter_pre_sparsify if sparsify else max_iter)
        lam = res["lam"]
        if sparsify:
            res = sc_weight_fw(Yc[:N0, :], zeta=zeta_lambda,
                               intercept=lambda_intercept,
                               lam=_sparsify(lam),
                               min_decrease=min_decrease,
                               max_iter=max_iter)
            lam = res["lam"]
        weights["lam"] = lam

    if update_omega:
        # Fit omega: match treated pre-periods to controls pre-periods
        Yc_T = Yc[:, :T0].T  # shape: T0 x (N0+1)
        res = sc_weight_fw(Yc_T, zeta=zeta_omega,
                           intercept=omega_intercept,
                           lam=weights.get("omega"),
                           min_decrease=min_decrease,
                           max_iter=max_iter_pre_sparsify if sparsify else max_iter)
        omega = res["lam"]
        if sparsify:
            res = sc_weight_fw(Yc_T, zeta=zeta_omega,
                               intercept=omega_intercept,
                               lam=_sparsify(omega),
                               min_decrease=min_decrease,
                               max_iter=max_iter)
            omega = res["lam"]
        weights["omega"] = omega

    omega = weights["omega"]
    lam = weights["lam"]
    # tau = [-omega, 1/N1]' Y [-lambda, 1/T1]
    left = np.concatenate([-omega, np.full(N1, 1.0 / N1)])
    right = np.concatenate([-lam, np.full(T1, 1.0 / T1)])
    tau = float(left @ Y @ right)

    return {
        "tau": tau,
        "omega": omega,
        "lam": lam,
        "Y": Y, "N0": N0, "T0": T0,
        "zeta_omega": zeta_omega, "zeta_lambda": zeta_lambda,
        "eta_omega": eta_omega, "eta_lambda": eta_lambda,
        "noise_level": noise_level,
        "omega_intercept": omega_intercept,
        "lambda_intercept": lambda_intercept,
        "sparsify": sparsify,
        "update_omega": update_omega, "update_lambda": update_lambda,
        "min_decrease": min_decrease, "max_iter": max_iter,
        "estimator": "sdid",
    }


def sc_estimate(Y, N0, T0, **kwargs):
    """Pure synthetic control: no time weights (uniform = 0 on pre weights,
    and lam intercept off, omega intercept off). Matches R sc_estimate which
    passes weights=list(lambda=rep(0,T0)), omega.intercept=FALSE, eta.omega=1e-6."""
    N, T = Y.shape
    T1 = T - T0
    kwargs.setdefault("eta_omega", 1e-6)
    kwargs.setdefault("omega_intercept", False)
    est = synthdid_estimate(
        Y, N0, T0,
        weights={"omega": None, "lam": np.zeros(T0)},
        update_lambda=False,
        **kwargs,
    )
    # With lam = 0, the left formula becomes: treated post average minus
    # sum(omega * Y_ctrl_post_mean). Verify: left @ Y @ right with lam=0 gives
    # -omega @ Y @ [0...0, 1/T1...] + (1/N1) * sum(Y_treated_post)/N1
    # That equals (treated post mean) - omega @ (control post means).
    est["estimator"] = "sc"
    return est


def did_estimate(Y, N0, T0, **kwargs):
    """Difference-in-differences: uniform unit and time weights."""
    est = synthdid_estimate(
        Y, N0, T0,
        weights={"omega": np.full(N0, 1.0 / N0), "lam": np.full(T0, 1.0 / T0)},
        update_omega=False, update_lambda=False,
        **kwargs,
    )
    est["estimator"] = "did"
    return est


def difp_estimate(Y, N0, T0, **kwargs):
    """De-meaned synthetic control (DIFP / Doudchenko-Imbens):
    omega with intercept, no time weights, and minimal regularization."""
    N, T = Y.shape
    T0_val = T0
    kwargs.setdefault("eta_omega", 1e-6)
    est = synthdid_estimate(
        Y, N0, T0,
        weights={"omega": None, "lam": np.full(T0_val, 1.0 / T0_val)},
        update_lambda=False,
        **kwargs,
    )
    est["estimator"] = "difp"
    return est


# ---------------------------------------------------------------------------
# Placebo standard error (Algorithm 4)
# ---------------------------------------------------------------------------

def placebo_se(est, replications=200, seed=12345):
    """Port of R placebo_se (Algorithm 4 of Arkhangelsky et al.).

    Permute the N0 control units, take the first (N0 - N1) as placebo
    controls and the last N1 as placebo "treated" (still under control), then
    re-run the same estimator. R's implementation passes the *original*
    fitted omega (restricted to the new placebo controls and renormalized)
    as a warm start, and preserves the original update.* flags — so for
    SDID/SC/DIFP, omega is re-fit on the new subproblem from that warm start
    (with sparsify re-applied); lambda is handled analogously. DID uses
    fixed uniform weights.
    """
    rng = np.random.default_rng(seed)
    Y = est["Y"]
    N0 = est["N0"]
    T0 = est["T0"]
    N, T = Y.shape
    N1 = N - N0
    if N0 <= N1:
        raise ValueError("Need N0 > N1 for placebo SE")

    omega = est["omega"]
    lam = est["lam"]
    estimator = est["estimator"]

    vals = np.empty(replications)
    for r in range(replications):
        ind = rng.permutation(N0)
        new_N0 = N0 - N1
        sub_omega_init = omega[ind[:new_N0]].copy()
        s = sub_omega_init.sum()
        if s != 0:
            sub_omega_init = sub_omega_init / s
        else:
            sub_omega_init = np.full_like(sub_omega_init, 1.0 / len(sub_omega_init))
        Y_sub = Y[ind]  # rows reordered by ind

        if estimator == "sdid":
            est2 = synthdid_estimate(
                Y_sub, new_N0, T0,
                weights={"omega": sub_omega_init, "lam": lam.copy()},
                update_omega=True, update_lambda=True,
                noise_level=est["noise_level"],
                eta_omega=est["eta_omega"], eta_lambda=est["eta_lambda"],
                omega_intercept=est["omega_intercept"],
                lambda_intercept=est["lambda_intercept"],
                min_decrease=est["min_decrease"], max_iter=est["max_iter"],
                sparsify=True,
            )
        elif estimator == "sc":
            # lam fixed at zero (original update_lambda=False),
            # omega re-fit with sub_omega_init warm start
            est2 = synthdid_estimate(
                Y_sub, new_N0, T0,
                weights={"omega": sub_omega_init, "lam": np.zeros(T0)},
                update_omega=True, update_lambda=False,
                noise_level=est["noise_level"],
                eta_omega=est["eta_omega"], eta_lambda=est["eta_lambda"],
                omega_intercept=False, lambda_intercept=est["lambda_intercept"],
                min_decrease=est["min_decrease"], max_iter=est["max_iter"],
                sparsify=True,
            )
        elif estimator == "did":
            est2 = synthdid_estimate(
                Y_sub, new_N0, T0,
                weights={"omega": np.full(new_N0, 1.0 / new_N0),
                         "lam": np.full(T0, 1.0 / T0)},
                update_omega=False, update_lambda=False,
                noise_level=est["noise_level"],
            )
        elif estimator == "difp":
            est2 = synthdid_estimate(
                Y_sub, new_N0, T0,
                weights={"omega": sub_omega_init, "lam": np.full(T0, 1.0 / T0)},
                update_omega=True, update_lambda=False,
                noise_level=est["noise_level"],
                eta_omega=est["eta_omega"], eta_lambda=est["eta_lambda"],
                omega_intercept=est["omega_intercept"],
                lambda_intercept=est["lambda_intercept"],
                min_decrease=est["min_decrease"], max_iter=est["max_iter"],
                sparsify=True,
            )
        else:
            raise ValueError(f"unknown estimator {estimator}")
        vals[r] = est2["tau"]

    se = np.sqrt((replications - 1) / replications) * vals.std(ddof=1)
    return float(se), vals


# ---------------------------------------------------------------------------
# Jackknife SE (Algorithm 3) — for robustness comparison
# ---------------------------------------------------------------------------

def jackknife_se(est):
    """Fixed-weights jackknife: drop each unit once, refit with fixed
    (renormalized) weights, compute SE per Algorithm 3."""
    Y = est["Y"]; N0 = est["N0"]; T0 = est["T0"]
    N, T = Y.shape
    N1 = N - N0
    if N0 == N - 1:
        return np.nan
    omega = est["omega"]; lam = est["lam"]
    n = N
    vals = np.empty(n)
    for i in range(n):
        keep = [j for j in range(n) if j != i]
        Y_sub = Y[keep]
        if i < N0:
            sub_omega = np.delete(omega, i)
            s = sub_omega.sum()
            sub_omega = sub_omega / s if s != 0 else np.full_like(sub_omega, 1.0 / len(sub_omega))
            new_N0 = N0 - 1
        else:
            sub_omega = omega
            new_N0 = N0
        est2 = synthdid_estimate(
            Y_sub, new_N0, T0,
            weights={"omega": sub_omega, "lam": lam},
            update_omega=False, update_lambda=False,
            noise_level=est["noise_level"],
            eta_omega=est["eta_omega"], eta_lambda=est["eta_lambda"],
            omega_intercept=est["omega_intercept"],
            lambda_intercept=est["lambda_intercept"],
        )
        vals[i] = est2["tau"]
    se = float(np.sqrt((n - 1) / n * (n - 1) * np.var(vals, ddof=1)))
    return se, vals


# ---------------------------------------------------------------------------
# Data loader for California Prop 99
# ---------------------------------------------------------------------------

def load_california():
    df = pd.read_csv(DATA_CSV, sep=";")
    return df
