"""Shared params, solvers, and simulators for Caballero-Simsek (openICPSR-177541).

A Note on Temporary Supply Shocks with Aggregate Demand Inertia.
Port of the Matlab code in 177541-V1/Matlab code/.
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Params:
    rho: float = 0.02
    eta: float = 0.8
    eta_low: float = 0.5
    yHStar: float = 1.0
    yLStar: float = 0.95
    phi: float = 1.0
    lam: float = 0.5
    # NKPC
    kappa: float = 0.5
    phi_y: float = 1.0
    phi_pi: float = 1.0
    psi: float = 1.0
    # Inertia
    b: float = 0.9
    # Grids + simulation
    yArr: np.ndarray = field(default_factory=lambda: np.arange(0.0, 1.0 + 1e-12, 0.001))
    piArr: Optional[np.ndarray] = None
    sInit: int = 1   # 1 = low, 2 = high
    T: int = 7
    tTrans: int = 4
    yLastInit: Optional[float] = None
    piLastInit: float = 0.0

    @property
    def beta(self) -> float:
        return float(np.exp(-self.rho))


def default_params() -> Params:
    p = Params()
    # For the inertial model we also need a pi grid with step 0.005
    piMin = -p.kappa / (1 - p.b) * (p.yHStar - p.yLStar)
    piMax = p.kappa / (1 - p.b) * (p.yHStar - p.yLStar)
    p.piArr = np.arange(piMin, piMax + 1e-12, 0.005)
    return p


# ---------------------------------------------------------------------------
# Baseline model (no inflation)
# ---------------------------------------------------------------------------

def solve_baseline(p: Params, eta: Optional[float] = None, type_: int = 1) -> dict:
    """Port of solveEquilibriumBaseline.m."""
    if eta is None:
        eta = p.eta
    beta = p.beta
    yArr = p.yArr
    yHStar, yLStar = p.yHStar, p.yLStar
    phi, lam, rho = p.phi, p.lam, p.rho

    eq: dict = {}

    if type_ == 1:
        # High supply state: Taylor rule -> endogenous convergence rate gammaH
        poly = [1.0, -1.0 / (1 - eta) - phi, eta / (1 - eta)]
        roots = np.roots(poly)
        gammaH = float(np.min(roots.real))
        thetaH = gammaH ** 2 / (1 - beta * gammaH ** 2)

        YHArr = yHStar + gammaH * (yArr - yHStar)
        VHArr = -thetaH * (yArr - yHStar) ** 2
        iHArr = rho + phi * (YHArr - yHStar)

        # Low supply state
        yL = (1.0 / (1 + beta * lam * thetaH) * yLStar
              + beta * lam * thetaH / (1 + beta * lam * thetaH) * yHStar)
        yLBar = yL - (1 - eta) / eta * (lam * (1 - gammaH) + phi) * (yHStar - yL)

        # Compute cutoffs (appendix A.1.2)
        step = (1 - eta) / eta * phi * (yHStar - yL)
        KL = int(np.ceil(yLBar / step))
        kLArr = np.arange(0, KL + 1)
        yLBarArr = np.zeros(len(kLArr))
        yLBarArr[0] = yLBar
        for i in range(1, len(kLArr)):
            yLBarCur = yLBarArr[i - 1]
            yLBarNext = yL if i == 1 else yLBarArr[i - 2]
            yLBarArr[i] = (1.0 / eta) * (
                yLBarCur * (1 + (1 - eta) * (phi - lam * gammaH))
                - (1 - eta) * (phi * yHStar + lam * (1 - gammaH) * yHStar
                               + (1 - lam) * yLBarNext)
            )
        neg = np.where(yLBarArr < 0)[0]
        if len(neg):
            cut = neg[0] + 1   # keep first negative too, matches matlab find(...,1,'first') then trim
            kLArr = kLArr[:cut]
            yLBarArr = yLBarArr[:cut]

        YLArr = np.zeros_like(yArr)
        iLArr = np.zeros_like(yArr)
        for i in range(len(yArr) - 1, -1, -1):
            y = yArr[i]
            if y >= yLBar - 0.0001:
                YLArr[i] = yL
                iLArr[i] = rho + lam * (1 - gammaH) * (yHStar - yL) - eta / (1 - eta) * (yL - y)
            else:
                # Find j > i such that tempArr(j) first turns positive
                # tempArr(j) = yArr(j) - 1/(1+(1-eta)(phi-lam gammaH)) * ( eta*y + (1-eta)*( phi*yHStar + lam*(1-gammaH)*yHStar + (1-lam)*YLArr(j) ) )
                denom = 1 + (1 - eta) * (phi - lam * gammaH)
                inner_const = eta * y + (1 - eta) * (phi * yHStar + lam * (1 - gammaH) * yHStar)
                # tempArr = yArr - (inner_const + (1-eta)*(1-lam)*YLArr) / denom
                tempArr = np.zeros_like(yArr)
                jslice = np.arange(i + 1, len(yArr))
                tempArr[jslice] = yArr[jslice] - (inner_const + (1 - eta) * (1 - lam) * YLArr[jslice]) / denom
                pos = np.where(tempArr > 0)[0]
                ind = pos[0]
                YLArr[i] = yArr[ind]
                iLArr[i] = rho + phi * (yArr[ind] - yHStar)

        eq.update(type='baseline', gammaH=gammaH, thetaH=thetaH, yArr=yArr,
                  YHArr=YHArr, VHArr=VHArr, iHArr=iHArr,
                  yL=yL, yLBar=yLBar,
                  YLArr=YLArr, iLArr=iLArr,
                  kLArr=kLArr, yLBarArr=yLBarArr)
        return eq

    if type_ == 2:
        eq['type'] = 'first best'
        eq['yArr'] = yArr
        eq['YHArr'] = yHStar + 0 * yArr
        eq['iHArr'] = rho + eta / (1 - eta) * (yArr - yHStar)
        eq['YLArr'] = yLStar + 0 * yArr
        eq['iLArr'] = rho + eta / (1 - eta) * (yArr - yLStar) + lam * (yHStar - yLStar)
        return eq

    if type_ == 3:
        eq['type'] = 'myopic'
        eqTemp = solve_baseline(p, eta=eta, type_=1)
        eq['gammaH'] = eqTemp['gammaH']
        eq['thetaH'] = eqTemp['thetaH']
        eq['yArr'] = yArr
        eq['YHArr'] = eqTemp['YHArr']
        eq['iHArr'] = eqTemp['iHArr']
        yLMyopic = yLStar
        YHMyopic = yHStar + eq['gammaH'] * (yLMyopic - yHStar)
        eq['YLArr'] = 0 * yArr + yLStar
        eq['iLArr'] = rho + eta / (1 - eta) * (yArr - yLMyopic) + lam * (YHMyopic - yLMyopic)
        return eq

    raise ValueError(f"Unknown type_: {type_}")


def _interp(grid: np.ndarray, values: np.ndarray, x: float) -> float:
    return float(np.interp(x, grid, values))


def simulate_baseline(p: Params, eq: dict) -> dict:
    T = p.T
    timeArr = np.arange(-1, T + 1)
    n = len(timeArr)
    sTimeArr = np.zeros(n, dtype=int)
    yTimeArr = np.zeros(n)
    iTimeArr = np.zeros(n)

    sTimeArr[0] = p.sInit
    sTimeArr[1] = p.sInit
    for i in range(2, n):
        if sTimeArr[i - 1] == 1 and timeArr[i] < p.tTrans:
            sTimeArr[i] = 1
        else:
            sTimeArr[i] = 2

    yTimeArr[0] = p.yLastInit
    iTimeArr[0] = np.nan
    yArr = eq['yArr'] if 'yArr' in eq else p.yArr

    for i in range(1, n):
        yLast = p.yLastInit if i == 1 else yTimeArr[i - 1]
        if sTimeArr[i] == 1:
            yTimeArr[i] = _interp(yArr, eq['YLArr'], yLast)
            iTimeArr[i] = _interp(yArr, eq['iLArr'], yLast)
        else:
            yTimeArr[i] = _interp(yArr, eq['YHArr'], yLast)
            iTimeArr[i] = _interp(yArr, eq['iHArr'], yLast)

    return {'timeArr': timeArr, 'sTimeArr': sTimeArr,
            'yTimeArr': yTimeArr, 'iTimeArr': iTimeArr}


# ---------------------------------------------------------------------------
# NKPC model
# ---------------------------------------------------------------------------

def solve_nkpc(p: Params, type_: int = 1) -> dict:
    eta = p.eta
    beta = p.beta
    yArr = p.yArr
    yHStar, yLStar = p.yHStar, p.yLStar
    lam, rho = p.lam, p.rho
    kappa, phi_y, phi_pi, psi = p.kappa, p.phi_y, p.phi_pi, p.psi

    eq: dict = {}
    if type_ == 1:
        poly = np.zeros(4)
        poly[0] = 1
        poly[1] = -(1 / (1 - eta) + phi_y + (1 + kappa) / beta)
        poly[2] = (1 / (1 - eta) + phi_y) / beta + phi_pi * kappa / beta + eta / (1 - eta)
        poly[3] = -1 / beta * eta / (1 - eta)
        roots = np.roots(poly)
        stable = [r.real for r in roots if 0 < r.real < 1 and abs(r.imag) < 1e-9]
        assert len(stable) == 1, f"expected 1 stable root, got {roots}"
        gammaH = stable[0]
        thetaH = gammaH ** 2 / (1 - beta * gammaH ** 2) * (1 + psi * (kappa / (1 - beta * gammaH)) ** 2)

        YHArr = yHStar + gammaH * (yArr - yHStar)
        VHArr = -thetaH * (yArr - yHStar) ** 2
        piHBold = kappa * gammaH / (1 - beta * gammaH)
        PiHArr = piHBold * (yArr - yHStar)
        iHArr = rho + phi_y * (YHArr - yHStar) + phi_pi * PiHArr
        PiHNextArr = gammaH * PiHArr
        rHArr = iHArr - PiHNextArr

        # Low state closed form
        omegaL = 1 + psi * (kappa + beta * lam * piHBold) * kappa / (1 - beta * (1 - lam))
        omegaH = beta * lam * (thetaH + psi * (kappa + beta * lam * piHBold) * piHBold / (1 - beta * (1 - lam)))
        yL = omegaL / (omegaL + omegaH) * yLStar + omegaH / (omegaL + omegaH) * yHStar
        piL = (kappa * (yL - yLStar) + beta * lam * piHBold * (yL - yHStar)) / (1 - beta * (1 - lam))

        YLArr = yL + 0 * yArr
        PiLArr = piL + 0 * yArr
        rLArr = rho + lam * (1 - gammaH) * (yHStar - yL) - eta / (1 - eta) * (yL - yArr)
        iLArr = rLArr + lam * PiHArr + (1 - lam) * piL

        eq.update(type='baseline', gammaH=gammaH, thetaH=thetaH, piHBold=piHBold,
                  yArr=yArr, YHArr=YHArr, VHArr=VHArr, PiHArr=PiHArr,
                  iHArr=iHArr, rHArr=rHArr,
                  yL=yL, piL=piL,
                  YLArr=YLArr, PiLArr=PiLArr, iLArr=iLArr, rLArr=rLArr)
        return eq

    if type_ == 2:
        eq['type'] = 'first best'
        eq['yArr'] = yArr
        eq['YHArr'] = yHStar + 0 * yArr
        eq['PiHArr'] = 0 * yArr
        eq['iHArr'] = rho + eta / (1 - eta) * (yArr - yHStar)
        eq['rHArr'] = eq['iHArr'].copy()
        eq['YLArr'] = yLStar + 0 * yArr
        eq['PiLArr'] = 0 * yArr
        eq['iLArr'] = rho + eta / (1 - eta) * (yArr - yLStar) + lam * (yHStar - yLStar)
        eq['rLArr'] = eq['iLArr'].copy()
        return eq

    raise ValueError


def simulate_nkpc(p: Params, eq: dict) -> dict:
    T = p.T
    timeArr = np.arange(-1, T + 1)
    n = len(timeArr)
    sTimeArr = np.zeros(n, dtype=int)
    y = np.zeros(n); pi = np.zeros(n); i_ = np.zeros(n); r = np.zeros(n)

    sTimeArr[0] = p.sInit
    sTimeArr[1] = p.sInit
    for k in range(2, n):
        sTimeArr[k] = 1 if (sTimeArr[k - 1] == 1 and timeArr[k] < p.tTrans) else 2

    y[0] = p.yLastInit
    pi[0] = np.nan; i_[0] = np.nan; r[0] = np.nan

    yArr = eq['yArr'] if 'yArr' in eq else p.yArr
    for k in range(1, n):
        yLast = y[k - 1]
        if sTimeArr[k] == 1:
            y[k] = _interp(yArr, eq['YLArr'], yLast)
            pi[k] = _interp(yArr, eq['PiLArr'], yLast)
            i_[k] = _interp(yArr, eq['iLArr'], yLast)
            r[k] = _interp(yArr, eq['rLArr'], yLast)
        else:
            y[k] = _interp(yArr, eq['YHArr'], yLast)
            pi[k] = _interp(yArr, eq['PiHArr'], yLast)
            i_[k] = _interp(yArr, eq['iHArr'], yLast)
            r[k] = _interp(yArr, eq['rHArr'], yLast)

    return {'timeArr': timeArr, 'sTimeArr': sTimeArr,
            'yTimeArr': y, 'piTimeArr': pi, 'iTimeArr': i_, 'rTimeArr': r}


# ---------------------------------------------------------------------------
# Inertial inflation model
# ---------------------------------------------------------------------------

def solve_inertia(p: Params, type_: int = 1) -> dict:
    eta = p.eta
    beta = p.beta
    yArr = p.yArr
    piArr = p.piArr
    yHStar, yLStar = p.yHStar, p.yLStar
    lam, rho = p.lam, p.rho
    kappa, phi_y, psi, b = p.kappa, p.phi_y, p.psi, p.b
    phi_pi = b  # per loadVariablesInflationInertia.m

    eq: dict = {}
    if type_ == 1:
        # Convergence rate in state H
        poly = np.zeros(3)
        poly[0] = 1 + kappa
        poly[1] = -(1 / (1 - eta) + phi_y)
        poly[2] = eta / (1 - eta)
        roots = np.roots(poly)
        stable = [r.real for r in roots if 0 < r.real < 1 and abs(r.imag) < 1e-9]
        assert len(stable) == 1
        gammaH = stable[0]
        PsiH = b ** 2 / (1 - beta * b ** 2) * psi
        IH = gammaH * b / (1 - beta * gammaH * b) * (psi + beta * PsiH) * kappa
        thetaH = gammaH ** 2 / (1 - beta * gammaH ** 2) * (1 + (psi + beta * PsiH) * kappa ** 2 + 2 * beta * IH * kappa)

        ny, npi = len(yArr), len(piArr)
        YHArr = np.zeros((ny, npi))
        VHArr = np.zeros((ny, npi))
        PiHArr = np.zeros((ny, npi))
        for j in range(npi):
            YHArr[:, j] = yHStar + gammaH * (yArr - yHStar)
            piPast = piArr[j]
            PiHArr[:, j] = kappa * gammaH * (yArr - yHStar) + b * piPast
            VHArr[:, j] = (-thetaH / 2 * (yArr - yHStar) ** 2
                           - PsiH / 2 * piPast
                           - IH * (yArr - yHStar) * piPast)

        iHArr = rho + phi_y * (YHArr - yHStar) + phi_pi * PiHArr
        PiHNextArr = kappa * gammaH * (YHArr - yHStar) + b * PiHArr
        rHArr = iHArr - PiHNextArr

        # Low state steady state
        A = 1 + beta * lam * (thetaH + kappa * IH)
        B = kappa * psi + beta * lam * (IH + kappa * PsiH)
        C = beta * (1 - lam) * b * (1 + beta * lam * thetaH)
        D = beta * (1 - lam) * b * beta * lam * IH
        E = beta * lam * (thetaH + kappa * IH - beta * (1 - lam) * b * thetaH)

        poly2 = np.zeros(3)
        poly2[0] = 1
        poly2[1] = -(A + B * kappa + b * C) / (C + D * kappa)
        poly2[2] = A * b / (C + D * kappa)
        r2 = np.roots(poly2)
        stable2 = [r.real for r in r2 if 0 < r.real < 1 and abs(r.imag) < 1e-9]
        assert len(stable2) == 1
        gammaL = stable2[0]
        assert gammaL < b

        omegaH_ = E / (A - C + (B - D) * kappa / (1 - b))
        yLBar = yLStar + omegaH_ * (yHStar - yLStar)
        piLBar = kappa / (1 - b) * (yLBar - yLStar)

        PiLArr = np.zeros((ny, npi))
        YLArr = np.zeros((ny, npi))
        for i in range(ny):
            PiLArr[i, :] = piLBar + gammaL * (piArr - piLBar)
            YLArr[i, :] = yLBar - (b - gammaL) / kappa * (piArr - piLBar)

        rLArr = np.zeros((ny, npi))
        iLArr = np.zeros((ny, npi))
        for i in range(ny):
            for j in range(npi):
                yPast = yArr[i]
                yL = YLArr[i, j]
                piLv = PiLArr[i, j]
                YH = yHStar + gammaH * (yL - yHStar)
                PiH = kappa * gammaH * (yL - yHStar) + b * piLv
                YL = yLBar + gammaL * (yL - yLBar)
                PiL = piLBar + gammaL * (piLv - piLBar)

                rL = rho + lam * (YH - yL) + (1 - lam) * (YL - yL) - eta / (1 - eta) * (yL - yPast)
                iL = rL + lam * PiH + (1 - lam) * PiL
                rLArr[i, j] = rL
                iLArr[i, j] = iL

        eq.update(type='baseline', A=A, B=B, C=C, D=D, E=E,
                  gammaH=gammaH, gammaL=gammaL, thetaH=thetaH, PsiH=PsiH, IH=IH,
                  yLBar=yLBar, piLBar=piLBar,
                  yArr=yArr, piArr=piArr,
                  YHArr=YHArr, VHArr=VHArr, PiHArr=PiHArr, iHArr=iHArr, rHArr=rHArr,
                  YLArr=YLArr, PiLArr=PiLArr, iLArr=iLArr, rLArr=rLArr)
        return eq

    if type_ == 2:
        eq['type'] = 'first best'
        ny, npi = len(yArr), len(piArr)
        yGrid = np.tile(yArr[:, None], (1, npi))
        eq['yArr'] = yArr
        eq['piArr'] = piArr
        eq['YHArr'] = yHStar + 0 * yGrid
        eq['PiHArr'] = 0 * yGrid
        eq['iHArr'] = rho + eta / (1 - eta) * (yGrid - yHStar)
        eq['rHArr'] = eq['iHArr'].copy()
        eq['YLArr'] = yLStar + 0 * yGrid
        eq['PiLArr'] = 0 * yGrid
        eq['iLArr'] = rho + lam * (yHStar - yLStar) + eta / (1 - eta) * (yGrid - yLStar)
        eq['rLArr'] = eq['iLArr'].copy()
        return eq

    raise ValueError


def _interp2(yArr, piArr, Z, yLast, piLast):
    """Bilinear interpolation matching Matlab interp2(piArr, yArr, Z, piLast, yLast)."""
    # Z[i,j] indexed by (y,pi)
    i = np.searchsorted(yArr, yLast) - 1
    i = max(0, min(i, len(yArr) - 2))
    j = np.searchsorted(piArr, piLast) - 1
    j = max(0, min(j, len(piArr) - 2))
    y0, y1 = yArr[i], yArr[i + 1]
    p0, p1 = piArr[j], piArr[j + 1]
    ty = (yLast - y0) / (y1 - y0)
    tp = (piLast - p0) / (p1 - p0)
    z = (Z[i, j] * (1 - ty) * (1 - tp)
         + Z[i + 1, j] * ty * (1 - tp)
         + Z[i, j + 1] * (1 - ty) * tp
         + Z[i + 1, j + 1] * ty * tp)
    return float(z)


def simulate_inertia(p: Params, eq: dict) -> dict:
    T = p.T
    timeArr = np.arange(-1, T + 1)
    n = len(timeArr)
    sTimeArr = np.zeros(n, dtype=int)
    y = np.zeros(n); pi = np.zeros(n); i_ = np.zeros(n); r = np.zeros(n)

    sTimeArr[0] = p.sInit
    sTimeArr[1] = p.sInit
    for k in range(2, n):
        sTimeArr[k] = 1 if (sTimeArr[k - 1] == 1 and timeArr[k] < p.tTrans) else 2

    y[0] = p.yLastInit
    pi[0] = p.piLastInit
    i_[0] = np.nan
    r[0] = np.nan

    yArr = eq['yArr']; piArr = eq['piArr']
    for k in range(1, n):
        yLast = y[k - 1]
        piLast = pi[k - 1]
        if sTimeArr[k] == 1:
            Y, Pi, I, R = eq['YLArr'], eq['PiLArr'], eq['iLArr'], eq['rLArr']
        else:
            Y, Pi, I, R = eq['YHArr'], eq['PiHArr'], eq['iHArr'], eq['rHArr']
        y[k] = _interp2(yArr, piArr, Y, yLast, piLast)
        pi[k] = _interp2(yArr, piArr, Pi, yLast, piLast)
        i_[k] = _interp2(yArr, piArr, I, yLast, piLast)
        r[k] = _interp2(yArr, piArr, R, yLast, piLast)

    return {'timeArr': timeArr, 'sTimeArr': sTimeArr,
            'yTimeArr': y, 'piTimeArr': pi, 'iTimeArr': i_, 'rTimeArr': r}


# ---------------------------------------------------------------------------
# ZLB model (Online Appendix B)
# ---------------------------------------------------------------------------

def solve_zlb(p: Params, eta: Optional[float] = None, type_: int = 1) -> dict:
    if eta is None:
        eta = p.eta
    beta = p.beta
    yArr = p.yArr
    yHStar, yLStar = p.yHStar, p.yLStar
    lam, rho = p.lam, p.rho
    assert rho - eta / (1 - eta) * (yHStar - yLStar) < 0

    eq: dict = {'yArr': yArr}
    if type_ == 1:
        # ---- High state with ZLB ----
        yHBar = yHStar - (1 - eta) / eta * rho
        step = (1 - eta) * rho / eta
        KH = int(np.ceil(yHBar / step))
        kHArr = np.arange(0, KH + 1)
        yHBarArr = np.zeros(len(kHArr))
        yHBarArr[0] = yHBar
        for i in range(1, len(kHArr)):
            yHBarCur = yHBarArr[i - 1]
            yHBarNext = yHStar if i == 1 else yHBarArr[i - 2]
            yHBarArr[i] = yHBarCur - (1 - eta) / eta * (rho + yHBarNext - yHBarCur)
        neg = np.where(yHBarArr < 0)[0]
        if len(neg):
            cut = neg[0] + 1
            kHArr = kHArr[:cut]; yHBarArr = yHBarArr[:cut]

        aArr = np.zeros(len(kHArr)); bArr = np.zeros(len(kHArr))
        aArr[0] = 0; bArr[0] = yHStar
        for i in range(1, len(kHArr)):
            aArr[i] = eta / (1 - (1 - eta) * aArr[i - 1])
            bArr[i] = aArr[i] * (1 - eta) / eta * (rho + bArr[i - 1])

        YHArr = np.zeros_like(yArr)
        kHindArr = np.zeros(len(yArr), dtype=int)
        for i, y in enumerate(yArr):
            if y >= yHBar:
                kHind = 0
                YH = yHStar
            else:
                # yHBarArr is decreasing; want first k where y >= yHBarArr(k)
                idx = np.where(y >= yHBarArr)[0]
                kHind = idx[0]
                YH = aArr[kHind] * y + bArr[kHind]
            kHindArr[i] = kHind
            YHArr[i] = YH

        VHArr = np.zeros_like(yArr)
        dVHArr = np.zeros_like(yArr)
        for i in range(len(yArr) - 1, -1, -1):
            y = yArr[i]
            if y >= yHBar:
                VHArr[i] = 0; dVHArr[i] = 0
            else:
                kHind = kHindArr[i]
                a = aArr[kHind]; bb = bArr[kHind]
                ay_b = a * y + bb
                VHNext = _interp(yArr, VHArr, ay_b)
                dVHNext = _interp(yArr, dVHArr, ay_b)
                VHArr[i] = -0.5 * (ay_b - yHStar) ** 2 + beta * VHNext
                dVHArr[i] = -(ay_b - yHStar) * a + beta * dVHNext * a

        iHArr = np.zeros_like(yArr)
        for i, y in enumerate(yArr):
            if y <= yHBar:
                iHArr[i] = 0
            else:
                iHArr[i] = rho + eta / (1 - eta) * (y - yHStar)

        eq.update(YHArr=YHArr, VHArr=VHArr, dVHArr=dVHArr, iHArr=iHArr,
                  kHArr=kHArr, yHBarArr=yHBarArr, aArr=aArr, bArr=bArr)

        # ---- Low state ----
        # unconstrained yL
        testArr = yArr - yLStar - beta * lam * dVHArr
        pos = np.where(testArr > 0)[0]
        ind = pos[0]
        yL = yArr[ind]
        YHyL = YHArr[ind]
        eq['yL'] = yL

        yLBar = yL - (1 - eta) / eta * (rho + lam * (YHyL - yL))
        KL = int(np.ceil(yLBar / ((1 - eta) * rho / eta)))
        kLArr = np.arange(0, KL + 1)
        yLBarArr = np.zeros(len(kLArr))
        yLBarArr[0] = yLBar
        for i in range(1, len(kLArr)):
            yLBarCur = yLBarArr[i - 1]
            yLBarNext = yL if i == 1 else yLBarArr[i - 2]
            YHNext = _interp(yArr, YHArr, yLBarCur)
            yLBarArr[i] = yLBarCur - (1 - eta) / eta * (rho + lam * YHNext + (1 - lam) * yLBarNext - yLBarCur)
        neg = np.where(yLBarArr < 0)[0]
        if len(neg):
            cut = neg[0] + 1
            kLArr = kLArr[:cut]; yLBarArr = yLBarArr[:cut]

        YLArr = np.zeros_like(yArr); iLArr = np.zeros_like(yArr)
        for i in range(len(yArr) - 1, -1, -1):
            y = yArr[i]
            if y >= yLBar:
                YLArr[i] = yL
                iLArr[i] = rho + lam * (YHyL - yL) + eta / (1 - eta) * (y - yL)
            else:
                tempArr = np.zeros_like(yArr)
                jslice = np.arange(i + 1, len(yArr))
                tempArr[jslice] = (yArr[jslice] - eta * y
                                   - (1 - eta) * (rho + lam * YHArr[jslice] + (1 - lam) * YLArr[jslice]))
                pos = np.where(tempArr > 0)[0]
                ind2 = pos[0]
                YLArr[i] = yArr[ind2]
                iLArr[i] = 0

        eq.update(YLArr=YLArr, iLArr=iLArr, kLArr=kLArr, yLBarArr=yLBarArr, yLBar=yLBar)
        return eq

    if type_ == 2:
        eq['type'] = 'first best'
        eq['YHArr'] = yHStar + 0 * yArr
        eq['iHArr'] = rho + eta / (1 - eta) * (yArr - yHStar)
        eq['YLArr'] = yLStar + 0 * yArr
        eq['iLArr'] = rho + eta / (1 - eta) * (yArr - yLStar) + lam * (yHStar - yLStar)
        return eq

    if type_ == 3:
        eq['type'] = 'myopic'
        eqTemp = solve_zlb(p, eta=eta, type_=1)
        eq['YHArr'] = eqTemp['YHArr']
        eq['iHArr'] = eqTemp['iHArr']
        yLMyopic = yLStar
        idxArr = np.where(yArr - yLMyopic >= 0)[0]
        indMyopic = idxArr[0]
        YHMyopic = eq['YHArr'][indMyopic]
        eq['YLArr'] = 0 * yArr + yLMyopic
        eq['iLArr'] = rho + eta / (1 - eta) * (yArr - yLMyopic) + lam * (YHMyopic - yLMyopic)
        return eq

    raise ValueError


def build_baseline_params() -> Params:
    """Load baseline params & compute yLastInit as in loadVariablesBaseline.m."""
    p = default_params()
    eqOpt = solve_baseline(p, eta=p.eta, type_=1)
    p.yLastInit = eqOpt['yLBar']
    return p
