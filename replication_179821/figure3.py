"""
Port of the authors' notebook "Code for Figure 3.ipynb" for paper 179821
(Onuchic & Ray 2023, "Signaling and Discrimination in Collaborative Projects",
AER 113(1), 210-252).

The paper is pure theory. Figure 3 is the only simulation: it illustrates
Proposition 1 under an exponential specification of the idea distribution
g(w, 1) ~ Exp(lambda_1), g(w, 0) ~ Exp(lambda_0). For each joint output z in
a grid, the script locates the three fixed points of the equilibrium
collaboration-threshold map psi (two asymmetric + one symmetric), then plots
collaboration probability and collaboration-set "size" in symmetric vs
asymmetric equilibria.

Running this script reproduces the two panels of Figure 3 and saves them to
`replication_179821/figure3_prob.png` and `replication_179821/figure3_size.png`.
"""

import math
import os

import matplotlib.pyplot as plt
import numpy as np

OUT = os.path.dirname(os.path.abspath(__file__))

# Parameters per Figure 3 caption (p=q=0.5, alpha=0.1, lambda_0=4, lambda_1=1)
l0, l1 = 4.0, 1.0
p, q, alpha = 0.5, 0.5, 0.1

N_Z = 50
N_A = 1000
Z = np.linspace(1.6, 5.0, N_Z)

psi = np.zeros((N_Z, N_A))
psiI = np.zeros((N_Z, N_A), dtype=int)

for n in range(N_Z):
    z = Z[n]
    AA = np.linspace(1e-6, z, N_A)

    Beta = np.zeros((N_A, N_A))
    b = np.zeros(N_A)

    for i in range(N_A):
        b[i] = 1.0 / (p + (1 - p) * (l0 / l1) * np.exp(-(l0 - l1) * AA[i]))
        for j in range(N_A - i - 1):
            lx = AA[i]
            bx = AA[i + j + 1]
            A = q * l1 * (np.exp(-(l0 - l1) * lx) - np.exp(-(l0 - l1) * bx)) \
                + (1 - q) * l0 * (l0 - l1) * (bx - lx) * np.exp(-(l0 - l1) * z)
            B = q * l1 * (l0 - l1) * (bx - lx) \
                + (1 - q) * l0 * (np.exp(-(l0 - l1) * (z - bx))
                                  - np.exp(-(l0 - l1) * (z - lx)))
            Beta[i, i + j + 1] = 1.0 / (p + (1 - p) * (l0 / l1) * (A / B))

    BR = np.zeros(N_A)
    BRI = np.zeros(N_A, dtype=int)

    for i in range(N_A):
        BRI[i] = i
        BR[i] = AA[i]
        for j in range(N_A - i - 1):
            Dif = Beta[i, i + j + 1] - b[i + j + 1] + (alpha / p) * (z - AA[i + j + 1])
            if Dif >= 0:
                BRI[i] = i + j + 1
                BR[i] = AA[i + j + 1]

    for i in range(N_A):
        a = BRI[i]
        aa = math.floor(N_A - a)
        c = BRI[aa]
        psiI[n, i] = math.floor(N_A - c)
        psi[n, i] = AA[math.floor(N_A - c)]

eqs = np.zeros((N_Z, 4))
eqsI = np.zeros((N_Z, 3), dtype=int)

for n in range(N_Z):
    z = Z[n]
    AA = np.linspace(1e-6, z, N_A)

    count = 0
    last = 1.0
    for i in range(N_A):
        new = psi[n, i] - AA[i] + 1e-13
        if new * last < 0:
            eqs[n, count] = AA[i]
            eqsI[n, count] = i
            if count == 2:
                eqs[n, count] = AA[N_A - i]
            if count == 1:
                eqs[n, 3] = AA[N_A - i]
            count += 1
        last = new

Difsym = np.zeros(N_Z)
Difasym = np.zeros(N_Z)
Diffsym = np.zeros(N_Z)
Diffasym = np.zeros(N_Z)


def make_gamma(z):
    def Gam(a):
        A = p * (l1 ** 2) * np.exp(-l1 * z) * (
            q * a + (1 - q) * (l0 / (l1 * (l0 - l1)))
            * (np.exp(-(l0 - l1) * (z - a)) - np.exp(-(l0 - l1) * z))
        )
        A = A + (1 - p) * (l0 ** 2) * np.exp(-l0 * z) * (
            (1 - q) * a + q * (l1 / (l0 * (l0 - l1)))
            * (np.exp((l0 - l1) * z) - np.exp((l0 - l1) * (z - a)))
        )
        B = p * (l1 ** 2) * np.exp(-l1 * z) * (
            q * z + (1 - q) * (l0 / (l1 * (l0 - l1)))
            * (1 - np.exp(-(l0 - l1) * z))
        )
        B = B + (1 - p) * (l0 ** 2) * np.exp(-l0 * z) * (
            (1 - q) * z + q * (l1 / (l0 * (l0 - l1)))
            * (np.exp((l0 - l1) * z) - 1)
        )
        return A / B
    return Gam


for n in range(N_Z):
    Gam = make_gamma(Z[n])
    Difsym[n] = Gam(eqs[n, 3]) - Gam(eqs[n, 1])
    Difasym[n] = Gam(eqs[n, 2]) - Gam(eqs[n, 0])
    Diffsym[n] = eqs[n, 3] - eqs[n, 1]
    Diffasym[n] = eqs[n, 2] - eqs[n, 0]

# Panel A: probability of collaboration
fig, ax = plt.subplots(dpi=150)
ax.set_xlabel("Joint Output $z$")
ax.set_ylabel("Probability of Collaboration")
ax.plot(Z, Difsym, "black", label="symmetric")
ax.plot(Z, Difasym, color="black", linestyle="dashed", label="asymmetric")
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "figure3_prob.png"))
plt.close(fig)

# Panel B: size of collaboration set
fig, ax = plt.subplots(dpi=150)
ax.set_xlabel("Joint Output $z$")
ax.set_ylabel("Size of Collaboration Set")
ax.plot(Z, Diffsym, "black", label="symmetric")
ax.plot(Z, Diffasym, color="black", linestyle="dashed", label="asymmetric")
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "figure3_size.png"))
plt.close(fig)

print("Z range:", Z[0], Z[-1])
print("Prob collab symmetric  — first, middle, last:",
      Difsym[0], Difsym[N_Z // 2], Difsym[-1])
print("Prob collab asymmetric — first, middle, last:",
      Difasym[0], Difasym[N_Z // 2], Difasym[-1])
print("Set size symmetric     — first, middle, last:",
      Diffsym[0], Diffsym[N_Z // 2], Diffsym[-1])
print("Set size asymmetric    — first, middle, last:",
      Diffasym[0], Diffasym[N_Z // 2], Diffasym[-1])
print("Saved:", os.path.join(OUT, "figure3_prob.png"))
print("Saved:", os.path.join(OUT, "figure3_size.png"))
