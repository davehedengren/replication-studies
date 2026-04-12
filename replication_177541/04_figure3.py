"""Figure 3: Inertial inflation model."""
import matplotlib.pyplot as plt
from utils import build_baseline_params, solve_inertia, simulate_inertia

p = build_baseline_params()

eqOpt = solve_inertia(p, type_=1)
eqFB = solve_inertia(p, type_=2)
simOpt = simulate_inertia(p, eqOpt)
simFB = simulate_inertia(p, eqFB)
t = simOpt['timeArr']

fig, axes = plt.subplots(4, 1, figsize=(10, 13))
titles = [r'Output $y_t$', r'Inflation $\pi_t$', r'Nominal interest rate $i_t$',
          r'Real interest rate $r_t$']
keys = ['yTimeArr', 'piTimeArr', 'iTimeArr', 'rTimeArr']
for ax, title, key in zip(axes, titles, keys):
    ax.plot(t, simOpt[key], 'b-', lw=3, label='equilibrium (inertia)')
    ax.plot(t, simFB[key], 'k:', lw=3, label='first-best (inertia)')
    ax.set_title(title, fontsize=18)
axes[-1].set_xlabel(r'Period $t$', fontsize=16)
axes[0].legend(loc='lower right', fontsize=10)
fig.tight_layout()
fig.savefig('replication_177541/Figure3.pdf')
print('Saved replication_177541/Figure3.pdf')
print(f"gammaH   = {eqOpt['gammaH']:.6f}")
print(f"gammaL   = {eqOpt['gammaL']:.6f}")
print(f"thetaH   = {eqOpt['thetaH']:.6f}")
print(f"PsiH     = {eqOpt['PsiH']:.6f}")
print(f"IH       = {eqOpt['IH']:.6f}")
print(f"yLBar    = {eqOpt['yLBar']:.6f}")
print(f"piLBar   = {eqOpt['piLBar']:.6f}")
for k, tt in enumerate(t):
    print(f"  t={tt:>2d}  y={simOpt['yTimeArr'][k]:.6f}  "
          f"pi={simOpt['piTimeArr'][k]:.6f}  "
          f"i={simOpt['iTimeArr'][k]:.6f}  "
          f"r={simOpt['rTimeArr'][k]:.6f}")
