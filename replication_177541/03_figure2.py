"""Figure 2: NKPC model."""
import matplotlib.pyplot as plt
from utils import build_baseline_params, solve_nkpc, simulate_nkpc

p = build_baseline_params()

eqOpt = solve_nkpc(p, type_=1)
eqFB = solve_nkpc(p, type_=2)
simOpt = simulate_nkpc(p, eqOpt)
simFB = simulate_nkpc(p, eqFB)
t = simOpt['timeArr']

fig, axes = plt.subplots(4, 1, figsize=(10, 13))
titles = [r'Output $y_t$', r'Inflation $\pi_t$', r'Nominal interest rate $i_t$',
          r'Real interest rate $r_t$']
keys = ['yTimeArr', 'piTimeArr', 'iTimeArr', 'rTimeArr']
for ax, title, key in zip(axes, titles, keys):
    ax.plot(t, simOpt[key], 'b-', lw=3, label='equilibrium (NKPC)')
    ax.plot(t, simFB[key], 'k:', lw=3, label='first-best (NKPC)')
    ax.set_title(title, fontsize=18)
axes[-1].set_xlabel(r'Period $t$', fontsize=16)
axes[0].legend(loc='lower right', fontsize=10)
fig.tight_layout()
fig.savefig('replication_177541/Figure2.pdf')
print('Saved replication_177541/Figure2.pdf')
print(f"gammaH  = {eqOpt['gammaH']:.6f}")
print(f"thetaH  = {eqOpt['thetaH']:.6f}")
print(f"piHBold = {eqOpt['piHBold']:.6f}")
print(f"yL      = {eqOpt['yL']:.6f}")
print(f"piL     = {eqOpt['piL']:.6f}")
for k, tt in enumerate(t):
    print(f"  t={tt:>2d}  y={simOpt['yTimeArr'][k]:.6f}  "
          f"pi={simOpt['piTimeArr'][k]:.6f}  "
          f"i={simOpt['iTimeArr'][k]:.6f}  "
          f"r={simOpt['rTimeArr'][k]:.6f}")
