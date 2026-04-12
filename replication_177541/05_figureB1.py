"""Figure B.1: ZLB-constrained baseline model."""
import matplotlib.pyplot as plt
from utils import build_baseline_params, solve_zlb, simulate_baseline

p = build_baseline_params()

eqOpt = solve_zlb(p, eta=p.eta, type_=1)
eqFB = solve_zlb(p, eta=p.eta, type_=2)
eqMyopic = solve_zlb(p, eta=p.eta, type_=3)
eqOptLow = solve_zlb(p, eta=p.eta_low, type_=1)

simOpt = simulate_baseline(p, eqOpt)
simFB = simulate_baseline(p, eqFB)
simMyopic = simulate_baseline(p, eqMyopic)
simOptLow = simulate_baseline(p, eqOptLow)
t = simOpt['timeArr']

fig, axes = plt.subplots(2, 1, figsize=(10, 12))
ax = axes[0]
ax.plot(t, simOpt['yTimeArr'], 'b-', lw=3, label='equilibrium (ZLB)')
ax.plot(t, simFB['yTimeArr'], 'k:', lw=3, label='first-best (ZLB)')
ax.plot(t, simOptLow['yTimeArr'], 'm-.', lw=1, label='equilibrium, less inertia')
ax.plot(t, simMyopic['yTimeArr'], 'r--', lw=3, label='myopic (ZLB)')
ax.set_ylim(p.yLStar - 0.007, p.yHStar + 0.007)
ax.set_title(r'Output $y_t$', fontsize=18)
ax.legend(loc='lower right', fontsize=10)

ax = axes[1]
ax.plot(t, simOpt['iTimeArr'], 'b-', lw=3)
ax.plot(t, simFB['iTimeArr'], 'k:', lw=3)
ax.plot(t, simOptLow['iTimeArr'], 'm-.', lw=1)
ax.plot(t, simMyopic['iTimeArr'], 'r--', lw=3)
ax.set_title(r'Interest rate $i_t$', fontsize=18)
ax.set_xlabel(r'Period $t$', fontsize=16)

fig.tight_layout()
fig.savefig('replication_177541/FigureB1.pdf')
print('Saved replication_177541/FigureB1.pdf')
print(f"yLBar    = {eqOpt['yLBar']:.6f}")
print(f"yL       = {eqOpt['yL']:.6f}")
print(f"kHArr    = {eqOpt['kHArr']}")
print(f"yHBarArr = {eqOpt['yHBarArr']}")
print(f"kLArr    = {eqOpt['kLArr']}")
print(f"yLBarArr = {eqOpt['yLBarArr']}")
for k, tt in enumerate(t):
    print(f"  t={tt:>2d}  y={simOpt['yTimeArr'][k]:.6f}  "
          f"yFB={simFB['yTimeArr'][k]:.6f}  "
          f"yM={simMyopic['yTimeArr'][k]:.6f}  "
          f"yLow={simOptLow['yTimeArr'][k]:.6f}")
