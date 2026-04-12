"""Figure 1: baseline no-inflation model."""
import matplotlib.pyplot as plt
from utils import build_baseline_params, solve_baseline, simulate_baseline

p = build_baseline_params()

eqOpt = solve_baseline(p, eta=p.eta, type_=1)
eqFB = solve_baseline(p, eta=p.eta, type_=2)
eqMyopic = solve_baseline(p, eta=p.eta, type_=3)
eqOptLow = solve_baseline(p, eta=p.eta_low, type_=1)

simOpt = simulate_baseline(p, eqOpt)
simFB = simulate_baseline(p, eqFB)
simMyopic = simulate_baseline(p, eqMyopic)
simOptLow = simulate_baseline(p, eqOptLow)

t = simOpt['timeArr']

fig, axes = plt.subplots(2, 1, figsize=(10, 12))
ax = axes[0]
ax.plot(t, simOpt['yTimeArr'], 'b-', lw=3, label='equilibrium')
ax.plot(t, simFB['yTimeArr'], 'k:', lw=3, label='first-best benchmark')
ax.plot(t, simOptLow['yTimeArr'], 'm-.', lw=1, label='equilibrium, less inertia')
ax.plot(t, simMyopic['yTimeArr'], 'r--', lw=3, label='myopic benchmark')
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
out = 'replication_177541/Figure1.pdf'
fig.savefig(out)
print(f'Saved {out}')

# Key numerics
print(f"gammaH          = {eqOpt['gammaH']:.6f}")
print(f"thetaH          = {eqOpt['thetaH']:.6f}")
print(f"yL              = {eqOpt['yL']:.6f}")
print(f"yLBar           = {eqOpt['yLBar']:.6f}")
print(f"eta_low gammaH  = {eqOptLow['gammaH']:.6f}")
print(f"eta_low yL      = {eqOptLow['yL']:.6f}")
print("Simulation paths (t, y_opt, y_FB, y_myopic, y_optLow):")
for k, tt in enumerate(t):
    print(f"  t={tt:>2d}  y={simOpt['yTimeArr'][k]:.6f}  "
          f"yFB={simFB['yTimeArr'][k]:.6f}  "
          f"yM={simMyopic['yTimeArr'][k]:.6f}  "
          f"yLow={simOptLow['yTimeArr'][k]:.6f}")
