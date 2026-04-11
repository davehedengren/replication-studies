"""Local Average Treatment Effect of driving on teenage mortality.

The paper reports LATE = β_mortality / θ_driving using a uniform bandwidth.
In Appendix Table A.14, they report three LATEs corresponding to three
first-stage measures. Here we replicate the headline LATE on motor vehicle
fatalities per 100 million vehicle miles driven, which is what the
paper's abstract emphasizes (10.1-14.5 deaths per 100M VMD)."""
import numpy as np

from utils import load_addhealth, load_mortality, rd_mse_opt

# For the LATE, paper imposes a uniform bandwidth (Appendix A.1).
# We reproduce with MSE-optimal individually, which will be close but not
# exactly identical.

mort = load_mortality("none")
add = load_addhealth("none")

mva = rd_mse_opt(mort, "cod_MVA", covs=True)
lic = rd_mse_opt(add, "DriverLicense", covs=True)
vmd150 = rd_mse_opt(add, "VehicleMiles_150", covs=True)
vmd265 = rd_mse_opt(add, "VehicleMiles_265", covs=True)

# Motor vehicle fatalities per 100M vehicle miles driven.
# deaths per 100k pop-year = MVA RD in per 100k
# miles per person-year = annual miles driven at cutoff (VMD150 or VMD265 coef)
# LATE (per 100M VMD) = (β_MVA / 100k) / (θ_VMD / 100M) = β * 1000 / θ
print("First-stage estimates (MSE-optimal):")
print(f"  ΔDriverLicense   = {lic['conv']:.4f}  (paper: 0.186)")
print(f"  ΔVMD (150)       = {vmd150['conv']:.1f} mi/yr (paper: 375)")
print(f"  ΔVMD (265)       = {vmd265['conv']:.1f} mi/yr (paper: 575)")
print()
print(f"Reduced form ΔMVA = {mva['conv']:.3f} per 100k (paper: 4.92)")
print()
print("LATE estimates (deaths per 100 million vehicle miles driven):")
late_150 = mva['conv'] * 1000 / vmd150['conv']
late_265 = mva['conv'] * 1000 / vmd265['conv']
late_lic = mva['conv'] / lic['conv']
print(f"  Using baseline VMD : {late_150:.2f}  (paper: 14.5)")
print(f"  Using alt VMD      : {late_265:.2f}  (paper: 10.1)")
print(f"  Per 100k per licensee: {late_lic:.2f}  (paper: 29.9)")
