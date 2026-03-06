"""Run build pipeline from Phase 2 onward (Phase 1 already complete)."""
import sys
sys.path.insert(0, '.')
from importlib import import_module

# Import the build module
build = import_module('01_build')

# Phase 2: Population
pop_data = build.phase2_population()

# Phase 3: Patents
patents = build.phase3_patents(pop_data)

# Phase 4: QCEW, Dynamism
build.phase4_outcomes()

# Phase 5: Push-pull components
iig_data = build.phase5_instruments()

# Phase 6: 3-stage instruments
instruments = build.phase6_instruments(iig_data)

# Phase 7: Final assembly
if instruments is not None:
    final = build.phase7_final(instruments, patents)
else:
    print("\nWARNING: Instrument construction failed.")

print("\n" + "=" * 70)
print("BUILD COMPLETE")
print("=" * 70)
