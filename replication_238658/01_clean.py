"""
01_clean.py — Load and validate all 4 datasets + literature survey.

Danieli et al. (2025) "Negative Control Falsification Tests for IV Designs"
"""

import numpy as np
import pandas as pd
from utils import (OUTPUT_DIR, load_adh, load_adh_preperiod, load_deming,
                   load_ashraf_galor, load_nunn_qian, load_literature_survey)

print("=" * 60)
print("01_clean.py — Data loading and validation")
print("=" * 60)

# ══════════════════════════════════════════════════════════════════════
# 1. ADH (Autor, Dorn, Hanson 2013)
# ══════════════════════════════════════════════════════════════════════

print("\n── 1. ADH: China Trade Shock ──\n")

adh = load_adh()
adh_pre = load_adh_preperiod()

print(f"  ADH main: {adh.shape[0]} rows, {adh.shape[1]} columns")
print(f"  ADH pre-period: {adh_pre.shape[0]} rows, {adh_pre.shape[1]} columns")
assert adh.shape[0] == 1444, f"Expected 1444, got {adh.shape[0]}"

# 1990 cross-section
adh_1990 = adh[adh['yr'] == 1990]
print(f"  1990 cross-section: {len(adh_1990)} commuting zones")
assert len(adh_1990) == 722

# Key variables
print(f"  IV (d_tradeotch_pw_lag): mean={adh_1990['d_tradeotch_pw_lag'].mean():.4f}")
print(f"  Endogenous (d_tradeusch_pw): mean={adh_1990['d_tradeusch_pw'].mean():.4f}")
print(f"  Outcome (d_sh_empl_mfg): mean={adh_1990['d_sh_empl_mfg'].mean():.4f}")

# ══════════════════════════════════════════════════════════════════════
# 2. DEMING (2014)
# ══════════════════════════════════════════════════════════════════════

print("\n── 2. Deming: School Choice ──\n")

dem = load_deming()
print(f"  Deming: {dem.shape[0]} rows, {dem.shape[1]} columns")

# Key variables
if 'lottery' in dem.columns:
    print(f"  lottery: {dem['lottery'].mean():.3f} (fraction won)")
if 'lott_VA' in dem.columns:
    print(f"  lott_VA: mean={dem['lott_VA'].mean():.4f}, sd={dem['lott_VA'].std():.4f}")

# ══════════════════════════════════════════════════════════════════════
# 3. ASHRAF & GALOR (2013)
# ══════════════════════════════════════════════════════════════════════

print("\n── 3. Ashraf-Galor: Genetic Diversity ──\n")

ag = load_ashraf_galor()
print(f"  Ashraf-Galor: {ag.shape[0]} rows, {ag.shape[1]} columns")

# Key variables
for v in ['mdist_hgdp', 'ln_pd1500', 'ln_yst', 'ln_arable', 'ln_abslat', 'ln_suitavg']:
    if v in ag.columns:
        valid = ag[v].dropna()
        print(f"  {v}: N={len(valid)}, mean={valid.mean():.3f}")

# Clean sample
if 'cleanpd1500' in ag.columns:
    ag_clean = ag[ag['cleanpd1500'] == 1]
    print(f"  Clean sample (cleanpd1500==1): {len(ag_clean)} countries")

# ══════════════════════════════════════════════════════════════════════
# 4. NUNN & QIAN (2014)
# ══════════════════════════════════════════════════════════════════════

print("\n── 4. Nunn-Qian: Food Aid & Conflict ──\n")

nq = load_nunn_qian()
print(f"  Nunn-Qian: {nq.shape[0]} rows, {nq.shape[1]} columns")

if 'instrument' in nq.columns:
    print(f"  instrument: mean={nq['instrument'].mean():.4f}")
if 'intra_state' in nq.columns:
    print(f"  intra_state (×1000): mean={nq['intra_state'].mean():.4f}")

# ══════════════════════════════════════════════════════════════════════
# 5. LITERATURE SURVEY
# ══════════════════════════════════════════════════════════════════════

print("\n── 5. Literature Survey ──\n")

lit = load_literature_survey()
print(f"  Papers surveyed: {len(lit)}")
print(f"  Columns: {list(lit.columns)[:10]}...")

print("\n" + "=" * 60)
print("01_clean.py — DONE")
print("=" * 60)
