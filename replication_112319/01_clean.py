"""
01_clean.py - Load and validate experimental data for Charness & Levin (2005)
"""
import sys
sys.path.insert(0, '.')
from utils import *

print("=" * 70)
print("PHASE 1: Data Loading and Validation")
print("=" * 70)

df = load_data_fast()
print(f"\nTotal observations: {len(df)}")
print(f"Total individuals: {df['Individual'].nunique()}")

# ── Validate participant counts ────────────────────────────────────────
print("\n--- Participant Counts by Treatment ---")
for t in [1, 2, 3]:
    n = df[df['treatment'] == t]['Individual'].nunique()
    expected = {1: 59, 2: 54, 3: 52}[t]
    match = "MATCH" if n == expected else "MISMATCH"
    print(f"  Treatment {t}: {n} participants (expected {expected}) [{match}]")

# ── Validate rounds per participant ────────────────────────────────────
print("\n--- Rounds per Participant ---")
rounds = df.groupby(['treatment', 'Individual'])['Round'].count()
for t in [1, 2, 3]:
    sub = rounds.loc[t]
    expected = {1: 60, 2: 60, 3: 80}[t]
    all_match = (sub == expected).all()
    print(f"  Treatment {t}: {sub.min()}-{sub.max()} rounds (expected {expected}) "
          f"[{'ALL MATCH' if all_match else 'MISMATCH'}]")

# ── Validate phase structure ───────────────────────────────────────────
print("\n--- Phase Structure ---")
for t in [1, 2]:
    sub = df[df['treatment'] == t]
    for p in ['I', 'II', 'III']:
        ps = sub[sub['phase'] == p]
        print(f"  T{t} Phase {p}: rounds {ps['Round'].min()}-{ps['Round'].max()}, "
              f"n_obs={len(ps)}, n_per_ind={len(ps)//({1:59,2:54}[t])}")

sub = df[df['treatment'] == 3]
for p in ['I', 'II']:
    ps = sub[sub['phase'] == p]
    print(f"  T3 Phase {p}: rounds {ps['Round'].min()}-{ps['Round'].max()}, "
          f"n_obs={len(ps)}, n_per_ind={len(ps)//52}")

# ── Validate Phase I restrictions ──────────────────────────────────────
print("\n--- Phase I First-Draw Restrictions ---")
for t in [1, 2]:
    sub = df[(df['treatment'] == t) & (df['phase'] == 'I')]
    # Odd rounds: must start Left; Even rounds: must start Right
    odd = sub[sub['Round'] % 2 == 1]
    even = sub[sub['Round'] % 2 == 0]
    print(f"  T{t} Phase I odd rounds: Left1 mean = {odd['Left1'].mean():.4f} (expect 1.0)")
    print(f"  T{t} Phase I even rounds: Left1 mean = {even['Left1'].mean():.4f} (expect 0.0)")

sub = df[(df['treatment'] == 3) & (df['phase'] == 'I')]
print(f"  T3 Phase I: Left1 mean = {sub['Left1'].mean():.4f} (expect 1.0 - always Left)")

# ── Validate urn compositions ──────────────────────────────────────────
print("\n--- Urn Composition Validation (via draw probabilities) ---")
for t, urns in [(1, T1_URNS), (2, T2_URNS)]:
    sub = df[df['treatment'] == t]
    for state_name, up_val in [('Up', 1), ('Down', 0)]:
        for urn_name, left_val in [('Left', 1), ('Right', 0)]:
            mask = (sub['Up'] == up_val) & (sub['Left1'] == left_val)
            obs = sub[mask]
            if len(obs) > 0:
                b, w = urns[(state_name, urn_name)]
                expected_p = b / (b + w) if (b + w) > 0 else 0
                actual_p = obs['Cyan1'].mean()
                print(f"  T{t} {state_name}/{urn_name}: P(Cyan)={actual_p:.4f} "
                      f"(expected ~{expected_p:.4f}, n={len(obs)})")

# ── Validate T3 first draw doesn't pay ─────────────────────────────────
print("\n--- T3 Cyan_Pays Randomization ---")
t3 = df[df['treatment'] == 3]
print(f"  T3 Cyan_Pays distribution: {t3['Cyan_Pays'].value_counts().to_dict()}")
print(f"  T3 Cyan_Pays mean: {t3['Cyan_Pays'].mean():.4f} (expect ~0.5)")

# ── Gender distribution ────────────────────────────────────────────────
print("\n--- Gender Distribution ---")
gender = df.groupby(['treatment', 'Individual'])['Female'].first()
for t in [1, 2, 3]:
    sub = gender.loc[t]
    print(f"  Treatment {t}: {(sub==0).sum()} male, {(sub==1).sum()} female")

# ── Key table verification counts ──────────────────────────────────────
print("\n--- Table 1 Verification: T1 Switching Errors ---")
t1 = df[df['treatment'] == 1]

# After Right draw
for phase_name, phase_val in [('I', 'I'), ('II', 'II'), ('III', 'III')]:
    for draw_name, black_val in [('Black', 1), ('White', 0)]:
        mask = (t1['phase'] == phase_val) & (t1['Left1'] == 0) & (t1['first_draw_black'] == black_val)
        sub = t1[mask]
        errors = sub['switching_error'].sum()
        total = len(sub)
        pct = errors / total * 100 if total > 0 else 0
        print(f"  Right {draw_name} Phase {phase_name}: {int(errors)}/{total} ({pct:.1f}%)")

print()
for phase_name, phase_val in [('I', 'I'), ('II', 'II')]:
    for draw_name, black_val in [('Black', 1), ('White', 0)]:
        mask = (t1['phase'] == phase_val) & (t1['Left1'] == 1) & (t1['first_draw_black'] == black_val)
        sub = t1[mask]
        errors = sub['switching_error'].sum()
        total = len(sub)
        pct = errors / total * 100 if total > 0 else 0
        print(f"  Left {draw_name} Phase {phase_name}: {int(errors)}/{total} ({pct:.1f}%)")

# Phase III Left draws for T1
for draw_name, black_val in [('Black', 1), ('White', 0)]:
    mask = (t1['phase'] == 'III') & (t1['Left1'] == 1) & (t1['first_draw_black'] == black_val)
    sub = t1[mask]
    errors = sub['switching_error'].sum()
    total = len(sub)
    pct = errors / total * 100 if total > 0 else 0
    print(f"  Left {draw_name} Phase III: {int(errors)}/{total} ({pct:.1f}%)")

# ── Save cleaned data ──────────────────────────────────────────────────
output_path = os.path.join(OUTPUT_DIR, 'cleaned_data.csv')
df.to_csv(output_path, index=False)
print(f"\nCleaned data saved to {output_path}")
print(f"Total rows: {len(df)}, Columns: {list(df.columns)}")
