"""
Shared utilities for replication of Charness & Levin (2005)
"When Optimal Choices Feel Wrong: A Laboratory Study of Bayesian Updating,
Complexity, and Affect"

Paper ID: 112319
"""
import os
import pandas as pd
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', '112319-V1')
DATA_FILE = os.path.join(DATA_DIR, 'new_big_data.xls')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Treatment definitions ──────────────────────────────────────────────
# Individual IDs by treatment
T1_IDS = range(1, 60)      # 59 participants
T2_IDS = range(60, 114)    # 54 participants
T3_IDS = range(114, 166)   # 52 participants

# Urn compositions: {(state, urn): (black_balls, white_balls)}
# Treatment 1
T1_URNS = {
    ('Up', 'Left'):   (4, 2),   # 4 black, 2 white
    ('Up', 'Right'):  (6, 0),   # 6 black, 0 white
    ('Down', 'Left'): (3, 3),   # 3 black, 3 white
    ('Down', 'Right'):(0, 6),   # 0 black, 6 white
}

# Treatment 2 (Left Down changed to 2B/4W)
T2_URNS = {
    ('Up', 'Left'):   (4, 2),
    ('Up', 'Right'):  (6, 0),
    ('Down', 'Left'): (2, 4),   # Changed from T1
    ('Down', 'Right'):(0, 6),
}

# Treatment 3 (same as T2)
T3_URNS = T2_URNS.copy()

# Phase structure
# T1/T2: Phase I = rounds 1-20, Phase II = rounds 21-50, Phase III = rounds 51-60
# T3: Phase I = rounds 1-70, Phase II = rounds 71-80
def get_phase_t12(round_num):
    if round_num <= 20:
        return 'I'
    elif round_num <= 50:
        return 'II'
    else:
        return 'III'

def get_phase_t3(round_num):
    if round_num <= 70:
        return 'I'
    else:
        return 'II'

# Payoffs per experimental unit
# T1/T2 Phases I-II: Left black = 1, Right black = 7/6
# T1 Phase III: Left black = 7/6, Right black = 1 (reversed)
# T2 Phase III: Left black = 7/6, Right black = 1 (reversed)
# T3: first draw doesn't pay; second draw from Left = 1, Right = 7/6 (Phase I)
#     Phase II: Left = 7/6, Right = 1

PAYOFF_LEFT_BLACK = 1.0
PAYOFF_RIGHT_BLACK = 7.0 / 6.0
PAYOFF_WHITE = 0.0

# ── BEU Predictions ────────────────────────────────────────────────────
# Treatment 1/2 Phase II: BEU-optimal first draw = Right
# Treatment 1 Phase III: BEU-optimal first draw = Left (payoffs reversed)
# Treatment 2 Phase III: BEU-optimal first draw = Right (barely)

# Switching predictions (T1 Phases I-II):
# After Right Black: stay with Right (Up state revealed) -> Left2=0 is correct
# After Right White: switch to Left (Down state revealed) -> Left2=1 is correct
# After Left Black: switch to Right (BEU, counter-reinforcement) -> Left2=0 is correct
# After Left White: stay with Left (BEU, counter-reinforcement) -> Left2=1 is correct

# T1 Phase III (payoffs reversed):
# After Left Black: stay with Left (no switch) -> Left2=1 is correct
# After Left White: stay with Left -> Left2=1 is correct
# (BEU says never switch from Left in Phase III of T1)

def beu_optimal_second_draw_left(treatment, phase, first_draw_left, first_draw_black):
    """
    Returns the BEU-optimal Left2 value (1=draw from Left, 0=draw from Right).
    """
    if not first_draw_left:
        # Started from Right
        if first_draw_black:
            # Right Black -> state is Up -> Right is better (6B vs 4B)
            if treatment in (1, 2) and phase in ('I', 'II'):
                return 0  # Stay Right
            elif treatment == 1 and phase == 'III':
                # Payoffs reversed: Left pays 7/6, Right pays 1
                # Up state: Right has 6B but pays 1 each, Left has 4B paying 7/6 each
                # EU(Right) = 6/6 * 1 = 1, EU(Left) = 4/6 * 7/6 = 28/36 = 0.778
                # Wait - in Phase III, black from Right pays 1, black from Left pays 7/6
                # Up state: Right 6B -> EU = 6/6 * 1 = 1. Left 4B/2W -> EU = 4/6 * 7/6 = 0.778
                # So stay Right
                return 0  # Stay Right
            elif treatment == 2 and phase == 'III':
                # T2 Phase III: payoffs reversed. After RB (Up state):
                # BEU-optimal is to switch to Left (Left pays 7/6 per black)
                # Actually in T2 Phase III, it's BEU-optimal to SWITCH to Right after LB
                # Let me re-check from paper: "*In Phase III of Treatment 2, it is
                # BEU-optimal to switch to Right after a Black draw from Left"
                return 0  # Stay Right
        else:
            # Right White -> state is Down
            if treatment in (1, 2) and phase in ('I', 'II'):
                return 1  # Switch to Left (Left has 3B/3W or 2B/4W, Right has 0B)
            elif treatment == 1 and phase == 'III':
                return 1  # Switch to Left
            elif treatment == 2 and phase == 'III':
                return 1  # Switch to Left
    else:
        # Started from Left
        if first_draw_black:
            # Left Black -> likely Up state (updates toward Up)
            if treatment in (1, 2) and phase in ('I', 'II'):
                return 0  # Switch to Right (BEU: counter-intuitive!)
            elif treatment == 1 and phase == 'III':
                # Payoffs reversed: stay Left always
                return 1  # Stay Left
            elif treatment == 2 and phase == 'III':
                # Paper footnote: "In Phase III of Treatment 2, it is BEU-optimal
                # to switch to Right after a Black draw from Left"
                return 0  # Switch to Right
        else:
            # Left White -> likely Down state (updates toward Down)
            if treatment in (1, 2) and phase in ('I', 'II'):
                return 1  # Stay with Left (BEU: counter-intuitive!)
            elif treatment == 1 and phase == 'III':
                return 1  # Stay Left
            elif treatment == 2 and phase == 'III':
                return 1  # Stay Left
    return None  # shouldn't reach here


def beu_optimal_second_draw_left_t3(phase, first_draw_favorable):
    """
    T3: first draw always from Left, no payoff on first draw.
    'favorable' = same color as what's desired for second draw.

    Phase I: second draw Left pays 1, Right pays 7/6
    BEU: after favorable -> switch to Right (0)
         after unfavorable -> stay Left (1)
    Phase II: payoffs reversed, Left pays 7/6, Right pays 1
    BEU: after favorable -> switch to Right (counter-intuitive)
         after unfavorable -> stay Left
    """
    if phase == 'I':
        if first_draw_favorable:
            return 0  # Switch to Right
        else:
            return 1  # Stay Left
    else:  # Phase II
        if first_draw_favorable:
            return 0  # Switch to Right (BEU-optimal in Phase II)
        else:
            return 1  # Stay Left


# ── Data Loading ───────────────────────────────────────────────────────
def load_data():
    """Load and annotate the experimental data."""
    df = pd.read_excel(DATA_FILE, sheet_name='Sheet1')

    # Assign treatment
    df['treatment'] = 0
    df.loc[df['Individual'].isin(T1_IDS), 'treatment'] = 1
    df.loc[df['Individual'].isin(T2_IDS), 'treatment'] = 2
    df.loc[df['Individual'].isin(T3_IDS), 'treatment'] = 3

    # Assign phase
    df['phase'] = ''
    mask_t12 = df['treatment'].isin([1, 2])
    mask_t3 = df['treatment'] == 3
    df.loc[mask_t12, 'phase'] = df.loc[mask_t12, 'Round'].apply(get_phase_t12)
    df.loc[mask_t3, 'phase'] = df.loc[mask_t3, 'Round'].apply(get_phase_t3)

    # Derive success/failure of first draw
    # For T1/T2: Cyan_Pays=1 always, so success = Cyan1
    # For T3: success = (Cyan1 == Cyan_Pays) -- favorable draw
    df['first_draw_black'] = np.where(
        df['treatment'].isin([1, 2]),
        df['Cyan1'],
        (df['Cyan1'] == df['Cyan_Pays']).astype(int)
    )

    # For T3, 'favorable' means the first draw color matches the paying color
    df['first_draw_favorable'] = (df['Cyan1'] == df['Cyan_Pays']).astype(int)

    # Did participant switch urns? (second draw different from first)
    df['switched'] = (df['Left1'] != df['Left2']).astype(int)

    # Compute BEU-optimal second draw
    df['beu_left2'] = np.nan
    for idx, row in df.iterrows():
        if row['treatment'] in (1, 2):
            df.loc[idx, 'beu_left2'] = beu_optimal_second_draw_left(
                row['treatment'], row['phase'],
                row['Left1'] == 1, row['first_draw_black'] == 1
            )
        else:  # T3
            df.loc[idx, 'beu_left2'] = beu_optimal_second_draw_left_t3(
                row['phase'], row['first_draw_favorable'] == 1
            )

    # Switching error: did participant deviate from BEU?
    df['switching_error'] = (df['Left2'] != df['beu_left2']).astype(int)

    # Starting error (T1/T2 only, Phase II and III)
    # T1 Phase II: BEU start = Right (Left1=0)
    # T1 Phase III: BEU start = Left (Left1=1)
    # T2 Phase II: BEU start = Right (Left1=0)
    # T2 Phase III: BEU start = Right (Left1=0) -- barely
    df['starting_error'] = np.nan
    mask = (df['treatment'] == 1) & (df['phase'] == 'II')
    df.loc[mask, 'starting_error'] = df.loc[mask, 'Left1']  # Error if Left1=1
    mask = (df['treatment'] == 1) & (df['phase'] == 'III')
    df.loc[mask, 'starting_error'] = (1 - df.loc[mask, 'Left1'])  # Error if Left1=0 (should start Left)
    mask = (df['treatment'] == 2) & (df['phase'] == 'II')
    df.loc[mask, 'starting_error'] = df.loc[mask, 'Left1']  # Error if Left1=1
    mask = (df['treatment'] == 2) & (df['phase'] == 'III')
    df.loc[mask, 'starting_error'] = df.loc[mask, 'Left1']  # Error if Left1=1 (BEU = Right)

    return df


def load_data_fast():
    """Load data with vectorized BEU computation (faster than row-by-row)."""
    df = pd.read_excel(DATA_FILE, sheet_name='Sheet1')

    # Assign treatment
    df['treatment'] = np.where(df['Individual'] <= 59, 1,
                      np.where(df['Individual'] <= 113, 2, 3))

    # Assign phase
    df['phase'] = ''
    mask_t12 = df['treatment'].isin([1, 2])
    mask_t3 = df['treatment'] == 3
    df.loc[mask_t12 & (df['Round'] <= 20), 'phase'] = 'I'
    df.loc[mask_t12 & (df['Round'] > 20) & (df['Round'] <= 50), 'phase'] = 'II'
    df.loc[mask_t12 & (df['Round'] > 50), 'phase'] = 'III'
    df.loc[mask_t3 & (df['Round'] <= 70), 'phase'] = 'I'
    df.loc[mask_t3 & (df['Round'] > 70), 'phase'] = 'II'

    # First draw outcome
    df['first_draw_black'] = np.where(
        df['treatment'].isin([1, 2]),
        df['Cyan1'],
        (df['Cyan1'] == df['Cyan_Pays']).astype(int)
    )
    df['first_draw_favorable'] = (df['Cyan1'] == df['Cyan_Pays']).astype(int)

    # Switched urns
    df['switched'] = (df['Left1'] != df['Left2']).astype(int)

    # BEU optimal Left2 (vectorized)
    df['beu_left2'] = np.nan

    # T1/T2 Phase I,II: Right start
    m = df['treatment'].isin([1, 2]) & df['phase'].isin(['I', 'II'])
    # After Right Black -> stay Right (beu_left2=0)
    df.loc[m & (df['Left1'] == 0) & (df['first_draw_black'] == 1), 'beu_left2'] = 0
    # After Right White -> switch Left (beu_left2=1)
    df.loc[m & (df['Left1'] == 0) & (df['first_draw_black'] == 0), 'beu_left2'] = 1
    # After Left Black -> switch Right (beu_left2=0) -- counter-reinforcement!
    df.loc[m & (df['Left1'] == 1) & (df['first_draw_black'] == 1), 'beu_left2'] = 0
    # After Left White -> stay Left (beu_left2=1) -- counter-reinforcement!
    df.loc[m & (df['Left1'] == 1) & (df['first_draw_black'] == 0), 'beu_left2'] = 1

    # T1 Phase III: payoffs reversed, BEU says stay Left always
    m = (df['treatment'] == 1) & (df['phase'] == 'III')
    df.loc[m & (df['Left1'] == 0) & (df['first_draw_black'] == 1), 'beu_left2'] = 0  # RB: stay Right
    df.loc[m & (df['Left1'] == 0) & (df['first_draw_black'] == 0), 'beu_left2'] = 1  # RW: switch Left
    df.loc[m & (df['Left1'] == 1) & (df['first_draw_black'] == 1), 'beu_left2'] = 1  # LB: stay Left
    df.loc[m & (df['Left1'] == 1) & (df['first_draw_black'] == 0), 'beu_left2'] = 1  # LW: stay Left

    # T2 Phase III: BEU-optimal to switch to Right after LB
    m = (df['treatment'] == 2) & (df['phase'] == 'III')
    df.loc[m & (df['Left1'] == 0) & (df['first_draw_black'] == 1), 'beu_left2'] = 0  # RB: stay Right
    df.loc[m & (df['Left1'] == 0) & (df['first_draw_black'] == 0), 'beu_left2'] = 1  # RW: switch Left
    df.loc[m & (df['Left1'] == 1) & (df['first_draw_black'] == 1), 'beu_left2'] = 0  # LB: switch Right
    df.loc[m & (df['Left1'] == 1) & (df['first_draw_black'] == 0), 'beu_left2'] = 1  # LW: stay Left

    # T3: always start Left
    m = (df['treatment'] == 3) & (df['phase'] == 'I')
    # Favorable -> switch Right (0), Unfavorable -> stay Left (1)
    df.loc[m & (df['first_draw_favorable'] == 1), 'beu_left2'] = 0
    df.loc[m & (df['first_draw_favorable'] == 0), 'beu_left2'] = 1

    m = (df['treatment'] == 3) & (df['phase'] == 'II')
    # Phase II payoffs reversed: favorable -> switch Right, unfavorable -> stay Left
    df.loc[m & (df['first_draw_favorable'] == 1), 'beu_left2'] = 0
    df.loc[m & (df['first_draw_favorable'] == 0), 'beu_left2'] = 1

    # Switching error
    df['switching_error'] = (df['Left2'] != df['beu_left2']).astype(int)

    # Starting error (T1/T2 only)
    df['starting_error'] = np.nan
    df.loc[(df['treatment'] == 1) & (df['phase'] == 'II'), 'starting_error'] = df['Left1']
    df.loc[(df['treatment'] == 1) & (df['phase'] == 'III'), 'starting_error'] = 1 - df['Left1']
    df.loc[(df['treatment'] == 2) & (df['phase'] == 'II'), 'starting_error'] = df['Left1']
    df.loc[(df['treatment'] == 2) & (df['phase'] == 'III'), 'starting_error'] = df['Left1']

    return df
