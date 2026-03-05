"""
03_figures.py — Reproduce Figures 1-3 and Appendix Figure A3.

Horn & Loewenstein (2025) "Underestimating Learning by Doing"
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import OUTPUT_DIR, STUDIES, load_study
import os

print("=" * 60)
print("03_figures.py — Reproducing all figures")
print("=" * 60)


# ══════════════════════════════════════════════════════════════════════
# FIGURE 1: Actual vs Predicted Performance across Rounds
# ══════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(3, 1, figsize=(8, 12))

for idx, (study_num, ax) in enumerate(zip([1, 2, 3], axes)):
    cfg = STUDIES[study_num]
    df = load_study(study_num)
    max_r = cfg['max_rounds']
    ahead_list = cfg['ahead']
    rounds = list(range(1, max_r + 1))

    # Actual performance
    actual = [df[f'r{r}_perf'].mean() for r in rounds]
    ax.plot(rounds, actual, 'k-o', linewidth=2, markersize=5, label='Actual Performance')

    # Predicted 1 round before
    pred1 = []
    for r in rounds:
        t = r - 1
        col = f't{t}_pred_r{r}'
        pred1.append(df[col].mean() if col in df.columns else np.nan)
    ax.plot(rounds, pred1, '-s', color='#377eb8', linewidth=1.5, markersize=4,
            label=f'Predicted Performance 1 Round Before')

    # Predicted ahead[1] rounds before
    ahead2 = ahead_list[1]
    pred2 = []
    for r in rounds:
        t = r - ahead2
        if t >= 0:
            col = f't{t}_pred_r{r}'
            pred2.append(df[col].mean() if col in df.columns else np.nan)
        else:
            pred2.append(np.nan)
    ax.plot(rounds, pred2, '--^', color='#e41a1c', linewidth=1.5, markersize=4,
            label=f'Predicted Performance {ahead2} Rounds Before')

    panel = chr(65 + idx)
    ax.set_title(f'Panel {panel}: {cfg["name"]}', fontsize=12, fontweight='bold')
    ax.set_xlabel('Round')
    y_label = 'Mean Number of Images' if study_num != 2 else 'Mean Score'
    ax.set_ylabel(y_label)
    ax.set_ylim(2, 6)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'Figure1.png'), dpi=150)
plt.close()
print("  Figure1.png saved")


# ══════════════════════════════════════════════════════════════════════
# FIGURE 2: Actual vs Predicted Learning across Round Intervals
# ══════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(3, 1, figsize=(8, 12))

for idx, (study_num, ax) in enumerate(zip([1, 2, 3], axes)):
    cfg = STUDIES[study_num]
    df = load_study(study_num)
    max_r = cfg['max_rounds']
    ahead = cfg['ahead'][1]  # 4, 3, or 4

    intervals = []
    actual_learning = []
    predicted_learning = []

    max_t = max_r - ahead
    for t in range(0, max_t + 1):
        r1 = t + 1
        r2 = t + ahead
        interval_label = f'R{r1}->R{r2}'
        intervals.append(interval_label)

        learn_col = f'learning_r{r1}r{r2}'
        pred_col = f't{t}_pred_r{r1}r{r2}'

        actual_learning.append(df[learn_col].mean() if learn_col in df.columns else np.nan)
        predicted_learning.append(df[pred_col].mean() if pred_col in df.columns else np.nan)

    x = range(len(intervals))
    ax.plot(x, actual_learning, '--o', color='black', linewidth=2, markersize=5,
            label='Actual Learning')
    ax.plot(x, predicted_learning, '--s', color='#377eb8', linewidth=1.5, markersize=4,
            label='Predicted Learning')

    panel = chr(65 + idx)
    ax.set_title(f'Panel {panel}: {cfg["name"]}', fontsize=12, fontweight='bold')
    ax.set_xticks(list(x))
    ax.set_xticklabels(intervals, fontsize=8)
    ax.set_xlabel('Round Interval')
    y_label = 'Mean Number of Images' if study_num != 2 else 'Mean Score'
    ax.set_ylabel(y_label)
    ax.set_ylim(-0.5, 2.0)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'Figure2.png'), dpi=150)
plt.close()
print("  Figure2.png saved")


# ══════════════════════════════════════════════════════════════════════
# FIGURE 3: Distribution of Overprediction Indices
# ══════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(3, 3, figsize=(14, 10))

for col_idx, (study_num, study_label) in enumerate(
    [(1, 'Low-Difficulty\nMirror Tracing'),
     (2, 'Latin Translation'),
     (3, 'High-Difficulty\nMirror Tracing')]):

    cfg = STUDIES[study_num]
    df = load_study(study_num)
    max_r = cfg['max_rounds']
    ahead_list = cfg['ahead']

    # Row 0: Predicted Performance 1 Round Before
    ax = axes[0, col_idx]
    diffs = []
    for t in range(1, max_r):
        r = t + 1
        col = f't{t}r1_perfdiff'
        if col in df.columns:
            diffs.append(df[col])
    if diffs:
        person_means = pd.concat(diffs, axis=1).mean(axis=1)
        ax.hist(person_means.dropna(), bins=20, color='#377eb8', alpha=0.7, edgecolor='white')
        ax.axvline(0, color='black', linestyle=':', linewidth=1)
    ax.set_title(f'Panel {chr(65 + col_idx)}: {study_label}', fontsize=9)
    if col_idx == 0:
        ax.set_ylabel('Frequency\n(Pred 1R Before)', fontsize=9)

    # Row 1: Predicted Performance ahead[1] Rounds Before
    ax = axes[1, col_idx]
    ahead2 = ahead_list[1]
    diffs = []
    for t in range(0, max_r - ahead2 + 1):
        r = t + ahead2
        col = f't{t}r{ahead2}_perfdiff'
        if col in df.columns:
            diffs.append(df[col])
    if diffs:
        person_means = pd.concat(diffs, axis=1).mean(axis=1)
        ax.hist(person_means.dropna(), bins=20, color='#e41a1c', alpha=0.7, edgecolor='white')
        ax.axvline(0, color='black', linestyle=':', linewidth=1)
    panel_letter = chr(68 + col_idx)
    ax.set_title(f'Panel {panel_letter}', fontsize=9)
    if col_idx == 0:
        ax.set_ylabel(f'Frequency\n(Pred {ahead2}R Before)', fontsize=9)

    # Row 2: Predicted Learning
    ax = axes[2, col_idx]
    learn_cols = sorted([c for c in df.columns if c.endswith('_learndiff')])
    if learn_cols:
        person_means = df[learn_cols].mean(axis=1)
        ax.hist(person_means.dropna(), bins=20, color='#4daf4a', alpha=0.7, edgecolor='white')
        ax.axvline(0, color='black', linestyle=':', linewidth=1)
    panel_letter = chr(71 + col_idx)
    ax.set_title(f'Panel {panel_letter}', fontsize=9)
    ax.set_xlabel('Overprediction Index', fontsize=9)
    if col_idx == 0:
        ax.set_ylabel('Frequency\n(Learning)', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'Figure3.png'), dpi=150)
plt.close()
print("  Figure3.png saved")


# ══════════════════════════════════════════════════════════════════════
# FIGURE A3: Performers vs Predictors (Study 4)
# ══════════════════════════════════════════════════════════════════════

df4 = load_study(4)
performers = df4[df4['performer'] == 1].copy()
predictors = df4[df4['performer'] == 0].copy()
rounds = list(range(1, 11))

fig, axes = plt.subplots(3, 1, figsize=(8, 12))

# Panel A: Predicted Performance 1 Round Ahead
ax = axes[0]
actual = [performers[f'r{r}_perf'].mean() for r in rounds]
perf_pred1 = [performers[f't{r-1}_pred_r{r}'].mean() if f't{r-1}_pred_r{r}' in performers.columns
              else np.nan for r in rounds]
pred_pred1 = [predictors[f't{r-1}_pred_r{r}'].mean() if f't{r-1}_pred_r{r}' in predictors.columns
              else np.nan for r in rounds]

ax.plot(rounds, actual, 'k-o', linewidth=2, markersize=5, label='Actual')
ax.plot(rounds, perf_pred1, '-s', color='#377eb8', linewidth=1.5, markersize=4,
        label='Performer Predictions')
ax.plot(rounds, pred_pred1, '--^', color='#e41a1c', linewidth=1.5, markersize=4,
        label='Predictor Predictions')
ax.set_title('Panel A: Predicted Performance One Round Ahead', fontweight='bold')
ax.set_ylabel('Mean Number of Images')
ax.set_ylim(2, 6)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Panel B: Predicted Performance 4 Rounds Ahead
ax = axes[1]
perf_pred4 = [performers[f't{r-4}_pred_r{r}'].mean() if r >= 4 and f't{r-4}_pred_r{r}' in performers.columns
              else np.nan for r in rounds]
pred_pred4 = [predictors[f't{r-4}_pred_r{r}'].mean() if r >= 4 and f't{r-4}_pred_r{r}' in predictors.columns
              else np.nan for r in rounds]

ax.plot(rounds, actual, 'k-o', linewidth=2, markersize=5, label='Actual')
ax.plot(rounds, perf_pred4, '-s', color='#377eb8', linewidth=1.5, markersize=4,
        label='Performer Predictions')
ax.plot(rounds, pred_pred4, '--^', color='#e41a1c', linewidth=1.5, markersize=4,
        label='Predictor Predictions')
ax.set_title('Panel B: Predicted Performance Four Rounds Ahead', fontweight='bold')
ax.set_ylabel('Mean Number of Images')
ax.set_ylim(2, 6)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Panel C: Learning
ax = axes[2]
intervals = []
actual_learn = []
perf_learn = []
pred_learn = []

for t in range(0, 7):
    r1 = t + 1
    r2 = t + 4
    intervals.append(f'R{r1}->R{r2}')

    learn_col = f'learning_r{r1}r{r2}'
    pred_col = f't{t}_pred_r{r1}r{r2}'

    actual_learn.append(performers[learn_col].mean() if learn_col in performers.columns else np.nan)
    perf_learn.append(performers[pred_col].mean() if pred_col in performers.columns else np.nan)
    pred_learn.append(predictors[pred_col].mean() if pred_col in predictors.columns else np.nan)

x = range(len(intervals))
ax.plot(x, actual_learn, '--o', color='black', linewidth=2, markersize=5, label='Actual Learning')
ax.plot(x, perf_learn, '-s', color='#377eb8', linewidth=1.5, markersize=4,
        label='Performer Predictions')
ax.plot(x, pred_learn, '--^', color='#e41a1c', linewidth=1.5, markersize=4,
        label='Predictor Predictions')
ax.set_title('Panel C: Learning', fontweight='bold')
ax.set_xticks(list(x))
ax.set_xticklabels(intervals, fontsize=8)
ax.set_ylabel('Mean Number of Images')
ax.set_ylim(-0.5, 2.0)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'FigureA3.png'), dpi=150)
plt.close()
print("  FigureA3.png saved")


print("\n" + "=" * 60)
print("03_figures.py — DONE")
print("=" * 60)
