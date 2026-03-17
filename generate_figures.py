#!/usr/bin/env python3
"""
Generate publication-quality figures for the drone sound detection article.
Focus: Model v8 and v9 results on real data.
Labels: English. Article text: Ukrainian.
"""

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────
OUTPUT_DIR = Path("/home/dremian/article_sound_detector/figures")
V8_CSV = Path("/home/dremian/drone-sound-detector/model_v8/runs/v8_640_20260213_204528/results.csv")
V9_CSV = Path("/home/dremian/drone-sound-detector/model_v9/runs/v9_640_20260216_052040/results.csv")
COMPARISON_JSON = Path("/home/dremian/article_sound_detector/comparison_results.json")
EXCLUDE_FOLDERS = {"16_ke41QysEhuQ"}  # leaf blower excluded

# ── Colors ─────────────────────────────────────────────────────────────────
C_V8 = '#2171b5'       # blue
C_V9 = '#e6550d'       # orange
C_CLEAN = '#2ca02c'    # green
C_AUG = '#d62728'      # red
C_DRONE = '#1f77b4'
C_NOT_DRONE = '#ff7f0e'


def setup_style():
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['DejaVu Serif', 'STIXGeneral', 'Times New Roman'],
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'axes.titleweight': 'bold',
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.15,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': False,
    })


def save_figure(fig, name):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        path = OUTPUT_DIR / f"{name}.{ext}"
        fig.savefig(path)
    print(f"  Saved: {name}.png / .pdf")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# Fig 1: Overall Results Summary
# ═══════════════════════════════════════════════════════════════════════════
def fig1_overall_results():
    print("Fig 1: Overall Results Summary")
    metrics = ['Accuracy', 'Drone Recall', 'Specificity']
    v8_vals = [92.5, 86.7, 95.7]
    v9_vals = [92.2, 89.4, 94.9]

    x = np.arange(len(metrics))
    w = 0.32

    fig, ax = plt.subplots(figsize=(7, 4))
    bars1 = ax.bar(x - w/2, v8_vals, w, label='Model v8', color=C_V8, edgecolor='white')
    bars2 = ax.bar(x + w/2, v9_vals, w, label='Model v9', color=C_V9, edgecolor='white')

    ax.set_ylabel('Score (%)')
    ax.set_title('Classification Performance on Unseen Validation Data')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(60, 100)
    ax.legend()

    for bars in (bars1, bars2):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f'{h:.1f}%', xy=(bar.get_x() + bar.get_width()/2, h),
                        xytext=(0, 4), textcoords='offset points',
                        ha='center', va='bottom', fontsize=9, fontweight='bold')

    save_figure(fig, 'fig1_overall_results')


# ═══════════════════════════════════════════════════════════════════════════
# Fig 2: Training Curves
# ═══════════════════════════════════════════════════════════════════════════
def fig2_training_curves():
    print("Fig 2: Training Curves")
    df8 = pd.read_csv(V8_CSV)
    df9 = pd.read_csv(V9_CSV)
    # Strip whitespace from column names
    df8.columns = df8.columns.str.strip()
    df9.columns = df9.columns.str.strip()

    fig, axes = plt.subplots(2, 3, figsize=(11, 6))

    configs = [
        (0, df8, 'Model v8', C_V8),
        (1, df9, 'Model v9', C_V9),
    ]

    for row, df, title, color in configs:
        epochs = df['epoch'].values

        # Training Loss
        ax = axes[row, 0]
        ax.plot(epochs, df['train/loss'], color=color, linewidth=1.5)
        ax.set_ylabel('Loss')
        ax.set_xlabel('Epoch')
        ax.set_title(f'{title} — Training Loss')

        # Validation Loss
        ax = axes[row, 1]
        ax.plot(epochs, df['val/loss'], color=color, linewidth=1.5)
        ax.set_ylabel('Loss')
        ax.set_xlabel('Epoch')
        ax.set_title(f'{title} — Validation Loss')

        # Accuracy
        ax = axes[row, 2]
        acc = df['metrics/accuracy_top1'].values * 100
        ax.plot(epochs, acc, color=color, linewidth=1.5)
        ax.set_ylabel('Accuracy (%)')
        ax.set_xlabel('Epoch')
        ax.set_title(f'{title} — Top-1 Accuracy')
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f'))

    save_figure(fig, 'fig2_training_curves')


# ═══════════════════════════════════════════════════════════════════════════
# Fig 3: Distance Detection
# ═══════════════════════════════════════════════════════════════════════════
def fig3_distance_detection():
    print("Fig 3: Distance Detection")
    v8_dist = [20, 30, 50, 75, 100]
    v8_recall = [89.8, 93.9, 82.7, 85.5, 81.7]

    v9_dist = [20, 30, 50, 75, 100, 150, 200, 300]
    v9_recall = [94.1, 96.7, 98.4, 98.5, 98.4, 100.0, 100.0, 100.0]

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.plot(v8_dist, v8_recall, '-o', color=C_V8, linewidth=2, markersize=7,
            label='Model v8', zorder=3)
    ax.plot(v9_dist, v9_recall, '-s', color=C_V9, linewidth=2, markersize=7,
            label='Model v9', zorder=3)

    # Annotate v8
    for d, r in zip(v8_dist, v8_recall):
        ax.annotate(f'{r:.1f}%', (d, r), textcoords='offset points',
                    xytext=(0, -14), ha='center', fontsize=8, color=C_V8)
    # Annotate v9
    for d, r in zip(v9_dist, v9_recall):
        ax.annotate(f'{r:.1f}%', (d, r), textcoords='offset points',
                    xytext=(0, 8), ha='center', fontsize=8, color=C_V9)

    ax.set_xlabel('Distance (m)')
    ax.set_ylabel('Drone Recall (%)')
    ax.set_title('Drone Detection Recall vs. Distance')
    ax.set_ylim(70, 105)
    ax.set_xlim(0, 320)
    ax.axhline(y=80, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
    ax.legend(loc='lower right')

    save_figure(fig, 'fig3_distance_detection')


# ═══════════════════════════════════════════════════════════════════════════
# Helper: load YouTube comparison data
# ═══════════════════════════════════════════════════════════════════════════
def _load_comparison_data():
    with open(COMPARISON_JSON) as f:
        data = json.load(f)
    return [r for r in data if r['folder'] not in EXCLUDE_FOLDERS]


# ═══════════════════════════════════════════════════════════════════════════
# Fig 4: YouTube Test — Per-Category Accuracy (v8 vs v9 on same data)
# ═══════════════════════════════════════════════════════════════════════════
def fig4_youtube_per_category():
    print("Fig 4: YouTube Test — Per-Category Accuracy")
    data = _load_comparison_data()

    # Aggregate by category
    cat_stats = {}
    for r in data:
        cat = r['category']
        if cat not in cat_stats:
            cat_stats[cat] = {'n': 0, 'v8_correct': 0, 'v9_correct': 0}
        n = r['total_segments']
        cat_stats[cat]['n'] += n
        if r['expected'] == 'drone':
            cat_stats[cat]['v8_correct'] += r['v8_drone_segments']
            cat_stats[cat]['v9_correct'] += r['v9_drone_segments']
        else:
            cat_stats[cat]['v8_correct'] += r['v8_not_drone_segments']
            cat_stats[cat]['v9_correct'] += r['v9_not_drone_segments']

    # Sort by average accuracy ascending
    items = sorted(cat_stats.items(),
                   key=lambda x: (x[1]['v8_correct'] + x[1]['v9_correct']) / (2 * x[1]['n']))

    categories = [k for k, _ in items]
    v8_acc = [v['v8_correct'] / v['n'] * 100 for _, v in items]
    v9_acc = [v['v9_correct'] / v['n'] * 100 for _, v in items]
    counts = [v['n'] for _, v in items]

    fig, ax = plt.subplots(figsize=(8, 4))
    y = np.arange(len(categories))
    h = 0.35
    bars1 = ax.barh(y - h/2, v8_acc, h, label='Model v8', color=C_V8, edgecolor='white')
    bars2 = ax.barh(y + h/2, v9_acc, h, label='Model v9', color=C_V9, edgecolor='white')

    for i in range(len(categories)):
        max_val = max(v8_acc[i], v9_acc[i])
        # n= label to the right
        ax.text(max_val + 2, y[i],
                f'n={counts[i]}', va='center', fontsize=8, color='#555555')
        # % inside v8 bar
        if v8_acc[i] > 15:
            ax.text(v8_acc[i] - 2, y[i] - h/2,
                    f'{v8_acc[i]:.0f}%', va='center', ha='right', fontsize=8,
                    color='white', fontweight='bold')
        # % inside v9 bar
        if v9_acc[i] > 15:
            ax.text(v9_acc[i] - 2, y[i] + h/2,
                    f'{v9_acc[i]:.0f}%', va='center', ha='right', fontsize=8,
                    color='white', fontweight='bold')

    ax.set_yticks(y)
    ax.set_yticklabels(categories)
    ax.set_xlabel('Correct Classification (%)')
    ax.set_title('YouTube Test — Accuracy by Sound Category (Same Data)')
    ax.set_xlim(0, 120)
    ax.invert_yaxis()
    ax.legend(loc='center right', bbox_to_anchor=(1.15, 0.5))

    save_figure(fig, 'fig4_youtube_per_category')


# ═══════════════════════════════════════════════════════════════════════════
# Fig 5: YouTube Test — Per-File Detailed Results
# ═══════════════════════════════════════════════════════════════════════════
def fig5_youtube_per_file():
    print("Fig 5: YouTube Test — Per-File Detailed Results")
    data = _load_comparison_data()

    # Sort: drones first, then not-drones; within group by v8 correct ascending
    drones = [r for r in data if r['expected'] == 'drone']
    not_drones = [r for r in data if r['expected'] == 'not_drone']
    drones.sort(key=lambda r: r['v8_correct_pct'])
    not_drones.sort(key=lambda r: r['v8_correct_pct'])
    ordered = not_drones + drones  # not_drones at top (inverted y-axis)

    labels = [r['description'] for r in ordered]
    v8_correct = [r['v8_correct_pct'] for r in ordered]
    v9_correct = [r['v9_correct_pct'] for r in ordered]
    expected = [r['expected'] for r in ordered]
    counts = [r['total_segments'] for r in ordered]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    y = np.arange(len(labels))
    h = 0.35

    bars1 = ax.barh(y - h/2, v8_correct, h, label='Model v8', color=C_V8, edgecolor='white')
    bars2 = ax.barh(y + h/2, v9_correct, h, label='Model v9', color=C_V9, edgecolor='white')

    for i in range(len(labels)):
        max_val = max(v8_correct[i], v9_correct[i])
        # n= label to the right
        ax.text(max_val + 2, y[i],
                f'n={counts[i]}', va='center', fontsize=7, color='#555555')
        # % inside v8 bar
        if v8_correct[i] > 15:
            ax.text(v8_correct[i] - 2, y[i] - h/2,
                    f'{v8_correct[i]:.0f}%', va='center', ha='right', fontsize=7,
                    color='white', fontweight='bold')
        # % inside v9 bar
        if v9_correct[i] > 15:
            ax.text(v9_correct[i] - 2, y[i] + h/2,
                    f'{v9_correct[i]:.0f}%', va='center', ha='right', fontsize=7,
                    color='white', fontweight='bold')

    # Add separator line between drone and not-drone groups
    n_not_drone = len(not_drones)
    sep_y = n_not_drone - 0.5
    ax.axhline(y=sep_y, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)

    # Place labels outside the plot area (right side)
    ax.annotate('DRONE\n(expected)', xy=(1.01, (sep_y + len(labels) - 0.5) / 2 / (len(labels) - 1)),
                xycoords=('axes fraction', 'axes fraction'),
                fontsize=7, style='italic', color='#2ca02c', va='center',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='#e8f5e9', edgecolor='#2ca02c', alpha=0.7))
    ax.annotate('NOT DRONE\n(expected)', xy=(1.01, (sep_y / 2) / (len(labels) - 1)),
                xycoords=('axes fraction', 'axes fraction'),
                fontsize=7, style='italic', color='#d62728', va='center',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='#fde8e8', edgecolor='#d62728', alpha=0.7))

    # Color y-labels by expected class
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    for i, tick_label in enumerate(ax.get_yticklabels()):
        if expected[i] == 'drone':
            tick_label.set_color('#2ca02c')
        else:
            tick_label.set_color('#333333')

    ax.set_xlabel('Correct Classification (%)')
    ax.set_title('YouTube Test — Per-File Results (v8 vs v9 on Identical Data)')
    ax.set_xlim(0, 120)
    ax.invert_yaxis()
    ax.legend(loc='lower right', bbox_to_anchor=(1.15, 0.0))

    save_figure(fig, 'fig5_youtube_per_file')


# ═══════════════════════════════════════════════════════════════════════════
# Fig 6: YouTube Test — False Positive Analysis (non-drone files only)
# ═══════════════════════════════════════════════════════════════════════════
def fig6_youtube_false_positives():
    print("Fig 6: YouTube Test — False Positive Rates")
    data = _load_comparison_data()

    not_drones = [r for r in data if r['expected'] == 'not_drone']
    # Sort by max FP rate descending
    not_drones.sort(key=lambda r: max(r['v8_drone_pct'], r['v9_drone_pct']), reverse=True)

    labels = [r['description'] for r in not_drones]
    v8_fp = [r['v8_drone_pct'] for r in not_drones]
    v9_fp = [r['v9_drone_pct'] for r in not_drones]
    counts = [r['total_segments'] for r in not_drones]
    cats = [r['category'] for r in not_drones]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    y = np.arange(len(labels))
    h = 0.35

    bars1 = ax.barh(y - h/2, v8_fp, h, label='Model v8', color=C_V8, edgecolor='white')
    bars2 = ax.barh(y + h/2, v9_fp, h, label='Model v9', color=C_V9, edgecolor='white')

    for i in range(len(labels)):
        max_val = max(v8_fp[i], v9_fp[i])
        # n= label: offset from the largest bar + space for % text
        if i < 6:
            n_offset = max(max_val + 2, 1.5)
        else:
            n_offset = max(max_val + 4, 3)
        ax.text(n_offset, y[i],
                f'n={counts[i]}', va='center', fontsize=8, color='#555555')
        # v8: % inside bar if wide, or text outside if narrow/zero
        if v8_fp[i] > 5:
            ax.text(v8_fp[i] - 0.5, y[i] - h/2,
                    f'{v8_fp[i]:.1f}%', va='center', ha='right', fontsize=7,
                    color='white', fontweight='bold')
        else:
            ax.text(v8_fp[i] + 0.5, y[i] - h/2,
                    f'{v8_fp[i]:.1f}%', va='center', ha='left', fontsize=7,
                    color=C_V8)
        # v9: same logic
        if v9_fp[i] > 5:
            ax.text(v9_fp[i] - 0.5, y[i] + h/2,
                    f'{v9_fp[i]:.1f}%', va='center', ha='right', fontsize=7,
                    color='white', fontweight='bold')
        else:
            ax.text(v9_fp[i] + 0.5, y[i] + h/2,
                    f'{v9_fp[i]:.1f}%', va='center', ha='left', fontsize=7,
                    color=C_V9)

    ax.set_yticks(y)
    ax.set_yticklabels([f'{l}\n({c})' for l, c in zip(labels, cats)], fontsize=8)
    ax.set_xlabel('False Positive Rate — Segments Classified as Drone (%)')
    ax.set_title('YouTube Test — False Positives on Non-Drone Audio')
    ax.set_xlim(0, max(max(v8_fp), max(v9_fp)) * 1.6)
    ax.invert_yaxis()
    ax.legend(loc='lower right')

    save_figure(fig, 'fig6_youtube_false_positives')


# ═══════════════════════════════════════════════════════════════════════════
# Fig 7: Confusion Matrices
# ═══════════════════════════════════════════════════════════════════════════
def fig7_confusion_matrices():
    print("Fig 7: Confusion Matrices")
    # v8: TP=1565, FN=240, FP=142, TN=3151
    cm_v8 = np.array([[1565, 240], [142, 3151]])
    # v9: TP=2737, FN=326, FP=166, TN=3071
    cm_v9 = np.array([[2737, 326], [166, 3071]])

    labels = ['Drone', 'Not Drone']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))

    for ax, cm, title in [(ax1, cm_v8, 'Model v8 (n=5,098)'),
                           (ax2, cm_v9, 'Model v9 (n=6,300)')]:
        totals = cm.sum(axis=1, keepdims=True)
        pcts = cm / totals * 100

        im = ax.imshow(cm, cmap='Blues', aspect='auto')

        for i in range(2):
            for j in range(2):
                color = 'white' if cm[i, j] > cm.max() * 0.5 else 'black'
                ax.text(j, i, f'{cm[i, j]}\n({pcts[i, j]:.1f}%)',
                        ha='center', va='center', fontsize=11,
                        fontweight='bold', color=color)

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title(title)
        # Re-enable spines for matrix
        for spine in ax.spines.values():
            spine.set_visible(True)

    save_figure(fig, 'fig7_confusion_matrices')


# ═══════════════════════════════════════════════════════════════════════════
# Fig 8: Robustness — Clean vs Augmented
# ═══════════════════════════════════════════════════════════════════════════
def fig8_robustness():
    print("Fig 8: Robustness — Clean vs Augmented")
    metrics = ['Accuracy', 'Drone\nRecall', 'Specificity']
    v8_clean = [92.5, 86.7, 95.7]
    v8_aug = [89.7, 83.6, 93.1]
    v9_clean = [92.2, 89.4, 94.9]
    v9_aug = [86.2, 89.7, 82.9]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    x = np.arange(len(metrics))
    w = 0.32

    for ax, clean, aug, title, color in [
        (ax1, v8_clean, v8_aug, 'Model v8', C_V8),
        (ax2, v9_clean, v9_aug, 'Model v9', C_V9),
    ]:
        b1 = ax.bar(x - w/2, clean, w, label='Clean', color=C_CLEAN, edgecolor='white')
        b2 = ax.bar(x + w/2, aug, w, label='Augmented', color=C_AUG, edgecolor='white', alpha=0.8)

        # Delta annotations
        for i in range(len(metrics)):
            delta = aug[i] - clean[i]
            sign = '+' if delta >= 0 else ''
            mid_x = x[i]
            top_y = max(clean[i], aug[i]) + 1.5
            ax.text(mid_x, top_y, f'{sign}{delta:.1f}%',
                    ha='center', va='bottom', fontsize=8, color='#555555')

        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.set_ylabel('Score (%)')
        ax.set_title(title)
        ax.set_ylim(70, 102)
        ax.legend(loc='lower left')

    fig.suptitle('Robustness: Clean vs. Augmented Validation Conditions', fontweight='bold', y=1.02)

    save_figure(fig, 'fig8_robustness')


# ═══════════════════════════════════════════════════════════════════════════
# Fig 9: Dataset Pipeline (block diagram)
# ═══════════════════════════════════════════════════════════════════════════
def fig9_dataset_pipeline():
    print("Fig 9: Dataset Pipeline (block diagram)")
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    stages = [
        ('Raw Audio\nFiles', '24,502', '#6baed6'),
        ('Cleaning', '21,646', '#4292c6'),
        ('1-sec\nSegments', '62,007', '#2171b5'),
        ('Balancing', '50,497', '#08519c'),
        ('Augmentation', '227,950', '#08306b'),
    ]
    val_stage = ('Validation\nSet', '5,098', '#e6550d')

    fig, ax = plt.subplots(figsize=(12, 4.0))
    ax.set_xlim(-0.5, 12.5)
    ax.set_ylim(-2.0, 2.5)
    ax.set_aspect('equal')
    ax.axis('off')

    box_w, box_h = 2.0, 1.4
    gap = 0.6
    x_start = 0.0
    main_y = 0.5  # raise main row to make room for validation below

    # Draw main pipeline boxes
    box_positions = []
    for i, (label, count, color) in enumerate(stages):
        x = x_start + i * (box_w + gap)
        y = main_y
        box = FancyBboxPatch((x, y), box_w, box_h,
                             boxstyle="round,pad=0.1",
                             facecolor=color, edgecolor='white',
                             linewidth=2, alpha=0.9)
        ax.add_patch(box)
        # Label
        ax.text(x + box_w/2, y + box_h/2 + 0.15, label,
                ha='center', va='center', fontsize=9,
                fontweight='bold', color='white')
        # Count
        ax.text(x + box_w/2, y + 0.2, count,
                ha='center', va='center', fontsize=8, color='#e0e0e0')
        box_positions.append((x, y))

        # Arrow between boxes
        if i < len(stages) - 1:
            ax.annotate('', xy=(x + box_w + gap * 0.15, y + box_h/2),
                        xytext=(x + box_w + 0.05, y + box_h/2),
                        arrowprops=dict(arrowstyle='->', color='#333333',
                                        lw=1.8))

    # Validation set branch (below, branching from stage 2 "1-sec Segments")
    branch_x = box_positions[2][0] + box_w / 2
    val_x = branch_x - box_w / 2
    val_y = main_y - box_h - 0.7  # place below main row with gap
    box = FancyBboxPatch((val_x, val_y), box_w, box_h,
                         boxstyle="round,pad=0.1",
                         facecolor=val_stage[2], edgecolor='white',
                         linewidth=2, alpha=0.9)
    ax.add_patch(box)
    ax.text(val_x + box_w/2, val_y + box_h/2 + 0.15, val_stage[0],
            ha='center', va='center', fontsize=9,
            fontweight='bold', color='white')
    ax.text(val_x + box_w/2, val_y + 0.2, val_stage[1],
            ha='center', va='center', fontsize=8, color='#e0e0e0')

    # Arrow from Segments down to Validation
    ax.annotate('', xy=(branch_x, val_y + box_h),
                xytext=(branch_x, main_y),
                arrowprops=dict(arrowstyle='->', color='#333333', lw=1.8))

    # Label on branch arrow
    mid_y = (main_y + val_y + box_h) / 2
    ax.text(branch_x + 0.15, mid_y, 'source-level\nsplit',
            ha='left', va='center', fontsize=7, color='#555555', style='italic')

    # Notes above main boxes
    notes = [
        '10+ sources',
        '−11.5%',
        '16 kHz mono',
        '1:1 ratio',
        '19 types, ×5',
    ]
    for i, note in enumerate(notes):
        x = box_positions[i][0] + box_w / 2
        ax.text(x, main_y + box_h + 0.25, note, ha='center', va='center',
                fontsize=7, color='#666666', style='italic')

    ax.set_title('Data Processing Pipeline', fontsize=12, fontweight='bold', pad=20)

    save_figure(fig, 'fig9_dataset_pipeline')


# ═══════════════════════════════════════════════════════════════════════════
# Fig 10: System Architecture
# ═══════════════════════════════════════════════════════════════════════════
def fig10_system_architecture():
    print("Fig 10: System Architecture")
    from matplotlib.patches import FancyBboxPatch

    fig, ax = plt.subplots(figsize=(12, 3.0))
    ax.set_xlim(-0.5, 14.5)
    ax.set_ylim(-0.5, 2.5)
    ax.set_aspect('equal')
    ax.axis('off')

    blocks = [
        ('Microphone\n16 kHz, mono', '#7fb3d8', 'Input'),
        ('1-sec\nSegmentation', '#5a9bd5', 'Preprocessing'),
        ('Mel-Spectrogram\n640×640 px', '#3a7cc0', 'Feature extraction'),
        ('YOLOv8n-cls\n1.44M params', '#1a5c9e', 'Classification'),
        ('Temporal\nSmoothing', '#08407a', 'Post-processing'),
        ('Drone /\nNot Drone', '#2ca02c', 'Output'),
    ]

    box_w, box_h = 2.0, 1.4
    gap = 0.5
    x_start = 0.0

    for i, (label, color, subtitle) in enumerate(blocks):
        x = x_start + i * (box_w + gap)
        y = 0.5

        box = FancyBboxPatch((x, y), box_w, box_h,
                             boxstyle="round,pad=0.1",
                             facecolor=color, edgecolor='white',
                             linewidth=2)
        ax.add_patch(box)

        # Main label
        ax.text(x + box_w/2, y + box_h/2 + 0.1, label,
                ha='center', va='center', fontsize=9,
                fontweight='bold', color='white')

        # Subtitle below box
        ax.text(x + box_w/2, y - 0.15, subtitle,
                ha='center', va='top', fontsize=7, color='#666666',
                style='italic')

        # Arrow
        if i < len(blocks) - 1:
            ax.annotate('', xy=(x + box_w + gap * 0.15, y + box_h/2),
                        xytext=(x + box_w + 0.05, y + box_h/2),
                        arrowprops=dict(arrowstyle='->', color='#333333',
                                        lw=1.8))

    ax.set_title('System Architecture: Audio-Based Drone Detection Pipeline',
                 fontsize=12, fontweight='bold', pad=15)

    save_figure(fig, 'fig10_system_architecture')


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════
def main():
    setup_style()
    print(f"\nGenerating figures → {OUTPUT_DIR}/\n")

    fig1_overall_results()
    fig2_training_curves()
    fig3_distance_detection()
    fig4_youtube_per_category()
    fig5_youtube_per_file()
    fig6_youtube_false_positives()
    fig7_confusion_matrices()
    fig8_robustness()
    fig9_dataset_pipeline()
    fig10_system_architecture()

    n_files = len(list(OUTPUT_DIR.glob('*')))
    print(f"\nDone! {n_files} files saved to {OUTPUT_DIR}/")


if __name__ == '__main__':
    main()
