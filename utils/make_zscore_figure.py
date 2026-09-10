"""
Combined Z-score figure: original Beyond-the-Hype (BtH) models + new 4096-feature experiments.

Each model is represented by its COMBINED score = mean of global and per-target value,
for MCC & BEDROC on the Random (R) and Temporal (T) splits.
  - Old BtH models: values are already combined (taken as-is).
  - New models: combined = (global + per-target) / 2.
Per metric, a z-score is computed jointly across ALL models (so it sums to zero);
each model's Combined Z is the mean of its four metric z-scores, with SEM = std/sqrt(4).
"""
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from utils.make_zscore_table import load_new, load_old

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT = os.path.join(ROOT, "output",
                           "zscore_combined_original_BtH_plus_new.png")

# The figure is drawn at the manuscript's text width and included at
# \linewidth, so this is the printed font size, a little below the 11 pt
# body text.
TEXT_WIDTH = 455.24 / 72.27    # inches
FONT_SIZE = 9

FIGURE_LABELS = {
    "RF (4096)": "RF (4096 features)",
    "DNN (4096)": "DNN (4096 features)",
}

def make_zscore_figure():
    # Obtain z-score table from the summary first
    orig = load_old()
    new = {FIGURE_LABELS.get(name, name): values for name, values in load_new().items()}

    ### JOINT Z-SCORE OVER ALL MODELS ###
    names  = list(orig) + list(new)
    is_new = [False] * len(orig) + [True] * len(new)
    M = np.array([{**orig, **new}[n] for n in names], float)

    Z    = (M - M.mean(0)) / M.std(0, ddof=1)     # per-metric z-score across all models
    comb = Z.mean(1)                              # combined z-score per model
    sem  = Z.std(1, ddof=1) / np.sqrt(Z.shape[1]) # SEM over the 4 metrics

    # sort ascending for the horizontal bar chart
    order   = np.argsort(comb)
    names_s = [names[i] for i in order]
    comb_s  = comb[order]
    sem_s   = sem[order]
    new_s   = [is_new[i] for i in order]

    ### FIGURE ###
    COL_ORIG, COL_NEW = "#5B8FB9", "#E07B39"
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH, 5.0), layout="constrained")

    y = np.arange(len(names_s))
    ax.barh(
        y, comb_s, xerr=sem_s,
        color=[COL_NEW if n else COL_ORIG for n in new_s],
        edgecolor="black", linewidth=0.6,
        error_kw=dict(ecolor="0.3", capsize=3, lw=1),
    )
    ax.set_yticks(y)
    ax.set_yticklabels(names_s)
    ax.tick_params(labelsize=FONT_SIZE)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Combined Z-score (mean of Z over combined\n"
                  "MCC & BEDROC, Random & Temporal)",
                  fontsize=FONT_SIZE)
    # Centred on the figure rather than the axes: the long model names push the
    # axes right, and the title would run off the page.
    fig.suptitle(
        "Z-score ranking — original BtH models + new 4096-feature experiments\n"
        "(combined = mean of global & per-target; Z computed jointly, sums to zero)",
        fontsize=FONT_SIZE,
    )
    ax.legend(
        handles=[
            Patch(facecolor=COL_ORIG, edgecolor="black", label="Original BtH models"),
            Patch(facecolor=COL_NEW,  edgecolor="black", label="New experiments (4096 features)"),
        ],
        loc="upper left", frameon=True, fontsize=FONT_SIZE,
    )
    ax.grid(axis="x", ls=":", alpha=0.5)

    plt.savefig(OUTPUT, dpi=600)
    plt.close(fig)
    print("saved", OUTPUT)
