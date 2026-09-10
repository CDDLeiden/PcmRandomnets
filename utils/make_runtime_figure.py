"""
Runtime figure (Original BtH / CHEMBL35).

Reads the runtimes from output/results_randomnets_beyond_the_hype.xlsx (Sheet1)
and produces one grouped bar chart with a row per dataset: A) Original BtH,
B) ChEMBL 35. Each row has two panels (Random / Temporal split); within each
panel the 256-feature group has 3 bars (RF, DNN, RN 1-layer) and the
4096-feature group has 4 bars (+ RN 2-layers).
"""
import matplotlib
import numpy as np
import openpyxl

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

MODELS = ["RandomForest", "DNN", "Randomnets (1 layer)", "Randomnets (2 layers)"]
LABELS = {
    "RandomForest": "RF",
    "DNN": "DNN",
    "Randomnets (1 layer)": "RN (1 layer)",
    "Randomnets (2 layers)": "RN (2 layers)",
}
COLORS = {
    "RandomForest": "#5B8FB9",
    "DNN": "#E07B39",
    "Randomnets (1 layer)": "#6FB07F",
    "Randomnets (2 layers)": "#B96FB0",
}
DATASETS = {"Original BTH": "Original BtH", "CHEMBL35": "ChEMBL 35"}
SPLITS = ["Random", "Temporal"]
# The figure is drawn at the manuscript's text width and included at
# \linewidth, so this is the printed font size, a little below the 11 pt
# body text.
TEXT_WIDTH = 455.24 / 72.27    # inches
FONT_SIZE = 9

def parse(path):
    ws = openpyxl.load_workbook(path, data_only=True)["Sheet1"]
    rows = [r for r in ws.iter_rows(values_only=True)]
    data = {}          # dataset -> split -> fp_size -> algo -> runtime (s)
    dataset = split = fp = None
    for r in rows:
        first = r[0]
        if first in ("Original BTH", "CHEMBL35"):
            dataset = first
            data[dataset] = {}
        elif first in ("Random", "Temporal"):
            split = first
            data[dataset][split] = {}
        elif first in (256, 4096):
            fp = first
            data[dataset][split][fp] = {}
        elif first in MODELS:
            val = r[6]
            data[dataset][split][fp][first] = None if val == "N/A" else float(val)
    return data

def plot_split(ax, split, splitdata):
    group_gap = 3.2         # distance between the 256 and 4096 group centers
    bar_w = 0.7

    centers = {256: 0.0, 4096: group_gap}
    for fp in (256, 4096):
        present = [a for a in MODELS if splitdata[fp].get(a) is not None]
        n = len(present)
        offsets = (np.arange(n) - (n - 1) / 2) * bar_w
        for a, off in zip(present, offsets):
            hrs = splitdata[fp][a] / 3600.0
            x = centers[fp] + off
            ax.bar(x, hrs, width=bar_w * 0.9, color=COLORS[a],
                   edgecolor="black", linewidth=0.5)
            ax.text(x, hrs, f"{hrs:.1f}", ha="center", va="bottom", fontsize=FONT_SIZE)
    ax.set_xticks(list(centers.values()))
    ax.set_xticklabels([f"{fp} bits" for fp in centers])
    ax.tick_params(labelsize=FONT_SIZE)
    ax.set_title(f"{split} split", fontsize=FONT_SIZE)
    ax.grid(axis="y", ls=":", alpha=0.5)
    ax.margins(x=0.05)

def plot_dataset(row, letter, name, dsdata):
    axes = row.subplots(1, 2, sharey=True)
    cap = max(
        v / 3600.0
        for split in SPLITS
        for fp in (256, 4096)
        for v in dsdata[split][fp].values()
        if v is not None
    ) * 1.08

    for ax, split in zip(axes, SPLITS):
        plot_split(ax, split, dsdata[split])
    axes[0].set_ylim(0, cap * 1.18)
    axes[0].set_ylabel("Runtime (hours)", fontsize=FONT_SIZE)
    row.suptitle(f"{letter}) {name}", x=0.01, ha="left",
                 fontsize=FONT_SIZE, fontweight="bold")

def make_figures():
    data = parse("output/results_randomnets_beyond_the_hype.xlsx")
    out = "output/runtime.png"

    fig = plt.figure(figsize=(TEXT_WIDTH, 5.6), layout="constrained")
    rows = fig.subfigures(len(DATASETS), 1)
    for row, letter, (dataset, name) in zip(rows, "AB", DATASETS.items()):
        plot_dataset(row, letter, name, data[dataset])

    handles = [Patch(facecolor=COLORS[a], edgecolor="black", label=LABELS[a]) for a in MODELS]
    fig.legend(handles=handles, loc="outside upper center", ncol=4, frameon=True,
               fontsize=FONT_SIZE)
    plt.savefig(out, dpi=600)
    print("saved", out)
    plt.close(fig)
