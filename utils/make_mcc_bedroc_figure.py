"""
MCC/BEDROC figures per dataset (the paper's Figure with the paired blue/red
bars; the paper shows the CHEMBL35 one).

Reads output/results_randomnets_beyond_the_hype.xlsx (Sheet1) and, for each
dataset at 4096 features, stacks one panel per split (A: Random, B: Temporal),
each plotting one bar per model:

    MCC    = mean(global, per-target)   on the left axis
    BEDROC = mean(global, per-target)   on the right axis

with an error bar of stdev/sqrt(2) = |global - per-target| / 2.
"""
import math

import matplotlib
import numpy as np
import openpyxl

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import FormatStrFormatter

FP_SIZE = 4096
MODELS = ["RandomForest", "DNN", "Randomnets (1 layer)", "Randomnets (2 layers)"]
LABELS = {
    "RandomForest": "RF",
    "DNN": "DNN",
    "Randomnets (1 layer)": "RN (1 layer)",
    "Randomnets (2 layers)": "RN (2 layers)",
}
# Blue for MCC, red for BEDROC.
MCC_COLOR = "#4F81BD"
BEDROC_COLOR = "#C0504D"

DATASETS = {"Original BTH": "original_bth", "CHEMBL35": "chembl35"}
SPLITS = ["Random", "Temporal"]
STEP = 0.05          # axis limits are rounded outwards to a multiple of this
# The figure is drawn at the manuscript's text width and included at
# \linewidth, so this is the printed font size, a little below the 11 pt
# body text.
TEXT_WIDTH = 455.24 / 72.27    # inches
FONT_SIZE = 9

def parse(path="output/results_randomnets_beyond_the_hype.xlsx"):
    """dataset -> split -> model -> (mcc, mcc_err, bedroc, bedroc_err)."""
    ws = openpyxl.load_workbook(path, data_only=True)["Sheet1"]
    data = {}
    dataset = split = fp = None
    for r in ws.iter_rows(values_only=True):
        first = r[0]
        if first in DATASETS:
            dataset = first
            data[dataset] = {}
        elif first in SPLITS:
            split = first
            data[dataset][split] = {}
        elif first in (256, 4096):
            fp = first
        elif first in MODELS and fp == FP_SIZE:
            # C..F = MCC global, MCC per-target, BEDROC global, BEDROC per-target
            mcc_g, mcc_pt, bed_g, bed_pt = r[2:6]
            if "N/A" in (mcc_g, mcc_pt, bed_g, bed_pt):
                continue
            data[dataset][split][first] = (
                (mcc_g + mcc_pt) / 2, abs(mcc_g - mcc_pt) / 2,
                (bed_g + bed_pt) / 2, abs(bed_g - bed_pt) / 2,
            )
    return data

def axis_range(series, hard_max=None):
    """Round the (value ± error) span outwards to a multiple of STEP."""
    lo = min(v - e for v, e in series)
    hi = max(v + e for v, e in series)
    lo = math.floor(lo / STEP) * STEP
    hi = math.ceil(hi / STEP) * STEP
    if hard_max is not None:
        hi = min(hi, hard_max)
    return lo, hi

def split_ranges(data):
    """split -> (mcc_range, bedroc_range), shared across both datasets."""
    ranges = {}
    for split in SPLITS:
        cells = [v for ds in data.values() for v in ds[split].values()]
        ranges[split] = (
            axis_range([(m, me) for m, me, _, _ in cells]),
            axis_range([(b, be) for _, _, b, be in cells], hard_max=1.0),
        )
    return ranges

def plot_split(ax, splitdata, mcc_range, bedroc_range):
    present = [a for a in MODELS if a in splitdata]
    x = np.arange(len(present))
    width = 0.38

    ax2 = ax.twinx()

    ebar = dict(ecolor="dimgray", elinewidth=1.0, capsize=3)
    ax.bar(x - width / 2, [splitdata[a][0] for a in present], width,
           yerr=[splitdata[a][1] for a in present], color=MCC_COLOR,
           edgecolor="black", linewidth=0.6, error_kw=ebar)
    ax2.bar(x + width / 2, [splitdata[a][2] for a in present], width,
            yerr=[splitdata[a][3] for a in present], color=BEDROC_COLOR,
            edgecolor="black", linewidth=0.6, error_kw=ebar)

    ax.set_ylim(*mcc_range)
    ax2.set_ylim(*bedroc_range)
    ax.set_ylabel("MCC", fontsize=FONT_SIZE)
    ax2.set_ylabel("BEDROC", fontsize=FONT_SIZE)
    # Two decimals on every axis so the stacked panels line up.
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax2.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[a] for a in present])
    ax.tick_params(labelsize=FONT_SIZE)
    ax2.tick_params(labelsize=FONT_SIZE)
    ax.grid(axis="y", ls=":", alpha=0.5)
    ax.set_axisbelow(True)

def plot(dsdata, ranges, out):
    """One row per split: A) Random, B) Temporal."""
    fig = plt.figure(figsize=(TEXT_WIDTH, 5.4), layout="constrained")
    rows = fig.subfigures(len(SPLITS), 1)
    for row, letter, split in zip(rows, "AB", SPLITS):
        plot_split(row.subplots(), dsdata[split], *ranges[split])
        row.suptitle(f"{letter}) {split} split", x=0.01, ha="left",
                     fontsize=FONT_SIZE, fontweight="bold")

    handles = [Patch(facecolor=MCC_COLOR, edgecolor="black", label="MCC"),
               Patch(facecolor=BEDROC_COLOR, edgecolor="black", label="BEDROC")]
    fig.legend(handles=handles, loc="outside upper center", ncol=2, frameon=True,
               fontsize=FONT_SIZE)

    plt.savefig(out, dpi=600)
    print(f"saved {out}  " + "  ".join(
        f"{s}: MCC{ranges[s][0]} BEDROC{ranges[s][1]}" for s in SPLITS))
    plt.close(fig)

def make_figures():
    data = parse()
    ranges = split_ranges(data)
    for dataset, code in DATASETS.items():
        plot(data[dataset], ranges, f"output/mcc_bedroc_{code}_{FP_SIZE}.png")