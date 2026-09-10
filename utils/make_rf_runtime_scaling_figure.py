#!/usr/bin/env python
"""RandomForest runtime-scaling figure (supplementary).

Reads output/results_per_target.csv and plots the RandomForest fit time against
the number of parallel jobs (\\texttt{n\\_jobs}) for the ChEMBL 35 / 4096-bit /
random-split setting, i.e. the sweep produced by config/runtime_scaling_rf.json.
Each point is the single timed run for that n_jobs value; a dashed line shows
ideal linear speed-up relative to the single-job run for reference.
"""

import csv
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_PER_TARGET_CSV = os.path.join(ROOT, "output", "results_per_target.csv")
OUTPUT = os.path.join(ROOT, "output", "rf_runtime_scaling.png")

DATASET = "chembl35"
FP_SIZE = 4096
SPLIT = "70-30"
RF_COLOR = "#5B8FB9"

def collect(path):
    """Return {n_jobs: runtime_seconds} for the RF n_jobs sweep."""
    runtimes = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if not row["id"].startswith("Randomforest"):
                continue
            if row["dataset"] != DATASET or int(row["fp_size"]) != FP_SIZE:
                continue
            if int(row["replicate"]) != 0:
                continue
            params = json.loads(row["params"])
            if params.get("split") != SPLIT or "n_jobs" not in params:
                continue
            runtimes[int(params["n_jobs"])] = float(row["runtime"])
    return runtimes

def make_rf_runtime_scaling_figure():
    """Plot RF fit time vs n_jobs; skip if the sweep has not been run."""
    runtimes = collect(RESULTS_PER_TARGET_CSV)
    if not runtimes or len(runtimes) != 5:
        print("no RandomForest n_jobs sweep rows found; skipping runtime-scaling figure")
        return

    jobs = sorted(runtimes)
    times = [runtimes[j] for j in jobs]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(jobs, times, marker="o", color=RF_COLOR, linewidth=1.5, label="Measured")
    # Ideal linear speed-up anchored at the smallest job count that was run.
    ideal = [times[0] * jobs[0] / j for j in jobs]
    ax.plot(jobs, ideal, ls="--", color="dimgray", linewidth=1.0,
            label="Ideal linear speed-up")

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(jobs)
    ax.set_xticklabels([str(j) for j in jobs])
    ax.set_xlabel("Number of parallel jobs (n_jobs)")
    ax.set_ylabel("Fit time (s)")
    ax.set_title("RandomForest runtime scaling\n(ChEMBL 35, 4096 bits, random split)")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend(frameon=True)

    fig.tight_layout()
    plt.savefig(OUTPUT, dpi=200)
    print(f"saved {OUTPUT}  (single run per n_jobs: {jobs})")
    plt.close(fig)
