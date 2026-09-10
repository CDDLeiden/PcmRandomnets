#!/usr/bin/env python
"""Regenerate every figure, table and reported number in the PcmRandomnets paper.

Run once the model runs have finished (run_qsprpred_bth.py has written
output/results.csv and output/results_per_target.csv):

    python analyze_results.py

Every figure and table lands in output/. The target-class / Papyrus steps print
their numbers to output/paper_numbers.log.
"""
import contextlib
import io
import os
import openpyxl

from utils.make_bth_summary import generate_summary
from utils.make_zscore_table import main as build_zscore_table
from utils.make_zscore_figure import make_zscore_figure
from utils.make_mcc_bedroc_figure import make_figures as make_mcc_bedroc
from utils.make_supplementary_tables import make_supplementary_tables
from utils.make_rf_runtime_scaling_figure import make_rf_runtime_scaling_figure
from utils.make_runtime_figure import make_figures as make_runtime
from utils.make_umap_figure import make_umap_figures
from utils.make_split_similarity import report as split_similarity
from utils.make_target_class_distribution import main as chembl35_classes
from utils.make_bth_target_classes import classify
from utils.make_papyrus_composition import analyze

ROOT = os.path.dirname(os.path.abspath(__file__))
SUMMARY_CSV = os.path.join(ROOT, "output", "results_randomnets_beyond_the_hype.csv")
SUMMARY_XLSX = os.path.join(ROOT, "output", "results_randomnets_beyond_the_hype.xlsx")
NUMBERS_LOG = os.path.join(ROOT, "output", "paper_numbers.log")

def _read_cell(text):
    """Reproduce Excel's import coercion: int, then float, then string.

    make_bth_summary writes fingerprint sizes as ints, metrics with the leading
    zero stripped (".613") and "N/A" for missing cells; the xlsx-reading scripts
    test cells with e.g. ``first in (256, 4096)`` and ``float(val)``, so the
    types must be restored.
    """
    if text == "":
        return None
    if text == "N/A":
        return "N/A"
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def sync_summary_xlsx():
    """Write the summary CSV into Sheet1 of the summary xlsx (the table/figure
    scripts read the xlsx, not the CSV). Only Sheet1 is touched; the derived
    sheets (Z-scores, MCC_BEDROC, Combined Z) are left as-is.
    """

    with open(SUMMARY_CSV, newline="", encoding="utf-8") as f:
        rows = [line.rstrip("\r\n").split(";") for line in f]
    wb = openpyxl.load_workbook(SUMMARY_XLSX)
    ws = wb["Sheet1"]
    for r, row in enumerate(rows, start=1):
        for c, cell in enumerate(row, start=1):
            ws.cell(row=r, column=c).value = _read_cell(cell)
    wb.save(SUMMARY_XLSX)

def _numbers(title, fn):
    """Run a reported-numbers step: print the heading, echo the step's output to
    the terminal, and append (heading + output) to output/paper_numbers.log.
    """
    print(title)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        fn()
    out = buf.getvalue()
    print(out, end="")
    with open(NUMBERS_LOG, "a", encoding="utf-8") as log:
        log.write(f"\n{'='*70}\n# {title}\n{'='*70}\n{out}")

def main():
    os.chdir(ROOT)
    open(NUMBERS_LOG, "w").close()  # fresh log each run

    print("Building the results summary CSV from the raw run outputs")
    generate_summary()

    # Excel for manual inspection of the results so far (also output is read in subsequent steps)
    print("Syncing the summary into Sheet1 of the xlsx")
    sync_summary_xlsx()

    print("Table 1: z-score table -> output/zscore_table.tex")
    build_zscore_table()

    print("Figure: z-score bar chart -> output/zscore_combined_original_BtH_plus_new.png")
    make_zscore_figure()

    print("Figure: MCC/BEDROC bars -> output/mcc_bedroc_*.png")
    make_mcc_bedroc()

    print("Supplementary per-model tables -> output/supplementary_tables.tex")
    make_supplementary_tables()

    print("Supplementary figure: RF runtime scaling -> output/rf_runtime_scaling.png")
    make_rf_runtime_scaling_figure()

    print("Figure: runtime bars -> output/runtime.png")
    make_runtime()

    print("Figure: UMAP + train/test overlap % -> output/umap.png")
    make_umap_figures()

    _numbers("Numbers: train/test nearest-neighbour Tanimoto similarity", split_similarity)
    _numbers("Numbers: ChEMBL 35 L1/L2 target-class distribution", chembl35_classes)
    _numbers("Numbers: original-BtH target classes", classify)
    _numbers("Numbers: Papyrus++ 05.7 composition", analyze)

if __name__ == "__main__":
    main()