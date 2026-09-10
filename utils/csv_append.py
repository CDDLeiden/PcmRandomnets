"""Append a row to a CSV file, writing the header first if the file is new.
Used to populate output/results.csv and output/results_per_target.csv from
run_qsprpred_bth.py.
"""

import csv
import os


def append_row(path, row):
    """Append the dict `row` to `path`, writing `row.keys()` as header if new."""
    is_new = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if is_new:
            writer.writeheader()
        writer.writerow(row)
