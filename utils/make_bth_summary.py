#!/usr/bin/env python
"""Generate the human-readable Beyond-The-Hype summary spreadsheet.

Takes the two raw result files written by ``run_qsprpred_bth.py``:

    output/results.csv             - global metrics (one row per run)
    output/results_per_target.csv  - per-target metrics + runtime (one row per run)

and writes:

    output/results_randomnets_beyond_the_hype.csv

The Randomnets numbers are reported for 25 epochs only (the 50-epoch runs are
not reported). Any (model, dataset, split, fp_size) combination with no matching
run is written as N/A. Also skips the no-ensemble and rf scaling sweep runs.

Entrypoint: ``generate_summary()``
"""

import csv
import json
import os
import statistics
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_CSV = os.path.join(ROOT, 'output', 'results.csv')
RESULTS_PER_TARGET_CSV = os.path.join(ROOT, 'output', 'results_per_target.csv')
OUTPUT = os.path.join(ROOT, 'output', 'results_randomnets_beyond_the_hype.csv')

### RAW FILE PARSING ###
# Columns required to interpret a row. A file missing any of them cannot be
# read, so we say so rather than skipping rows.
RESULTS_COLUMNS = {'id', 'dataset', 'fp_size', 'params', 'mcc', 'bedroc'}
PER_TARGET_COLUMNS = {'id', 'dataset', 'fp_size', 'params', 'mcc_avg',
                      'bedroc_avg', 'runtime'}

# See docstring: only results with these params are relevant for the z-figures and mcc/bedroc figures
# Anything else is just for the supplementary materials
PARAMS_PHYSCHEM = 'full'
PARAMS_RN = {'n_epochs': 25, 'dim': 2048, 'n_nns': 25, 'mask_thr': 0.5, 'batch_size': 8}
PARAMS_DNN_EPOCHS = 2000
PARAMS_RF_N_JOBS = 32

def _load_json_params(raw):
    """Parse the JSON params field of a row."""
    try:
        params = json.loads(raw)
    except (json.JSONDecodeError, ValueError, TypeError):
        return None
    return params if isinstance(params, dict) else None

def _open_rows(path, required):
    """Yield rows from a headered result file, checking the schema up front."""
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise SystemExit(
                f'{path}: missing column(s) {sorted(missing)}. '
                f'Regenerate it with run_qsprpred_bth.py.'
            )
        yield from reader

def _aggregate(samples, skipped, label):
    """Average each metric over replicates and report what was skipped."""
    out = {}
    for key, metrics in samples.items():
        out[key] = {name: statistics.mean(values)
                    for name, values in metrics.items()}
    counts = sorted({len(next(iter(m.values()))) for m in samples.values()})
    print(f'{label}: {len(out)} cells from {sum(len(next(iter(m.values()))) for m in samples.values())} '
          f'rows (replicates per cell: {counts or [0]}), {skipped} row(s) skipped')
    return out

def model_kind(model_id):
    """Map a model id (e.g. 'RandomnetsBthModel_0') to a short kind."""
    if model_id.startswith('Randomforest'):
        return 'rf'
    if model_id.startswith('DNN'):
        return 'dnn'
    if model_id.startswith('Randomnets'):
        return 'rn'
    return None

def parse_results(path):
    """Return {key: {'mcc', 'bedroc'}} from the global results file."""
    samples = defaultdict(lambda: defaultdict(list))
    skipped = 0
    for row in _open_rows(path, RESULTS_COLUMNS):
        params = _load_json_params(row['params'])
        key = _make_key(row['id'], row['dataset'], int(row['fp_size']), params) \
            if params is not None else None
        if key is None:
            skipped += 1
            continue
        samples[key]['mcc'].append(float(row['mcc']))
        samples[key]['bedroc'].append(float(row['bedroc']))
    return _aggregate(samples, skipped, os.path.basename(path))

def parse_per_target(path):
    """Return {key: {'mcc', 'bedroc', 'runtime'}} from the per-target file."""
    samples = defaultdict(lambda: defaultdict(list))
    skipped = 0
    for row in _open_rows(path, PER_TARGET_COLUMNS):
        params = _load_json_params(row['params'])
        key = _make_key(row['id'], row['dataset'], int(row['fp_size']), params) \
            if params is not None else None
        # bedroc_avg is empty for classification runs, which the summary layout
        # has no cell for.
        if key is None or not row['bedroc_avg']:
            skipped += 1
            continue
        samples[key]['mcc'].append(float(row['mcc_avg']))
        samples[key]['bedroc'].append(float(row['bedroc_avg']))
        samples[key]['runtime'].append(float(row['runtime']))
    out = _aggregate(samples, skipped, os.path.basename(path))
    for metrics in out.values():
        metrics['runtime'] = int(round(metrics['runtime']))
    return out

def _make_key(model_id, dataset, fp_size, params):
    """Key-value pairs to filter out any rows not covered by the configurations above"""
    kind = model_kind(model_id)
    if kind is None:
        return None
    if params.get('physchem', PARAMS_PHYSCHEM) != PARAMS_PHYSCHEM:
        return None
    split = params.get('split')
    if kind == 'rn':
        layers = params.get('layers') or []
        if (params.get('n_epochs') != PARAMS_RN['n_epochs']
                or not layers or layers[0] != PARAMS_RN['dim']
                or params.get('n_nns', PARAMS_RN['n_nns']) != PARAMS_RN['n_nns']
                or params.get('mask_thr', PARAMS_RN['mask_thr']) != PARAMS_RN['mask_thr']
                or (params.get('batch_size') or PARAMS_RN['batch_size'])
                != PARAMS_RN['batch_size']):
            return None
        n_layers = len(layers)
    elif kind == 'dnn':
        if params.get('n_epochs') != PARAMS_DNN_EPOCHS:
            return None
        n_layers = 0
    else:
        if params.get('n_jobs', PARAMS_RF_N_JOBS) != PARAMS_RF_N_JOBS:
            return None
        n_layers = 0
    return (kind, dataset, fp_size, split, n_layers)

### OUTPUT LAYOUT ###
DATASETS = [('original_bth', 'Original BTH'), ('chembl35', 'CHEMBL35')]
SPLITS = [('70-30', 'Random'), ('temporal', 'Temporal')]
FP_SIZES = [256, 4096]

# (label, kind, n_layers) in display order within each fingerprint-size block.
MODEL_ROWS = [
    ('RandomForest', 'rf', 0),
    ('DNN', 'dnn', 0),
    ('Randomnets (1 layer)', 'rn', 1),
    ('Randomnets (2 layers)', 'rn', 2),
]

HEADER_CELLS = ['MCC Global', 'MCC per target', 'BEDROC Global',
                'BEDROC per target', 'Runtime']

def fmt_metric(value):
    """Format a metric to 3 decimals with the leading zero stripped (.613)."""
    s = f'{value:.3f}'
    if s.startswith('0.'):
        return s[1:]
    if s.startswith('-0.'):
        return '-' + s[2:]
    return s

def build_rows(global_metrics, per_target_metrics):
    """Build the full list of output rows (each a list of 7 string cells)."""
    rows = []
    first_block = True
    for ds_key, ds_label in DATASETS:
        for split_idx, (split_key, split_label) in enumerate(SPLITS):
            if not first_block:
                rows.append(['', '', '', '', '', '', ''])  # blank separator
            first_block = False
            if split_idx == 0:
                rows.append([ds_label, '', '', '', '', '', ''])  # dataset header
            rows.append([split_label, ''] + HEADER_CELLS)        # split header
            for fp_size in FP_SIZES:
                rows.append([str(fp_size), '', '', '', '', '', ''])  # fp header
                for label, kind, n_layers in MODEL_ROWS:
                    key = (kind, ds_key, fp_size, split_key, n_layers)
                    g = global_metrics.get(key)
                    pt = per_target_metrics.get(key)
                    cells = [
                        fmt_metric(g['mcc']) if g else 'N/A',
                        fmt_metric(pt['mcc']) if pt else 'N/A',
                        fmt_metric(g['bedroc']) if g else 'N/A',
                        fmt_metric(pt['bedroc']) if pt else 'N/A',
                        str(pt['runtime']) if pt else 'N/A',
                    ]
                    rows.append([label, ''] + cells)
    return rows

def write_csv(rows, path):
    """Write rows as a semicolon-delimited, CRLF file (Excel-import style)."""
    with open(path, 'w', newline='', encoding='utf-8') as f:
        for row in rows:
            f.write(';'.join(row) + '\r\n')

def generate_summary():
    """Build the summary spreadsheet from the raw result files."""
    global_metrics = parse_results(RESULTS_CSV)
    per_target_metrics = parse_per_target(RESULTS_PER_TARGET_CSV)
    rows = build_rows(global_metrics, per_target_metrics)
    write_csv(rows, OUTPUT)
    print(f'Wrote {OUTPUT} ({len(rows)} rows)')
