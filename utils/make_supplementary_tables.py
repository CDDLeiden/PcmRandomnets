#!/usr/bin/env python
"""Generate the supplementary per-model result tables (LaTeX).

Reads the two raw result files written by ``run_qsprpred_bth.py``:

    output/results.csv             - global metrics (one row per replicate)
    output/results_per_target.csv  - per-target metrics (one row per replicate)

With output:

    output/supplementary_tables.tex

The output is a LaTeX file with four tables: the 256- and 4096-bit Morgan
fingerprints, each split into a random and a temporal table, styled after Tables
S1/S2 of Lenselink et al. (Beyond the Hype). Each cell is the mean over the
replicate runs; the sample standard deviation is appended as ``mean $\\pm$ std``
unless it rounds to zero, in which case only the mean is shown (as for the
deterministic RandomForest). A model/dataset/split cell with no matching run is
written as ``--``. Not every model is evaluated at 256 bits, so the 256-bit
tables carry only the rows that were run there.
"""

import csv
import json
import os
import statistics
from collections import defaultdict

# Shared with the main-text summary so both describe the same experiments.
from utils.make_bth_summary import (
    PARAMS_DNN_EPOCHS,
    PARAMS_PHYSCHEM,
    PARAMS_RF_N_JOBS,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_CSV = os.path.join(ROOT, 'output', 'results.csv')
RESULTS_PER_TARGET_CSV = os.path.join(ROOT, 'output', 'results_per_target.csv')
OUTPUT = os.path.join(ROOT, 'output', 'supplementary_tables.tex')

### ROW LAYOUT ###
# Datasets and splits in display order (csv value -> display label).
DATASETS = [('original_bth', 'Original BtH'), ('chembl35', 'ChEMBL 35')]

# (label, kind, layers, n_nns, n_epochs, fps) in display order within a dataset
# block.
MODEL_ROWS = [
    ('RF', 'rf', None, None, None, (256, 4096)),
    ('DNN', 'dnn', None, None, None, (256, 4096)),
    ('RN (1 layer, 25 epochs)', 'rn', (2048,), 25, 25, (256, 4096)),
    ('RN (1 layer, 50 epochs)', 'rn', (2048,), 25, 50, (256, 4096)),
    ('RN (2 layers, 25 epochs)', 'rn', (2048, 2048), 25, 25, (4096,)),
    ('RN (2 layers, 50 epochs)', 'rn', (2048, 2048), 25, 50, (4096,)),
    ('RN (2 layers, 4000 units, 25 epochs)', 'rn', (4000, 4000), 25, 25, (4096,)),
    ('RN (1 layer, no ensemble, 25 epochs)', 'rn', (2048,), 1, 25, (4096,)),
    ('RN (2 layers, no ensemble, 25 epochs)', 'rn', (2048, 2048), 1, 25, (4096,)),
    ('RN (2 layers, no ensemble, 500 epochs, batch 256)', 'rn', (2048, 2048), 1, 500, (4096,)),
]

# (fp_size, split csv value, random-table label, table label) in output order.
TABLES = [
    (256, '70-30', 'tab:si_256_random', 'tab:si_256_random'),
    (256, 'temporal', 'tab:si_256_random', 'tab:si_256_temporal'),
    (4096, '70-30', 'tab:si_4096_random', 'tab:si_4096_random'),
    (4096, 'temporal', 'tab:si_4096_random', 'tab:si_4096_temporal'),
]

# Extra caption sentence(s) per fingerprint size (random table only).
FP_NOTE = {
    256: (
        r'Only the models evaluated at this fingerprint size are shown; the '
        r'two-layer, wide (4000-unit) and ensembling-off RandomNets variants '
        r'were run only at 4096 bits. '
    ),
    4096: (
        r'All RandomNets are trained with a batch size of 8; the 500-epoch '
        r'control instead uses 256, which is the reason that longer schedule '
        r'is tractable. '
    ),
}

def _caption(fp_size, split_key, random_label):
    """Build the caption for one table."""
    bits = f'{fp_size}-bit'
    if split_key == '70-30':
        return (
            f'Results on the random split (70--30) for the {bits} Morgan '
            r'fingerprints. Shown is the mean Matthews correlation coefficient '
            r'(MCC) and BEDROC ($\alpha{=}20$) computed per target, alongside '
            r'the same metrics computed on the merged (global) predictions. RN '
            r'denotes RandomNets; hidden layers have 2048 units and implicit '
            r'ensembling (\texttt{n\_nns}${=}$25) is used unless stated '
            r'otherwise. RF and DNN are the reference models from the main '
            r'text. ' + FP_NOTE[fp_size] +
            r'Values are the mean $\pm$ standard deviation over three replicate '
            r'runs; where the standard deviation is zero only the mean is shown.'
        )
    return (
        f'Results on the temporal split for the {bits} Morgan fingerprints. '
        r'Columns, models and formatting are as in Table~\ref{' + random_label + '}.'
    )

### RAW FILE PARSING ###
def _kind(model_id):
    """Map a model id to a short kind, or None if unrecognised."""
    if model_id.startswith('Randomforest'):
        return 'rf'
    if model_id.startswith('DNN'):
        return 'dnn'
    if model_id.startswith('Randomnets'):
        return 'rn'
    return None

def _key(model_id, params):
    """Full variant key: (kind, layers, n_nns, n_epochs)."""
    kind = _kind(model_id)
    if kind is None:
        return None
    if params.get('physchem', PARAMS_PHYSCHEM) != PARAMS_PHYSCHEM:
        return None
    if kind == 'dnn':
        if params.get('n_epochs') != PARAMS_DNN_EPOCHS:
            return None
        return (kind, None, None, None)
    if kind == 'rf':
        if params.get('n_jobs', PARAMS_RF_N_JOBS) != PARAMS_RF_N_JOBS:
            return None
        return (kind, None, None, None)
    return (kind, tuple(params.get('layers', [])),
            params.get('n_nns'), params.get('n_epochs'))

def _collect(path, metric_cols):
    """Return {(key, dataset, split, fp_size): {metric: [values over reps]}}.

    ``metric_cols`` maps output metric name to the csv column to read.
    """
    samples = defaultdict(lambda: defaultdict(list))
    with open(path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            params = json.loads(row['params'])
            key = _key(row['id'], params)
            if key is None:
                continue
            cell = (key, row['dataset'], params.get('split'), int(row['fp_size']))
            for name, col in metric_cols.items():
                samples[cell][name].append(float(row[col]))
    return samples

### CELL FORMATTING ###
def _fmt(values):
    """Format a replicate list as 'mean' or 'mean $\\pm$ std' (3 decimals).

    The sample standard deviation is shown only when it rounds to a non-zero
    value at 3 decimals.
    """
    if not values:
        return '--'
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    if f'{std:.3f}' == '0.000':
        return f'{mean:.3f}'
    return f'{mean:.3f} $\\pm$ {std:.3f}'

def _cells(spec, dataset, split, fp_size, global_s, target_s):
    """Return the four formatted metric cells for one model row."""
    _, kind, layers, n_nns, n_epochs, _fps = spec
    key = (kind, layers, n_nns, n_epochs)
    g = global_s.get((key, dataset, split, fp_size), {})
    t = target_s.get((key, dataset, split, fp_size), {})
    return [_fmt(g.get('mcc')), _fmt(g.get('bedroc')),
            _fmt(t.get('mcc')), _fmt(t.get('bedroc'))]

### LATEX EMISSION ###
def _table(fp_size, split_key, caption, label, global_s, target_s):
    """Build one LaTeX table environment for a given fingerprint size and split."""
    rows = [s for s in MODEL_ROWS if fp_size in s[5]]
    lines = [
        r'\begin{table}[ht]',
        r'\centering',
        r'\caption{' + caption + '}',
        r'\label{' + label + '}',
        r'\begin{tabular}{@{}>{\raggedright\arraybackslash}p{4.6cm} cc cc@{}}',
        r'\toprule',
        r' & \multicolumn{2}{c}{Global} & \multicolumn{2}{c}{Per target} \\',
        r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}',
        r'Model & MCC & BEDROC & MCC & BEDROC \\',
        r'\midrule',
    ]
    for ds_i, (ds_key, ds_label) in enumerate(DATASETS):
        if ds_i:
            lines.append(r'\cmidrule(l){1-5}')
        lines.append(r'\multicolumn{5}{@{}l}{\textit{' + ds_label + r'}}\\')
        for spec in rows:
            cells = _cells(spec, ds_key, split_key, fp_size, global_s, target_s)
            lines.append(f'{spec[0]} & ' + ' & '.join(cells) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    return '\n'.join(lines)

def _report_counts(global_s):
    """Print the replicate count behind each populated cell so short cells show."""
    print('Replicate counts per populated cell (model / dataset / split / fp):')
    labels = {(s[1], s[2], s[3], s[4]): s[0] for s in MODEL_ROWS}
    for (key, dataset, split, fp), metrics in sorted(global_s.items(), key=lambda x: str(x[0])):
        name = labels.get(key)
        if name is None:
            continue
        print(f'  {name:40s} {dataset:12s} {split:8s} {fp:>5} n={len(metrics["mcc"])}')

def make_supplementary_tables():
    """Write (and print) the supplementary LaTeX tables from the raw results."""
    global_s = _collect(RESULTS_CSV, {'mcc': 'mcc', 'bedroc': 'bedroc'})
    target_s = _collect(RESULTS_PER_TARGET_CSV, {'mcc': 'mcc_avg', 'bedroc': 'bedroc_avg'})

    tables = '\n\n'.join(
        _table(fp_size, split_key, _caption(fp_size, split_key, random_label),
               label, global_s, target_s)
        for fp_size, split_key, random_label, label in TABLES
    )
    with open(OUTPUT, 'w', encoding='utf-8') as f:
        f.write(tables + '\n')

    _report_counts(global_s)
    print(f'Wrote {OUTPUT} (copy-paste into supplementary.tex)')
    return tables