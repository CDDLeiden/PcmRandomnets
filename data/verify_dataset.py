"""Verify that a prepared dataset directory can build without NaN values.
Any dataset with NaN values would either crash the model in the first epoch
or be caught by the model's sanity checks.

Data-preparation sanity check (not part of the manuscript pipeline). Run from
the repo root against a directory produced by the data/prepare_*.py scripts:

    python data/verify_dataset.py --data-dir data/chembl35
    python data/verify_dataset.py --data-dir data/original_bth --physchem original
"""
import argparse
import glob
import os
import re
import sys

import pandas as pd

PHYSCHEM_SETS = {'full': 'fps_cols_full.txt', 'original': 'fps_cols_original.txt'}

def verify(data_dir='./data/original_bth', physchem='full'):
    """Verify a prepared beyond_the_hype dataset directory. Returns True if OK."""
    DATA_DIR   = data_dir
    MAIN_CSV   = f'{DATA_DIR}/compounds_with_smiles_and_split2.csv'
    FPS_COLS_NAME = PHYSCHEM_SETS[physchem]
    FPS_COLS   = f'{DATA_DIR}/{FPS_COLS_NAME}'
    PDESC_FILE = f'{DATA_DIR}/Protein_Descriptors_50_sections.txt'
    CDESC_FILE = f'{DATA_DIR}/compound_additional_physchem_features.txt'

    ok = True

    def fail(msg):
        nonlocal ok
        print(f'  FAIL {msg}')
        ok = False

    def check_id_prefix(series, col, fname):
        ids = series.astype(str)
        if not ids.iloc[0].startswith('CHEMBL'):
            fail(f'{fname}: {col} lacks "CHEMBL" prefix (e.g. {ids.iloc[0]!r})')
            return False
        return True

    def check_header_alignment(path, sep='\t'):
        with open(path) as f:
            h = f.readline().rstrip('\n').split(sep)
            r = f.readline().rstrip('\n').split(sep)
        if len(h) == len(r):
            return True
        expanded = []
        for field in h:
            expanded.extend(p for p in re.split(r'\s+', field.strip()) if p)
        if len(expanded) == len(r):
            fail(f'{os.path.basename(path)}: header has {len(h)} tab fields but {len(r)} data fields '
                 f'— space-merged column names detected. Re-run prepare_data.py --force to fix.')
        else:
            fail(f'{os.path.basename(path)}: header has {len(h)} fields, data rows have {len(r)} '
                 f'— unrecoverable mismatch.')
        return False

    def check_nan_numeric(df, id_col, fname):
        desc_cols = [c for c in df.columns if c != id_col]
        numeric = df[desc_cols].apply(pd.to_numeric, errors='coerce')
        nan_mask = numeric.isna().any(axis=1)
        bad_ids = df.loc[nan_mask, id_col].tolist()
        if bad_ids:
            examples = ', '.join(str(x) for x in bad_ids[:5])
            suffix = f' ... and {len(bad_ids) - 5} more' if len(bad_ids) > 5 else ''
            fail(f'{fname}: {len(bad_ids)} row(s) contain non-numeric / NaN descriptor values '
                 f'— e.g. {examples}{suffix}')
            return False
        return True

    def coverage_check(label, missing, total):
        if missing:
            examples = sorted(missing)[:3]
            fail(f'{label}: {len(missing)}/{total} IDs have no descriptor match (e.g. {examples})')
            return False
        print(f'  OK   {label}: all {total} IDs covered')
        return True

    print(f'=== Verifying {DATA_DIR} (physchem: {physchem}) ===\n')

    # fps_cols selection list
    if not os.path.exists(FPS_COLS):
        fail(f'{FPS_COLS_NAME} not found in {DATA_DIR}')
        return False
    with open(FPS_COLS) as f:
        fps_cols = [l.strip() for l in f]
    print(f'{FPS_COLS_NAME}: {len(fps_cols)} feature columns')

    # Main CSV
    print(f'\nMain dataset ({os.path.basename(MAIN_CSV)}):')
    main_df = pd.read_csv(MAIN_CSV)
    print(f'  {len(main_df)} rows, {main_df["CMP_CHEMBL_ID"].nunique()} unique compounds, '
          f'{main_df["TGT_CHEMBL_ID"].nunique()} unique targets')
    check_id_prefix(main_df['CMP_CHEMBL_ID'], 'CMP_CHEMBL_ID', os.path.basename(MAIN_CSV))
    check_id_prefix(main_df['TGT_CHEMBL_ID'], 'TGT_CHEMBL_ID', os.path.basename(MAIN_CSV))

    # Fingerprint files (all *_fix.txt available)
    fps_files = sorted(glob.glob(f'{DATA_DIR}/compound_features_*_fix.txt'))
    if not fps_files:
        fail(f'No compound_features_*_fix.txt files found in {DATA_DIR}')
    else:
        for fps_path in fps_files:
            fps_name = os.path.basename(fps_path)
            print(f'\nFingerprints ({fps_name}):')
            with open(fps_path) as _f:
                _header = _f.readline()
            fps_sep = '\t' if _header.count('\t') > _header.count(',') else ','
            fps_df = pd.read_csv(fps_path, sep=fps_sep)
            fps_df['CMP_CHEMBL_ID'] = fps_df['CMP_CHEMBL_ID'].astype(str)
            print(f'  {len(fps_df)} rows, {fps_df["CMP_CHEMBL_ID"].nunique()} unique compounds, '
                  f'{len(fps_df.columns) - 1} bit columns')
            check_id_prefix(fps_df['CMP_CHEMBL_ID'], 'CMP_CHEMBL_ID', fps_name)
            main_cmp = set(main_df['CMP_CHEMBL_ID'].astype(str))
            fps_cmp  = set(fps_df['CMP_CHEMBL_ID'])
            coverage_check(f'fingerprints {fps_name} (by CMP_CHEMBL_ID)',
                           main_cmp - fps_cmp, len(main_cmp))

    # Protein descriptors
    print(f'\nProtein descriptors ({os.path.basename(PDESC_FILE)}):')
    if not check_header_alignment(PDESC_FILE):
        ok = False
    pdesc_df = pd.read_csv(PDESC_FILE, sep='\t')
    pdesc_df = pdesc_df[pdesc_df.columns.intersection(fps_cols + ['TGT_CHEMBL_ID'])]
    print(f'  {len(pdesc_df)} rows, {pdesc_df["TGT_CHEMBL_ID"].nunique()} unique targets, '
          f'{len(pdesc_df.columns) - 1} descriptor columns')
    check_id_prefix(pdesc_df['TGT_CHEMBL_ID'], 'TGT_CHEMBL_ID', os.path.basename(PDESC_FILE))
    main_tgt_ids = set(main_df['TGT_CHEMBL_ID'].astype(str))
    check_nan_numeric(pdesc_df[pdesc_df['TGT_CHEMBL_ID'].astype(str).isin(main_tgt_ids)].reset_index(drop=True),
                      'TGT_CHEMBL_ID', os.path.basename(PDESC_FILE))

    # Physchem descriptors
    print(f'\nPhyschem descriptors ({os.path.basename(CDESC_FILE)}):')
    if not check_header_alignment(CDESC_FILE):
        ok = False
    cdesc_df = pd.read_csv(CDESC_FILE, sep='\t', low_memory=False)
    cdesc_df = cdesc_df[cdesc_df.columns.intersection(fps_cols + ['CMP_CHEMBL_ID'])]
    cdesc_df['CMP_CHEMBL_ID'] = cdesc_df['CMP_CHEMBL_ID'].astype(str)
    print(f'  {len(cdesc_df)} rows, {cdesc_df["CMP_CHEMBL_ID"].nunique()} unique compounds, '
          f'{len(cdesc_df.columns) - 1} descriptor columns')
    check_id_prefix(cdesc_df['CMP_CHEMBL_ID'], 'CMP_CHEMBL_ID', os.path.basename(CDESC_FILE))
    main_cmp_ids = set(main_df['CMP_CHEMBL_ID'].astype(str))
    check_nan_numeric(cdesc_df[cdesc_df['CMP_CHEMBL_ID'].isin(main_cmp_ids)].reset_index(drop=True),
                      'CMP_CHEMBL_ID', os.path.basename(CDESC_FILE))

    # Join coverage
    print('\n=== Checking join coverage ===')
    main_cmp = set(main_df['CMP_CHEMBL_ID'].astype(str))
    main_tgt = set(main_df['TGT_CHEMBL_ID'].astype(str))
    pdesc_tgt = set(pdesc_df['TGT_CHEMBL_ID'].astype(str))
    cdesc_cmp = set(cdesc_df['CMP_CHEMBL_ID'])

    coverage_check('protein desc  (by TGT_CHEMBL_ID)', main_tgt - pdesc_tgt, len(main_tgt))
    coverage_check('physchem desc (by CMP_CHEMBL_ID)',  main_cmp - cdesc_cmp, len(main_cmp))

    # Sample join NaN check
    print('\n=== Simulating descriptor join (sample of 500 rows) ===')
    sample = main_df.sample(min(500, len(main_df)), random_state=42)[
        ['CMP_CHEMBL_ID', 'TGT_CHEMBL_ID']
    ].copy()

    if fps_files:
        with open(fps_files[0]) as _f:
            _header = _f.readline()
        fps_sep = '\t' if _header.count('\t') > _header.count(',') else ','
        fps_df = pd.read_csv(fps_files[0], sep=fps_sep)
        fps_df['CMP_CHEMBL_ID'] = fps_df['CMP_CHEMBL_ID'].astype(str)
        fps_name = os.path.basename(fps_files[0])

        for label, merged, id_col in [
            (f'fingerprints ({fps_name})',     sample.merge(fps_df,   on='CMP_CHEMBL_ID', how='left'), 'CMP_CHEMBL_ID'),
            ('protein desc',                   sample.merge(pdesc_df, on='TGT_CHEMBL_ID', how='left'), 'TGT_CHEMBL_ID'),
            ('physchem desc',                  sample.merge(cdesc_df, on='CMP_CHEMBL_ID', how='left'), 'CMP_CHEMBL_ID'),
        ]:
            desc_cols = [c for c in merged.columns if c not in ('CMP_CHEMBL_ID', 'TGT_CHEMBL_ID')]
            nan_rows = merged[desc_cols].isna().any(axis=1).sum()
            if nan_rows:
                fail(f'{label}: {nan_rows}/500 rows have NaN after join')
            else:
                print(f'  OK   {label}: no NaN in sample join')

    # Result
    print()
    if ok:
        print('PASSED — dataset should build without NaN values.')
    else:
        print('FAILED — see issues above.')
    return ok

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Verify a prepared beyond_the_hype dataset directory.')
    ap.add_argument('--data-dir', default='./data/original_bth',
                    help='Path to the prepared dataset directory (default: ./data/original_bth)')
    ap.add_argument('--physchem', choices=sorted(PHYSCHEM_SETS), default='full',
                    help='Compound physicochemical descriptor selection to verify (default: full)')
    args = ap.parse_args()
    sys.exit(0 if verify(args.data_dir, args.physchem) else 1)
