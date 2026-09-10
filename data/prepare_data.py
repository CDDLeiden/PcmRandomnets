"""Reproduce all derived data files from the raw supplementary material.

Required input files (place in input/original_bth/):
  curated_set_with_publication_year.sd
  compound_features_*.txt
  compound_additional_physchem_features.txt
  Protein_Descriptors_50_sections.txt
  70Training.csv
  30Val.csv

Produced output files (written to data/original_bth/):
  curated_set_with_publication_year.csv  -- properties extracted from SD file
  molecules.smi                          -- SMILES extracted from SD molblocks,
                                            row-aligned with the CSV above
  compound_features_*_fix.txt            -- deduplicated, Compound_ID renamed to
                                            CMP_CHEMBL_ID
  compounds_with_smiles_and_split.csv    -- curated data + SMILES + datasplit column
  compounds_with_smiles_and_split2.csv   -- same, with NaN rows removed
  compound_additional_physchem_features.txt  -- copied from input (pass-through)
  Protein_Descriptors_50_sections.txt        -- copied from input (pass-through)
  fps_cols_full.txt                          -- generated: every compound physicochemical
  fps_cols_original.txt                         descriptor / the 6 BtH ones, each plus
                                                every protein descriptor

run_qsprpred_bth.py picks between the two via the config's "physchem" key
("full" or "original"); see descriptor_selection.py.
"""

import argparse
import glob
import os
import re
import sys

import pandas as pd
from rdkit import Chem
from rdkit.Chem import MolToSmiles

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from descriptor_selection import build_fps_cols

_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
_ROOT    = os.path.dirname(_SCRIPT_DIR)

INPUT_DIR     = os.path.join(_ROOT, 'input', 'original_bth')
BASE_DIR      = os.path.join(_ROOT, 'data',  'original_bth')
SHARED_DIR    = os.path.join(_ROOT, 'data')

def ipath(*parts):
    return os.path.join(INPUT_DIR, *parts)

def path(*parts):
    return os.path.join(BASE_DIR, *parts)

def _fix_tsv_header(src, dst):
    """Copy a TSV file, expanding any header fields that contain space-merged column names."""
    with open(src, 'r') as f_in:
        header_line = f_in.readline()
        rest = f_in.read()

    raw_fields = header_line.rstrip('\n').split('\t')
    fixed_fields = []
    for field in raw_fields:
        parts = [p for p in re.split(r'\s+', field.strip()) if p]
        fixed_fields.extend(parts)

    if len(fixed_fields) != len(raw_fields):
        print(f'  WARNING: {os.path.basename(src)}: header had {len(raw_fields)} tab-separated '
              f'fields but {len(fixed_fields)} column names after expanding space-merged names — '
              f'writing fixed header to output.')

    with open(dst, 'w') as f_out:
        f_out.write('\t'.join(fixed_fields) + '\n')
        f_out.write(rest)

def write_fps_cols(force=False):
    """Generate both descriptor-selection lists read by run_qsprpred_bth.py.

    Columns are read straight from the two descriptor files' headers; see
    descriptor_selection.py for what each list holds.
    """
    prot_cols = list(pd.read_csv(path('Protein_Descriptors_50_sections.txt'),
                                 sep='\t', nrows=0).columns)
    cmp_cols = list(pd.read_csv(path('compound_additional_physchem_features.txt'),
                                sep='\t', nrows=0).columns)
    for physchem in ('full', 'original'):
        fname = f'fps_cols_{physchem}.txt'
        dst = path(fname)
        if not force and os.path.exists(dst):
            print(f'  skipping {fname}: already exists')
            continue
        cols = build_fps_cols(physchem, cmp_cols, prot_cols, 'CMP_LOGP')
        n_prot = sum(c.startswith('Prot_') for c in cols)
        with open(dst, 'w') as f:
            f.write('\n'.join(cols) + '\n')
        print(f'  generated {fname} ({len(cols) - n_prot} cmp + {n_prot} prot)')

def copy_passthrough_files(force=False):
    """Copy files that run_qsprpred_bth.py reads directly from data_dir."""
    files = [
        'compound_additional_physchem_features.txt',
        'Protein_Descriptors_50_sections.txt',
    ]
    for fname in files:
        dst = path(fname)
        if not force and os.path.exists(dst):
            print(f'  skipping {fname}: already exists')
            continue
        _fix_tsv_header(ipath(fname), dst)
        print(f'  copied {fname}')

    write_fps_cols(force=force)

def extract_sd_file(force=False):
    """Extract property table and SMILES from the SD file."""
    csv_out = path('curated_set_with_publication_year.csv')
    smi_out = path('molecules.smi')
    if not force and os.path.exists(csv_out) and os.path.exists(smi_out):
        print('  skipping: output files already exist')
        return

    supplier = Chem.SDMolSupplier(ipath('curated_set_with_publication_year.sd'),
                                   removeHs=False, sanitize=False)

    rows = []
    smiles_list = []
    columns = None
    for mol in supplier:
        if mol is None:
            rows.append(None)
            smiles_list.append(None)
            continue
        props = mol.GetPropsAsDict()
        if columns is None:
            columns = list(props.keys())
        rows.append(props)
        try:
            Chem.SanitizeMol(mol)
            smiles_list.append(MolToSmiles(mol))
        except Exception:
            smiles_list.append(None)

    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(csv_out, index=False)
    print(f'  wrote {len(df)} rows to {csv_out}')

    smi_df = pd.DataFrame({'SMILES': smiles_list})
    smi_df.to_csv(smi_out, index=False)
    print(f'  wrote {len(smi_df)} rows to {smi_out}')

def dedup_compound_features(force=False, fp_sizes=None):
    """Deduplicate compound feature files and rename Compound_ID -> CMP_CHEMBL_ID."""
    input_files = glob.glob(ipath('compound_features_*.txt'))
    input_files = [f for f in input_files if '_fix' not in f]
    if fp_sizes is not None:
        input_files = [f for f in input_files
                       if any(f'_{n}.' in os.path.basename(f) for n in fp_sizes)]

    for inp in sorted(input_files):
        base = os.path.basename(inp).replace('.txt', '_fix.txt')
        out = path(base)
        if not force and os.path.exists(out):
            print(f'  skipping {base}: already exists')
            continue
        df = pd.read_csv(inp, sep='\t')
        id_candidates = [c for c in df.columns
                         if c == 'CMP_CHEMBL_ID' or c.lower() in ('compound_id', 'chembl_id')]
        if not id_candidates:
            raise ValueError(f'{inp}: cannot find compound ID column; columns are {list(df.columns)}')
        if id_candidates[0] != 'CMP_CHEMBL_ID':
            df = df.rename(columns={id_candidates[0]: 'CMP_CHEMBL_ID'})
        ids = df['CMP_CHEMBL_ID'].astype(str)
        if not ids.iloc[0].startswith('CHEMBL'):
            print(f'  WARNING: {os.path.basename(inp)}: CMP_CHEMBL_ID values lack "CHEMBL" prefix '
                  f'(e.g. {ids.iloc[0]!r}) — prepending "CHEMBL" automatically. '
                  f'Check the source file for malformed compound IDs.')
            ids = 'CHEMBL' + ids
        df['CMP_CHEMBL_ID'] = ids
        df = df.drop_duplicates(subset=['CMP_CHEMBL_ID'])
        df.to_csv(out, index=False)
        print(f'  wrote {len(df)} rows to {out}')

def build_split_dataset(force=False):
    """Join curated data with SMILES and 70/30 train/test split assignments."""
    out = path('compounds_with_smiles_and_split.csv')
    if not force and os.path.exists(out):
        print('  skipping: output file already exists')
        return

    curated = pd.read_csv(path('curated_set_with_publication_year.csv'))
    smiles  = pd.read_csv(path('molecules.smi'))
    train   = pd.read_csv(ipath('70Training.csv'))
    val     = pd.read_csv(ipath('30Val.csv'))

    train = train[['TC_key']].assign(datasplit='train')
    val   = val[['TC_key']].assign(datasplit='test')
    splits = pd.concat([train, val], ignore_index=True)

    df = curated.copy()
    df['SMILES'] = smiles['SMILES']
    df = df.merge(splits, on='TC_key', how='left')

    df.to_csv(out, index=False)
    print(f'  wrote {len(df)} rows to {out}')

ESSENTIAL_COLS = ['BIOACT_PCHEMBL_VALUE', 'SMILES', 'CMP_CHEMBL_ID', 'TGT_CHEMBL_ID', 'datasplit', 'DOC_YEAR']

def filter_nan(force=False):
    """Drop rows with NaN in essential columns or missing from any descriptor file."""
    out = path('compounds_with_smiles_and_split2.csv')
    if not force and os.path.exists(out):
        print('  skipping: output file already exists')
        return

    df = pd.read_csv(path('compounds_with_smiles_and_split.csv'))
    before = len(df)

    essential = [c for c in ESSENTIAL_COLS if c in df.columns]
    df = df.dropna(subset=essential)
    after_nan = len(df)
    if before - after_nan:
        print(f'  dropped {before - after_nan} rows with NaN in essential columns')

    # Drop compounds that would produce NaN descriptor features.
    descriptor_files = {
        'compound_features_256_fix.txt': ('CMP_CHEMBL_ID', ','),
        'compound_additional_physchem_features.txt': ('CMP_CHEMBL_ID', '\t'),
    }
    unique_cmp_ids = set(df['CMP_CHEMBL_ID'].astype(str))
    for fname, (id_col, sep) in descriptor_files.items():
        fpath = path(fname)
        if not os.path.exists(fpath):
            print(f'  skipping descriptor check for {fname}: file not found')
            continue
        desc = pd.read_csv(fpath, sep=sep, low_memory=False)
        desc[id_col] = desc[id_col].astype(str)

        # Compounds absent from the file entirely.
        missing_ids = sorted(unique_cmp_ids - set(desc[id_col]))
        coverage = 1 - len(missing_ids) / len(unique_cmp_ids)
        if missing_ids and coverage < 0.5:
            print(f'  WARNING: {fname}: only {coverage:.0%} of compounds covered — '
                  f'likely a file format issue (broken header?). Skipping descriptor check.')
            continue

        # Compounds present but with NaN in any descriptor column.
        # Columns may contain string '-nan' / '  -nan' instead of float NaN, so coerce first.
        desc_cols = [c for c in desc.columns if c != id_col]
        numeric = desc[desc_cols].apply(pd.to_numeric, errors='coerce')
        nan_ids = sorted(set(desc.loc[numeric.isna().any(axis=1), id_col]))

        bad_ids = sorted(set(missing_ids) | set(nan_ids))
        if not bad_ids:
            continue

        examples = ', '.join(
            f'{c} (SMILES: {df.loc[df["CMP_CHEMBL_ID"]==c, "SMILES"].iloc[0]})'
            for c in bad_ids[:5]
        )
        suffix = f' ... and {len(bad_ids)-5} more' if len(bad_ids) > 5 else ''
        print(f'  WARNING: {len(bad_ids)} compound(s) in {fname} have missing or NaN descriptors '
              f'— dropping all their rows: {examples}{suffix}')
        df = df[~df['CMP_CHEMBL_ID'].astype(str).isin(bad_ids)]

    print(f'  dropped {before - len(df)} rows total ({len(df)} remaining)')
    df.to_csv(out, index=False)
    print(f'  wrote {len(df)} rows to {out}')

def main():
    parser = argparse.ArgumentParser(description='Prepare beyond-the-hype dataset files.')
    parser.add_argument('--force', action='store_true', help='Overwrite existing output files')
    parser.add_argument('--fp-sizes', nargs='+', type=int, default=None,
                        metavar='N', help='Only process compound_features_N.txt (default: all)')
    args = parser.parse_args()

    os.makedirs(BASE_DIR, exist_ok=True)
    os.makedirs(SHARED_DIR, exist_ok=True)

    print('Step 1: Copy pass-through files')
    copy_passthrough_files(force=args.force)

    print('Step 2: Extract SD file -> curated CSV + molecules.smi')
    extract_sd_file(force=args.force)

    print('Step 3: Deduplicate compound feature files')
    dedup_compound_features(force=args.force, fp_sizes=args.fp_sizes)

    print('Step 4: Join with train/test splits')
    build_split_dataset(force=args.force)

    print('Step 5: Filter NaN rows')
    filter_nan(force=args.force)

    print('Done.')

if __name__ == '__main__':
    main()
