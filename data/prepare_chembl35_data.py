"""Prepare ChEMBL35 dataset files from bth_chembl35 SD files.

Required input files (place in input/chembl35/):
  training_set_with_desc.sd
  test_set_with_desc.sd
  2026_05_19_c35_for_descriptor_processing.sd  -- source of DOC_YEAR per Interaction_ID
  prot_desc_50_sections.txt.gz                 -- protein descriptors keyed by accession
  list_of_targets.txt                          -- TGT_CHEMBL_ID <-> accession mapping

Output directory: data/chembl35/

Produced output files:
  compounds_with_smiles_and_split.csv   -- all interactions with descriptors, SMILES,
                                           datasplit, DOC_YEAR, and TC_key columns
  compounds_with_smiles_and_split2.csv  -- NaN rows removed, then collapsed to one
                                           median row per (compound, target) pair per
                                           the BtH standardization, with a fresh
                                           per-target stratified train/test split
  compound_features_256_fix.txt         -- Morgan fingerprints (radius=2, ECFP4),
  compound_features_512_fix.txt            one row per compound, deduplicated on
  compound_features_2048_fix.txt           CMP_CHEMBL_ID, comma-separated
  compound_features_4096_fix.txt
  compound_additional_physchem_features.txt  -- physchem descriptors per compound,
                                                column names matching original BtH convention
  Protein_Descriptors_50_sections.txt   -- protein descriptors keyed by TGT_CHEMBL_ID
  accession_mapping.csv                 -- generated from list_of_targets.txt; used to
                                           analyze target classes
  fps_cols_full.txt                     -- generated: every compound physicochemical
  fps_cols_original.txt                    descriptor / the 6 BtH ones (logD), each plus
                                           every protein descriptor

run_qsprpred_bth.py picks between the two fps_cols files via the config's
"physchem" key ("full" or "original"); see descriptor_selection.py.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from descriptor_selection import build_fps_cols

_SCRIPT_DIR     = os.path.dirname(os.path.abspath(__file__))
_ROOT      = os.path.dirname(_SCRIPT_DIR)
INPUT_DIR       = os.path.join(_ROOT, 'input', 'chembl35')
OUT_DIR         = os.path.join(_ROOT, 'data',  'chembl35')
FINGERPRINT_SIZES = [256, 512, 2048, 4096]

# Mapping from BtH compound_additional_physchem_features column names
# to the corresponding field names in the bth_chembl35 SD files.
# Columns absent from the SD files are simply omitted from the output.
PHYSCHEM_RENAME = {
    'CMP_ATOMCOUNT_C':                      'C_Count',
    'CMP_ATOMCOUNT_C_sp3':                  'n_sp3_c',
    'CMP_ATOMCOUNT_C_sp2':                  'n_sp2_c',
    'CMP_ATOMCOUNT_C_sp':                   'n_sp_c',
    'CMP_ATOMS_C_sp3_FRAC':                 'SP3_Carbon_fraction',
    'CMP_ATOMS_C_sp2_FRAC':                 'SP2_Carbon_fraction',
    'CMP_ATOMS_C_sp_FRAC':                  'SP_Carbon_fraction',
    'CMP_CLASS_ACID':                       'CMP_ACID',
    'CMP_CLASS_BASE':                       'CMP_BASE',
    'CMP_CLASS_NEUTRAL':                    'CMP_NEUTRAL',
    'CMP_CLASS_ZWITTERION':                 'CMP_ZWITTERION',
    'CMP_FORMALCHARGE':                     'FormalCharge',
    'CMP_ATOMCOUNT_HEAVY':                  'CMP_HEAVY_ATOMS',
    'CMP_BONDS_TOTAL':                      'Num_Bonds',
    'CMP_ATOMCOUNT_HYDROGENS':              'Num_Hydrogens',
    'CMP_BONDS_EXPLICIT':                   'Num_ExplicitBonds',
    'CMP_ATOMCOUNT_POSITIVE':               'Num_PositiveAtoms',
    'CMP_ATOMCOUNT_NEGATIVE':               'Num_NegativeAtoms',
    'CMP_BONDS_RING':                       'Num_RingBonds',
    'CMP_BONDS_ROTATABLE':                  'Num_RotatableBonds',
    'CMP_BONDS_AROMATIC':                   'Num_AromaticBonds',
    'CMP_BONDS_BRIDGE':                     'Num_BridgeBonds',
    'CMP_NUM_RINGS':                        'Num_Rings',
    'CMP_NUM_AROMATICRINGS':               'Num_AromaticRings',
    'CMP_NUM_RINGASSEMBLIES':              'Num_RingAssemblies',
    'CMP_NUM_CHAINS':                      'Num_Chains',
    'CMP_NUM_CHAINASSEMBLIES':             'Num_ChainAssemblies',
    'CMP_ATOMCOUNT_METALATOMS':            'Num_MetalAtoms',
    'CMP_BONDS_PIBONDS':                   'Num_PiBonds',
    'CMP_BONDS_STEREOATOMS':               'Num_StereoAtoms',
    'CMP_BONDS_STEREOBONDS':               'Num_StereoBonds',
    'CMP_BONDS_SINGLEBONDS':               'Num_SingleBonds',
    'CMP_BONDS_DOUBLEBONDS':               'Num_DoubleBonds',
    'CMP_BONDS_TRIPLEBONDS':               'Num_TripleBonds',
    'CMP_NUM_ALIPHATICSINGLEBONDS':        'Num_AliphaticSingleBonds',
    'CMP_NUM_ALIPHATICDOUBLEBONDS':        'Num_AliphaticDoubleBonds',
    'CMP_NUM_HYDROGENBONS':                'Num_HydrogenBonds',  # note intentional typo matching BtH
    'CMP_NUM_TERMINALROTOMERS':            'Num_TerminalRotomers',
    'CMP_ATOMCOUNT_H_ACCEPTORS':           'Num_H_Acceptors',
    'CMP_ATOMCOUNT_H_DONORS':              'Num_H_Donors',
    'CMP_ATOMCOUNT_H':                     'H_Count',
    'CMP_ATOMCOUNT_N':                     'N_Count',
    'CMP_ATOMCOUNT_O':                     'O_Count',
    'CMP_ATOMCOUNT_F':                     'F_Count',
    'CMP_ATOMCOUNT_P':                     'P_Count',
    'CMP_ATOMCOUNT_S':                     'S_Count',
    'CMP_ATOMCOUNT_Cl':                    'Cl_Count',
    'CMP_ATOMCOUNT_Br':                    'Br_Count',
    'CMP_ATOMCOUNT_I':                     'I_Count',
    'CMP_MOLECULAR_SURFACEAREA':           'Molecular_SurfaceArea',
    'CMP_MOLECULAR_SASA':                  'Molecular_SASA',
    'CMP_LOGD':                            'CMP_CX_LOGD',
    'CMP_SOLUBILITY':                      'Molecular_Solubility',
    'CMP_ATOMCOUNT_TOTAL':                 'Num_Atoms',
    'CMP_NUM_ALIPHATICRINGS':              'Num_AliphaticRings',
    'CMP_BONDS_HEAVYATOMS':                'Heavy_Atom_Bonds',
    'CMP_BONDS_HYDROGENS':                 'Hydrogen_Atom_Bonds',
    'CMP_BONDS_SINGLE_FRAC':              'SingleBonds_Frac',
    'CMP_BONDS_DOUBLE_FRAC':              'DoubleBonds_Frac',
    'CMP_BONDS_TRIPLE_FRAC':              'TripleBonds_Frac',
    'CMP_BONDS_AROMATIC_FRAC':            'AromaticBonds_Frac',
    'CMP_BONDS_BRIDGE_FRAC':              'BridgeBonds_Frac',
    'CMP_BONDS_STEREO_FRAC':              'StereoBonds_Frac',
    'CMP_BONDS_RING_FRAC':                'RingBonds_Frac',
    'CMP_BONDS_ALIPHATIC_FRAC':           'Aliphatic_Ringbonds_Frac',
    'CMP_BONDS_ROTATABLE_FRAC':           'RotatableBonds_Frac',
    'CMP_ATOMCOUNT_HALOGENS':             'Num_Halogens',
    'CMP_ATOMS_C_FRAC':                   'Carbon_fraction',
    'CMP_ATOMS_H_FRAC':                   'Hydrogen_fraction',
    'CMP_ATOMS_N_FRAC':                   'Nitrogen_fraction',
    'CMP_ATOMS_O_FRAC':                   'Oxygen_fraction',
    'CMP_ATOMS_HETEROATOM_FRAC':          'Heteroatom_fraction',
    'CMP_ATOMS_S_FRAC':                   'Sulphur_fraction',
    'CMP_ATOMS_P_FRAC':                   'Phosphorus_fraction',
    'CMP_ATOMS_HALOGEN_FRAC':             'Halogen_fraction',
    'CMP_ATOMS_OTHER_FRAC':               'OtherAtom_fraction',
    'CMP_ATOMS_STEROATOM_FRAC':           'StereoAtom_Fraction',  # BtH typo: STERO not STEREO
    'CMP_ATOMS_POSITIVE_FRAC':            'PositiveAtom_Fraction',
    'CMP_ATOMS_NEGATIVE_FRAC':            'NegativeAtom_Fraction',
    'CMP_ATOMS_H_ACCEPTOR_FRAC':          'H_Acceptors_Fraction',
    'CMP_ATOMS_H_DONOR_FRAC':             'H_Donors_fraction',
    'CMP_RIGIDITY_INDEX':                 'Rigidity_Index',
    'CMP_MOLECULAR_WEIGHT':               'Molecular_Weight',
    'CMP_CLASS_LIPINSKI_PASS':            'Lipinski_Pass',
    'CMP_NUM_Ro5_Violations':             'CMP_NUM_RO5_VIOLATIONS',
    'CMP_MOLECULAR_POLAR_SURFACEAREA':    'Molecular_PolarSurfaceArea',
    'CMP_MOLECULAR_POLAR_SURFACEAREA_FRAC': 'Molecular_PolarSurfaceArea_Fraction',
    'Molecular_Weight':                   'Molecular_Weight',
}

def read_sd_file(name, split_label):
    path = os.path.join(INPUT_DIR, name)
    supplier = Chem.SDMolSupplier(path, removeHs=False, sanitize=False)
    rows = []
    for mol in supplier:
        if mol is None:
            continue
        rows.append(mol.GetPropsAsDict())
    df = pd.DataFrame(rows)
    df['datasplit'] = split_label
    print(f'  {split_label:5s}: {len(df)} rows from {name}')
    return df

def load_doc_year_mapping(name='2026_05_19_c35_for_descriptor_processing.sd'):
    """Fast text scan: build Interaction_ID -> DOC_YEAR without full RDKit parsing."""
    path = os.path.join(INPUT_DIR, name)
    mapping = {}
    current_id = None
    next_is_id = False
    next_is_year = False
    with open(path) as f:
        for line in f:
            line = line.rstrip('\n')
            if next_is_id:
                current_id = line
                next_is_id = False
            elif next_is_year:
                if current_id:
                    mapping[current_id] = line
                next_is_year = False
            elif line == '> <Interaction_ID>':
                next_is_id = True
            elif line == '> <DOC_YEAR>':
                next_is_year = True
    print(f'  loaded DOC_YEAR for {len(mapping):,} interactions')
    return mapping

def build_combined_dataset(force=False):
    out = os.path.join(OUT_DIR, 'compounds_with_smiles_and_split.csv')
    if not force and os.path.exists(out):
        print('  skipping: output file already exists')
        return pd.read_csv(out)

    print('  scanning for DOC_YEAR ...')
    doc_year = load_doc_year_mapping()

    train_df = read_sd_file('training_set_with_desc.sd', 'train')
    test_df = read_sd_file('test_set_with_desc.sd', 'test')
    df = pd.concat([train_df, test_df], ignore_index=True)
    df = df.drop(columns=['assay_id'], errors='ignore')
    if 'smiles' in df.columns and 'SMILES' not in df.columns:
        df = df.rename(columns={'smiles': 'SMILES'})

    df['DOC_YEAR'] = df['Interaction_ID'].map(doc_year)
    df['TC_key'] = df['TGT_CHEMBL_ID'] + ' - ' + df['CMP_CHEMBL_ID']

    missing = df['DOC_YEAR'].isna().sum()
    if missing:
        print(f'  warning: {missing} rows could not be matched to a DOC_YEAR')

    df.to_csv(out, index=False)
    print(f'  wrote {len(df):,} total rows to {out}')
    return df

ESSENTIAL_COLS = ['BIOACT_PCHEMBL_VALUE', 'SMILES', 'CMP_CHEMBL_ID', 'TGT_CHEMBL_ID', 'datasplit', 'DOC_YEAR']

def aggregate_and_resplit(df2, seed=42):
    """Collapse each (compound, target) pair to one row and draw a fresh split.

    Matches the original Beyond-the-Hype standardization (Lenselink et al. 2017,
    Methods): "If multiple measurements for a ligand-receptor data point were
    present, the median value was chosen and duplicates were removed." Without
    this step the same compound-target pair (often with an identical pChEMBL
    value) leaks across the train/test partitions.

      - BIOACT_PCHEMBL_VALUE -> median over the pair's measurements
      - DOC_YEAR             -> earliest (min) publication year of the pair
                               (used by the temporal split; BtH split newer data
                               into the test set, so the pair's first appearance
                               is the conservative choice)
      - all other columns    -> first value (physchem descriptors are constant
                               per compound; protein columns constant per target)

    The per-measurement train/test labels are discarded and a new per-target
    stratified random split (25% test) is drawn, so every target keeps both train
    and test pairs (needed for the per-target metrics).
    """
    test_size = 0.25
    key = ['CMP_CHEMBL_ID', 'TGT_CHEMBL_ID']
    df2 = df2.copy()
    df2['BIOACT_PCHEMBL_VALUE'] = pd.to_numeric(df2['BIOACT_PCHEMBL_VALUE'], errors='coerce')
    df2['DOC_YEAR'] = pd.to_numeric(df2['DOC_YEAR'], errors='coerce')
    n_before = len(df2)

    agg = {c: 'first' for c in df2.columns if c not in key}
    agg['BIOACT_PCHEMBL_VALUE'] = 'median'
    agg['DOC_YEAR'] = 'min'
    out = df2.groupby(key, as_index=False).agg(agg)
    out['TC_key'] = out['TGT_CHEMBL_ID'] + ' - ' + out['CMP_CHEMBL_ID']
    print(f'  aggregated {n_before:,} measurement rows -> {len(out):,} unique '
          f'(compound, target) pairs (median pChEMBL, earliest DOC_YEAR)')

    rng = np.random.default_rng(seed)
    label = np.array(['train'] * len(out), dtype=object)
    for _, pos in out.groupby('TGT_CHEMBL_ID').indices.items():
        n_test = int(round(len(pos) * test_size))
        if n_test:
            label[rng.choice(pos, size=n_test, replace=False)] = 'test'
    out['datasplit'] = label
    n_test = int((out['datasplit'] == 'test').sum())
    print(f'  fresh per-target stratified split (test_size={test_size}, seed={seed}): '
          f'train={len(out) - n_test:,} ({100 * (len(out) - n_test) / len(out):.1f}%)  '
          f'test={n_test:,} ({100 * n_test / len(out):.1f}%)')
    return out

def filter_nan(df, force=False, seed=42):
    """Drop rows with NaN in essential columns, rows for targets with no protein descriptor
    coverage, rows for compounds with unparseable SMILES, and rows for compounds with NaN
    in fps_cols physchem descriptors.
    """
    out = os.path.join(OUT_DIR, 'compounds_with_smiles_and_split2.csv')
    if not force and os.path.exists(out):
        print('  skipping: output file already exists')
        return

    before = len(df)
    essential = [c for c in ESSENTIAL_COLS if c in df.columns]
    df2 = df.dropna(subset=essential)
    after_essential = len(df2)
    if before - after_essential:
        print(f'  dropped {before - after_essential:,} rows with NaN in essential columns')

    # Drop compounds whose SMILES cannot be parsed by RDKit — they would be silently skipped
    # during fingerprint generation, leaving those compounds without fingerprint descriptors.
    invalid_cids = [
        cid for cid, smi in df2.drop_duplicates('CMP_CHEMBL_ID')[['CMP_CHEMBL_ID', 'SMILES']].itertuples(index=False)
        if Chem.MolFromSmiles(str(smi)) is None
    ]
    if invalid_cids:
        examples = ', '.join(
            f'{c} (SMILES: {df2.loc[df2["CMP_CHEMBL_ID"] == c, "SMILES"].iloc[0]!r})'
            for c in invalid_cids[:5]
        )
        suffix = f' ... and {len(invalid_cids) - 5} more' if len(invalid_cids) > 5 else ''
        print(f'  WARNING: {len(invalid_cids)} compound(s) have unparseable SMILES '
              f'— dropping all their rows: {examples}{suffix}')
        df2 = df2[~df2['CMP_CHEMBL_ID'].isin(invalid_cids)]

    # Drop compounds whose physchem descriptors map to fps_cols but have NaN/non-numeric values.
    # PHYSCHEM_RENAME maps {bth_output_col: sd_source_col}; fps_cols uses bth_output_col names.
    # Both selection lists are checked, so one dataset serves either "physchem" setting.
    fps_cols = set()
    for name in ('fps_cols_full.txt', 'fps_cols_original.txt'):
        fps_cols_path = os.path.join(OUT_DIR, name)
        if os.path.exists(fps_cols_path):
            with open(fps_cols_path) as f:
                fps_cols |= {l.strip() for l in f}
    if fps_cols:
        # sd_col -> bth_col for every bth_col that is in fps_cols and present in df
        sd_cols = {sd_col: bth_col
                   for bth_col, sd_col in PHYSCHEM_RENAME.items()
                   if bth_col in fps_cols and sd_col in df2.columns}
        if sd_cols:
            per_cmp = df2.drop_duplicates(subset='CMP_CHEMBL_ID')[
                ['CMP_CHEMBL_ID', 'SMILES'] + list(sd_cols.keys())
            ].copy()
            numeric = per_cmp[list(sd_cols.keys())].apply(pd.to_numeric, errors='coerce')
            bad_cids = sorted(per_cmp.loc[numeric.isna().any(axis=1), 'CMP_CHEMBL_ID'].tolist())
            if bad_cids:
                examples = ', '.join(
                    f'{c} (SMILES: {per_cmp.loc[per_cmp["CMP_CHEMBL_ID"] == c, "SMILES"].iloc[0]})'
                    for c in bad_cids[:5]
                )
                suffix = f' ... and {len(bad_cids) - 5} more' if len(bad_cids) > 5 else ''
                print(f'  WARNING: {len(bad_cids)} compound(s) in compound_additional_physchem_features '
                      f'have missing or NaN descriptors — dropping all their rows: {examples}{suffix}')
                df2 = df2[~df2['CMP_CHEMBL_ID'].isin(bad_cids)]

    prot_file = os.path.join(OUT_DIR, 'Protein_Descriptors_50_sections.txt')
    if os.path.exists(prot_file):
        prot_targets = set(pd.read_csv(prot_file, sep='\t', usecols=['TGT_CHEMBL_ID'])['TGT_CHEMBL_ID'])
        missing_mask = ~df2['TGT_CHEMBL_ID'].isin(prot_targets)
        n_missing = missing_mask.sum()
        if n_missing:
            print(f'  dropped {n_missing:,} rows with no protein descriptor coverage')
        df2 = df2[~missing_mask]

    print(f'  dropped {before - len(df2):,} rows total ({len(df2):,} remaining)')

    df2 = aggregate_and_resplit(df2, seed=seed)

    df2.to_csv(out, index=False)
    print(f'  wrote {len(df2):,} rows to {out}')

def generate_fingerprints(df, n_bits, force=False):
    out = os.path.join(OUT_DIR, f'compound_features_{n_bits}_fix.txt')
    if not force and os.path.exists(out):
        print(f'  skipping compound_features_{n_bits}_fix.txt: already exists')
        return

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=n_bits)
    ids, bit_rows = [], []
    for _, row in df.drop_duplicates(subset='CMP_CHEMBL_ID').iterrows():
        smi = row.get('SMILES')
        if not smi or str(smi) == 'nan':
            continue
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            continue
        fp = gen.GetFingerprintAsNumPy(mol)
        ids.append(row['CMP_CHEMBL_ID'])
        bit_rows.append(fp.tolist())

    cols = [f'bit_{i + 1}' for i in range(n_bits)]
    fp_df = pd.DataFrame(bit_rows, columns=cols)
    fp_df.insert(0, 'CMP_CHEMBL_ID', ids)
    fp_df.to_csv(out, index=False)  # comma-separated to match original BtH format
    print(f'  wrote {len(fp_df):,} compounds to {out}')

def build_physchem_file(df, force=False):
    """Extract and rename physchem columns to match original BtH naming convention."""
    out = os.path.join(OUT_DIR, 'compound_additional_physchem_features.txt')
    if not force and os.path.exists(out):
        print('  skipping compound_additional_physchem_features.txt: already exists')
        return

    cmp_df = df.drop_duplicates(subset='CMP_CHEMBL_ID')[['CMP_CHEMBL_ID']].copy()
    for bth_col, sd_col in PHYSCHEM_RENAME.items():
        if sd_col in df.columns:
            cmp_df[bth_col] = df.drop_duplicates(subset='CMP_CHEMBL_ID')[sd_col].values
        # Columns absent from the SD are simply omitted

    # Remove duplicate BtH column names caused by Molecular_Weight mapping to itself
    cmp_df = cmp_df.loc[:, ~cmp_df.columns.duplicated()]
    cmp_df.to_csv(out, sep='\t', index=False)
    print(f'  wrote {len(cmp_df):,} compounds, {len(cmp_df.columns) - 1} physchem features to {out}')

def build_protein_descriptors(force=False):
    """Bridge prot_desc (accession key) to TGT_CHEMBL_ID via list_of_targets.txt.

    Writes both Protein_Descriptors_50_sections.txt and accession_mapping.csv
    (the two-column TGT_CHEMBL_ID/accession bridge read by the target-class
    analysis, utils/make_target_class_distribution.py).
    Targets without a matching accession in prot_desc are omitted.
    """
    out = os.path.join(OUT_DIR, 'Protein_Descriptors_50_sections.txt')
    acc_out = os.path.join(OUT_DIR, 'accession_mapping.csv')
    if not force and os.path.exists(out) and os.path.exists(acc_out):
        print('  skipping Protein_Descriptors_50_sections.txt: already exists')
        return

    targets = pd.read_csv(
        os.path.join(INPUT_DIR, 'list_of_targets.txt'), sep='\t',
        usecols=['TGT_CHEMBL_ID', 'accession'],
    )
    prot_df = pd.read_csv(
        os.path.join(INPUT_DIR, 'prot_desc_50_sections.txt.gz'),
        sep='\t', compression='gzip',
    )
    prot_df = prot_df.drop(columns=['TGT_CHEMBL_ID'], errors='ignore')
    merged = targets.merge(prot_df, on='accession', how='inner').drop(columns=['accession'])
    merged.to_csv(out, sep='\t', index=False)
    print(f'  wrote {len(merged):,} targets to {out}')

    targets[['TGT_CHEMBL_ID', 'accession']].to_csv(acc_out, sep='\t', index=False)
    print(f'  wrote {len(targets):,} entries to {acc_out}')

def write_fps_cols(force=False):
    """Generate both descriptor-selection lists read by run_qsprpred_bth.py.

    Protein descriptors come from the file built in the previous step. The compound
    physicochemical columns are taken from PHYSCHEM_RENAME rather than from
    compound_additional_physchem_features.txt.

    chembl35's SD source carries CMP_LOGD (ChemAxon logD) but no CMP_LOGP, so logD is
    the lipophilicity descriptor of the 'original' selection here.
    """
    prot_cols = list(pd.read_csv(os.path.join(OUT_DIR, 'Protein_Descriptors_50_sections.txt'),
                                 sep='\t', nrows=0).columns)
    cmp_cols = list(PHYSCHEM_RENAME)
    for physchem in ('full', 'original'):
        fname = f'fps_cols_{physchem}.txt'
        dst = os.path.join(OUT_DIR, fname)
        if not force and os.path.exists(dst):
            print(f'  skipping {fname}: already exists')
            continue
        cols = build_fps_cols(physchem, cmp_cols, prot_cols, 'CMP_LOGD')
        n_prot = sum(c.startswith('Prot_') for c in cols)
        with open(dst, 'w') as f:
            f.write('\n'.join(cols) + '\n')
        print(f'  generated {fname} ({len(cols) - n_prot} cmp + {n_prot} prot)')

def main():
    parser = argparse.ArgumentParser(description='Prepare ChEMBL35 dataset files.')
    parser.add_argument('--force', action='store_true', help='Overwrite existing output files')
    parser.add_argument('--fp-sizes', nargs='+', type=int, default=FINGERPRINT_SIZES,
                        metavar='N', help='Fingerprint sizes to generate (default: all)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for the fresh split')
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    print('Step 1: Combine train + test SD files')
    df = build_combined_dataset(force=args.force)

    print('Step 2: Build protein descriptors file')
    build_protein_descriptors(force=args.force)

    print('Step 3: Generate descriptor-selection lists (fps_cols_full.txt, fps_cols_original.txt)')
    write_fps_cols(force=args.force)

    print('Step 4: Filter NaN rows (drops rows with missing protein descriptor coverage)')
    filter_nan(df, force=args.force, seed=args.seed)

    print('Step 5: Generate Morgan fingerprints')
    for n_bits in args.fp_sizes:
        print(f'  {n_bits} bits ...')
        generate_fingerprints(df, n_bits, force=args.force)

    print('Step 6: Build compound physchem features file')
    build_physchem_file(df, force=args.force)

    print('Done.')

if __name__ == '__main__':
    main()
