"""Shared definition of the fps_cols_*.txt descriptor-selection lists.

Can be used to filter the list of physicochemical descriptors of the Beyond-
the-Hype paper exactly, giving input files from the Pipeline Pilot server.
We opted to run our experiments with the full available list of descriptors.

  original -- the six compound physicochemical descriptors of the Beyond-the-Hype
              paper (Lenselink et al. 2017), plus every protein descriptor
  full     -- every compound physicochemical descriptor, plus every protein descriptor
"""

# The six compound descriptors used by the paper. Lipophilicity descriptor is
# given by CMP_LOGP for original_bth, CMP_LOGD for chembl35.
def original_cmp_descriptors(lipophilicity_col):
    return [
        'CMP_MOLECULAR_WEIGHT',
        lipophilicity_col,
        'CMP_MOLECULAR_POLAR_SURFACEAREA_FRAC',
        'CMP_ATOMCOUNT_H_ACCEPTORS',
        'CMP_ATOMCOUNT_H_DONORS',
        'CMP_BONDS_ROTATABLE',
    ]


# Compound columns that were removed from selection:
# CMP_LOGD is BtH's lipophilicity descriptor rather than CMP_LOGP,
# CMP_FULL_MWT is identical to the selected Molecular_Weight, and
# CMP_INORGANIC_FLAG is constant across all compounds.
FULL_CMP_EXCLUDE = set({
    'CMP_LOGP',
    'CMP_RO3_PASS',
    'CMP_FULL_MWT',
    'CMP_INORGANIC_FLAG',
})


def build_fps_cols(physchem, cmp_cols, prot_cols, lipophilicity_col):
    """Return the selection list for ``physchem`` ('full' or 'original').

    ``cmp_cols`` is the compound physicochemical descriptor file's column order (the
    ID column may be included; it is dropped) and ``prot_cols`` the protein
    descriptor file's.
    """
    if physchem == 'original':
        cmp_selected = original_cmp_descriptors(lipophilicity_col)
    elif physchem == 'full':
        cmp_selected = [c for c in cmp_cols
                        if c != 'CMP_CHEMBL_ID' and c not in FULL_CMP_EXCLUDE]
    else:
        raise ValueError(f'unknown physchem selection: {physchem!r}')
    return cmp_selected + [c for c in prot_cols if c.startswith('Prot_')]
