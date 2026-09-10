# Reproducing the results

Setup the conda environment and the datasets first, then run and analyze the experiments with the following commands:

python run_qsprpred_bth.py
python analyze_results.py

## Conda environment

The experiments run against **QSPRpred's `dev` branch on GitHub** (CDDLeiden/QSPRpred), installed as an editable checkout. `requirements-lock.txt` in the repository root is the frozen environment the results were produced with. QSPRpred itself is an editable install at a fixed commit, so rebuilding from it pins the library as well as its dependencies:
```
-e git+https://github.com/CDDLeiden/QSPRpred.git@dda37a341a398195ae93b1e452821c2211a40e8b#egg=qsprpred
```

There is a dependency conflict in the current dev version of QSPRpred: `ml2json`'s metadata caps `scikit-learn` at 1.4.0, while `papyrus_scripts` forces numpy 2, which needs `scikit-learn >= 1.4.2`. Because the lock file already pins every version, install it with `--no-deps` so pip does not re-resolve (and crash on) that conflict:

```bash
conda create -n PcmRandomnets python=3.11 -y
conda activate PcmRandomnets
pip install -r requirements-lock.txt --no-deps
```

Or, build the environment from scratch (this resolves against current PyPI, so it approximates rather than exactly reproduces the version table below):

```bash
conda create -n PcmRandomnets python=3.11 -y
conda activate PcmRandomnets

# CUDA build first, so the [deep] extra does not pull a CPU-only wheel
pip install torch --index-url https://download.pytorch.org/whl/cu124

git clone -b dev https://github.com/CDDLeiden/QSPRpred.git ~/Projects/github/QSPRpred
pip install -e "~/Projects/github/QSPRpred[deep,extra]"

# not declared by QSPRpred; needed by the Lightning Randomnets model and its logger
pip install pytorch_lightning tensorboard

# No --no-deps here: pip resolves the full tree, which pulls an older scikit-learn to
# satisfy ml2json's stale cap, so force it back up to the version numpy 2 requires
pip install -U "scikit-learn>=1.4.2"
```

| | Version |
|---|---|
| Python | 3.11 |
| QSPRpred | `dev` @ `dda37a341a398195ae93b1e452821c2211a40e8b` |
| torch | 2.6.0+cu124 |
| numpy | 2.4.6 |
| pandas | 3.0.3 |
| scikit-learn | 1.9.0 |
| scipy | 1.17.1 |
| rdkit | 2026.3.3 |
| pytorch-lightning | 2.6.5 |

## Input files for beyond_the_hype datasets

These files are not under source control. Place them under `input/` (in the
subdirectories shown below) before running the prepare scripts from `data/`.

### input/original_bth/

Input files for `prepare_data.py`. Outputs go to `data/original_bth/`.
All files come from the supplementary materials of Lenselink *et al.*, "Beyond the
Hype: Deep Neural Networks Outperform Established Methods Using a ChEMBL Bioactivity
Benchmark Set", *J. Cheminform.* **9**, 45 (2017),
[doi:10.1186/s13321-017-0232-0](https://doi.org/10.1186/s13321-017-0232-0).

| File | Description | Source |
|------|-------------|--------|
| `curated_set_with_publication_year.sd` | Curated bioactivity data with publication year | BtH supplementary materials |
| `compound_features_256.txt` | Morgan fingerprints, 256 bits | BtH supplementary materials |
| `compound_features_512.txt` | Morgan fingerprints, 512 bits | BtH supplementary materials |
| `compound_features_2048.txt` | Morgan fingerprints, 2048 bits | BtH supplementary materials |
| `compound_features_4096.txt` | Morgan fingerprints, 4096 bits | BtH supplementary materials |
| `compound_additional_physchem_features.txt` | Physchem descriptors per compound | BtH supplementary materials |
| `Protein_Descriptors_50_sections.txt` | Protein descriptors keyed by TGT\_CHEMBL\_ID | BtH supplementary materials |
| `fps_cols.txt` | Selected feature columns for model training. Unused: the prepare scripts generate `fps_cols_full.txt` and `fps_cols_original.txt` from the descriptor headers instead | BtH supplementary materials |
| `70Training.csv` | TC\_key list for the 70% training split | BtH supplementary materials |
| `30Val.csv` | TC\_key list for the 30% test split | BtH supplementary materials |

### input/chembl35/

Input files for `prepare_chembl35_data.py`. Outputs go to `data/chembl35/`.
All files are exported from the Pipeline Pilot server. They are attached to the
[v1.0.0 release](https://github.com/CDDLeiden/PcmRandomnets/releases/tag/v1.0.0) as
`PcmRandomnets_input_chembl35.tar.gz` (774 MB; 6.8 GB extracted). Extracting it in the
repository root places the files in `input/chembl35/`:

```bash
wget https://github.com/CDDLeiden/PcmRandomnets/releases/download/v1.0.0/PcmRandomnets_input_chembl35.tar.gz
tar -xzf PcmRandomnets_input_chembl35.tar.gz
```

These files are derived from ChEMBL release 35 and, like ChEMBL, are licensed under
[CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/). The MIT license of this
repository covers the code only.

| File | Description |
|------|-------------|
| `training_set_with_desc.sd` | Training set SD file with descriptors (random split) |
| `test_set_with_desc.sd` | Test set SD file with descriptors (random split) |
| `2026_05_19_c35_for_descriptor_processing.sd` | Full ChEMBL35 export; used to look up DOC\_YEAR per Interaction\_ID |
| `prot_desc_50_sections.txt.gz` | Protein descriptors keyed by UniProt accession |
| `list_of_targets.txt` | TGT\_CHEMBL\_ID ↔ accession mapping for ChEMBL35 targets |

# Hardware specifications

Hardware used for the reported experiments:
  GPU: 1× NVIDIA H200 NVL (Hopper, sm_90, 141 GB)
  CPU: AMD EPYC 9554P (64 cores / 128 threads)
  NVIDIA driver: 610.43.02 (CUDA 13.3); requires driver supporting CUDA ≥ 12.4
Note: GPU training is non-deterministic; on Ampere+ GPUs (incl. H200) cuDNN uses
TF32 for fp32 convolutions by default, so RandomNets metrics may vary by ~1e-3
across runs/hardware. Set torch.backends.cudnn.allow_tf32 = False to disable.