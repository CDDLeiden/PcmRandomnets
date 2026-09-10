"""Train/test nearest-neighbour Tanimoto similarity per split and dataset.

Quantifies how close the test compounds lie to the training set: for each test
compound we take its maximum Tanimoto similarity to any training compound
(4096-bit Morgan fingerprints, the same ones used for the UMAP). The training
reference is the complete set (exact nearest neighbour); only the test set is
subsampled, for a stable mean. Reported over all test compounds and over the
test-only subset (those absent from training).

Computed statistics are cached to output/split_similarity.json; pass
recompute=True (or delete the file) to redo them.
"""
import json
import os
import statistics
import sys

import numpy as np
from rdkit import DataStructs
from rdkit.DataStructs import ExplicitBitVect

from utils.make_umap_figure import load_fingerprints, per_compound_labels

CACHE = "output/split_similarity.json"
N_TEST = 8000
SEED = 42
FP_SIZE = 4096

def _log(msg):
    """Progress goes to stderr so it stays out of paper_numbers.log."""
    print(msg, file=sys.stderr, flush=True)

def _bvs_from_matrix(X):
    """Build one RDKit ExplicitBitVect per row of a 0/1 fingerprint matrix."""
    bvs = []
    for row in X:
        bv = ExplicitBitVect(int(X.shape[1]))
        bv.SetBitsFromList(np.nonzero(row)[0].tolist())
        bvs.append(bv)
    return bvs

def _stats(maxes):
    """[mean, median, n] for a list/array of nearest-neighbour similarities."""
    maxes = list(maxes)
    if not maxes:
        return [float("nan"), float("nan"), 0]
    return [statistics.mean(maxes), statistics.median(maxes), len(maxes)]

def _compute(datasets, n_test, seed, fp_size):
    """Return {dataset: {split: {'all': [...], 'test_only': [...]}}}."""
    rng = np.random.default_rng(seed)
    out = {}
    for ds in datasets:
        _log(f"[{ds}] loading labels and full fingerprints ...")
        labels = per_compound_labels(ds)
        fps = load_fingerprints(ds, labels.index.to_numpy(), fp_size=fp_size)
        labels = labels.loc[fps.index]
        _log(f"[{ds}] building {len(fps):,} bit vectors ...")
        bvs = _bvs_from_matrix(fps.to_numpy().astype(np.uint8))
        out[ds] = {}
        for split in ("random", "temporal"):
            cat = labels[split].to_numpy()
            train_bvs = [b for b, keep in zip(bvs, cat != "test-only") if keep]
            test_idx = np.nonzero(cat != "train-only")[0]
            take = rng.choice(test_idx, size=min(n_test, len(test_idx)), replace=False)
            _log(f"[{ds}] {split}: {len(test_idx):,} test compounds, "
                 f"sampling {len(take):,} against {len(train_bvs):,} training compounds")
            maxes = np.array([max(DataStructs.BulkTanimotoSimilarity(bvs[i], train_bvs))
                              for i in take])
            cats = cat[take]
            out[ds][split] = {
                "all": _stats(maxes),
                "test_only": _stats(maxes[cats == "test-only"]),
            }
    return out

def _load_or_compute(datasets, n_test, seed, fp_size, recompute):
    """Return the statistics dict, using the JSON cache when available."""
    if os.path.exists(CACHE) and not recompute:
        _log(f"loading cached similarities {CACHE}")
        with open(CACHE, encoding="utf-8") as f:
            return json.load(f)
    results = _compute(datasets, n_test, seed, fp_size)
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    with open(CACHE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    return results

def report(datasets=("original_bth", "chembl35"), n_test=N_TEST, seed=SEED,
           fp_size=FP_SIZE, recompute=False):
    """Print the nearest-neighbour Tanimoto similarities (for paper_numbers.log)."""
    results = _load_or_compute(datasets, n_test, seed, fp_size, recompute)
    print(f"Nearest-neighbour Tanimoto similarity of test compounds to the full "
          f"training set\n({fp_size}-bit Morgan, test subsampled to {n_test}, seed {seed}).\n")
    for ds in datasets:
        print(f"{ds}:")
        for split in ("random", "temporal"):
            a = results[ds][split]["all"]
            t = results[ds][split]["test_only"]
            print(f"  {split:9s} all test  mean={a[0]:.3f} median={a[1]:.3f} (n={a[2]})"
                  f"   test-only mean={t[0]:.3f} median={t[1]:.3f} (n={t[2]})")