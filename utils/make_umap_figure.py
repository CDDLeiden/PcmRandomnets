"""
UMAP of compound chemical space per dataset, recolored by split.

For each dataset (original_bth, chembl35):
  - one UMAP embedding is computed on the 4096-bit Morgan fingerprints of the
    UNIQUE compounds (subsampled for tractability/legibility);
  - two panels show the SAME coordinates recolored by (a) the fixed 70/30 split
    (the `datasplit` column, i.e. 70Training/30Val the models trained on) and
    (b) the temporal split (DOC_YEAR > cutoff -> test, matching QSPRpred's
    TemporalSplit).

Each unique compound is categorised as train-only / test-only / both, since the
splits are defined per interaction and a compound may appear in both subsets.

Run inside an env with umap-learn, scikit-learn, pandas, matplotlib.
"""
import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import umap
from matplotlib.lines import Line2D

DATA = "data"
TEMPORAL_CUTOFF = {"original_bth": 2012, "chembl35": 2020}
# original_bth keeps the fixed 70/30 split shipped with the BtH data; chembl35 is
# re-split per target (75/25) after collapsing duplicate compound-target pairs.
RANDOM_SPLIT_TITLE = {"original_bth": "Fixed 70/30 split", "chembl35": "Random split (75/25)"}
# Colourblind-safe qualitative colours (Okabe-Ito): sky blue / vermillion / green.
# Sky blue is clearly lighter (CIELAB L* 70) than vermillion and green (54, 58),
# so the minority test-only / both points stand out against the train-only bulk.
# Vermillion rather than red so the blue+other overlap blends to a neutral tone
# instead of a purple that could be mistaken for a category.
CAT_COLORS = {"train-only": "#56B4E9", "test-only": "#D55E00", "both": "#009E73"}
DATASET_NAMES = {"original_bth": "Original BtH", "chembl35": "ChEMBL 35"}
# The figure is drawn at the manuscript's text width and included at
# \linewidth, so this is the printed font size, a little below the 11 pt
# body text.
TEXT_WIDTH = 455.24 / 72.27    # inches
FONT_SIZE = 9


def _categorise(has_train, has_test):
    """Vectorised: 'both' | 'train-only' | 'test-only' from two boolean Series."""
    return np.where(has_train & has_test, "both",
                    np.where(has_train, "train-only", "test-only"))


def per_compound_labels(ds):
    """Return DataFrame indexed by CMP_CHEMBL_ID with random & temporal categories."""
    cut = TEMPORAL_CUTOFF[ds]
    f = f"{DATA}/{ds}/compounds_with_smiles_and_split2.csv"
    df = pd.read_csv(f, usecols=["CMP_CHEMBL_ID", "datasplit", "DOC_YEAR"])
    df["is_test_rnd"] = df["datasplit"].eq("test")
    df["is_test_tmp"] = df["DOC_YEAR"] > cut
    g = df.groupby("CMP_CHEMBL_ID")
    agg = g.agg(has_test_rnd=("is_test_rnd", "any"), has_train_rnd=("is_test_rnd", lambda s: (~s).any()),
                has_test_tmp=("is_test_tmp", "any"), has_train_tmp=("is_test_tmp", lambda s: (~s).any()))
    out = pd.DataFrame({
        "random": _categorise(agg["has_train_rnd"], agg["has_test_rnd"]),
        "temporal": _categorise(agg["has_train_tmp"], agg["has_test_tmp"]),
    }, index=agg.index)
    return out


def load_fingerprints(ds, wanted_ids, fp_size=4096, chunksize=20000):
    """Chunk-read compound_features to keep only the subsampled compound rows."""
    path = f"{DATA}/{ds}/compound_features_{fp_size}_fix.txt"
    wanted = set(wanted_ids)
    header = pd.read_csv(path, sep=",", nrows=0).columns
    dtypes = {c: np.uint8 for c in header if c.lower().startswith("bit")}
    frames = []
    for chunk in pd.read_csv(path, sep=",", chunksize=chunksize, dtype=dtypes, low_memory=False):
        chunk["CMP_CHEMBL_ID"] = chunk["CMP_CHEMBL_ID"].astype(str)
        if not chunk["CMP_CHEMBL_ID"].iloc[0].startswith("CHEMBL"):
            chunk["CMP_CHEMBL_ID"] = "CHEMBL" + chunk["CMP_CHEMBL_ID"]
        keep = chunk[chunk["CMP_CHEMBL_ID"].isin(wanted)]
        if len(keep):
            frames.append(keep)
    fps = pd.concat(frames).drop_duplicates("CMP_CHEMBL_ID").set_index("CMP_CHEMBL_ID")
    bit_cols = [c for c in fps.columns if c.lower().startswith("bit")]
    return fps[bit_cols].astype(np.uint8)


def subsample(labels, n, seed=42):
    if len(labels) <= n:
        return labels.index.to_numpy()
    rng = np.random.default_rng(seed)
    return rng.choice(labels.index.to_numpy(), size=n, replace=False)


def compute_embedding(ds, n_sample, seed, fp_size=4096, recompute=False):
    """Return labels DataFrame with x,y UMAP coords, caching the result to parquet."""
    cache = f"output/umap_{ds}_embedding.pkl"
    if os.path.exists(cache) and not recompute:
        print(f"[{ds}] loading cached embedding {cache}")
        return pd.read_pickle(cache)

    print(f"[{ds}] loading per-compound labels ...")
    labels = per_compound_labels(ds)
    print(f"[{ds}] unique compounds: {len(labels)}")
    for split in ("random", "temporal"):
        vc = labels[split].value_counts()
        pct = (100 * vc / len(labels)).round(1)
        print(f"   {split:9s}: " + ", ".join(f"{k}={vc[k]} ({pct[k]}%)" for k in vc.index))

    ids = subsample(labels, n_sample, seed)
    labels = labels.loc[ids]
    print(f"[{ds}] subsampled to {len(ids)} compounds; reading fingerprints ...")
    fps = load_fingerprints(ds, ids, fp_size=fp_size)
    labels = labels.loc[fps.index]                 # align, drop any missing FPs
    X = fps.to_numpy()
    print(f"[{ds}] fingerprint matrix {X.shape}; running UMAP ...")

    reducer = umap.UMAP(n_neighbors=25, min_dist=0.1, metric="jaccard", random_state=seed)
    emb = reducer.fit_transform(X)
    labels = labels.copy()
    labels["x"], labels["y"] = emb[:, 0], emb[:, 1]
    labels.to_pickle(cache)
    print(f"[{ds}] cached embedding -> {cache}")
    return labels


def render_row(row, letter, ds, labels, s, alpha):
    axes = row.subplots(1, 2, sharex=True, sharey=True)
    titles = {"random": RANDOM_SPLIT_TITLE[ds], "temporal": f"Temporal split (cutoff {TEMPORAL_CUTOFF[ds]})"}
    # single random z-order across all points so no category systematically
    # overplots another; the colour mix in a region then reflects true local
    # proportions (drawing by category would let "both" paint over "train-only")
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(labels))
    for ax, split in zip(axes, ("random", "temporal")):
        cols = labels[split].map(CAT_COLORS).to_numpy()
        ax.scatter(labels["x"].to_numpy()[perm], labels["y"].to_numpy()[perm],
                   s=s, c=cols[perm], alpha=alpha, linewidths=0, rasterized=True)
        ax.set_title(titles[split], fontsize=FONT_SIZE)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel("UMAP 1", fontsize=FONT_SIZE)
    axes[0].set_ylabel("UMAP 2", fontsize=FONT_SIZE)
    row.suptitle(f"{letter}) {DATASET_NAMES[ds]}", x=0.01, ha="left",
                 fontsize=FONT_SIZE, fontweight="bold")


def render(panels, out, s, alpha):
    """One row per dataset (A, B, ...); panels maps dataset -> labels with x, y."""
    fig = plt.figure(figsize=(TEXT_WIDTH, 6.2), layout="constrained")
    rows = fig.subfigures(len(panels), 1)
    for row, letter, (ds, labels) in zip(rows, "AB", panels.items()):
        render_row(row, letter, ds, labels, s, alpha)

    handles = [Line2D([0], [0], marker="o", ls="", ms=6, mfc=CAT_COLORS[c], mec="none", label=c)
               for c in ["train-only", "test-only", "both"]]
    fig.legend(handles=handles, loc="outside upper center", ncol=3, frameon=True,
               fontsize=FONT_SIZE)
    plt.savefig(out, dpi=600)
    print(f"saved {out}")
    plt.close(fig)


def make_umap_figures(datasets=("original_bth", "chembl35"), n_sample=40000, seed=42,
                      fp_size=4096, point_size=2, alpha=0.5, recompute=False):
    # point_size is in pt^2 at print size (the figure is drawn at text width).
    panels = {ds: compute_embedding(ds, n_sample, seed, fp_size, recompute) for ds in datasets}
    render(panels, "output/umap.png", point_size, alpha)
