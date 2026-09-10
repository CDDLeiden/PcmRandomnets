"""ChEMBL L1/L2 protein-class distribution for the chembl35 dataset targets.

Fetches each target's ChEMBL protein classification (via accession ->
target_component -> protein_classification), then reports the L1/L2 distribution
both by target and weighted by number of interactions (data points), so it can be
compared with the Papyrus paper. Results are cached to output/chembl35_target_classes.csv.
"""
import json
import os

import pandas as pd
from chembl_webresource_client.new_client import new_client

DATA = "data/chembl35"
CACHE = "output/chembl35_target_classes.csv"
TREE = "output/chembl_protein_class_tree.json"

def load_tree():
    """protein_class_id -> (class_level, pref_name, parent_id); cached to JSON."""
    if os.path.exists(TREE):
        raw = json.load(open(TREE))
        return {int(k): tuple(v) for k, v in raw.items()}
    nodes = {}
    for e in new_client.protein_classification.all():
        nodes[e["protein_class_id"]] = (e.get("class_level"), e.get("pref_name"), e.get("parent_id"))
    json.dump({str(k): v for k, v in nodes.items()}, open(TREE, "w"))
    return nodes

def resolve_levels(cid, tree):
    """Walk parent chain, return {level: pref_name}."""
    chain = {}
    while cid in tree and tree[cid][0] is not None:
        level, name, parent = tree[cid]
        chain[level] = name
        cid = parent
    return chain

def fetch_classes():
    am = pd.read_csv(f"{DATA}/accession_mapping.csv", sep="\t")
    am["TGT_CHEMBL_ID"] = am["TGT_CHEMBL_ID"].astype(str)
    accs = sorted({a for s in am["accession"].astype(str) for a in s.split(",")})

    # accession -> first protein_classification_id
    tc_api = new_client.target_component
    acc2cid = {}
    B = 200
    for i in range(0, len(accs), B):
        batch = accs[i:i + B]
        for e in tc_api.filter(accession__in=batch).only(["accession", "protein_classifications"]):
            pcs = e.get("protein_classifications") or []
            if pcs:
                acc2cid[e["accession"]] = pcs[0]["protein_classification_id"]
        print(f"  target_component {min(i+B,len(accs))}/{len(accs)}  (mapped {len(acc2cid)})", flush=True)

    # resolve L1/L2 via the classification tree (parent-chain walk)
    tree = load_tree()
    cid2lv = {cid: resolve_levels(cid, tree) for cid in set(acc2cid.values())}

    rows = []
    for _, r in am.iterrows():
        acc = str(r["accession"]).split(",")[0]
        lv = cid2lv.get(acc2cid.get(acc), {})
        rows.append({"TGT_CHEMBL_ID": r["TGT_CHEMBL_ID"], "accession": acc,
                     "L1": lv.get(1), "L2": lv.get(2)})
    out = pd.DataFrame(rows).drop_duplicates("TGT_CHEMBL_ID")
    out.to_csv(CACHE, index=False)
    print(f"  cached {len(out)} targets -> {CACHE}", flush=True)
    return out

def report(cls):
    # interactions per target (data-point weighting)
    counts = pd.read_csv(f"{DATA}/compounds_with_smiles_and_split2.csv",
                         usecols=["TGT_CHEMBL_ID"])["TGT_CHEMBL_ID"].astype(str).value_counts()
    cls = cls.copy()
    cls["n_interactions"] = cls["TGT_CHEMBL_ID"].map(counts).fillna(0).astype(int)
    n_t, n_i = len(cls), int(cls["n_interactions"].sum())
    unclassified = cls["L1"].isna().sum()
    print(f"\ntargets={n_t}  interactions={n_i}  unclassified targets={unclassified}")

    for lvl in ["L1", "L2"]:
        by_t = cls[lvl].fillna("Unclassified").value_counts()
        by_i = cls.groupby(cls[lvl].fillna("Unclassified"))["n_interactions"].sum().sort_values(ascending=False)
        print(f"\n=== {lvl} distribution ===")
        print(f"{'class':40s} {'targets':>8s} {'tgt%':>6s} {'datapts':>10s} {'dp%':>6s}")
        for k in by_i.index[:15]:
            print(f"{str(k)[:40]:40s} {by_t.get(k,0):8d} {100*by_t.get(k,0)/n_t:6.1f} "
                  f"{int(by_i[k]):10d} {100*by_i[k]/n_i:6.1f}")

def main():
    if os.path.exists(CACHE):
        print("loading cached", CACHE)
        cls = pd.read_csv(CACHE)
    else:
        cls = fetch_classes()
    report(cls)
