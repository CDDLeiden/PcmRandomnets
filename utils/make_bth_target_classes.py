"""Classify original BtH targets by ChEMBL L1/L2 (reuses the cached class tree)."""
import json

import pandas as pd
from chembl_webresource_client.new_client import new_client

DATA = "data/original_bth"
CACHE = "output/original_bth_target_classes.csv"
TREE = "output/chembl_protein_class_tree.json"

def _levels(cid, tree):
    chain = {}
    while cid in tree and tree[cid][0] is not None:
        lv, name, parent = tree[cid]
        chain[lv] = name
        cid = parent
    return chain

def classify():
    tree = {int(k): tuple(v) for k, v in json.load(open(TREE)).items()}

    targets = pd.read_csv(f"{DATA}/compounds_with_smiles_and_split2.csv",
                          usecols=["TGT_CHEMBL_ID"])["TGT_CHEMBL_ID"].astype(str)
    uniq = sorted(targets.unique())

    # 1) target -> accession
    tgt2acc = {}
    B = 200
    for i in range(0, len(uniq), B):
        batch = uniq[i:i + B]
        for e in new_client.target.filter(target_chembl_id__in=batch).only(["target_chembl_id", "target_components"]):
            accs = [c["accession"] for c in e.get("target_components", []) if c.get("accession")]
            if accs:
                tgt2acc[e["target_chembl_id"]] = accs[0]
        print(f"  target {min(i+B,len(uniq))}/{len(uniq)} (mapped {len(tgt2acc)})", flush=True)

    # 2) accession -> class id
    accs = sorted(set(tgt2acc.values()))
    acc2cid = {}
    for i in range(0, len(accs), B):
        for e in new_client.target_component.filter(accession__in=accs[i:i + B]).only(["accession", "protein_classifications"]):
            pcs = e.get("protein_classifications") or []
            if pcs:
                acc2cid[e["accession"]] = pcs[0]["protein_classification_id"]
        print(f"  target_component {min(i+B,len(accs))}/{len(accs)}", flush=True)

    rows = []
    for t in uniq:
        acc = tgt2acc.get(t)
        lv = _levels(acc2cid.get(acc), tree) if acc else {}
        rows.append({"TGT_CHEMBL_ID": t, "accession": acc, "L1": lv.get(1), "L2": lv.get(2)})
    pd.DataFrame(rows).to_csv(CACHE, index=False)
    print(f"  cached {len(rows)} -> {CACHE}", flush=True)
