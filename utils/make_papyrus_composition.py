#!/usr/bin/env python3
"""Regenerate the Papyrus++ composition analysis on the LATEST release (05.7 / ChEMBL 34).

Was used to generate the numbers in the Discussion paragraph on Papyrus.

Counting conventions (same as the notes):
  - target = target_id (mutants distinct)
  - compound = connectivity (stereo-free)
  - publication = distinct doc id from all_doc_ids (union across the target's rows)
  - a Papyrus++ row is already one aggregated (compound, target) pair -> "data point"
"""
import pandas as pd

BASE = "/home/chichi148/Data/RandomNetsExperiments/data/papyrus/05.7/"
PP   = BASE + "05.7++_combined_set_without_stereochemistry.tsv.xz"
PROT = BASE + "05.7_combined_set_protein_targets.tsv.xz"

def _l1(c):
    if not isinstance(c, str) or not c:
        return "Unclassified"
    return c.split(";")[0].split("->")[0]

def _l2(c):
    if not isinstance(c, str) or not c:
        return "Unclassified"
    parts = c.split(";")[0].split("->")
    return parts[1] if len(parts) > 1 else "Unclassified"

def analyze():
    print("loading Papyrus++ 05.7 ...")
    df = pd.read_csv(PP, sep="\t", compression="xz",
                     usecols=["source", "connectivity", "target_id", "Year",
                              "all_doc_ids", "accession", "Protein_Type"],
                     dtype=str)
    prot = pd.read_csv(PROT, sep="\t", compression="xz",
                       usecols=["target_id", "Classification", "Organism"], dtype=str)

    prot["L1"] = prot["Classification"].map(_l1)
    prot["L2"] = prot["Classification"].map(_l2)
    cls = prot.set_index("target_id")[["L1", "L2", "Classification"]]

    N = len(df)
    print(f"\n{'='*70}\nBASELINE  Papyrus++ 05.7 (full high-quality set)\n{'='*70}")
    n_tgt = df["target_id"].nunique()
    n_cmp = df["connectivity"].nunique()
    comp_per_tgt = df.groupby("target_id")["connectivity"].nunique()
    print(f"data points (aggregated pairs) : {N:,}")
    print(f"unique targets (target_id)     : {n_tgt:,}")
    print(f"unique compounds (connectivity): {n_cmp:,}")
    print(f"median compounds / target      : {comp_per_tgt.median():.0f}")
    print(f"targets >=30 compounds         : {(comp_per_tgt>=30).sum():,}")

    ### PUBLICATIONS PER TARGET (UNION OF DISTINCT DOC IDS) ###
    def pubs_per_target(sub):
        s = set()
        for v in sub.dropna():
            for d in str(v).split(";"):
                d = d.strip()
                if d:
                    s.add(d)
        return len(s)
    pub_per_tgt = df.groupby("target_id")["all_doc_ids"].apply(pubs_per_target)

    ### BTH COMPOSITION FILTERS ###
    print(f"\n{'='*70}\nBtH COMPOSITION FILTERS on Papyrus++ 05.7\n{'='*70}")
    ge30   = comp_per_tgt[comp_per_tgt >= 30].index
    ge2pub = pub_per_tgt[pub_per_tgt >= 2].index
    both   = comp_per_tgt.index[(comp_per_tgt >= 30) & (pub_per_tgt.reindex(comp_per_tgt.index).fillna(0) >= 2)]

    def summarize(name, tgt_index):
        sub = df[df["target_id"].isin(tgt_index)]
        print(f"{name:32s} pts={len(sub):>8,} ({100*len(sub)/N:5.1f}%)  targets={tgt_index.nunique():>6,}  compounds={sub['connectivity'].nunique():>8,}")
        return sub
    summarize("none (Papyrus++ 05.7)", comp_per_tgt.index)
    summarize(">=30 compounds/target", ge30)
    summarize(">=2 publications/target", ge2pub)
    sub_both = summarize("both (BtH criteria)", both)

    ### SOURCE COMPOSITION ###
    print(f"\n{'='*70}\nSOURCE COMPOSITION (each ++ row has a single source)\n{'='*70}")
    src_full = df["source"].value_counts()
    for s, c in src_full.items():
        print(f"  full ++      {s:16s} {c:>8,} ({100*c/N:5.1f}%)")
    print("  -- non-ChEMBL total: "
          f"{N - src_full.get('ChEMBL34',0):,} ({100*(N-src_full.get('ChEMBL34',0))/N:.1f}%)")

    print("\n  within the BtH-criteria subset:")
    src_both = sub_both["source"].value_counts()
    for s, c in src_both.items():
        print(f"  BtH-subset   {s:16s} {c:>8,} ({100*c/len(sub_both):5.1f}%)")

    # Where passing targets get their depth: chembl vs kinase augmentation
    print(f"\n{'='*70}\nPASSING TARGETS: ChEMBL34-alone vs kinase augmentation\n{'='*70}")
    KIN = {"Christmann2016", "Sharma2016"}
    # compounds per target from ChEMBL34 rows only
    chembl_rows = df[df["source"] == "ChEMBL34"]
    chembl_comp_per_tgt = chembl_rows.groupby("target_id")["connectivity"].nunique()
    pass_tgts = pd.Index(both)
    has_chembl   = pass_tgts.intersection(chembl_rows["target_id"].unique())
    clear30_chembl = pass_tgts.intersection(chembl_comp_per_tgt[chembl_comp_per_tgt >= 30].index)
    via_kinase   = pass_tgts.difference(clear30_chembl)
    absent_chembl = pass_tgts.difference(chembl_rows["target_id"].unique())
    print(f"passing targets (BtH criteria)            : {len(pass_tgts):,}")
    print(f"  have any ChEMBL34 data                  : {len(has_chembl):,} ({100*len(has_chembl)/len(pass_tgts):.0f}%)")
    print(f"  clear >=30 from ChEMBL34 alone          : {len(clear30_chembl):,} ({100*len(clear30_chembl)/len(pass_tgts):.0f}%)")
    print(f"  pass only via kinase-set augmentation   : {len(via_kinase):,} ({100*len(via_kinase)/len(pass_tgts):.0f}%)")
    print(f"    - absent from ChEMBL34 entirely       : {len(absent_chembl):,}")
    print(f"    - had ChEMBL34 data but <30, pushed over: {len(via_kinase)-len(absent_chembl):,}")

    # ligands contributed by kinase sets vs chembl among passing targets w/ kinase data
    kin_rows_all = df[df["source"].isin(KIN)]
    pass_with_kin = pass_tgts.intersection(kin_rows_all["target_id"].unique())
    kin_lig = df[(df["target_id"].isin(pass_with_kin)) & (df["source"].isin(KIN))]["connectivity"].nunique()
    chembl_lig = df[(df["target_id"].isin(pass_with_kin)) & (df["source"] == "ChEMBL34")]["connectivity"].nunique()
    print(f"\n  passing targets with any kinase-set data : {len(pass_with_kin):,}")
    print(f"    ligands from kinase sets (connectivity): {kin_lig:,}")
    print(f"    ligands from ChEMBL34   (connectivity): {chembl_lig:,}")

    ### TARGET-CLASS DISTRIBUTION (L1/L2) ###
    def class_dist(sub, level, label):
        j = sub.join(cls[level].rename("cls"), on="target_id")
        by_pts = j["cls"].value_counts(normalize=True) * 100
        by_tgt = j.drop_duplicates("target_id")["cls"].value_counts(normalize=True) * 100
        out = pd.DataFrame({"data-pt%": by_pts.round(1), "tgt%": by_tgt.round(1)}).fillna(0)
        print(f"\n--- {label}: {level} distribution ---")
        print(out.sort_values("data-pt%", ascending=False).head(12).to_string())
        return out
    print(f"\n{'='*70}\nTARGET-CLASS DISTRIBUTION\n{'='*70}")
    class_dist(df, "L1", "Papyrus++ 05.7 FULL")
    class_dist(sub_both, "L1", "Papyrus++ 05.7 after BtH criteria")
    class_dist(df, "L2", "Papyrus++ 05.7 FULL")
    class_dist(sub_both, "L2", "Papyrus++ 05.7 after BtH criteria")

    # organism spread among unique targets
    print(f"\n{'='*70}\nORGANISM SPREAD (unique targets, full ++)\n{'='*70}")
    org = prot.set_index("target_id")["Organism"]
    tgt_org = df.drop_duplicates("target_id").join(org, on="target_id")["Organism"]
    print(tgt_org.value_counts().head(8).to_string())
    print(f"distinct organisms: {tgt_org.nunique()}")