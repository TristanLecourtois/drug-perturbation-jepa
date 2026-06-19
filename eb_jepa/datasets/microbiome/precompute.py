"""
precompute.py - Build a compact cache for the microbiome JEPA from the raw
DIABIMMUNE / MicrobeAtlas data, KEEPING abundances (needed for the
abundance-weighted encoder, alpha-diversity, and soft-UniFrac phylo descriptor).

Output cache (torch.save) holds:
  - emb_table : float16 [U, 384]  unique ProkBERT OTU embeddings actually used
  - otu_to_idx: dict raw-otu-id -> row in emb_table
  - subjects  : list of {subject, label, timepoints:[{age, idx[int32], cnt[int32],
                                                      div, phylo[float16 384], milk}]}

It reuses the (tested) metadata linking + OTU->embedding-key resolver from the
sibling Microbiome-Modelling repo, and streams the 19GB mapped file once here to
recover per-OTU counts. Run once; the dataset then loads only the small cache.

Env:
  MICROBIOME_RAW  -> path to the raw data/ dir (default: the sibling repo's data/)
  MICROBIOME_CACHE-> output path (default: $EBJEPA_DSETS/microbiome/cache.pt)
"""

import os
import sys
import math

import h5py
import numpy as np
import torch

# --- locate raw data + reuse the sibling repo's loaders/resolver -------------
_DEFAULT_RAW = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "Microbiome-Modelling", "data")
)
RAW = os.environ.get("MICROBIOME_RAW", _DEFAULT_RAW)
_MM_REPO = os.path.abspath(os.path.join(RAW, ".."))
if _MM_REPO not in sys.path:
    sys.path.insert(0, _MM_REPO)

MAPPED = os.path.join(RAW, "microbeatlas", "samples-otus.97.mapped")
PROKBERT = os.path.join(RAW, "model", "prokbert_embeddings.h5")
SAMPLES = os.path.join(RAW, "diabimmune", "samples.csv")
MILK = os.path.join(RAW, "diabimmune", "milk.csv")
DIABETES = os.path.join(RAW, "diabimmune", "diabetes.csv")


def _safe_float(x):
    try:
        return float(str(x).strip())
    except Exception:
        return None


def _parse_csv(path):
    rows = []
    with open(path, errors="replace") as f:
        header = None
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split(",")
            if header is None:
                header = [h.lstrip("﻿") for h in parts]
                continue
            rows.append(dict(zip(header, parts)))
    return rows


def stream_counts(mapped_path, needed_srs):
    """Yield (srs, {otu97: count}) from the MicrobeAtlas mapped file."""
    cur, otus = None, {}
    with open(mapped_path, "r", errors="replace") as f:
        for line in f:
            if line.startswith(">"):
                if cur is not None and otus:
                    yield cur, otus
                header = line[1:].split()[0]
                srs = header.split(".")[-1]
                cur = srs if srs in needed_srs else None
                otus = {}
                continue
            if cur is None:
                continue
            parts = line.split("\t") if "\t" in line else line.split()
            first = parts[0]
            cnt = 1
            if len(parts) > 1:
                try:
                    cnt = int(float(parts[1]))
                except Exception:
                    cnt = 1
            otu97 = next((t for t in first.split(";") if t.startswith("97_")), None)
            if otu97 is None:
                toks = first.split(";")
                otu97 = toks[-1] if toks else None
            if otu97:
                otus[otu97] = otus.get(otu97, 0) + cnt
        if cur is not None and otus:
            yield cur, otus


def main():
    from scripts.diabimmune.utils import load_run_data
    from scripts import utils as mm

    out = os.environ.get("MICROBIOME_CACHE")
    if not out:
        droot = os.environ.get("EBJEPA_DSETS", os.path.join(os.path.dirname(__file__)))
        out = os.path.join(droot, "microbiome", "cache.pt")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    print("== Linking metadata (runs -> srs -> subject -> sample/age) ==")
    run_rows, SRA_to_micro, gid_to_sample, micro_to_subject, micro_to_sample = load_run_data(
        wgs_path=os.path.join(RAW, "diabimmune", "SraRunTable_wgs.csv"),
        extra_path=os.path.join(RAW, "diabimmune", "SraRunTable_extra.csv"),
        samples_path=SAMPLES,
        microbeatlas_path=os.path.join(RAW, "diabimmune", "microbeatlas_samples.tsv"),
    )
    needed = set(micro_to_subject.keys())
    print(f"   SRS with a subject: {len(needed)}")

    # sample -> age, subject -> milk / t1d label
    sample_age = {}
    for r in _parse_csv(SAMPLES):
        sid = r.get("sampleID", "").strip()
        if sid:
            sample_age[sid] = _safe_float(r.get("age_at_collection", ""))
    milk_lab = {r.get("subjectID", "").strip(): (r.get("milk_first_three_days", "") or "").strip()
                for r in _parse_csv(MILK)} if os.path.exists(MILK) else {}
    t1d_lab = {}
    if os.path.exists(DIABETES):
        for r in _parse_csv(DIABETES):
            sid = r.get("subjectID", "").strip()
            v = (r.get("t1d_diagnosed", "") or "").strip().lower()
            t1d_lab[sid] = 1 if v in ("1", "yes", "true", "t") else 0

    counts_cache = os.path.join(os.path.dirname(out), "srs_counts.pt")
    if os.path.exists(counts_cache):
        print(f"== Loading cached per-OTU counts from {counts_cache} ==")
        srs_counts = torch.load(counts_cache, weights_only=False)
    else:
        print("== Streaming mapped file for per-OTU counts (19GB, one pass) ==")
        srs_counts = {}
        for srs, otus in stream_counts(MAPPED, needed):
            srs_counts[srs] = otus
        torch.save(srs_counts, counts_cache)
    print(f"   samples with counts: {len(srs_counts)}")

    # OTU -> embedding-key resolver (prefer bacteria), then build embedding table.
    # NOTE: use ABSOLUTE paths -- mm.* defaults are relative to the MM repo cwd.
    RENAME = os.path.join(RAW, "microbeatlas", "otus.rename.map1")
    micro_to_otus = {srs: list(c.keys()) for srs, c in srs_counts.items()}
    rename_map = mm.load_otu_rename_map(RENAME) if os.path.exists(RENAME) else None
    resolver = (mm.build_otu_key_resolver(micro_to_otus, rename_map, PROKBERT, prefer="B")
                if rename_map else {})
    print(f"   resolver entries: {len(resolver)}")

    print("== Building embedding table (unique OTUs) ==")
    emb_rows, otu_to_idx = [], {}
    with h5py.File(PROKBERT, "r") as f:
        emb = f["embeddings"]
        for srs, counts in srs_counts.items():
            for oid in counts:
                if oid in otu_to_idx:
                    continue
                key = resolver.get(oid, oid)
                if key in emb:
                    otu_to_idx[oid] = len(emb_rows)
                    emb_rows.append(np.asarray(emb[key][()], dtype=np.float32))
    emb_table = np.stack(emb_rows, axis=0).astype(np.float16)  # [U, 384]
    print(f"   unique OTU embeddings: {emb_table.shape}")

    # group (subject, age) -> aggregated counts
    agg = {}
    for srs, counts in srs_counts.items():
        subject = micro_to_subject.get(srs)
        si = micro_to_sample.get(srs)
        sample_id = si.get("sample") if si else None
        age = sample_age.get(sample_id) if sample_id else None
        if subject is None or age is None:
            continue
        d = agg.setdefault((subject, age), {})
        for oid, c in counts.items():
            if oid in otu_to_idx:
                d[oid] = d.get(oid, 0) + c

    # per-subject timelines with diversity + phylo descriptor
    by_subject = {}
    for (subject, age), counts in agg.items():
        if not counts:
            continue
        idx = np.array([otu_to_idx[o] for o in counts], dtype=np.int32)
        cnt = np.array([counts[o] for o in counts], dtype=np.float32)
        p = cnt / cnt.sum()
        shannon = float(-(p * np.log(p + 1e-12)).sum())
        vecs = emb_table[idx].astype(np.float32)  # [n, 384]
        phylo = (p[:, None] * vecs).sum(0)  # abundance-weighted mean embedding
        phylo = phylo / (np.linalg.norm(phylo) + 1e-6)
        by_subject.setdefault(subject, []).append(
            dict(age=age, idx=idx, cnt=cnt, div=shannon,
                 phylo=phylo.astype(np.float16), milk=milk_lab.get(subject, "")),
        )

    subjects = []
    for subject, tps in by_subject.items():
        tps.sort(key=lambda d: d["age"])
        subjects.append(dict(subject=subject, label=t1d_lab.get(subject, 0), timepoints=tps))

    n_tp = sum(len(s["timepoints"]) for s in subjects)
    divs = [tp["div"] for s in subjects for tp in s["timepoints"]]
    print(f"== subjects={len(subjects)}  timepoints={n_tp}  "
          f"median_tp/subj={int(np.median([len(s['timepoints']) for s in subjects]))} "
          f"shannon[min/med/max]={min(divs):.2f}/{np.median(divs):.2f}/{max(divs):.2f}")

    torch.save(
        dict(emb_table=torch.from_numpy(emb_table), otu_to_idx=otu_to_idx, subjects=subjects),
        out,
    )
    print(f"saved cache -> {out}  ({os.path.getsize(out)/1e6:.0f} MB)")


if __name__ == "__main__":
    main()
