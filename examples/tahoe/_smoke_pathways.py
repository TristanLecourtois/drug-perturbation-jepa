"""
Smoke test for the MSigDB Hallmark pathways path (no download):
  1. fabricate a tiny cache (panel + KMeans modules + labels) and a gene_metadata CSV
     mapping some panel genes to REAL hallmark symbols;
  2. run precompute_pathways -> pathways.pt (panel-aligned hallmark membership);
  3. load TahoeDataset with data.pathways set and confirm it switches from KMeans to
     hallmark programs and rebuilds the per-cell activity P that PathwayCoherenceLoss eats.

  python -m examples.tahoe._smoke_pathways
"""
import os
import subprocess
import sys
import tempfile

import numpy as np
import torch

from eb_jepa.datasets.tahoe.dataset import TahoeConfig, TahoeDataset
from eb_jepa.datasets.tahoe.pathways import load_hallmark


def main():
    tmp = tempfile.mkdtemp()
    # take real symbols from two hallmark sets so the membership is non-empty + overlapping
    sets = load_hallmark()
    names = sorted(sets)[:3]
    syms = []
    for n in names:
        syms += sets[n][:6]
    syms = list(dict.fromkeys(syms))             # dedup, keep order
    K = len(syms) + 4                             # +4 panel genes with NO hallmark
    panel = torch.arange(3, 3 + K, dtype=torch.long)

    N = 50
    blob = dict(
        panel=panel, X=torch.rand(N, K),
        modules=torch.randint(0, 5, (K,)), n_modules=5,
        drug=torch.zeros(N, dtype=torch.long), moa=torch.zeros(N, dtype=torch.long),
        cell_line=torch.zeros(N, dtype=torch.long),
        drug_names=["DMSO"], moa_names=["x"], cl_names=["c0"], drug_fp=torch.zeros(1, 8),
    )
    cache = os.path.join(tmp, "cache.pt"); torch.save(blob, cache)

    meta = os.path.join(tmp, "gene_meta.csv")
    with open(meta, "w") as f:
        f.write("token_id,gene_symbol,ensembl_id\n")
        for i, t in enumerate(panel.tolist()):
            sym = syms[i] if i < len(syms) else ""     # last 4 unmapped -> belong to no set
            f.write(f"{t},{sym},ENSG{i:05d}\n")

    out = os.path.join(tmp, "pathways.pt")
    cmd = [sys.executable, "-m", "eb_jepa.datasets.tahoe.precompute_pathways",
           "--cache", cache, "--gene-meta", meta, "--out", out]
    print("$", " ".join(cmd)); subprocess.run(cmd, check=True)

    pw = torch.load(out, weights_only=False)
    M = pw["membership"].shape[1]
    assert pw["membership"].shape[0] == K, pw["membership"].shape
    assert M >= 1 and len(pw["names"]) == M
    assert (pw["membership"].sum(1) > 0).sum().item() == len(syms), "covered genes mismatch"

    # KMeans baseline (no pathways_path) vs hallmark (with it)
    base = TahoeDataset(TahoeConfig(cache_path=cache, drop_frac=0.0, noise_std=0.0))
    assert base.pathway_kind == "kmeans" and base.n_modules == 5
    hall = TahoeDataset(TahoeConfig(cache_path=cache, drop_frac=0.0, noise_std=0.0, pathways_path=out))
    assert hall.pathway_kind == "hallmark" and hall.n_modules == M
    assert hall.P.shape == (N, M), hall.P.shape          # per-cell activity, one row per cell

    print(f"\nhallmark: {M} sets x K={K} | covered {len(syms)}/{K} genes | "
          f"per-cell activity P={tuple(hall.P.shape)} (was KMeans {base.n_modules} programs)")
    print("SMOKE OK")


if __name__ == "__main__":
    main()
