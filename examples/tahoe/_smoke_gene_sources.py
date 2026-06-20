"""
Smoke test for the multi-source gene-init pipeline, WITHOUT any download:
  1. fabricate a tiny training cache (with `panel`), a gene_metadata CSV, and a
     synthetic source table ({symbol: vector}) standing in for ESM2;
  2. run precompute_gene_sources to align it onto the panel -> gene_sources.pt;
  3. register it on a SetTransformer and run forward + backward.

Proves the contract end-to-end so that, on Dalia, only real artifacts need swapping in.

  python -m examples.tahoe._smoke_gene_sources
"""
import os
import subprocess
import sys
import tempfile

import numpy as np
import torch

from eb_jepa.architectures import SetTransformer


def main():
    tmp = tempfile.mkdtemp()
    K, d = 12, 16
    panel = torch.arange(3, 3 + K, dtype=torch.long)              # token_ids 3..14
    torch.save({"panel": panel, "X": torch.randn(40, K)}, os.path.join(tmp, "cache.pt"))

    # gene_metadata: token_id -> symbol (cover 9 of 12 panel genes; 3 left unmapped)
    meta = os.path.join(tmp, "gene_meta.csv")
    with open(meta, "w") as f:
        f.write("token_id,gene_symbol,ensembl_id\n")
        for i, t in enumerate(panel.tolist()):
            sym = f"GENE{i}" if i < 9 else ""                      # last 3 have no symbol
            f.write(f"{t},{sym},ENSG{i:05d}\n")

    # synthetic ESM2-like table: {symbol: vector}; cover GENE0..GENE6 (so 7/12 panel hits)
    esm = os.path.join(tmp, "esm2.pt")
    torch.save({f"GENE{i}": np.random.randn(d).astype(np.float32) for i in range(7)}, esm)

    out = os.path.join(tmp, "gene_sources.pt")
    cmd = [sys.executable, "-m", "eb_jepa.datasets.tahoe.precompute_gene_sources",
           "--cache", os.path.join(tmp, "cache.pt"), "--gene-meta", meta,
           "--sources", "esm2", "--esm2-emb", esm, "--out", out]
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True)

    tables = torch.load(out, weights_only=False)
    assert "esm2" in tables, "esm2 table not written"
    assert tables["esm2"].shape == (K, d), tables["esm2"].shape
    nonzero = (tables["esm2"].abs().sum(1) > 0).sum().item()
    assert nonzero == 7, f"expected 7 covered genes, got {nonzero}"

    enc = SetTransformer(n_genes=K, out_d=8, d_model=16, n_latents=4, depth=1, heads=2)
    for name, tbl in tables.items():
        enc.register_gene_source(name, tbl)
    x = torch.randn(5, K, requires_grad=False)
    z = enc(x)
    assert z.shape == (5, 8), z.shape
    z.sum().backward()
    assert enc.src_proj["esm2"].weight.grad is not None, "projection got no gradient"
    # frozen source table must carry no gradient
    assert enc.get_buffer("src_esm2").requires_grad is False

    print(f"\naligned table covers {nonzero}/{K} panel genes (rest zero); "
          f"encoder forward {tuple(z.shape)} + backward OK")
    print("SMOKE OK")


if __name__ == "__main__":
    main()
