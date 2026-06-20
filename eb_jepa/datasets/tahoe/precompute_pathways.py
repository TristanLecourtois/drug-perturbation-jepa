"""
precompute_pathways.py — build the panel-aligned MSigDB Hallmark membership matrix
used (instead of KMeans co-expression modules) by PathwayCoherenceLoss.

Output (the contract the dataset expects via `data.pathways=<this .pt>`):

    pathways.pt  ==  { "membership": Tensor[K, M],   # 1.0 = panel gene g in hallmark p
                       "names":      [str, ...] }     # length M, the kept hallmark names

  K = len(cache["panel"]) (same gene order as the cache); M = #hallmark sets hitting
  the panel. Needs a gene_metadata table to translate panel token_ids -> gene symbols.

  # one command:
  python -m eb_jepa.datasets.tahoe.precompute_pathways \
      --cache artifacts/tahoe/cache.pt \
      --gene-meta $WORK/tahoe/gene_metadata.parquet \
      --out artifacts/tahoe/pathways.pt
"""
from __future__ import annotations

import argparse
import os

import torch

from eb_jepa.datasets.tahoe.pathways import build_panel_membership
from eb_jepa.datasets.tahoe.precompute_gene_sources import load_panel_keys


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", required=True, help="training cache .pt (provides `panel`)")
    ap.add_argument("--gene-meta", required=True, help="gene_metadata parquet/csv (token_id, gene_symbol)")
    ap.add_argument("--hallmark", default="", help="override hallmark.json path (default: vendored)")
    ap.add_argument("--out", default="artifacts/tahoe/pathways.pt")
    ap.add_argument("--keep-empty", action="store_true", help="keep hallmark sets that hit no panel gene")
    args = ap.parse_args()

    _, symbols, _ = load_panel_keys(args.cache, args.gene_meta)
    if not any(symbols):
        raise SystemExit("No gene symbols resolved from --gene-meta; cannot map hallmark gene sets.")

    membership, names = build_panel_membership(
        symbols, path=args.hallmark or None, drop_empty=not args.keep_empty)
    K, M = membership.shape
    per_gene = (membership.sum(1) > 0).sum().item()
    print(f"[pathways] {M} hallmark sets x K={K} genes | "
          f"{per_gene}/{K} panel genes hit >=1 set | mean genes/set={membership.sum(0).mean():.1f}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save({"membership": membership, "names": names}, args.out)
    print(f"saved -> {args.out}")
    print("use it:  python -m examples.tahoe.main --fname examples/tahoe/cfgs/train.yaml \\")
    print(f"           data.pathways={args.out}")


if __name__ == "__main__":
    main()
