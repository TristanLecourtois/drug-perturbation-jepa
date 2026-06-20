"""
precompute_gene_sources.py — build the multi-source gene-init tables for the
SetTransformer encoder (scGPT / KGE / ESM2 / Evo2), ALIGNED to the gene panel.

Output (the contract expected by examples/tahoe/main.py & ground.py):

    gene_sources.pt  ==  { "scgpt": Tensor[K, d_scgpt],
                           "kge":   Tensor[K, d_kge],
                           "esm2":  Tensor[K, d_esm2],
                           "evo2":  Tensor[K, d_evo2] }

  * K               = len(cache["panel"])  (same gene order as the training cache)
  * row g           = the embedding of panel gene g in that source
  * MISSING gene    = an all-zero row (the encoder's learned per-gene id still fires,
                      and the learned per-source projection can ignore zeros).

The encoder consumes it via  `model.encoder=settransformer data.gene_sources=<this .pt>`;
SetTransformer.register_gene_source keeps the table FROZEN and learns only a [d->d_model]
projection — so adding a source is always safe (never hurts capacity, can only help).

Design: each source is a small ADAPTER that returns a `{gene_key: vector}` dict; a single
aligner maps it onto the panel. An adapter whose input artifact is absent prints exactly
what to download and is SKIPPED — so this script always succeeds and writes whatever
sources are available right now. Run it again after downloading more; it merges.

  # one command, once you have the artifacts on disk (e.g. on Dalia):
  python -m eb_jepa.datasets.tahoe.precompute_gene_sources \
      --cache       $WORK/tahoe/cache.pt \
      --gene-meta   $WORK/tahoe/gene_metadata.parquet \
      --sources     scgpt kge esm2 evo2 \
      --scgpt-ckpt  $WORK/models/scGPT_human/best_model.pt \
      --scgpt-vocab $WORK/models/scGPT_human/vocab.json \
      --kge-file    $WORK/kg/primekg_gene_emb.pt \
      --esm2-emb    $WORK/esm2/gene_esm2.pt \
      --evo2-emb    $WORK/evo2/gene_evo2.pt \
      --out         artifacts/tahoe/gene_sources.pt
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch


# --------------------------------------------------------------------------- #
# Panel identity: map panel token_ids -> gene symbol / ensembl id             #
# --------------------------------------------------------------------------- #
def load_panel_keys(cache_path, gene_meta_path):
    """Return (panel_token_ids[K], symbols[K], ensembl[K]).

    The training cache stores `panel` = int64 gene token_ids. External sources are
    keyed by gene SYMBOL (scGPT, most KGE) or ENSEMBL id (Evo2), so we need the
    Tahoe `gene_metadata` table (token_id, gene_symbol, ensembl_id) to translate.
    Missing metadata -> symbols/ensembl are empty strings (those sources skip).
    """
    blob = torch.load(cache_path, weights_only=False)
    panel = blob["panel"].numpy().astype(np.int64)
    K = len(panel)
    symbols = [""] * K
    ensembl = [""] * K
    if gene_meta_path and os.path.exists(gene_meta_path):
        meta = _read_table(gene_meta_path)
        tid2sym = {int(t): str(s) for t, s in zip(meta["token_id"], meta["gene_symbol"])}
        tid2ens = {int(t): str(e) for t, e in zip(meta["token_id"], meta.get("ensembl_id", meta["gene_symbol"]))}
        symbols = [tid2sym.get(int(t), "") for t in panel]
        ensembl = [tid2ens.get(int(t), "") for t in panel]
        cov = sum(bool(s) for s in symbols)
        print(f"[panel] K={K} | symbol coverage {cov}/{K} ({100*cov/K:.1f}%) via {gene_meta_path}")
    else:
        print(f"[panel] K={K} | NO gene_metadata given -> symbol/ensembl sources will skip. "
              f"(pass --gene-meta <gene_metadata.parquet> to enable scGPT/KGE/ESM2/Evo2 alignment)")
    return panel, symbols, ensembl


def _read_table(path):
    """Load a parquet/csv gene_metadata table as a dict of columns (no hard pandas dep at import)."""
    if path.endswith(".parquet"):
        import pyarrow.parquet as pq
        return {k: list(v) for k, v in pq.read_table(path).to_pydict().items()}
    import csv
    cols: dict[str, list] = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            for k, v in row.items():
                cols.setdefault(k, []).append(v)
    return cols


def align_to_panel(keyed: dict, panel_keys: list, dim: int) -> tuple[torch.Tensor, int]:
    """Map a {gene_key: vector} dict onto the panel order -> (Tensor[K, dim], n_covered).

    Case-insensitive key match; missing genes get a zero row. `dim` is inferred from
    the first vector if None is passed.
    """
    if dim is None:
        dim = len(next(iter(keyed.values())))
    lut = {str(k).upper(): np.asarray(v, dtype=np.float32) for k, v in keyed.items()}
    table = np.zeros((len(panel_keys), dim), dtype=np.float32)
    covered = 0
    for i, key in enumerate(panel_keys):
        vec = lut.get(str(key).upper())
        if vec is not None and vec.shape[0] == dim:
            table[i] = vec
            covered += 1
    return torch.from_numpy(table), covered


# --------------------------------------------------------------------------- #
# Source adapters: each returns {gene_key: vector} or None (skip), + the key type #
# --------------------------------------------------------------------------- #
def adapter_scgpt(args):
    """scGPT gene embedding table, keyed by gene SYMBOL.

    scGPT ships a token vocab (vocab.json: symbol -> idx) and a checkpoint whose
    gene-embedding weight is the per-token table. We read both and emit symbol->vec.
    Checkpoint download: https://github.com/bowang-lab/scGPT (scGPT_human).
    """
    ckpt, vocab = args.scgpt_ckpt, args.scgpt_vocab
    if not (ckpt and vocab and os.path.exists(ckpt) and os.path.exists(vocab)):
        print("[scgpt] SKIP — need --scgpt-ckpt and --scgpt-vocab "
              "(download scGPT_human from github.com/bowang-lab/scGPT).")
        return None, "symbol"
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = sd.get("model_state_dict", sd)
    # gene embedding weight name varies across releases; pick the first that matches.
    key = next((k for k in sd if "gene_encoder.embedding.weight" in k
                or k.endswith("encoder.embedding.weight")), None)
    if key is None:
        print(f"[scgpt] SKIP — no gene-embedding weight found in {ckpt} (keys: {list(sd)[:5]}…)")
        return None, "symbol"
    W = sd[key].float().numpy()                       # [n_vocab, d]
    vmap = json.load(open(vocab))                     # symbol -> idx
    keyed = {sym: W[i] for sym, i in vmap.items() if 0 <= int(i) < W.shape[0]}
    print(f"[scgpt] loaded {len(keyed)} gene vectors (d={W.shape[1]}) from {ckpt}")
    return keyed, "symbol"


def adapter_kge(args):
    """Biomedical knowledge-graph embeddings (PrimeKG / Hetionet), keyed by SYMBOL.

    Accepts a .pt/.npz/.tsv of {gene_symbol: vector}. Build one from PrimeKG with any
    KGE method (TransE/ComplEx/node2vec) and dump the gene nodes keyed by symbol.
    """
    f = args.kge_file
    if not (f and os.path.exists(f)):
        print("[kge] SKIP — need --kge-file (a {symbol: vector} table from PrimeKG/Hetionet).")
        return None, "symbol"
    keyed = _load_keyed_table(f)
    print(f"[kge] loaded {len(keyed)} gene vectors from {f}")
    return keyed, "symbol"


def adapter_esm2(args):
    """ESM2 protein embeddings, keyed by gene SYMBOL (gene -> canonical protein -> ESM2).

    Heavy to compute, so we expect a PRECOMPUTED {symbol: vector} table (--esm2-emb).
    To build it: map each panel gene to its canonical protein sequence, run ESM2
    (facebookresearch/esm, e.g. esm2_t33_650M_UR50D), mean-pool over residues.
    """
    f = args.esm2_emb
    if not (f and os.path.exists(f)):
        print("[esm2] SKIP — need --esm2-emb (precomputed {symbol: vector}; "
              "compute via facebookresearch/esm on canonical protein sequences).")
        return None, "symbol"
    keyed = _load_keyed_table(f)
    print(f"[esm2] loaded {len(keyed)} protein vectors from {f}")
    return keyed, "symbol"


def adapter_evo2(args):
    """Evo2 DNA embeddings, keyed by ENSEMBL id (gene body DNA -> Evo2, layer mean-pool).

    The DNA modality eb_jepa uses and the one our design is otherwise missing. Heavy,
    so we expect a PRECOMPUTED {ensembl_id: vector} table (--evo2-emb). Build it by
    running arcinstitute/evo2 over each gene's canonical transcript DNA (layer 24,
    mean-pooled), keyed by ensembl_id.
    """
    f = args.evo2_emb
    if not (f and os.path.exists(f)):
        print("[evo2] SKIP — need --evo2-emb (precomputed {ensembl_id: vector}; "
              "compute via arcinstitute/evo2 on canonical transcript DNA).")
        return None, "ensembl"
    keyed = _load_keyed_table(f)
    print(f"[evo2] loaded {len(keyed)} DNA vectors from {f}")
    return keyed, "ensembl"


def _load_keyed_table(path):
    """Load {key: vector} from .pt (dict or (keys, matrix)), .npz, or .tsv/.csv."""
    if path.endswith(".pt"):
        obj = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(obj, dict) and "keys" in obj and "emb" in obj:
            return {k: np.asarray(v) for k, v in zip(obj["keys"], np.asarray(obj["emb"]))}
        return {k: np.asarray(v) for k, v in obj.items()}
    if path.endswith(".npz"):
        z = np.load(path, allow_pickle=True)
        return {k: v for k, v in zip(z["keys"], z["emb"])}
    sep = "\t" if path.endswith(".tsv") else ","
    keyed = {}
    with open(path) as fh:
        for line in fh:
            parts = line.rstrip("\n").split(sep)
            keyed[parts[0]] = np.asarray(parts[1:], dtype=np.float32)
    return keyed


ADAPTERS = {"scgpt": adapter_scgpt, "kge": adapter_kge, "esm2": adapter_esm2, "evo2": adapter_evo2}


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", required=True, help="training cache .pt (provides `panel`)")
    ap.add_argument("--gene-meta", default="", help="gene_metadata parquet/csv (token_id, gene_symbol, ensembl_id)")
    ap.add_argument("--sources", nargs="+", default=["scgpt", "kge", "esm2", "evo2"],
                    choices=list(ADAPTERS))
    ap.add_argument("--out", default="artifacts/tahoe/gene_sources.pt")
    # per-source inputs (all optional; a source whose input is missing is skipped)
    ap.add_argument("--scgpt-ckpt", default=""); ap.add_argument("--scgpt-vocab", default="")
    ap.add_argument("--kge-file", default="")
    ap.add_argument("--esm2-emb", default="")
    ap.add_argument("--evo2-emb", default="")
    ap.add_argument("--merge", action="store_true",
                    help="merge into an existing --out instead of overwriting")
    args = ap.parse_args()

    panel, symbols, ensembl = load_panel_keys(args.cache, args.gene_meta)
    keys_by_type = {"symbol": symbols, "ensembl": ensembl}

    tables = {}
    if args.merge and os.path.exists(args.out):
        tables = torch.load(args.out, weights_only=False)
        print(f"[merge] starting from {len(tables)} existing source(s) in {args.out}")

    for name in args.sources:
        keyed, key_type = ADAPTERS[name](args)
        if keyed is None:
            continue
        panel_keys = keys_by_type[key_type]
        if not any(panel_keys):
            print(f"[{name}] SKIP — panel has no {key_type} keys (give --gene-meta).")
            continue
        table, covered = align_to_panel(keyed, panel_keys, dim=None)
        if covered == 0:
            print(f"[{name}] SKIP — 0/{len(panel)} panel genes matched (key mismatch?).")
            continue
        tables[name] = table
        print(f"[{name}] -> table[{tuple(table.shape)}], covered {covered}/{len(panel)} "
              f"({100*covered/len(panel):.1f}%); missing genes are zero rows.")

    if not tables:
        print("\nNo sources built (all inputs absent). Nothing written. "
              "Re-run with at least one --*-emb / --*-ckpt artifact present.")
        return

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save(tables, args.out)
    print(f"\nsaved {list(tables)} -> {args.out}")
    print("use it:  python -m examples.tahoe.main --fname examples/tahoe/cfgs/train.yaml \\")
    print(f"           model.encoder=settransformer data.gene_sources={args.out}")


if __name__ == "__main__":
    main()
