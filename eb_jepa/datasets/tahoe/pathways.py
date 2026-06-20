"""
pathways.py — REAL biology gene programs (MSigDB Hallmark) for PathwayCoherenceLoss,
as a drop-in replacement for the data-driven KMeans co-expression `modules`.

The 50 MSigDB **Hallmark** sets are vendored under ``pathways/hallmark.json``
({pathway_name: [gene_symbol, ...]}). Given the training panel (gene token_ids) and
a token_id->symbol map, we build a panel-aligned **membership matrix** [K, M]:

    membership[g, p] = 1.0  iff panel gene g belongs to hallmark pathway p

This is the SAME object the dataset already builds from KMeans (`onehot` [K, n_mod]),
except: (1) it encodes *real* curated programs, and (2) it is OVERLAPPING — a gene can
belong to several hallmarks (or none), unlike the one-hot KMeans assignment. The rest
of the pipeline is unchanged: per-cell pathway activity P = X @ (membership / colsum),
and PathwayCoherenceLoss matches the latent geometry to P's geometry.
"""
from __future__ import annotations

import json
import os

import torch

_PATHWAY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pathways")
_HALLMARK_JSON = os.path.join(_PATHWAY_DIR, "hallmark.json")


def load_hallmark(path: str | None = None) -> dict[str, list[str]]:
    """Return {pathway_name: [gene_symbol, ...]} for the 50 hallmark sets."""
    with open(path or _HALLMARK_JSON, encoding="utf-8") as f:
        return json.load(f)["sets"]


def build_panel_membership(panel_symbols, path: str | None = None, drop_empty: bool = True):
    """Panel-aligned hallmark membership matrix.

    panel_symbols : list[str] of length K (gene symbol per panel position; "" if unknown).
    Returns (membership[K, M] float tensor, names[M]). With drop_empty, hallmark sets
    that hit no panel gene are removed (keeps M tight to the panel).
    """
    sets = load_hallmark(path)
    names = sorted(sets)
    upper = [str(s).upper() for s in panel_symbols]
    pos = {}
    for i, s in enumerate(upper):
        if s:
            pos.setdefault(s, []).append(i)        # a symbol may map to >1 panel slot
    K = len(panel_symbols)
    cols, kept = [], []
    for name in names:
        col = torch.zeros(K)
        for g in sets[name]:
            for i in pos.get(str(g).upper(), ()):
                col[i] = 1.0
        if drop_empty and col.sum() == 0:
            continue
        cols.append(col)
        kept.append(name)
    if not cols:
        return torch.zeros(K, 0), []
    return torch.stack(cols, dim=1), kept           # [K, M], names
