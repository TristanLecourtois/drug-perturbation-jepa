"""
dataset.py - Tahoe-100M cell-state dataset for two-view (image-JEPA style) SSL.

A "cell" is a dense expression vector over a fixed top-K gene panel. SSL views
are made by ecologically/biologically plausible augmentations: random gene
DROPOUT (mimics single-cell sequencing dropout) + multiplicative noise. The JEPA
pulls the two views of the same cell together (predictor = identity) and VICReg
prevents collapse. No temporal axis -> no temporal collapse (unlike the
microbiome world model), which is exactly why this modality trains cleanly.

__getitem__ -> (view1[K], view2[K], drug_id, moa_id, cell_line_id)
"""

import os
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


@dataclass
class TahoeConfig:
    cache_path: str = "artifacts/tahoe/cache.pt"
    drop_frac: float = 0.3      # gene dropout per view (sequencing-dropout augmentation)
    noise_std: float = 0.1      # multiplicative log-noise
    val_fraction: float = 0.2
    seed: int = 0
    split: str = "train"        # "train" | "val"
    pathways_path: str = ""     # optional pathways.pt {membership[K,M], names}; ""=KMeans modules


class TahoeDataset(Dataset):
    def __init__(self, cfg: TahoeConfig, stats=None):
        self.cfg = cfg
        blob = torch.load(cfg.cache_path, weights_only=False)
        X = blob["X"].float()                      # [N, K]
        # gene counts -> log1p; pretrained embeddings are already real-valued -> keep as is.
        if not blob.get("is_embedding", False):
            X = torch.log1p(torch.clamp(X, min=0.0))
        # then per-feature z-score (mandatory for the regularizer)
        if stats is None:
            mu = X.mean(0, keepdim=True); sd = X.std(0, keepdim=True) + 1e-6
            self.mu, self.sd = mu, sd
        else:
            self.mu, self.sd = stats
        self.X = (X - self.mu) / self.sd
        self.drug = blob["drug"]; self.moa = blob["moa"]; self.cell_line = blob["cell_line"]
        self.drug_names = blob["drug_names"]; self.moa_names = blob["moa_names"]
        self.cl_names = blob["cl_names"]; self.drug_fp = blob["drug_fp"]
        self.K = self.X.shape[1]

        # per-cell pathway descriptor = mean (standardized) expression per program.
        # Programs are EITHER real MSigDB Hallmark sets (if a pathways.pt is given) OR
        # the data-driven KMeans co-expression modules baked into the cache.
        if cfg.pathways_path and os.path.exists(cfg.pathways_path):
            pw = torch.load(cfg.pathways_path, weights_only=False)
            membership = pw["membership"].float()                 # [K, M] (overlapping)
            assert membership.shape[0] == self.K, \
                f"pathways K={membership.shape[0]} != panel K={self.K} (rebuild on this cache)"
            self.pathway_names = pw.get("names")
            self.pathway_kind = "hallmark"
        else:
            self.modules = blob["modules"]                        # [K] KMeans module id per gene
            membership = torch.zeros(self.K, int(blob["n_modules"]))
            membership[torch.arange(self.K), self.modules] = 1.0  # one-hot
            self.pathway_names = None
            self.pathway_kind = "kmeans"
        self.n_modules = membership.shape[1]
        self.P = (self.X @ membership) / membership.sum(0).clamp_min(1.0)  # [N, M] activity

        rng = np.random.default_rng(cfg.seed)
        idx = rng.permutation(len(self.X))
        n_val = int(round(len(idx) * cfg.val_fraction))
        self.ids = idx[n_val:] if cfg.split == "train" else idx[:n_val]

    def stats(self):
        return (self.mu, self.sd)

    def _augment(self, x, rng):
        m = (torch.rand(self.K) > self.cfg.drop_frac).float()
        noise = 1.0 + self.cfg.noise_std * torch.randn(self.K)
        return x * m * noise

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        j = int(self.ids[i])
        x = self.X[j]
        return (self._augment(x, None), self._augment(x, None), self.P[j],
                int(self.drug[j]), int(self.moa[j]), int(self.cell_line[j]))


def make_loaders(cfg: TahoeConfig, batch_size=512, num_workers=0):
    tr = TahoeDataset(TahoeConfig(**{**cfg.__dict__, "split": "train"}))
    va = TahoeDataset(TahoeConfig(**{**cfg.__dict__, "split": "val"}), stats=tr.stats())
    return (tr, va,
            DataLoader(tr, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True),
            DataLoader(va, batch_size=batch_size, shuffle=False, num_workers=num_workers))
