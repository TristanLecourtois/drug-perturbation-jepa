"""
dataset.py - Microbiome longitudinal dataset for the EB-JEPA world model.

Each subject is a time-ordered sequence of bacterial communities. A community is
a SET of OTUs; each OTU carries a fixed ProkBERT sequence embedding and its
relative abundance. We emit fixed-length windows so the data slots into the
library's [B, C, T, H, W] convention with H = N (OTU slots), W = 1.

__getitem__ returns a dict:
  observations : [emb_dim + 1, T, N, 1]  (channels = embedding + log-abundance)
  actions      : [A, T]                  (per-step feeding one-hot + dt)
  diversity    : [T]                     (Shannon alpha-diversity target)
  phylo        : [T, emb_dim]            (abundance-weighted mean embedding)
  label        : scalar                  (subject T1D label, for the probe eval)

Default collate stacks these into [B, ...]. Build the cache first with
eb_jepa/datasets/microbiome/precompute.py.
"""

import math
import os
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


@dataclass
class MicrobiomeConfig:
    cache_path: str = "eb_jepa/datasets/microbiome/cache.pt"
    n_window: int = 6          # T timepoints per training window
    stride: int = 1
    n_max: int = 64            # OTU slots per community (top-abundance kept)
    emb_dim: int = 384
    abundance_scale: float = 1.0e4
    split: str = "train"       # "train" | "val"
    val_fraction: float = 0.2
    seed: int = 0
    milk_vocab: Optional[List[str]] = None  # filled at load time


def _milk_categories(subjects):
    cats = sorted({(tp["milk"] or "none") for s in subjects for tp in s["timepoints"]})
    return cats


class MicrobiomeDataset(Dataset):
    def __init__(self, config: MicrobiomeConfig):
        self.cfg = config
        blob = torch.load(config.cache_path, weights_only=False)
        self.emb_table = blob["emb_table"].float()  # [U, 384]
        self.subjects = blob["subjects"]

        # feeding vocabulary -> action one-hot
        self.milk_vocab = config.milk_vocab or _milk_categories(self.subjects)
        self.action_dim = len(self.milk_vocab) + 1  # + dt feature

        # subject-disjoint split
        names = sorted(s["subject"] for s in self.subjects)
        rng = np.random.default_rng(config.seed)
        rng.shuffle(names)
        n_val = max(1, int(round(len(names) * config.val_fraction)))
        val = set(names[:n_val])
        keep = val if config.split == "val" else (set(names) - val)
        self.subjects = [s for s in self.subjects if s["subject"] in keep]

        # window index over (subject, start)
        self.windows = []
        T = config.n_window
        for si, s in enumerate(self.subjects):
            n = len(s["timepoints"])
            if n < T:
                continue
            for start in range(0, n - T + 1, config.stride):
                self.windows.append((si, start))

    def __len__(self):
        return len(self.windows)

    def _action_vec(self, tp, dt):
        v = np.zeros(self.action_dim, dtype=np.float32)
        cat = tp["milk"] or "none"
        if cat in self.milk_vocab:
            v[self.milk_vocab.index(cat)] = 1.0
        v[-1] = dt
        return v

    def __getitem__(self, i):
        cfg = self.cfg
        si, start = self.windows[i]
        s = self.subjects[si]
        tps = s["timepoints"][start : start + cfg.n_window]
        T, N, E = cfg.n_window, cfg.n_max, cfg.emb_dim

        obs = torch.zeros(E + 1, T, N, 1, dtype=torch.float32)
        div = torch.zeros(T, dtype=torch.float32)
        phylo = torch.zeros(T, E, dtype=torch.float32)
        act = torch.zeros(self.action_dim, T, dtype=torch.float32)

        for t, tp in enumerate(tps):
            idx = np.asarray(tp["idx"], dtype=np.int64)
            cnt = np.asarray(tp["cnt"], dtype=np.float32)
            p = cnt / (cnt.sum() + 1e-9)
            # keep the top-N most abundant OTUs (deterministic, dominant taxa)
            if len(idx) > N:
                top = np.argsort(-p)[:N]
                idx, p = idx[top], p[top]
            n = len(idx)
            emb = self.emb_table[torch.from_numpy(idx)]  # [n, E]
            logab = np.log1p(cfg.abundance_scale * p).astype(np.float32)  # >=0, monotone
            obs[:E, t, :n, 0] = emb.t()
            obs[E, t, :n, 0] = torch.from_numpy(logab)
            div[t] = float(tp["div"])
            phylo[t] = torch.from_numpy(np.asarray(tp["phylo"], dtype=np.float32))
            dt = (tps[t + 1]["age"] - tp["age"]) if t + 1 < len(tps) else 0.0
            act[:, t] = torch.from_numpy(self._action_vec(tp, float(dt) / 365.0))

        mean_age = float(np.mean([tp["age"] for tp in tps])) / 365.0  # years
        return {
            "observations": obs,
            "actions": act,
            "diversity": div,
            "phylo": phylo,
            "age": torch.tensor(mean_age, dtype=torch.float32),
            "label": torch.tensor(float(s["label"]), dtype=torch.float32),
        }


def make_loaders(cfg: MicrobiomeConfig, batch_size=64, num_workers=0):
    """Return (train_loader, val_loader) sharing the same feeding vocabulary."""
    train_cfg = MicrobiomeConfig(**{**cfg.__dict__, "split": "train"})
    train_ds = MicrobiomeDataset(train_cfg)
    val_cfg = MicrobiomeConfig(**{**cfg.__dict__, "split": "val", "milk_vocab": train_ds.milk_vocab})
    val_ds = MicrobiomeDataset(val_cfg)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, drop_last=False)
    return train_ds, val_ds, train_loader, val_loader
