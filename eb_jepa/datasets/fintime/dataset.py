"""FinTime dataset — financial multivariate time series (Track 5).

Memmap-backed windows produced once by ``prepare.py`` from the HF dataset
``thesven/fintime-decoder-dataset``. Each window is ``[C variates, T steps]``,
already per-feature z-scored, with ready-made supervised targets.

Data loading is PROVIDED (plumbing). The modelling choices you make on top of
these windows (encoder, SSL objective, probe) live in ``examples/fintime/`` and
are where the ``# TODO``s are.
"""
import os
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class FinTimeConfig:
    data_root: str = "/lustre/work/pdl17890/udl806719/datasets/Finance/fintime_prep"
    split: str = "train"            # train | validation | test
    mode: str = "ssl"               # ssl (unlabeled windows) | supervised ((x, y))
    target: str = "direction"       # direction(2-cls) | return(reg) | ...
    in_channels: int = 87
    # SSL augmentations (light — windows are already z-scored)
    aug_noise_std: float = 0.05
    aug_scale_jitter: float = 0.1
    batch_size: int = 256
    num_workers: int = 8


class FinTimeDataset(torch.utils.data.Dataset):
    def __init__(self, cfg: FinTimeConfig):
        self.cfg = cfg
        xp = os.path.join(cfg.data_root, f"{cfg.split}_X.npy")
        mp = os.path.join(cfg.data_root, f"{cfg.split}_meta.npz")
        if not os.path.exists(xp):
            raise FileNotFoundError(
                f"{xp} not found — generate it once with examples/fintime/prepare.py")
        self.X = np.load(xp, mmap_mode="r")           # [N, C, T] float32
        # Materialize labels into memory: a lazily-loaded .npz is a shared zip handle
        # and concurrent reads from forked DataLoader workers corrupt it (BadZipFile).
        meta = np.load(mp, allow_pickle=True)
        self._y_dir = np.asarray(meta["y_direction"], dtype=np.int64)
        self._y_ret = np.asarray(meta["y_return"], dtype=np.float32)
        self._rng = np.random.default_rng()

    def __len__(self):
        return self.X.shape[0]

    def _augment(self, x):
        c, rng = self.cfg, self._rng
        x = x.copy()
        if c.aug_scale_jitter > 0:
            x *= (1.0 + rng.uniform(-c.aug_scale_jitter, c.aug_scale_jitter,
                                    size=(x.shape[0], 1)).astype(np.float32))
        if c.aug_noise_std > 0:
            x += rng.normal(0, c.aug_noise_std, size=x.shape).astype(np.float32)
        return x

    def __getitem__(self, i):
        x = np.array(self.X[i], dtype=np.float32)     # copy (memmap is read-only)
        if self.cfg.mode == "supervised":
            y = self._y_dir[i] if self.cfg.target == "direction" else self._y_ret[i]
            return torch.from_numpy(x), y
        # SSL: one window (predictive JEPA) — for a two-view objective return
        # two augmented copies instead (see the # TODO in examples/fintime/main.py).
        self._rng = np.random.default_rng(torch.randint(0, 2**31 - 1, (1,)).item())
        return torch.from_numpy(self._augment(x))


def make_loader(cfg: FinTimeConfig, shuffle=None):
    ds = FinTimeDataset(cfg)
    is_train = cfg.split == "train"
    return torch.utils.data.DataLoader(
        ds, batch_size=cfg.batch_size, shuffle=is_train if shuffle is None else shuffle,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=is_train,
        persistent_workers=cfg.num_workers > 0)
