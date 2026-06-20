"""
Tahoe-JEPA: a cell-state Joint-Embedding Predictive model on Tahoe-100M.

A scaled, differentiated take on GeneJEPA (Litman et al. 2025). Same JEPA premise
(predict in representation space, no count reconstruction) but THREE deliberate
differences we ablate:
  1. SIGReg (BCS) regularizer        instead of GeneJEPA's VICReg
  2. PathwayCoherenceLoss            a gene-program structural prior they don't have
  3. biological two-view augmentation (gene dropout + noise) instead of masked-prediction

Encoder = MLP over a top-K gene panel; two augmented views are pulled together by
the regularizer; a linear probe on frozen embeddings is compared to raw-expression
and PCA baselines (the GeneJEPA evaluation protocol).

  python -m examples.tahoe.main --fname examples/tahoe/cfgs/train.yaml optim.epochs=30
"""

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from eb_jepa.architectures import Projector, SetTransformer
from eb_jepa.datasets.tahoe.dataset import TahoeConfig, make_loaders
from eb_jepa.losses import BCS, VICRegLoss, PathwayCoherenceLoss
from eb_jepa.schedulers import CosineWithWarmup


class CellEncoder(nn.Module):
    """MLP cell-state encoder over a fixed gene panel -> latent z."""
    def __init__(self, k, h, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(k, h), nn.BatchNorm1d(h), nn.GELU(),
            nn.Linear(h, h), nn.BatchNorm1d(h), nn.GELU(),
            nn.Linear(h, d),
        )

    def forward(self, x):
        return self.net(x)


@torch.no_grad()
def embed(enc, X, device, bs=4096):
    enc.eval()
    out = []
    for i in range(0, len(X), bs):
        out.append(enc(X[i:i+bs].to(device)).cpu())
    return torch.cat(out).numpy()


def probe(name, Xtr, ytr, Xva, yva):
    """Linear probe (logreg) macro-F1 + accuracy; drops classes absent from train."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score, accuracy_score
    keep = ytr >= 0
    if keep.sum() < 10 or len(np.unique(ytr[keep])) < 2:
        return None
    clf = LogisticRegression(max_iter=300, n_jobs=-1).fit(Xtr[keep], ytr[keep])
    m = yva >= 0
    pred = clf.predict(Xva[m])
    return dict(task=name, macro_f1=float(f1_score(yva[m], pred, average="macro")),
                acc=float(accuracy_score(yva[m], pred)))


def run(fname, overrides):
    cfg = OmegaConf.load(fname)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.meta.seed); np.random.seed(cfg.meta.seed)

    dcfg = TahoeConfig(cache_path=cfg.data.cache_path, drop_frac=cfg.data.drop_frac,
                       noise_std=cfg.data.noise_std, val_fraction=cfg.data.val_fraction,
                       seed=cfg.meta.seed, pathways_path=cfg.data.get("pathways", ""))
    tr, va, train_loader, val_loader = make_loaders(dcfg, batch_size=cfg.data.batch_size)
    K, D = tr.K, cfg.model.dstc
    print(f"== tahoe-jepa | device={device} | cells train={len(tr)} val={len(va)} | "
          f"genes={K} pathways={tr.n_modules}({tr.pathway_kind}) reg={cfg.loss.reg} ==")

    enc_kind = cfg.model.get("encoder", "mlp")
    if enc_kind == "settransformer":
        enc = SetTransformer(
            n_genes=K, out_d=D, d_model=cfg.model.get("d_model", 192),
            n_latents=cfg.model.get("n_latents", 32), depth=cfg.model.get("depth", 2),
            heads=cfg.model.get("heads", 4),
        )
        # optional: real per-gene source tables (scGPT / KGE / ESM2) aligned to the
        # gene panel, saved as {name: tensor[K, d]} (torch .pt). Frozen; projection learned.
        gs = cfg.data.get("gene_sources", "")
        if gs and os.path.exists(gs):
            tables = torch.load(gs, weights_only=False)
            for name, tbl in tables.items():
                enc.register_gene_source(name, tbl)
            print(f"  gene-init sources: {list(tables)}")
        enc = enc.to(device)
    else:
        enc = CellEncoder(K, cfg.model.henc, D).to(device)
    proj = Projector(f"{D}-{cfg.model.proj}-{cfg.model.proj}").to(device)
    if cfg.loss.reg == "sigreg":
        reg = BCS(num_slices=cfg.loss.num_slices, lmbd=cfg.loss.lmbd).to(device)
    else:
        reg = VICRegLoss(std_coeff=cfg.loss.std_coeff, cov_coeff=cfg.loss.cov_coeff).to(device)
    pathway = PathwayCoherenceLoss().to(device)
    pw = cfg.loss.pathway_coeff

    params = list(enc.parameters()) + list(proj.parameters())
    opt = torch.optim.AdamW(params, lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay)
    sched = CosineWithWarmup(opt, total_steps=max(1, len(train_loader) * cfg.optim.epochs),
                             warmup_ratio=0.1, min_lr=cfg.optim.lr * 0.01)

    for ep in range(1, cfg.optim.epochs + 1):
        enc.train(); proj.train()
        agg = {"reg": 0.0, "path": 0.0, "n": 0}
        for v1, v2, P, *_ in train_loader:
            v1, v2, P = v1.to(device), v2.to(device), P.to(device)
            z1, z2 = enc(v1), enc(v2)
            rl = reg(proj(z1), proj(z2))["loss"]
            pl = pathway(z1, P)
            total = rl + pw * pl
            opt.zero_grad(set_to_none=True); total.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step()
            b = v1.shape[0]; agg["reg"] += rl.item()*b; agg["path"] += pl.item()*b; agg["n"] += b
        n = max(1, agg["n"])
        msg = f"[ep {ep:03d}] reg={agg['reg']/n:.4f} pathway={agg['path']/n:.4f}"
        if ep % cfg.optim.probe_every == 0 or ep == cfg.optim.epochs:
            Etr = embed(enc, tr.X[tr.ids], device); Eva = embed(enc, va.X[va.ids], device)
            for tname, arr in [("drug", "drug"), ("moa", "moa"), ("cell_line", "cell_line")]:
                ytr = getattr(tr, arr)[tr.ids].numpy(); yva = getattr(va, arr)[va.ids].numpy()
                if tname == "moa":  # drop the dominant 'unclear' class
                    unclear = tr.moa_names.index("unclear") if "unclear" in tr.moa_names else -999
                    ytr = np.where(ytr == unclear, -1, ytr); yva = np.where(yva == unclear, -1, yva)
                r = probe(tname, Etr, ytr, Eva, yva)
                if r: msg += f" | {tname}: F1={r['macro_f1']:.3f} acc={r['acc']:.3f}"
        print(msg)

    # final baselines vs ours (GeneJEPA protocol: linear probe on frozen features)
    print("\n== final linear-probe comparison (macro-F1) ==")
    Xtr_raw = tr.X[tr.ids].numpy(); Xva_raw = va.X[va.ids].numpy()
    Etr = embed(enc, tr.X[tr.ids], device); Eva = embed(enc, va.X[va.ids], device)
    from sklearn.decomposition import PCA
    pca = PCA(n_components=min(D, 50)).fit(Xtr_raw)
    Ptr, Pva = pca.transform(Xtr_raw), pca.transform(Xva_raw)
    for tname, arr in [("cell_line", "cell_line"), ("drug", "drug"), ("moa", "moa")]:
        ytr = getattr(tr, arr)[tr.ids].numpy(); yva = getattr(va, arr)[va.ids].numpy()
        if tname == "moa":
            u = tr.moa_names.index("unclear") if "unclear" in tr.moa_names else -999
            ytr = np.where(ytr == u, -1, ytr); yva = np.where(yva == u, -1, yva)
        for feat, Xt, Xv in [("raw", Xtr_raw, Xva_raw), ("pca50", Ptr, Pva), ("JEPA(ours)", Etr, Eva)]:
            r = probe(tname, Xt, ytr, Xv, yva)
            if r: print(f"  {tname:10s} {feat:11s} F1={r['macro_f1']:.3f} acc={r['acc']:.3f}")

    ckpt = os.environ.get("EBJEPA_CKPTS", "checkpoints/tahoe")
    os.makedirs(ckpt, exist_ok=True)
    torch.save({"enc": enc.state_dict(), "cfg": OmegaConf.to_container(cfg)},
               os.path.join(ckpt, "tahoe_jepa.pt"))
    print(f"saved -> {os.path.join(ckpt, 'tahoe_jepa.pt')}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fname", default="examples/tahoe/cfgs/train.yaml")
    args, rest = ap.parse_known_args()
    run(args.fname, rest)
