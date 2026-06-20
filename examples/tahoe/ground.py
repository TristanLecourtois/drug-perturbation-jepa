"""
ground.py - Masked-GENE latent-prediction grounding for the cell-state encoder.

The JEPA-DNA idea (NVIDIA 2026), ported from raw DNA to single-cell RNA:
instead of reconstructing masked tokens, predict the GLOBAL latent of the masked
content against an EMA target encoder (cosine), kept from collapsing by VICReg.
Here the masked unit is a GENE of the top-K panel (not a nucleotide span), which
is exactly what the SetTransformer enables (every gene = a token):

  context encoder f_θ : SetTransformer over the cell with a subset of genes MASKED
  target encoder  f_ξ : EMA copy of f_θ, over the FULL cell (no mask)  [no grad]
  predictor       g_φ : LatentPredictor, pooled-ctx-latent -> pooled-target-latent
  loss                : (1 - cos(g_φ(z_ctx), z_tgt.detach())) + VICReg(var,cov)

Only random gene masking is used (genes in a panel have no sequential order, so
JEPA-DNA's *span* masking has no transcriptomic analogue — stated honestly); the
mask ratio is *scheduled* up over training, like their masking scheduler. We probe
the (EMA target) frozen embedding with a linear classifier vs raw / PCA baselines,
the GeneJEPA / JEPA-DNA linear-probing protocol.

  python -m examples.tahoe.ground --fname examples/tahoe/cfgs/ground.yaml
"""

import argparse
import copy
import os

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from eb_jepa.architectures import SetTransformer, LatentPredictor
from eb_jepa.datasets.tahoe.dataset import TahoeConfig, make_loaders
from eb_jepa.losses import MaskedGeneJEPALoss


@torch.no_grad()
def ema_update(target: nn.Module, online: nn.Module, decay: float):
    for pt, po in zip(target.parameters(), online.parameters()):
        pt.data.mul_(decay).add_(po.data, alpha=1.0 - decay)
    for bt, bo in zip(target.buffers(), online.buffers()):
        bt.data.copy_(bo.data)  # frozen source tables stay identical


def gene_mask(B, K, frac, device, gen):
    """Random per-gene mask, ~frac of K genes masked, >=1 masked & >=1 visible."""
    m = torch.rand(B, K, generator=gen, device=device) < frac
    all_masked = m.all(dim=1)
    if all_masked.any():
        m[all_masked, 0] = False                       # keep >=1 visible
    none_masked = ~m.any(dim=1)
    if none_masked.any():
        m[none_masked, 0] = True                       # keep >=1 masked
    return m


@torch.no_grad()
def embed(enc, X, device, bs=4096):
    enc.eval()
    out = [enc(X[i:i + bs].to(device)).cpu() for i in range(0, len(X), bs)]
    return torch.cat(out).numpy()


def probe(name, Xtr, ytr, Xva, yva):
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

    dcfg = TahoeConfig(cache_path=cfg.data.cache_path, drop_frac=0.0, noise_std=0.0,
                       val_fraction=cfg.data.val_fraction, seed=cfg.meta.seed)
    tr, va, _, _ = make_loaders(dcfg, batch_size=cfg.data.batch_size)
    K, D = tr.K, cfg.model.dstc
    Xtr, Xva = tr.X[tr.ids], va.X[va.ids]                  # standardized expression
    print(f"== tahoe-ground (masked-gene JEPA) | device={device} | cells train={len(tr)} "
          f"val={len(va)} | genes={K} D={D} ==")

    def make_enc():
        e = SetTransformer(n_genes=K, out_d=D, d_model=cfg.model.get("d_model", 192),
                           n_latents=cfg.model.get("n_latents", 32),
                           depth=cfg.model.get("depth", 2), heads=cfg.model.get("heads", 4))
        gs = cfg.data.get("gene_sources", "")
        if gs and os.path.exists(gs):
            for nm, tbl in torch.load(gs, weights_only=False).items():
                e.register_gene_source(nm, tbl)
            print(f"  gene-init sources: registered")
        return e

    online = make_enc().to(device)
    target = copy.deepcopy(online).to(device)
    for p in target.parameters():
        p.requires_grad_(False)
    predictor = LatentPredictor(D, depth=cfg.model.get("pred_depth", 3)).to(device)
    crit = MaskedGeneJEPALoss(var_coeff=cfg.loss.var_coeff, cov_coeff=cfg.loss.cov_coeff).to(device)

    params = list(online.parameters()) + list(predictor.parameters())
    if cfg.optim.get("kind", "adamw") == "sgdm":  # JEPA-DNA found SGDM best for *grounding a pretrained* backbone
        opt = torch.optim.SGD(params, lr=cfg.optim.lr, momentum=0.9, weight_decay=cfg.optim.weight_decay)
    else:
        opt = torch.optim.AdamW(params, lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay)

    gen = torch.Generator(device=device).manual_seed(cfg.meta.seed)
    n = len(Xtr); bs = cfg.data.batch_size
    m0, m1 = cfg.mask.frac_start, cfg.mask.frac_end
    decay = cfg.optim.ema_decay
    E = cfg.optim.epochs

    for ep in range(1, E + 1):
        frac = m0 + (m1 - m0) * (ep - 1) / max(1, E - 1)   # scheduled masking ratio
        online.train(); predictor.train()
        perm = torch.randperm(n)
        agg = {"jepa": 0.0, "var": 0.0, "cov": 0.0, "n": 0}
        for i in range(0, n - bs + 1, bs):
            xb = Xtr[perm[i:i + bs]].to(device)
            gm = gene_mask(xb.shape[0], K, frac, device, gen)
            z_ctx = online(xb, gene_mask=gm)               # context (masked) global
            z_pred = predictor(z_ctx)
            with torch.no_grad():
                z_tgt = target(xb)                         # EMA target, full cell
            z_var = online(xb)                             # deterministic full pass -> VICReg
            out = crit(z_pred, z_tgt, var_emb=z_var)
            opt.zero_grad(set_to_none=True); out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
            ema_update(target, online, decay)
            b = xb.shape[0]
            for k in ("jepa", "var", "cov"):
                agg[k] += out[k].item() * b
            agg["n"] += b
        nn_ = max(1, agg["n"])
        msg = (f"[ep {ep:03d}] mask={frac:.2f} jepa={agg['jepa']/nn_:.4f} "
               f"var={agg['var']/nn_:.4f} cov={agg['cov']/nn_:.4f}")
        if ep % cfg.optim.probe_every == 0 or ep == E:
            Etr, Eva = embed(target, Xtr, device), embed(target, Xva, device)
            for tname in ("drug", "moa", "cell_line"):
                ytr = getattr(tr, tname)[tr.ids].numpy(); yva = getattr(va, tname)[va.ids].numpy()
                if tname == "moa" and "unclear" in tr.moa_names:
                    u = tr.moa_names.index("unclear")
                    ytr = np.where(ytr == u, -1, ytr); yva = np.where(yva == u, -1, yva)
                r = probe(tname, Etr, ytr, Eva, yva)
                if r: msg += f" | {tname}: F1={r['macro_f1']:.3f}"
        print(msg)

    # final comparison vs raw / PCA (linear-probe protocol)
    print("\n== final linear-probe (macro-F1): raw / pca50 / JEPA-ground(EMA target) ==")
    from sklearn.decomposition import PCA
    Xtr_raw, Xva_raw = Xtr.numpy(), Xva.numpy()
    pca = PCA(n_components=min(D, 50)).fit(Xtr_raw)
    Ptr, Pva = pca.transform(Xtr_raw), pca.transform(Xva_raw)
    Etr, Eva = embed(target, Xtr, device), embed(target, Xva, device)
    for tname in ("cell_line", "drug", "moa"):
        ytr = getattr(tr, tname)[tr.ids].numpy(); yva = getattr(va, tname)[va.ids].numpy()
        if tname == "moa" and "unclear" in tr.moa_names:
            u = tr.moa_names.index("unclear")
            ytr = np.where(ytr == u, -1, ytr); yva = np.where(yva == u, -1, yva)
        for feat, Xt, Xv in [("raw", Xtr_raw, Xva_raw), ("pca50", Ptr, Pva), ("JEPA-ground", Etr, Eva)]:
            r = probe(tname, Xt, ytr, Xv, yva)
            if r: print(f"  {tname:10s} {feat:12s} F1={r['macro_f1']:.3f} acc={r['acc']:.3f}")

    ckpt = os.environ.get("EBJEPA_CKPTS", "checkpoints/tahoe"); os.makedirs(ckpt, exist_ok=True)
    torch.save({"target": target.state_dict(), "online": online.state_dict(),
                "cfg": OmegaConf.to_container(cfg)}, os.path.join(ckpt, "tahoe_ground.pt"))
    print(f"saved -> {os.path.join(ckpt, 'tahoe_ground.pt')}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fname", default="examples/tahoe/cfgs/ground.yaml")
    args, rest = ap.parse_known_args()
    run(args.fname, rest)
