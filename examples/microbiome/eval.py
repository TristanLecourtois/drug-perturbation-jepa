"""
Evaluate a trained Microbiome-JEPA: report latent prediction skill + the
downstream T1D linear probe, and draw the latent space (PCA) colored by host
phenotype and by alpha-diversity.

  python -m examples.microbiome.eval --ckpt checkpoints/microbiome/microbiome_jepa.pt
"""

import argparse
import os

import numpy as np
import torch
from omegaconf import OmegaConf

from eb_jepa.datasets.microbiome.dataset import MicrobiomeConfig, make_loaders
from examples.microbiome.main import build_jepa, evaluate
from eb_jepa.losses import AlphaDiversityLoss, PhyloDispersionLoss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/microbiome/microbiome_jepa.pt")
    ap.add_argument("--out", default="checkpoints/microbiome/latent_space.png")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    blob = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(blob["cfg"])

    dcfg = MicrobiomeConfig(
        cache_path=cfg.data.cache_path, n_window=cfg.data.n_window,
        n_max=cfg.data.n_max, emb_dim=cfg.model.emb_dim,
        val_fraction=cfg.data.val_fraction, seed=cfg.meta.seed,
    )
    train_ds, val_ds, _, val_loader = make_loaders(dcfg, batch_size=cfg.data.batch_size)
    jepa = build_jepa(cfg, train_ds.action_dim, device)
    jepa.load_state_dict(blob["jepa"])
    jepa.eval()
    alpha = AlphaDiversityLoss(cfg.model.dstc).to(device)
    phylo = PhyloDispersionLoss().to(device)

    metrics = evaluate(jepa, alpha, phylo, val_loader, cfg, device)
    print("== val metrics ==")
    for k, v in metrics.items():
        print(f"  {k:18s} {v:.4f}")

    # gather pooled latents for the figure
    feats, labels, divs = [], [], []
    with torch.no_grad():
        for batch in val_loader:
            state = jepa.encoder(batch["observations"].to(device))  # [B,D,T,1,1]
            feats.append(state.mean(dim=2)[..., 0, 0].cpu().numpy())
            labels.append(batch["label"].numpy())
            divs.append(batch["diversity"].mean(1).numpy())
    X = np.concatenate(feats); y = np.concatenate(labels); dv = np.concatenate(divs)
    Xc = X - X.mean(0); _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    P = Xc @ Vt[:2].T

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    s0 = ax[0].scatter(P[:, 0], P[:, 1], c=dv, cmap="viridis", s=14)
    ax[0].set_title("latent space — colored by α-diversity"); plt.colorbar(s0, ax=ax[0])
    for lab, col, name in [(0, "tab:blue", "no T1D"), (1, "tab:red", "T1D")]:
        m = y == lab
        ax[1].scatter(P[m, 0], P[m, 1], c=col, s=16, alpha=0.7, label=name)
    ax[1].set_title(f"latent space — host phenotype (AUROC={metrics.get('t1d_auroc', float('nan')):.2f})")
    ax[1].legend()
    for a in ax:
        a.set_xlabel("PC1"); a.set_ylabel("PC2")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.tight_layout(); fig.savefig(args.out, dpi=150)
    print(f"figure -> {args.out}")


if __name__ == "__main__":
    main()
