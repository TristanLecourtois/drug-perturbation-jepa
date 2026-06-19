"""
Microbiome-JEPA: an energy-based JEPA world model for gut bacterial communities.

This is the headline EB-JEPA challenge -- carrying the recipe to a new, noisy,
high-dimensional biological modality -- with TWO microbiome-specific losses:

  * AlphaDiversityLoss  (diversity preservation): the latent must keep ecological
    alpha-diversity decodable, so imagined futures don't collapse the community's
    diversity structure.
  * PhyloDispersionLoss (soft-UniFrac): latent geometry must respect microbial
    phylogeny, using abundance-weighted mean ProkBERT embeddings as a tree-free
    phylogenetic descriptor.

Architecture (pure JEPA, no imposter / no reconstruction):
  encoder   = SetEncoder           (permutation-invariant, abundance-weighted DeepSets)
  predictor = RNNPredictor         (action-conditioned latent dynamics; diet/feeding)
  regularizer = VC_IDM_Sim_Regularizer (var+cov anti-collapse + temporal + inverse-dynamics)
  predcost  = SquareLossSeq        (prediction energy in latent space)

Run (smoke test):
  python -m examples.microbiome.main --fname examples/microbiome/cfgs/train.yaml \
      optim.epochs=2 logging.log_wandb=false
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from eb_jepa.architectures import (
    InverseDynamicsModel,
    Projector,
    RNNPredictor,
    SetEncoder,
)
from eb_jepa.datasets.microbiome.dataset import MicrobiomeConfig, make_loaders
from eb_jepa.jepa import JEPA
from eb_jepa.losses import (
    AlphaDiversityLoss,
    PhyloDispersionLoss,
    SquareLossSeq,
    VC_IDM_Sim_Regularizer,
)
from eb_jepa.schedulers import CosineWithWarmup


def build_jepa(cfg, action_dim, device):
    D = cfg.model.dstc
    encoder = SetEncoder(emb_dim=cfg.model.emb_dim, h_d=cfg.model.henc, out_d=D)
    predictor = RNNPredictor(hidden_size=D, action_dim=action_dim,
                             final_ln=nn.LayerNorm(D))
    action_encoder = nn.Identity()
    idm = InverseDynamicsModel(state_dim=D, hidden_dim=cfg.model.hpre, action_dim=action_dim)
    projector = Projector(f"{D}-{4*D}-{4*D}")
    regularizer = VC_IDM_Sim_Regularizer(
        cov_coeff=cfg.loss.cov_coeff,
        std_coeff=cfg.loss.std_coeff,
        sim_coeff_t=cfg.loss.sim_coeff_t,
        idm_coeff=cfg.loss.idm_coeff,
        idm=idm,
        projector=projector,
        first_t_only=False,
    )
    predcost = SquareLossSeq()
    return JEPA(encoder, action_encoder, predictor, regularizer, predcost).to(device)


@torch.no_grad()
def _gather(jepa, alpha_loss, phylo_loss, loader, cfg, device):
    """Pooled latents + labels + prediction/identity/div/phylo accumulators."""
    tot = {"pred": 0.0, "ident": 0.0, "div": 0.0, "phylo": 0.0, "tvar": 0.0, "n": 0}
    feats, labels, ages = [], [], []
    for batch in loader:
        obs = batch["observations"].to(device)
        act = batch["actions"].to(device)
        state = jepa.encoder(obs)  # [B, D, T, 1, 1]
        preds, _ = jepa.unroll(obs, act, nsteps=cfg.model.nsteps,
                               unroll_mode="autoregressive", compute_loss=False,
                               return_all_steps=False)
        Tn = min(state.shape[2], preds.shape[2])
        b = obs.shape[0]
        tot["pred"] += torch.mean((preds[:, :, 1:Tn] - state[:, :, 1:Tn]) ** 2).item() * b
        tot["ident"] += torch.mean((state[:, :, : Tn - 1] - state[:, :, 1:Tn]) ** 2).item() * b
        # temporal variance: how much a trajectory moves in latent over time (collapse monitor)
        tot["tvar"] += state[..., 0, 0].var(dim=2).mean().item() * b
        tot["div"] += alpha_loss(state, batch["diversity"].to(device)).item() * b
        tot["phylo"] += phylo_loss(state, batch["phylo"].to(device)).item() * b
        tot["n"] += b
        feats.append(state.mean(dim=2)[..., 0, 0].cpu().numpy())  # [B, D] subject-pooled
        labels.append(batch["label"].numpy())
        ages.append(batch["age"].numpy())
    return (tot, np.concatenate(feats), np.concatenate(labels).astype(int),
            np.concatenate(ages))


def evaluate(jepa, alpha_loss, phylo_loss, val_loader, cfg, device, fit_loader=None):
    """Val metrics. The T1D probe is fit on `fit_loader` (train) and scored on
    `val_loader` -- subject-disjoint, no leakage."""
    jepa.eval()
    tot, Xv, yv, av = _gather(jepa, alpha_loss, phylo_loss, val_loader, cfg, device)
    n = max(1, tot["n"])
    metrics = {k: v / n for k, v in tot.items() if k != "n"}
    metrics["skill_vs_identity"] = metrics["ident"] / max(1e-9, metrics["pred"])
    if fit_loader is not None:
        try:
            from sklearn.linear_model import LogisticRegression, Ridge
            from sklearn.metrics import r2_score, roc_auc_score
            _, Xt, yt, at = _gather(jepa, alpha_loss, phylo_loss, fit_loader, cfg, device)
            # microbiome aging clock: predict host age from the community latent
            reg = Ridge(alpha=1.0).fit(Xt, at)
            metrics["age_r2"] = float(r2_score(av, reg.predict(Xv)))
            # secondary, harder probe: host T1D phenotype
            if len(np.unique(yt)) == 2 and len(np.unique(yv)) == 2:
                clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(Xt, yt)
                metrics["t1d_auroc"] = float(roc_auc_score(yv, clf.predict_proba(Xv)[:, 1]))
        except Exception:
            pass
    return metrics


def run(fname, overrides):
    cfg = OmegaConf.load(fname)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.meta.seed)
    np.random.seed(cfg.meta.seed)

    dcfg = MicrobiomeConfig(
        cache_path=cfg.data.cache_path, n_window=cfg.data.n_window,
        n_max=cfg.data.n_max, emb_dim=cfg.model.emb_dim,
        val_fraction=cfg.data.val_fraction, seed=cfg.meta.seed,
    )
    train_ds, val_ds, train_loader, val_loader = make_loaders(
        dcfg, batch_size=cfg.data.batch_size, num_workers=cfg.data.num_workers
    )
    A = train_ds.action_dim
    print(f"== microbiome-jepa | device={device} | action_dim={A} | "
          f"train_windows={len(train_ds)} val_windows={len(val_ds)} ==")

    jepa = build_jepa(cfg, A, device)
    alpha_loss = AlphaDiversityLoss(state_dim=cfg.model.dstc).to(device)
    phylo_loss = PhyloDispersionLoss().to(device)

    params = list(jepa.parameters()) + list(alpha_loss.parameters())
    opt = torch.optim.AdamW(params, lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay)
    sched = CosineWithWarmup(opt, total_steps=max(1, len(train_loader) * cfg.optim.epochs),
                             warmup_ratio=0.1, min_lr=cfg.optim.lr * 0.01)

    ld, lp = cfg.loss.div_coeff, cfg.loss.phylo_coeff
    for ep in range(1, cfg.optim.epochs + 1):
        jepa.train()
        agg = {"loss": 0.0, "rloss": 0.0, "ploss": 0.0, "div": 0.0, "phylo": 0.0, "n": 0}
        for batch in train_loader:
            obs = batch["observations"].to(device)
            act = batch["actions"].to(device)
            preds, (loss, rloss, runw, rdict, ploss) = jepa.unroll(
                obs, act, nsteps=cfg.model.nsteps, unroll_mode="autoregressive",
                compute_loss=True,
            )
            state = jepa.encoder(obs)
            l_div = alpha_loss(state, batch["diversity"].to(device))
            l_phylo = phylo_loss(state, batch["phylo"].to(device))
            total = loss + ld * l_div + lp * l_phylo

            opt.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()

            b = obs.shape[0]
            ploss_val = ploss.item() if torch.is_tensor(ploss) else float(ploss)
            agg["loss"] += total.item() * b; agg["rloss"] += rloss.item() * b
            agg["ploss"] += ploss_val * b; agg["div"] += l_div.item() * b
            agg["phylo"] += l_phylo.item() * b; agg["n"] += b
        n = max(1, agg["n"])
        val = evaluate(jepa, alpha_loss, phylo_loss, val_loader, cfg, device,
                       fit_loader=train_loader)
        print(f"[ep {ep:03d}] train loss={agg['loss']/n:.3f} pred={agg['ploss']/n:.4f} "
              f"div={agg['div']/n:.4f} phylo={agg['phylo']/n:.4f} reg={agg['rloss']/n:.3f} "
              f"|| val skill={val['skill_vs_identity']:.3f}x tvar={val['tvar']:.3f} "
              f"age_r2={val.get('age_r2', float('nan')):.3f} "
              f"t1d_auroc={val.get('t1d_auroc', float('nan')):.3f}")

    ckpt_dir = os.environ.get("EBJEPA_CKPTS", "checkpoints/microbiome")
    os.makedirs(ckpt_dir, exist_ok=True)
    out = os.path.join(ckpt_dir, "microbiome_jepa.pt")
    torch.save({"jepa": jepa.state_dict(), "cfg": OmegaConf.to_container(cfg),
                "milk_vocab": train_ds.milk_vocab}, out)
    print(f"saved -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fname", default="examples/microbiome/cfgs/train.yaml")
    args, rest = ap.parse_known_args()
    run(args.fname, rest)
