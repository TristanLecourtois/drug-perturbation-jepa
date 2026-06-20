"""
Tahoe perturbation WORLD MODEL — a true EB-JEPA with a frozen pretrained encoder.

  encoder f_θ   = FROZEN MosaicFM-3B embedding (identity over precomputed vectors)
  predictor g_φ = RNNPredictor, ACTION-conditioned on the drug (Morgan fingerprint)
  energy        = || g_φ(z_control, drug) − z_perturbed ||²   via eb_jepa.JEPA.unroll
  biology losses: PerturbationSignatureLoss (drug-signature consistency)
                  + PathwayCoherenceLoss   (gene-program / embedding-module structure)

Only the predictor is trained (the encoder is frozen) — exactly the "frozen
pretrained encoder + learned predictor" world-model setting. The drug induces a
large, controlled state change, so the no-change baseline is beatable (skill > 1),
unlike the slow microbiome trajectories.

  python -m examples.tahoe.perturb --fname examples/tahoe/cfgs/perturb.yaml
"""

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf

from eb_jepa.architectures import RNNPredictor
from eb_jepa.jepa import JEPA
from eb_jepa.losses import SquareLossSeq, PathwayCoherenceLoss, PerturbationSignatureLoss
from eb_jepa.datasets.tahoe.pert_dataset import PertConfig, make_loaders


class FrozenIdentityEncoder(nn.Module):
    """The state IS the frozen MosaicFM embedding -> identity, no trainable params."""
    def forward(self, x):
        return x


class NoReg(nn.Module):
    """No anti-collapse term needed: the encoder is frozen, so it cannot collapse."""
    def forward(self, state, actions=None):
        z = (state.sum() * 0.0)
        return z, z, {}


def _states(obs):
    z_ctrl = obs[:, :, 0, 0, 0]
    z_pert = obs[:, :, 1, 0, 0]
    return z_ctrl, z_pert


@torch.no_grad()
def evaluate(jepa, loader, device, train_shift=None):
    jepa.eval()
    tot = {"pred": 0.0, "ident": 0.0, "mshift": 0.0, "n": 0}
    for b in loader:
        obs = b["observations"].to(device); act = b["actions"].to(device)
        z_ctrl, z_pert = _states(obs)
        preds, _ = jepa.unroll(obs, act, nsteps=1, unroll_mode="autoregressive", compute_loss=False)
        pred = preds[:, :, -1, 0, 0]
        n = obs.shape[0]
        tot["pred"] += torch.mean((pred - z_pert) ** 2).item() * n
        tot["ident"] += torch.mean((z_ctrl - z_pert) ** 2).item() * n       # no-effect baseline
        if train_shift is not None:
            ms = z_ctrl + train_shift[b["drug"].to(device)]                 # mean-shift baseline
            tot["mshift"] += torch.mean((ms - z_pert) ** 2).item() * n
        tot["n"] += n
    n = max(1, tot["n"])
    m = {k: v / n for k, v in tot.items() if k != "n"}
    m["skill_vs_identity"] = m["ident"] / max(1e-9, m["pred"])
    m["skill_vs_meanshift"] = m["mshift"] / max(1e-9, m["pred"]) if train_shift is not None else float("nan")
    return m


def compute_mean_shift(loader, n_drugs, D, device):
    """Average (z_pert - z_ctrl) per drug over the training set (mean-shift baseline)."""
    s = torch.zeros(n_drugs, D, device=device); c = torch.zeros(n_drugs, device=device)
    for b in loader:
        obs = b["observations"].to(device); z_ctrl, z_pert = _states(obs)
        d = b["drug"].to(device)
        s.index_add_(0, d, (z_pert - z_ctrl)); c.index_add_(0, d, torch.ones_like(d, dtype=s.dtype))
    return s / c.clamp_min(1).unsqueeze(1)


def run(fname, overrides):
    cfg = OmegaConf.load(fname)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.meta.seed); np.random.seed(cfg.meta.seed)

    dcfg = PertConfig(cache_path=cfg.data.cache_path, val_fraction=cfg.data.val_fraction, seed=cfg.meta.seed)
    tr, va, train_loader, val_loader = make_loaders(dcfg, batch_size=cfg.data.batch_size,
                                                    num_workers=cfg.data.num_workers)
    D, A = tr.D, tr.action_dim
    print(f"== tahoe-perturb EB-JEPA | device={device} | D={D} action_dim={A} | "
          f"train={len(tr)} val={len(va)} | control={'DMSO' if tr.has_ctrl else 'centroid'} ==")

    encoder = FrozenIdentityEncoder()
    n_layers = int(cfg.model.get("layers", 1)) if "model" in cfg else 1
    predictor = RNNPredictor(hidden_size=D, action_dim=A, num_layers=n_layers, final_ln=nn.LayerNorm(D))
    jepa = JEPA(encoder, nn.Identity(), predictor, NoReg(), SquareLossSeq()).to(device)
    sig = PerturbationSignatureLoss().to(device)
    path = PathwayCoherenceLoss().to(device)
    ls, lp = cfg.loss.sig_coeff, cfg.loss.path_coeff
    lc = float(cfg.loss.get("cos_coeff", 0.0))   # JEPA-DNA: latent COSINE alignment, added to the
                                                 # MSE prediction loss (their hybrid > either alone)

    opt = torch.optim.AdamW(predictor.parameters(), lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay)

    train_shift = compute_mean_shift(train_loader, len(tr.drug_names), D, device)

    best = {"skill_vs_meanshift": -1.0}
    ckpt = os.environ.get("EBJEPA_CKPTS", "checkpoints/tahoe"); os.makedirs(ckpt, exist_ok=True)
    for ep in range(1, cfg.optim.epochs + 1):
        jepa.train()
        agg = {"pred": 0.0, "sig": 0.0, "path": 0.0, "cos": 0.0, "n": 0}
        for b in train_loader:
            obs = b["observations"].to(device); act = b["actions"].to(device)
            z_ctrl, z_pert = _states(obs)
            preds, (loss, rloss, _, _, ploss) = jepa.unroll(
                obs, act, nsteps=1, unroll_mode="autoregressive", compute_loss=True)
            pred = preds[:, :, -1, 0, 0]
            l_sig = sig(pred - z_ctrl, b["drug"].to(device))
            l_path = path(pred, b["pathway"].to(device))
            # JEPA-DNA latent alignment: predicted state should point the same way as the
            # true perturbed state (cosine), complementing the MSE on magnitude.
            l_cos = (1.0 - F.cosine_similarity(pred, z_pert, dim=-1)).mean()
            total = loss + ls * l_sig + lp * l_path + lc * l_cos
            opt.zero_grad(set_to_none=True); total.backward()
            torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0); opt.step()
            n = obs.shape[0]
            agg["pred"] += (ploss.item() if torch.is_tensor(ploss) else float(ploss)) * n
            agg["sig"] += l_sig.item() * n; agg["path"] += l_path.item() * n
            agg["cos"] += l_cos.item() * n; agg["n"] += n
        n = max(1, agg["n"])
        if ep % cfg.optim.probe_every == 0 or ep == cfg.optim.epochs:
            m = evaluate(jepa, val_loader, device, train_shift)
            star = ""
            if m["skill_vs_meanshift"] > best["skill_vs_meanshift"]:
                best = {**m, "epoch": ep}; star = " *BEST*"
                torch.save({"jepa": jepa.state_dict(), "cfg": OmegaConf.to_container(cfg)},
                           os.path.join(ckpt, "tahoe_perturb.pt"))
            print(f"[ep {ep:03d}] train pred={agg['pred']/n:.4f} sig={agg['sig']/n:.4f} "
                  f"path={agg['path']/n:.4f} cos={agg['cos']/n:.4f} || val pred={m['pred']:.4f} "
                  f"skill_vs_noeffect={m['skill_vs_identity']:.3f}x "
                  f"skill_vs_meanshift={m['skill_vs_meanshift']:.3f}x{star}")
        else:
            print(f"[ep {ep:03d}] train pred={agg['pred']/n:.4f} sig={agg['sig']/n:.4f} "
                  f"path={agg['path']/n:.4f} cos={agg['cos']/n:.4f}")

    print(f"== BEST (ep {best.get('epoch','?')}): skill_vs_noeffect={best.get('skill_vs_identity', float('nan')):.3f}x "
          f"skill_vs_meanshift={best['skill_vs_meanshift']:.3f}x ==")
    print(f"saved best -> {os.path.join(ckpt, 'tahoe_perturb.pt')}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fname", default="examples/tahoe/cfgs/perturb.yaml")
    args, rest = ap.parse_known_args()
    run(args.fname, rest)
