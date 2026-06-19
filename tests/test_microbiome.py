"""Tests for the microbiome JEPA additions (SetEncoder + biological losses +
end-to-end JEPA wiring). These run on random tensors -- no dataset/cache needed."""

import torch
import torch.nn as nn

from eb_jepa.architectures import (
    InverseDynamicsModel,
    Projector,
    RNNPredictor,
    SetEncoder,
)
from eb_jepa.jepa import JEPA
from eb_jepa.losses import (
    AlphaDiversityLoss,
    PhyloDispersionLoss,
    SquareLossSeq,
    VC_IDM_Sim_Regularizer,
)

B, E, T, N, D, A = 4, 384, 6, 32, 16, 5


def _obs():
    # [B, E+1, T, N, 1]: embedding channels + a non-negative abundance channel
    x = torch.randn(B, E + 1, T, N, 1)
    x[:, E] = torch.rand(B, T, N, 1)  # abundance >= 0
    return x


def test_set_encoder_shapes_and_permutation_invariance():
    enc = SetEncoder(emb_dim=E, h_d=32, out_d=D)
    x = _obs()
    z = enc(x)
    assert z.shape == (B, D, T, 1, 1)
    # permuting OTUs (the N axis) must not change the pooled representation
    perm = torch.randperm(N)
    z2 = enc(x[:, :, :, perm, :])
    assert torch.allclose(z, z2, atol=1e-5)


def test_padding_is_ignored():
    enc = SetEncoder(emb_dim=E, h_d=32, out_d=D)
    x = _obs()
    z = enc(x)
    # append padded (abundance-0) OTUs: representation must be unchanged
    pad = torch.randn(B, E + 1, T, 5, 1)
    pad[:, E] = 0.0
    z_pad = enc(torch.cat([x, pad], dim=3))
    assert torch.allclose(z, z_pad, atol=1e-5)


def test_alpha_diversity_loss_scalar():
    loss = AlphaDiversityLoss(state_dim=D)
    state = torch.randn(B, D, T, 1, 1)
    div = torch.rand(B, T)
    out = loss(state, div)
    assert out.ndim == 0 and out.item() >= 0


def test_phylo_dispersion_loss_scalar():
    loss = PhyloDispersionLoss(max_samples=64)
    state = torch.randn(B, D, T, 1, 1)
    phylo = torch.randn(B, T, E)
    out = loss(state, phylo)
    assert out.ndim == 0 and out.item() >= 0


def test_jepa_unroll_end_to_end():
    enc = SetEncoder(emb_dim=E, h_d=32, out_d=D)
    pred = RNNPredictor(hidden_size=D, action_dim=A, final_ln=nn.LayerNorm(D))
    idm = InverseDynamicsModel(state_dim=D, hidden_dim=32, action_dim=A)
    reg = VC_IDM_Sim_Regularizer(
        cov_coeff=1.0, std_coeff=1.0, sim_coeff_t=1.0, idm_coeff=1.0,
        idm=idm, projector=Projector(f"{D}-{2*D}-{2*D}"), first_t_only=False,
    )
    jepa = JEPA(enc, nn.Identity(), pred, reg, SquareLossSeq())
    obs = _obs()
    act = torch.randn(B, A, T)
    preds, losses = jepa.unroll(obs, act, nsteps=3, unroll_mode="autoregressive",
                                compute_loss=True)
    total = losses[0]
    assert torch.isfinite(total)
    total.backward()  # gradients flow through encoder + predictor + regularizer
    assert any(p.grad is not None for p in enc.parameters())
