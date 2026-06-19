# Microbiome-JEPA — an energy-based world model for the gut microbiome

Carrying the EB-JEPA recipe to a new, noisy, high-dimensional **biological**
modality: longitudinal gut bacterial communities (DIABIMMUNE infants). Pure
JEPA — prediction in latent space, **no reconstruction, no imposter/contrastive
target** — with two **microbiome-specific losses**.

## Why this stresses the recipe

A community is an unordered **set** of bacteria (OTUs), each a ProkBERT DNA
embedding + a relative abundance; communities are compositional, sparse, and
extremely noisy (sequencing depth varies wildly). Predicting raw abundances is
hopeless — exactly the regime where predicting in a learned latent space wins.

## The model (one `JEPA`, three swapped components)

| Component | Choice | File |
|---|---|---|
| Encoder `f_θ` | `SetEncoder` — permutation-invariant, **abundance-weighted** DeepSets over OTU embeddings | `eb_jepa/architectures.py` |
| Predictor `g_φ` | `RNNPredictor` — **action-conditioned** latent dynamics (action = feeding + Δt) | `eb_jepa/architectures.py` |
| Regularizer `R` | `VC_IDM_Sim_Regularizer` — variance+covariance (anti-collapse) + temporal-sim + inverse-dynamics | `eb_jepa/losses.py` |
| Prediction cost | `SquareLossSeq` — energy in latent space | `eb_jepa/losses.py` |

### The two microbiome-specific losses (the contribution)

- **`AlphaDiversityLoss`** *(diversity preservation)* — a linear head must
  recover Shannon α-diversity from the latent. Forces the encoder to keep
  ecological diversity decodable, so imagined futures don't collapse it.
- **`PhyloDispersionLoss`** *(soft-UniFrac)* — latent pairwise distances must
  mirror **phylogenetic** dissimilarity between communities, using each
  community's abundance-weighted mean ProkBERT embedding as a **tree-free**
  phylogenetic descriptor.

Total objective:
`L = L_pred(g_φ(z,u), z') + λ_v·var + λ_c·cov + λ_s·sim_t + λ_i·IDM + λ_d·divloss + λ_p·phyloloss`

## Hypothesis (the judged claim)

> Adding ecologically-grounded losses (diversity preservation + phylogenetic
> structure) yields a microbiome world model whose latents predict community
> change better than the no-change baseline **and** transfer better to a
> downstream host-phenotype probe (T1D), versus a plain VICReg JEPA.

Headline metrics: **`skill_vs_identity`** (latent prediction vs "no change")
and **`t1d_auroc`** (linear probe on frozen latents).

## Run

```bash
# 1. build the compact cache from raw DIABIMMUNE/MicrobeAtlas data (once)
python -m eb_jepa.datasets.microbiome.precompute        # streams the 19GB mapped file once

# 2. train (smoke test)
python -m examples.microbiome.main --fname examples/microbiome/cfgs/train.yaml \
    optim.epochs=2

# 3. evaluate (metrics + latent figure)
python -m examples.microbiome.eval --ckpt checkpoints/microbiome/microbiome_jepa.pt
```

## The controlled comparison (the deliverable)

One change at a time, three seeds each (`meta.seed`):

```bash
# baseline: generic JEPA (no bio losses)
python -m examples.microbiome.main loss.div_coeff=0 loss.phylo_coeff=0
# + diversity only
python -m examples.microbiome.main loss.phylo_coeff=0
# + phylo only
python -m examples.microbiome.main loss.div_coeff=0
# full
python -m examples.microbiome.main
```

Report `skill_vs_identity` and `t1d_auroc` (mean ± std over seeds) → one bar
chart. Watch `std`/`cov` in the logs: if `std` pins high while `pred` → 0, the
encoder is collapsing — add regularization, don't remove it.
