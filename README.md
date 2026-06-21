<h1 align="center">DynaAMIcs — a Drug-Perturbation JEPA World-Model</h1>

<p align="center">
  <b>An energy-based JEPA <i>world-model of drug perturbations</i> on Tahoe-100M.</b><br>
  <code>(control cell, drug) → perturbed cell</code>, predicted in representation space.
</p>

<p align="center">
  <a href="https://docs.google.com/presentation/d/17WaQSNrnvW2XvhzPFOqw-5u7euWNNd3cDUlr0zF8ey4/edit?usp=sharing"><b>📊 Slides / Presentation</b></a>
  &nbsp;·&nbsp;
  Built on <a href="https://github.com/marinabar/eb_jepa">eb-JEPA</a>
  &nbsp;·&nbsp;
  Hack The World(s) / Vivatech
  &nbsp;·&nbsp;
  single-cell transcriptomics
</p>

---

## TL;DR

We learn a **world-model** that predicts how a cell's transcriptomic state changes
**under a drug intervention**, entirely in latent space (JEPA: no count reconstruction).

$$\mathcal{E} = \lVert\, g_\phi(f_\theta(x),\, q_\omega(a)) - f_\theta(x') \,\rVert^2 \;+\; \lambda\, R(z)$$

- $f_\theta$ encoder · $g_\phi$ drug-conditioned predictor · $q_\omega$ action (drug) encoder · $R$ anti-collapse
- $x$ = control cell, $a$ = drug, $x'$ = the same cell **after** the drug.

**Why a world-model and not just an encoder?** From-scratch SSL learns *cell identity*
(probe F1 ≈ 0.93) but **not** the *drug* (F1 ≈ 0.02). Conditioning the dynamics on the
action is what makes the drug effect learnable — and lets us **screen drugs in silico**.

> **Honesty constraint (non-negotiable):** no invented numbers. Every comparison is a
> baseline we actually ran. We do **not** claim to beat GeneJEPA on its protocol — we
> measure our own baselines and differentiators cleanly, and report negative results too.

---

## The model in one picture

```
ENCODER f_θ (frozen)                 PREDICTOR g_φ (trained)            ENERGY

 genes/emb ──► f_θ ──► z_ctrl ┐
 drug ──Morgan fp──────────────├──► g_φ(z_ctrl, drug) ──► ẑ_pert ──┐
 genes/emb ──► f_θ ──► z_pert ─┘                                   ├──◇ ‖ẑ_pert − z_pert‖²
                                                                   │   + biology priors
 (z_pert is only a TARGET — at inference we give (z_ctrl, drug) and read ẑ_pert)
```

Only the predictor is trained; the encoder is **frozen** (so it cannot collapse and the
target is fixed). A drug induces a large controlled state change → the no-change baseline
is beatable (skill > 1), unlike slow microbiome trajectories.

---

## Three encoder regimes (the encoder `f_θ`)

| Regime | Encoder `f_θ` | Trained how | Status |
|---|---|---|---|
| **E1** | **MosaicFM-3B** embeddings (2560-d), frozen | pretrained, off-the-shelf | ✅ default, runs today |
| **E2** | `SetTransformer` trained end-to-end with the predictor | one stage | ✅ available |
| **E3** | `SetTransformer` **grounded** (2-step), then frozen | masked-gene JEPA → freeze | ✅ wired (needs raw-gene cache) |

`SetTransformer` (Perceiver, [`eb_jepa/architectures.py`](eb_jepa/architectures.py)) treats **every gene as a token**:

```
token[g] = id_emb(g)  +  Σ_sources Wₛ · sourceₛ(g)  +  value_proj(expression[g])
```

The `sourceₛ` are **frozen per-gene tables** (scGPT / KGE / ESM2 / Evo2) with a learned
projection — the multi-source **gene-init**. None are required (it trains on the learned
gene-id alone); real sources plug in via `register_gene_source`.

---

## The 2-step training (E3, JEPA-DNA → RNA)

**JEPA ≠ world-model.** JEPA is a *training principle* (predict in representation space).
A world-model is a *model type* (state + action → next state). Step 1 is a pure encoder;
step 2 is the world-model.

```
STEP 1 — ground.py     (masked-gene JEPA, à la JEPA-DNA / GeneJEPA)
   genes ─mask─► 🟩 SetTransformer (online) ─► g_φ ─► ẑ ──cosine──► 🧊 EMA target (full cell)
                                                          + VICReg(var,cov)  →  tahoe_ground.pt

STEP 2 — perturb.py    (world-model, encoder FROZEN)
   genes_ctrl ─► 🧊 SetTransformer (frozen) ─► z_ctrl ┐
   drug fp ─────────────────────────────────────────├─► 🟩 g_φ ─► ẑ_pert ──◇ ‖ẑ_pert − z_pert‖²
   genes_pert ─► 🧊 SetTransformer (frozen) ─► z_pert ┘
```

```bash
# step 1 — ground the encoder
python -m examples.tahoe.ground  --fname examples/tahoe/cfgs/ground.yaml

# step 2 — world-model on the frozen grounded encoder (E3)
python -m examples.tahoe.perturb --fname examples/tahoe/cfgs/perturb.yaml \
    model.encoder=settransformer model.ground_ckpt=checkpoints/tahoe/tahoe_ground.pt \
    data.cache_path=<raw-gene perturbation cache>
# default (E1): omit model.encoder → frozen MosaicFM embeddings
```

---

## Losses (biology priors — our differentiators vs GeneJEPA)

| Loss | Role |
|---|---|
| `SquareLossSeq` | eb-JEPA energy `‖ẑ_pert − z_pert‖²` (via `JEPA.unroll`) |
| `PerturbationSignatureLoss` | the predicted shift `ẑ_pert − z_ctrl` must be consistent **per drug** (supervised-contrastive) |
| `PathwayCoherenceLoss` | latent geometry ≈ **gene-program** geometry (KMeans modules **or** real MSigDB Hallmark sets) |
| `MaskedGeneJEPALoss` | step-1 grounding: cosine to EMA target + VICReg anti-collapse |
| **sliced-Wasserstein OT** | match the *predicted* vs *true* perturbed **distribution** per `(drug, cell_line)` stratum (ported from eb_jepa) — fixes pseudo-pairing; toggle `loss.ot_coeff` |
| JEPA-DNA cosine | latent direction alignment `(1 − cos(ẑ_pert, z_pert))`; hybrid with MSE; toggle `loss.cos_coeff` |

Deliberately **no** `ImposterRepulsionLoss` — assumed pure JEPA.

---

## Plug-in scaffolding (one command each)

Everything below is **wired and smoke-tested**; only the real artifacts need to be dropped in.

```bash
# Real biology programs: panel-aligned MSigDB Hallmark membership (vs KMeans modules)
make pathways          # → pathways.pt ; then: ... data.pathways=artifacts/tahoe/pathways.pt

# Multi-source gene-init: aligns scGPT / KGE / ESM2 / Evo2 to the panel (skips missing sources)
make gene_sources      # → gene_sources.pt ; then: ... model.encoder=settransformer data.gene_sources=…

# End-to-end validation, no downloads needed:
make smoke_pathways  smoke_gene_sources  smoke_perturb_e3
```

See builder headers for the artifact paths:
[`precompute_pathways.py`](eb_jepa/datasets/tahoe/precompute_pathways.py) ·
[`precompute_gene_sources.py`](eb_jepa/datasets/tahoe/precompute_gene_sources.py).

---

## Data pipeline

| Script | Output |
|---|---|
| [`precompute.py`](eb_jepa/datasets/tahoe/precompute.py) | top-K raw-gene cell cache (`panel`, `X[N,K]`, KMeans modules) |
| [`precompute_emb.py`](eb_jepa/datasets/tahoe/precompute_emb.py) | MosaicFM-embedding cell cache (E1) |
| [`precompute_pert.py`](eb_jepa/datasets/tahoe/precompute_pert.py) | perturbation cache (ctrl/pert, Morgan fp, centroids) |
| [`precompute_pbmc.py`](eb_jepa/datasets/tahoe/precompute_pbmc.py) | PBMC3k transfer benchmark |

Drug actions: **Morgan fingerprints** (RDKit) from `canonical_smiles`, fallback one-hot.
Control = DMSO of the same cell line (else the line centroid as pseudo-control).

---

## Evaluation (always report the *pair* `(probe F1, skill)`)

- **Encoder** → linear-probe **Macro-F1** (drug / moa / cell_line) vs `raw` / `PCA-50` /
  `SetTransformer random-init` / `MosaicFM`.
- **World-model** → `skill = MSE_baseline / MSE_pred` (scale-invariant) vs **no-effect**
  (`ẑ_pert = z_ctrl`) and **mean-shift** (`z_ctrl + meanΔ(drug)`).

> ⚠️ Skill alone can be gamed by a degenerate encoder. Always report `(F1, skill)` together:
> a good encoder has **both** high.

```bash
# full ablation driver (biology losses × seeds, scaling, zero-shot drugs, in-silico screening)
python -m examples.tahoe.experiments --cache <cache_pert.pt> --fp <drug_fp.pt> \
    --out artifacts/tahoe/exp --epochs 12
```

---

## Measured findings (honest)

- **Headline:** the world-model beats **no-effect** (~1.20×) and **mean-shift** (~1.19×).
- **Motivation:** from-scratch SSL learns cell identity (F1 0.93), not the drug (0.02).
- **PBMC3k:** 0.92 in-domain (≠ comparable to GeneJEPA's 0.69 *frozen-transfer*; stated explicitly).
- **Collapse ablation:** SIGReg std 1.14 / acc 0.94 **vs** none std 0.002 / acc 0.43.
- **Microbiome:** honest *negative* result (temporal collapse persists despite TemporalVarianceLoss).

---

## Repo layout (Tahoe)

```
examples/tahoe/
  ground.py         step-1 masked-gene grounding (SetTransformer + EMA + VICReg)
  perturb.py        step-2 world-model (E1 MosaicFM / E3 frozen grounded SetTransformer)
  main.py           representation JEPA (two-view SIGReg/VICReg + probe vs raw/PCA)
  experiments.py    ablations · scaling · zero-shot · in-silico screening
  embed_viz.py      UMAP/t-SNE of predicted state & drug-specific shift
  cfgs/             ground.yaml · perturb.yaml · train.yaml
  _smoke_*.py       CPU smoke tests (no data/download needed)
eb_jepa/
  architectures.py  SetTransformer, RNNPredictor, LatentPredictor, MultiSourceFusion …
  losses.py         Pathway/Signature/MaskedGeneJEPA losses + sliced-Wasserstein OT
  datasets/tahoe/   datasets + precompute (cells, perturbations, gene-sources, pathways)
```

Roadmap & full work log: [`examples/tahoe/NEXT_STEPS.md`](examples/tahoe/NEXT_STEPS.md) ·
[`UPDATETRISTAN.md`](UPDATETRISTAN.md).

---

## Data & model sources

**Tahoe-100M** (100M scRNA-seq cells, ~1000 cancer lines, ~3000 drugs) ·
**Tahoe-x1 / MosaicFM-3B** (frozen cell embeddings) ·
**GeneJEPA** (Litman 2025, bioRxiv 2025.10.14.682378 — direct comparison, no SOTA claim) ·
**JEPA-DNA** (NVIDIA 2026 — the 2-step grounding idea) ·
**RDKit** · **PBMC3k** (scanpy) · **scGPT / KGE / ESM2 / Evo2** (gene-init sources).

Framework: [eb-JEPA](https://github.com/marinabar/eb_jepa) (encoder / predictor / regularizer / `unroll`).
