# UPDATE TRISTAN — récap complet du travail (branche `tristan`)

> Document de synthèse : **tout ce qui a été créé / ajouté, les sources de données, et la logique**
> derrière chaque pièce. Pour la liste des tâches restantes → [examples/tahoe/NEXT_STEPS.md](examples/tahoe/NEXT_STEPS.md).
>
> ⚠️ **Contrainte d'honnêteté (non négociable) :** aucune valeur inventée. Toute comparaison = un
> baseline qu'on a **réellement** fait tourner. On ne prétend **pas** battre GeneJEPA sur son
> protocole ; on mesure proprement nos baselines + nos différenciateurs.

---

## 0. Vue d'ensemble — ce qu'on construit

**Headline : un world-model énergie-based (EB-JEPA) de perturbation médicamenteuse.**
`(cellule contrôle, médicament) → cellule perturbée`, sur Tahoe-100M.

Principe JEPA : on **prédit dans l'espace de représentation** (pas de reconstruction de comptages).
Énergie = `‖ g_φ(f_θ(x), q_ω(a)) − f_θ(x') ‖²  +  λ·R(z)`.
- `f_θ` encoder, `g_φ` predictor (drug-conditionné), `q_ω` action-encoder, `R` anti-collapse.

Deux régimes d'encoder coexistent dans le repo (source de confusion fréquente — clarifié) :
| Modèle | Encoder `f_θ` | Anti-collapse |
|---|---|---|
| **Tahoe perturbation world-model** (headline) | **GELÉ** (embeddings MosaicFM pré-calculés) | non (rien à régulariser) |
| Tahoe représentation (`main.py`, PBMC3k) | **ENTRAÎNÉ** | oui (**SIGReg**) |
| Microbiome SetEncoder | **ENTRAÎNÉ** | oui (VICReg / IDM) |

« Encoder gelé » = on n'exécute **aucun** réseau : l'embedding MosaicFM **EST** l'état `z`, on le lit.

---

## 1. Encoders ajoutés — [eb_jepa/architectures.py](eb_jepa/architectures.py) (+178 lignes)

### `SetTransformer` (Perceiver) — l'encoder à pousser
- Chaque **gène = un token**. Token init =
  `learned-id(gène) + Σ_sources W_s·source_s(gène) + value_proj(expression)`.
- M latents apprennent + **cross-attention** vers les K gènes (O(K·M), scalable), puis self-attn, pooling → `z`.
- **`register_gene_source(name, table[K,d])`** : table **gelée** (scGPT / KGE / ESM2), projection **apprise**.
- **Logique** : c'est ce qui **permet le gene-init multi-sources**. Avec un MLP il n'y a pas de tokens →
  l'init par gène n'a nulle part où vivre. Recette inspirée de GeneJEPA (Perceiver + masked-gene).
- Testé : forward/backward OK, sources scGPT(512-d)+KGE(128-d) s'enregistrent, gradients OK, tables gelées.
- Branché dans `main.py` via `model.encoder=settransformer` + `data.gene_sources=<.pt>`.

### `MultiSourceFusion`
- Fusionne plusieurs sources d'embeddings → un latent, avec **fallback appris par source** (n'importe
  quel sous-ensemble peut être présent). Projection Linear/source + concat + MLP.
- **Logique** : permet de fusionner MosaicFM + scGPT + KGE… (niveau cellule) OU plusieurs représentations
  du **médicament** (fingerprint + descripteurs + drug-KGE) côté action.

### `SetEncoder` (DeepSets, microbiome)
- Encoder pondéré-par-abondance, invariant permutation, sur tokens OTU. Sortie `[B,D,T,1,1]` (contrat eb_jepa).

### `CellEncoder` (MLP) — baseline rapide (dans `examples/tahoe/main.py`).

---

## 2. Losses ajoutées — [eb_jepa/losses.py](eb_jepa/losses.py) (+164 lignes)

| Loss | Domaine | Logique |
|---|---|---|
| **`PathwayCoherenceLoss`** | Tahoe | distances cosine par paires dans le latent ≈ distances d'activité des **modules de gènes** → prior de programme biologique (ce que GeneJEPA n'a pas). |
| **`PerturbationSignatureLoss`** | Tahoe | le **shift prédit** (`z_pert − z_ctrl`) doit être cohérent **par médicament** (supervised-contrastive) → force la signature du médicament. |
| `AlphaDiversityLoss`, `PhyloDispersionLoss`, `TemporalVarianceLoss` | microbiome | priors écologiques + anti-collapse temporel. |

> ⚠️ pas d'`ImposterRepulsionLoss` : choix assumé **pur JEPA sans imposter**.

---

## 3. Datasets & pipeline de précalcul — [eb_jepa/datasets/tahoe/](eb_jepa/datasets/tahoe/)

| Fichier | Rôle |
|---|---|
| `precompute_pert.py` | **cache de perturbation** depuis un shard d'embeddings : `X[N,2560]`, drug, cell_line, `is_control` (auto-détecte DMSO/vehicle/control), centroïdes par lignée (pseudo-contrôle), `drug_fp` (Morgan via RDKit, fallback one-hot), modules (KMeans sur les dims). Sur Dalia : 300k cellules, 95 drugs, 50 lignées, 859 contrôles réels. |
| `pert_dataset.py` | `PertDataset` : paires contrôle→perturbé `[D,2,1,1]` + action `[A,2]`. Contrôle = DMSO aléatoire de la même lignée, sinon centroïde. |
| `precompute_pbmc.py` | cache benchmark **PBMC3k** (type cellulaire dans le slot cell_line). |
| `precompute_emb.py` / `precompute.py` | construction des caches depuis embeddings / shards bruts. |
| `dataset.py` | cache cellule two-view (gene dropout + bruit) pour `main.py`. |

---

## 4. Modèles & drivers — [examples/tahoe/](examples/tahoe/)

| Fichier | Rôle |
|---|---|
| **`perturb.py`** | le **world-model** : `FrozenIdentityEncoder` + `RNNPredictor` via `JEPA.unroll`. Pertes `PerturbationSignatureLoss` + `PathwayCoherenceLoss`. `evaluate()` calcule **skill_vs_identity** & **skill_vs_meanshift**. |
| **`main.py`** | cellule **représentation** JEPA (two-view + SIGReg/VICReg + PathwayCoherence + probe vs raw/PCA). Encoder sélectionnable `mlp`/`settransformer`. |
| **`experiments.py`** | driver batch : ablation `baseline/+pathway/+perturbsig/full/IDM` × seeds {1,1000,10000}, scaling, **zero-shot** (drugs held-out via fingerprint), **screening** in-silico. |
| `slides/` | deck jury LaTeX moderne + scripts de figures (`make_figs.py`, `make_umap.py`, `collapse_demo.py`). |
| `*.slurm` | lanceurs Dalia. |

---

## 5. Sources de données & modèles (références)

| Source | Quoi | Usage |
|---|---|---|
| **Tahoe-100M** | 100M cellules scRNA-seq, ~1000 lignées cancéreuses, ~3000 médicaments | dataset principal |
| **Tahoe-x1 / MosaicFM-3B** | embeddings cellulaires pré-calculés (2560-d) | **notre encoder gelé** |
| **GeneJEPA** (Litman 2025, bioRxiv 2025.10.14.682378) | JEPA transcriptome (Perceiver + Fourier + EMA + VICReg, 4×H100) | **concurrent direct** — comparaison honnête, pas de claim SOTA |
| **eb_jepa** | framework EB-JEPA (encoder/predictor/regularizer/unroll) | base de tout |
| **RDKit** | fingerprints Morgan + descripteurs depuis SMILES | actions médicament (94/95 drugs) |
| **PBMC3k** (scanpy) | benchmark type cellulaire | probe transfert |
| **scGPT / KGE / ESM2** | embeddings gène (à télécharger + aligner) | **sources gene-init** (à faire, cf NEXT_STEPS §3) |

---

## 6. Résultats / findings (honnêtes)

- **Headline positif** : le world-model bat **no-effect** (~1.20×) et **mean-shift** (~1.19×).
- **Motivation** : SSL from-scratch apprend l'**identité cellulaire** (F1 0.93), pas le **médicament** (0.02)
  → justifie le world-model conditionné par l'action.
- **PBMC3k** : 0.92 in-domain (≠ comparable au 0.69 transfert-gelé de GeneJEPA — précisé).
- **Ablation collapse** (`collapse_demo.py`) : SIGReg std 1.14 / acc 0.94 **vs** none std 0.002 / acc 0.43.
- **Microbiome** : finding négatif honnête (collapse temporel persistant malgré TemporalVarianceLoss).

---

## 7. À reprendre de la branche `bnz` (Belgacem) — analysé, à porter

| Pièce | Intérêt pour Tahoe |
|---|---|
| **`eval_collapse.py`** (probe fraîche de décodabilité d'action) | ⭐ **probe de drug-decodability** : décoder le médicament depuis `[z_ctrl,z_pert]` → preuve mécanistique que le modèle retient l'intervention. Branche sur l'ablation IDM. Astuce **floored-standardize** à copier. |
| Framing **IDM + dépendance-au-régime** | résultat mécanistique honnête (IDM rescue +0.23 R² en régime collapse). |
| **`plan_glv.py`** (MPPI latent) | → **planning/screening de médicaments** : choisir le drug qui rapproche la cellule d'un état-cible. La démo "whaouh". |
| `diagnose_planning.py` (oracle/solvability) | rigueur : prouver qu'un négatif vient du task-spec, pas du modèle. |
| Structure **`REPORT.md`** (MEASURED/PENDING, négatifs diagnostiqués) | template narratif qui gagne **sans SOTA**. |
| `SetTransformerEncoder` (PMA pooling + masque, contrat 5D) | cross-polliniser : ajouter pooling PMA + support masque à notre `SetTransformer`. |

À **ne pas** prendre : `ImposterRepulsionLoss` (pur JEPA), simulateur gLV / real-data infants (microbiome-spécifique).

---

## 8. Comment lancer

```bash
# Dalia (1 seul GPU dispo → jobs sérialisent ; UN job batché)
sbatch --wrap "uv run python -u -m examples.tahoe.experiments \
  --cache $WORK/tahoe/cache_pert_small.pt --fp $WORK/tahoe/drug_fp.pt \
  --out $WORK/tahoe/exp --epochs 12"          # → results.json

# Local : encoder set-transformer + (option) sources gene-init
python -m examples.tahoe.main --fname examples/tahoe/cfgs/train.yaml \
  data.cache_path=artifacts/tahoe/cache.pt model.encoder=settransformer \
  model.d_model=128 model.n_latents=24 model.depth=2 optim.epochs=10 \
  [data.gene_sources=artifacts/tahoe/gene_sources.pt]
```

Env local : `~/miniconda3/envs/ml_env/bin/python` (torch 2.8, rdkit, scanpy, umap, pyarrow).
Dalia : `$WORK=/lustre/work/vivatech-dynamics/tlecourto`, partition `defq`, `--reservation=Vivatech`,
venv par arch (`eb_jepa_$ARCH`). pyarrow uniquement sur login (x86) → précalcul sur login, train lit le cache.

---

## 9. Prochaine action recommandée
1. **Porter `eval_collapse` → drug-decodability probe** (petit, gros gain narratif), branché sur l'ablation IDM.
2. **Planning/screening latent** (single-step) = démo.
3. Mirrorer le style **REPORT** dans le deck.

(Détail complet des todos → [examples/tahoe/NEXT_STEPS.md](examples/tahoe/NEXT_STEPS.md).)

---

## 10. Entraînement en 2 ÉTAPES — JEPA-DNA porté au RNA (ajout récent)

**Inspiration : JEPA-DNA** (Daniel et al., NVIDIA 2026) — *ground* un backbone génomique gelé en
prédisant, dans l'espace latent, la représentation **globale** de segments masqués (alignement
**cosine** vers une cible **EMA**, anti-collapse **VICReg**). On garde l'**idée** mais l'unité masquée
devient un **GÈNE** (pas un nucléotide) — adapté à la transcriptomique.

### 10.1 JEPA ≠ world-model (clarification importante)
- **JEPA** = un *principe* d'entraînement (prédire dans l'espace des représentations).
- **World-model** = un *type de modèle* : prédit comment un état évolue **sous une ACTION** (état + action → état futur).

| | Étape 1 (`ground.py`) | Étape 2 (`perturb.py`) |
|---|---|---|
| Principe JEPA | ✅ | ✅ |
| Action ? | ❌ non | ✅ le médicament (action-conditionné) |
| Prédit un futur/conséquence ? | ❌ gènes masqués de la **même** cellule | ✅ la cellule **après** le drug |
| Donc c'est… | un **encodeur** (représentation) | un **world-model** |

→ L'étape 1 n'est **pas** un world-model : pas d'action, pas de dynamique. Sa sortie = un **encodeur entraîné**.

### 10.2 Le pipeline en 2 étapes (E3)
```
 ÉTAPE A — pré-entraînement (ground.py, masked-gene JEPA)
   gènes ──masque programmé──► 🟩 SetTransformer (online)  ──cosine──► cible EMA (full)
                                       + VICReg(var,cov)  → sauve l'encodeur (target EMA)

 ÉTAPE B — world-model (perturb.py) avec CET encodeur GELÉ
   genes_ctrl ─► 🧊 SetTransformer(figé) ─► z_ctrl ┐
   drug fp ────────────────────────────────────────├─► 🟩 g_φ(z_ctrl, drug) ─► ẑ_pert
   genes_pert ─► 🧊 SetTransformer(figé) ─► z_pert ─┴────◇ ‖ẑ_pert − z_pert‖² (énergie eb-JEPA)
```
- Encodeur **gelé** en étape B ⇒ aucun collapse, cible `z_pert` fixe.
- **Condition data** : le world-model doit tourner sur des **gènes bruts** (le SetTransformer mange `[2000]`),
  pas sur MosaicFM ⇒ il faut un **cache de perturbation avec gènes bruts** (re-precompute, cf. §10.6).

### 10.3 `z_pert` n'apparaît QUE dans la loss (principe JEPA)
Le predictor reçoit **seulement** `(z_ctrl, drug)` et sort `ẑ_pert`. Le **vrai** `z_pert` (encodage de la
cellule traitée) sert **uniquement de cible** dans `‖ẑ_pert − z_pert‖²`. En inférence/screening, `z_pert`
n'est pas nécessaire : on donne `(z_ctrl, drug)` → on lit `ẑ_pert`. La loss eb-JEPA = `SquareLossSeq` via
`JEPA.unroll` (`ploss`) ; `R = NoReg` car l'encodeur est gelé (rien à régulariser).

### 10.4 Modèle 1 amélioré (terme cosine JEPA-DNA)
`perturb.py` : ajout d'un terme **cosine** d'alignement latent `(1 − cos(ẑ_pert, z_pert))` en plus de la
MSE (knob `loss.cos_coeff` ; 0.0 = baseline MSE-seule, 1.0 = hybride). Mirroir du finding JEPA-DNA :
l'alignement latent cosine est un meilleur signal que la reconstruction seule, le **hybride** est le meilleur.

### 10.5 À quoi comparer — baselines & benchmarks (honnête)
On évalue **deux choses séparément** :
- **L'encodeur** (étape 1) → **Macro-F1** d'un probe linéaire (drug/moa/cell_line) vs :
  `raw` / `PCA-50` / **`SetTransformer random-init gelé`** (prouve que le pré-entraînement apporte qqch) / `MosaicFM gelé`.
- **Le world-model** (étape 2) → **skill = `MSE_baseline / MSE_pred`** (ratio, scale-invariant) vs :
  - **no-effect** (`ẑ_pert = z_ctrl`, le drug ne fait rien) → `skill_vs_identity` ;
  - **mean-shift** (`z_ctrl + effet_moyen(drug)`, baseline forte) → `skill_vs_meanshift` ;
  - *(à ajouter)* régression linéaire `(z_ctrl, drug) → z_pert`.
- **Comparaison d'encodeurs dans le world-model** (skill comparable car ratio) :
  **E1** MosaicFM gelé · **E2** SetTransformer entraîné bout-en-bout · **E3** notre SetTransformer pré-entraîné gelé (le 2-step).
- **Généralisation** (déjà dans `experiments.py`) : **zero-shot drugs** (held-out via fingerprint), **scaling**, **screening**.

> ⚠️ **Piège honnête** : le skill seul peut tromper (un encodeur dégénéré/faible-dim peut gonfler le skill
> si le mean-shift est faible dans son espace). **Toujours reporter le couple `(probe F1, skill)`** : un bon
> encodeur a F1 **et** skill élevés. Empêche de « gagner » par effondrement.

### 10.6 Fichiers ajoutés / modifiés (cette session)
| Fichier | Rôle |
|---|---|
| `eb_jepa/architectures.py` | `SetTransformer.forward(x, gene_mask)` + `mask_token` (re-masking) ; `LatentPredictor` |
| `eb_jepa/losses.py` | `MaskedGeneJEPALoss` = cosine + VICReg(var,cov) |
| `examples/tahoe/ground.py` + `cfgs/ground.yaml` + `ground.slurm` | driver grounding masked-gene (cible EMA, masquage programmé 0.15→0.45, probe vs raw/PCA) |
| `examples/tahoe/perturb.py` + `cfgs/perturb.yaml` | terme cosine JEPA-DNA (`loss.cos_coeff`) + knob profondeur predictor |
| `examples/tahoe/perturb_ablation.slurm` | ablation `cos=0` vs `cos=1` en une allocation |
| `examples/tahoe/_smoke_ground.py`, `_smoke_perturb.py` | smoke tests synthétiques (passent : `SMOKE OK`) |

### 10.7 TODO 2-step
- [ ] **Re-precompute** un cache de perturbation avec **gènes bruts** (paires ctrl/pert) pour brancher E2/E3.
- [ ] Charger `tahoe_ground.pt` (encodeur étape 1) **gelé** dans `perturb.py` (E3).
- [ ] Câbler baselines manquantes : **SetTransformer random-init gelé** (probe) + **régression linéaire** (skill).
- [ ] Ablation `skill(E1) vs E2 vs E3` + reporter `(F1, skill)` ensemble.
