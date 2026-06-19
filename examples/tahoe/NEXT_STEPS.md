√# Tahoe-JEPA — ce qu'il reste à faire

État du modèle : world-model de perturbation médicamenteuse `(cellule contrôle, drug) → cellule perturbée`
construit sur eb-JEPA. Deux encoders disponibles :
- **`mlp`** (`CellEncoder`) — baseline, rapide.
- **`settransformer`** (`SetTransformer`, Perceiver) — chaque gène = token ; permet le
  **gene-init multi-sources** (learned-id + scGPT + KGE + ESM2). C'est l'encoder à pousser.

> ⚠️ **Contrainte d'honnêteté (non négociable).** Aucune valeur inventée. Toute comparaison
> = un baseline qu'on a **réellement** fait tourner (raw / PCA / VICReg / no-reg / scGPT-frozen
> si on obtient les poids). On ne prétend pas battre GeneJEPA sur son protocole exact — on
> mesure proprement nos propres baselines et différenciateurs.

---

## 1. Benchmarks (protocole = linear probe sur features gelées, façon GeneJEPA)

| Benchmark | Tâche | Commande | Comparer à |
|---|---|---|---|
| Tahoe in-domain | probe `cell_line` / `drug` / `moa` | `main.py` (affiche raw / pca50 / JEPA) | raw, PCA-50 |
| PBMC3k | probe type cellulaire (Macro-F1) | cache `precompute_pbmc.py` → `main.py` | raw, PCA, (scGPT si dispo) |
| Perturbation skill | MSE prédite vs identité & mean-shift | `perturb.py` → `evaluate()` | no-effect, mean-shift |

**À faire :**
- [ ] Régénérer le cache PBMC3k (`precompute_pbmc.py`) et lancer le probe → reporter Macro-F1.
- [ ] Confirmer le skill du world-model `> 1.20×` vs mean-shift (job perturb tuné, 2 layers).
- [ ] (Si possible) un vrai baseline **scGPT-frozen** sur PBMC3k pour une comparaison côte-à-côte honnête.

---

## 2. Ablations (driver : `experiments.py`, déjà écrit)

`experiments.py` couvre déjà, sur le cache de perturbation :
- **Pertes bio** : `baseline` / `+pathway` / `+perturbsig` / `full` / `IDM` × seeds `{1, 1000, 10000}`.
- **Scaling** : skill en fonction du nombre de cellules d'entraînement.
- **Zero-shot** : drugs held-out via fingerprint Morgan (action).
- **Screening in-silico** : ranking des drugs par effet prédit.

```bash
# Dalia (1 seul GPU dispo → les jobs sérialisent ; lancer UN job batché)
sbatch --wrap "uv run python -u -m examples.tahoe.experiments \
  --cache $WORK/tahoe/cache_pert_small.pt --fp $WORK/tahoe/drug_fp.pt \
  --out $WORK/tahoe/exp --epochs 12"
# sortie → $WORK/tahoe/exp/results.json
```

**À faire :**
- [ ] Récupérer `results.json` du batch (job `74812`).
- [ ] **Ablation encoder** : ajouter `mlp` vs `settransformer` dans `experiments.py` (même seeds).
- [ ] **Ablation régularizer** (collapse) : `sigreg` vs `vicreg` vs `none` — la figure `collapse.png`
      existe (std 1.14/acc 0.94 vs std 0.002/acc 0.43) ; l'étendre à 3 seeds.
- [ ] Générer les figures (`make_figs.py`) et les insérer dans `slides/main.tex`.

---

## 3. Multi-sources / gene-init — **est-ce que ça améliore ?** (la vraie expérience)

Question : ajouter scGPT / KGE comme sources d'init des tokens-gènes améliore-t-il le probe
et/ou le skill ? Réponse = une **ablation propre** :

| Variante | sources gene-init | mesure |
|---|---|---|
| A | learned-id seul | F1 probe + skill |
| B | learned-id + scGPT | idem |
| C | learned-id + KGE | idem |
| D | learned-id + scGPT + KGE | idem |

Code déjà prêt : `SetTransformer.register_gene_source(name, table[K,d])` (table gelée, projection apprise).
Brancher via le config : `data.gene_sources=<chemin .pt>` où le `.pt` = `{name: tensor[K, d]}`
**aligné sur le panel de gènes** (`cache['panel']`).

**Ce qui manque = obtenir les tables réelles, alignées au panel :**
- [ ] **scGPT** : télécharger le checkpoint → extraire la table d'embeddings de gènes → mapper
      par symbole de gène sur `cache['panel']` → `tensor[K, d_scgpt]`. Gènes absents = vecteur 0.
- [ ] **KGE** : embeddings gènes d'un KG biomédical (Hetionet/PrimeKG) → aligner par symbole/Entrez.
- [ ] **ESM2** (optionnel) : gène → protéine → embedding ESM2 → `tensor[K, d_esm]`.
- [ ] Sauver `{ 'scgpt':..., 'kge':... }` dans `artifacts/tahoe/gene_sources.pt`.
- [ ] Lancer A/B/C/D (mêmes seeds) → **tableau d'amélioration honnête** (+Δ F1, +Δ skill, ou nul).

> Résultat **nul ou négatif = OK et publiable** : « la fusion de sources n'aide pas au-delà de
> l'encoder appris » est un finding honnête (cf. le finding de collapse du microbiome).

### Variante côté ACTION (plus rapide à obtenir, sans download lourd)
On a déjà `pubchem_cid` → fusionner plusieurs représentations du **médicament** :
- [ ] fingerprint Morgan (on l'a) + descripteurs physico-chimiques RDKit (gratuits, calculables).
- [ ] (si on trouve un fichier) **drug-KGE** aligné par `pubchem_cid`.
- [ ] Mesurer l'effet sur le **zero-shot** vers nouveaux médicaments (là où la fusion devrait aider le plus).
- Module prêt : `MultiSourceFusion(source_dims)`.

---

## 4. Slides / livrable
- [ ] Ajouter au deck (`slides/main.tex`) : ablation pertes (3 seeds), scaling, zero-shot, screening,
      ablation encoder MLP↔SetTransformer, ablation sources A/B/C/D (même si nul).
- [ ] Schéma archi : marquer **gelé vs entraîné**, et l'insertion du gene-init multi-sources.

## 5. Ordre conseillé
1. Récupérer `results.json` (job batché) → figures → slides. *(débloque le narratif principal)*
2. Ablation encoder MLP vs SetTransformer (pas de download requis). 
3. Obtenir **une** vraie source (scGPT le plus simple) → ablation A/B.
4. Le reste (KGE, ESM2, drug-KGE) si le temps le permet.

## Commandes locales utiles
```bash
# encoder set-transformer en local (smoke / dev)
python -m examples.tahoe.main --fname examples/tahoe/cfgs/train.yaml \
  data.cache_path=artifacts/tahoe/cache.pt model.encoder=settransformer \
  model.d_model=128 model.n_latents=24 model.depth=2 optim.epochs=10

# avec sources gene-init (une fois gene_sources.pt construit)
python -m examples.tahoe.main ... model.encoder=settransformer \
  data.gene_sources=artifacts/tahoe/gene_sources.pt
```
