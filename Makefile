.ONESHELL:
.PHONY: help
.DEFAULT_GOAL := help

help: ## Show this help message
	@grep -hE '^[A-Za-z0-9_ \-]*?:.*##.*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

run_image_jepa: ## Run the image JEPA example
	uv run python -m examples.image_jepa.main

run_video_jepa: ## Run the video JEPA example
	uv run python -m examples.video_jepa.main

run_ac_video_jepa: ## Run the action-conditioned video JEPA example
	uv run python -m examples.ac_video_jepa.main

# --- Tahoe gene-init multi-sources (scGPT / KGE / ESM2 / Evo2) --------------- #
# Override the artifact paths once you have them on disk (defaults under $WORK).
CACHE        ?= artifacts/tahoe/cache.pt
GENE_META    ?= $(WORK)/tahoe/gene_metadata.parquet
GENE_SRC_OUT ?= artifacts/tahoe/gene_sources.pt
SCGPT_CKPT   ?= $(WORK)/models/scGPT_human/best_model.pt
SCGPT_VOCAB  ?= $(WORK)/models/scGPT_human/vocab.json
KGE_FILE     ?= $(WORK)/kg/primekg_gene_emb.pt
ESM2_EMB     ?= $(WORK)/esm2/gene_esm2.pt
EVO2_EMB     ?= $(WORK)/evo2/gene_evo2.pt

gene_sources: ## Build aligned gene-init tables (skips any source whose artifact is missing)
	uv run python -m eb_jepa.datasets.tahoe.precompute_gene_sources \
	  --cache $(CACHE) --gene-meta $(GENE_META) \
	  --sources scgpt kge esm2 evo2 \
	  --scgpt-ckpt $(SCGPT_CKPT) --scgpt-vocab $(SCGPT_VOCAB) \
	  --kge-file $(KGE_FILE) --esm2-emb $(ESM2_EMB) --evo2-emb $(EVO2_EMB) \
	  --out $(GENE_SRC_OUT)

smoke_gene_sources: ## Validate the gene-init pipeline end-to-end (no downloads)
	uv run python -m examples.tahoe._smoke_gene_sources

PATHWAYS_OUT ?= artifacts/tahoe/pathways.pt

pathways: ## Build panel-aligned MSigDB Hallmark membership (real biology programs)
	uv run python -m eb_jepa.datasets.tahoe.precompute_pathways \
	  --cache $(CACHE) --gene-meta $(GENE_META) --out $(PATHWAYS_OUT)

smoke_pathways: ## Validate the hallmark-pathways path end-to-end (no downloads)
	uv run python -m examples.tahoe._smoke_pathways
