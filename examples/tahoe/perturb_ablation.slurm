#!/bin/bash
#SBATCH --job-name=tperturb_abl
#SBATCH --partition=defq
#SBATCH --reservation=Vivatech
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --output=tperturb_abl_%j.out
#SBATCH --error=tperturb_abl_%j.err
set -e
REPO="${EBJEPA_REPO:-$SLURM_SUBMIT_DIR}"
source "$REPO/env.sh"
cd "$REPO"
echo "=== host $(hostname) ==="
nvidia-smi --query-gpu=name --format=csv,noheader || true
CACHE="${CACHE:-$WORK/tahoe/cache_pert_small.pt}"
EP="${EPOCHS:-30}"
echo "######## ablation cosine JEPA-DNA on $CACHE ########"
echo ">>> [1/2] BASELINE  cos_coeff=0.0 (MSE only)"
uv run python -u -m examples.tahoe.perturb --fname examples/tahoe/cfgs/perturb.yaml \
  data.cache_path="$CACHE" optim.epochs="$EP" loss.cos_coeff=0.0
echo ">>> [2/2] HYBRID    cos_coeff=1.0 (MSE + latent cosine)"
uv run python -u -m examples.tahoe.perturb --fname examples/tahoe/cfgs/perturb.yaml \
  data.cache_path="$CACHE" optim.epochs="$EP" loss.cos_coeff=1.0
echo ">>> ablation done — compare the two BEST skill_vs_meanshift lines above"
