#!/bin/bash
#SBATCH --job-name=tahoe_ground
#SBATCH --partition=defq
#SBATCH --reservation=Vivatech
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=tahoe_ground_%j.out
#SBATCH --error=tahoe_ground_%j.err
set -e
REPO="${EBJEPA_REPO:-$SLURM_SUBMIT_DIR}"
source "$REPO/env.sh"
cd "$REPO"
echo "=== host $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo ">>> masked-gene JEPA grounding (JEPA-DNA idea ported to RNA)"
uv run python -u -m examples.tahoe.ground --fname examples/tahoe/cfgs/ground.yaml \
  data.cache_path="$WORK/tahoe/cache.pt" \
  data.batch_size="${BS:-512}" optim.epochs="${EPOCHS:-15}" \
  optim.probe_every="${PROBE_EVERY:-5}" ${EXTRA:-}
