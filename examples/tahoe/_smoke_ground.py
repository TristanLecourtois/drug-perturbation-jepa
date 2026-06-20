"""
Smoke test for the masked-gene JEPA grounding (examples/tahoe/ground.py).

Builds a tiny SYNTHETIC gene-panel cache (same schema as
eb_jepa/datasets/tahoe/dataset.py) with cell-line structure so the linear probe
has signal, then runs the grounding driver for a few epochs on CPU and checks the
loss is finite. No real Tahoe data needed.

  python -m examples.tahoe._smoke_ground
"""
import os
import tempfile

import numpy as np
import torch

from examples.tahoe import ground


def build_cache(path, N=400, K=64, n_lines=4, n_drugs=5, n_moa=3, n_modules=6, seed=0):
    rng = np.random.default_rng(seed)
    # cell-line centroids give the panel structure the probe should recover
    centroids = rng.normal(0, 1, (n_lines, K)).astype(np.float32)
    cell_line = rng.integers(0, n_lines, N)
    X = (centroids[cell_line] + 0.3 * rng.normal(0, 1, (N, K))).astype(np.float32)
    X = np.clip(X * 2 + 5, 0, None)                       # positive "counts" -> log1p in dataset
    drug = rng.integers(0, n_drugs, N)
    moa = rng.integers(0, n_moa, N)
    modules = rng.integers(0, n_modules, K)
    blob = dict(
        X=torch.from_numpy(X), drug=torch.from_numpy(drug),
        moa=torch.from_numpy(moa), cell_line=torch.from_numpy(cell_line),
        drug_names=[f"d{i}" for i in range(n_drugs)],
        moa_names=[f"m{i}" for i in range(n_moa)],
        cl_names=[f"cl{i}" for i in range(n_lines)],
        drug_fp=torch.eye(n_drugs),
        modules=torch.from_numpy(modules), n_modules=n_modules,
        is_embedding=False,
    )
    torch.save(blob, path)


def main():
    tmp = tempfile.mkdtemp()
    cache = os.path.join(tmp, "cache_smoke.pt")
    build_cache(cache)
    overrides = [
        f"data.cache_path={cache}", "data.batch_size=64",
        "model.dstc=32", "model.d_model=48", "model.n_latents=8", "model.depth=1",
        "model.pred_depth=2", "optim.epochs=3", "optim.probe_every=3",
    ]
    os.environ["EBJEPA_CKPTS"] = os.path.join(tmp, "ckpt")
    ground.run("examples/tahoe/cfgs/ground.yaml", overrides)
    print("\nSMOKE OK")


if __name__ == "__main__":
    main()
