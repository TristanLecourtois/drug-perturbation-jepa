"""
Smoke test for the perturbation world model (examples/tahoe/perturb.py), focused
on the new JEPA-DNA-inspired cosine alignment term. Builds a tiny SYNTHETIC
perturbation cache (schema of pert_dataset.py) where each drug applies a fixed
shift in frozen-embedding space (so a predictor CAN beat no-effect / mean-shift),
then trains a few epochs on CPU with cos_coeff>0 and checks it runs + skill finite.

  python -m examples.tahoe._smoke_perturb
"""
import os
import tempfile

import numpy as np
import torch

from examples.tahoe import perturb


def build_cache(path, n_lines=4, n_drugs=5, D=2560, per=60, n_modules=8, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.normal(0, 1, (n_lines, D)).astype(np.float32)        # cell-line baselines
    drug_shift = rng.normal(0, 1.5, (n_drugs, D)).astype(np.float32)  # per-drug fixed effect
    X, drug, cl, is_ctrl = [], [], [], []
    for line in range(n_lines):
        for _ in range(per):                                        # DMSO controls per line
            X.append(base[line] + 0.2 * rng.normal(0, 1, D)); drug.append(0); cl.append(line); is_ctrl.append(True)
        for d in range(1, n_drugs):                                 # treated cells
            for _ in range(per):
                X.append(base[line] + drug_shift[d] + 0.2 * rng.normal(0, 1, D))
                drug.append(d); cl.append(line); is_ctrl.append(False)
    X = np.stack(X).astype(np.float32)
    drug = np.array(drug, np.int64); cl = np.array(cl, np.int64); is_ctrl = np.array(is_ctrl, bool)
    centroid = np.stack([X[cl == c].mean(0) for c in range(n_lines)]).astype(np.float32)
    drug_names = ["DMSO"] + [f"d{i}" for i in range(1, n_drugs)]    # name 0 => detected as control
    blob = dict(
        X=torch.from_numpy(X), drug=torch.from_numpy(drug), cell_line=torch.from_numpy(cl),
        is_control=torch.from_numpy(is_ctrl), centroid=torch.from_numpy(centroid),
        drug_fp=torch.eye(n_drugs), drug_names=drug_names, cl_names=[f"cl{i}" for i in range(n_lines)],
        modules=torch.from_numpy(rng.integers(0, n_modules, D).astype(np.int64)), n_modules=n_modules,
    )
    torch.save(blob, path)


def main():
    tmp = tempfile.mkdtemp(); cache = os.path.join(tmp, "cache_pert_smoke.pt")
    build_cache(cache)
    os.environ["EBJEPA_CKPTS"] = os.path.join(tmp, "ckpt")
    overrides = [
        f"data.cache_path={cache}", "data.batch_size=128", "data.num_workers=0",
        "optim.epochs=4", "optim.probe_every=2", "loss.cos_coeff=1.0", "model.layers=1",
    ]
    perturb.run("examples/tahoe/cfgs/perturb.yaml", overrides)
    print("\nSMOKE OK")


if __name__ == "__main__":
    main()
