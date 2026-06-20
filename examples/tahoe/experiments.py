"""
experiments.py - all the controlled studies for the Tahoe perturbation EB-JEPA,
in one process (one GPU allocation):

  1. ablation of the biology losses (baseline / +Pathway / +PerturbSignature / full), 3 seeds
  2. IDM (known) vs PerturbationSignatureLoss (ours)
  3. scaling curve: skill vs #training cells
  4. zero-shot held-out drugs (requires chemical-fingerprint actions)
  5. in-silico screening: rank drugs by predicted effect toward a target state

Frozen MosaicFM encoder (identity over precomputed embeddings); we train only the
GRU predictor g_phi(z_ctrl, action) and compare prediction MSE to no-effect /
mean-shift baselines (skill = baseline_MSE / our_MSE, >1 is better).

  python -m examples.tahoe.experiments --cache $WORK/tahoe/cache_pert.pt \
      --fp artifacts/tahoe/drug_fp.pt --out artifacts/tahoe/exp --epochs 12
"""
import argparse, json, os
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from eb_jepa.architectures import RNNPredictor, InverseDynamicsModel
from eb_jepa.losses import PathwayCoherenceLoss, PerturbationSignatureLoss, grouped_sliced_wasserstein


def build_pairs(blob, fp_table, action, max_cells, holdout_drugs, seed):
    """Return train/val arrays of (z_ctrl, z_pert, action_vec, drug_id), with an
    optional set of drugs held out of training (zero-shot)."""
    X = blob["X"].float()
    X = (X - X.mean(0, keepdim=True)) / (X.std(0, keepdim=True) + 1e-6)
    drug = blob["drug"].numpy(); cl = blob["cell_line"].numpy()
    is_ctrl = blob["is_control"].numpy()
    centroid = blob["centroid"].float()
    centroid = (centroid - X.mean(0, keepdim=True)) / (X.std(0, keepdim=True) + 1e-6) \
        if centroid.shape[0] == 0 else centroid  # already raw; standardize below
    # standardize centroid with same stats
    mu, sd = blob["X"].float().mean(0, keepdim=True), blob["X"].float().std(0, keepdim=True) + 1e-6
    centroid = (blob["centroid"].float() - mu) / sd
    n_drugs = len(blob["drug_names"])

    if action == "fp" and fp_table is not None:
        A = fp_table                                  # [n_drugs, F]
    else:
        A = torch.eye(n_drugs)                         # one-hot
    act_dim = A.shape[1]

    treated = np.where(~is_ctrl)[0]
    rng = np.random.default_rng(seed); rng.shuffle(treated)
    if max_cells:
        treated = treated[:max_cells]
    # control pool per line
    has_ctrl = bool(is_ctrl.any())
    ctrl_by_line = {}
    if has_ctrl:
        for i in np.where(is_ctrl)[0]:
            ctrl_by_line.setdefault(int(cl[i]), []).append(i)

    held = set()
    if holdout_drugs:
        ud = sorted(set(drug[treated].tolist()))
        rng2 = np.random.default_rng(123)
        held = set(rng2.choice(ud, size=max(1, int(0.2 * len(ud))), replace=False).tolist())

    def ctrl_of(line, r):
        if has_ctrl and ctrl_by_line.get(line):
            return X[ctrl_by_line[line][r.integers(len(ctrl_by_line[line]))]]
        return centroid[line]

    r = np.random.default_rng(seed + 7)
    rows = []
    for j in treated:
        rows.append((int(j), int(cl[j]), int(drug[j]), int(drug[j]) in held))
    # build tensors lazily in the train loop via indices
    return dict(X=X, A=A, act_dim=act_dim, centroid=centroid, rows=rows, held=held,
                ctrl_of=ctrl_of, rng=r, n_drugs=n_drugs)


def make_batch(data, idx, device):
    X, A, ctrl_of, r = data["X"], data["A"], data["ctrl_of"], data["rng"]
    zc, zp, a, d = [], [], [], []
    for k in idx:
        j, line, dr, _ = data["rows"][k]
        zc.append(ctrl_of(line, r)); zp.append(X[j]); a.append(A[dr]); d.append(dr)
    return (torch.stack(zc).to(device), torch.stack(zp).to(device),
            torch.stack(a).to(device), torch.tensor(d, device=device))


def mean_shift(data, train_idx, device):
    D = data["X"].shape[1]; s = torch.zeros(data["n_drugs"], D); c = torch.zeros(data["n_drugs"])
    for k in train_idx:
        j, line, dr, _ = data["rows"][k]
        s[dr] += (data["X"][j] - data["ctrl_of"](line, data["rng"])); c[dr] += 1
    return (s / c.clamp_min(1).unsqueeze(1)).to(device)


def train_one(data, device, cfg):
    torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"])
    D = data["X"].shape[1]; A = data["act_dim"]
    pred = RNNPredictor(hidden_size=D, action_dim=A, final_ln=nn.LayerNorm(D)).to(device)
    idm = InverseDynamicsModel(D, 256, A).to(device) if cfg["idm"] > 0 else None
    sig = PerturbationSignatureLoss(); path = PathwayCoherenceLoss()
    params = list(pred.parameters()) + (list(idm.parameters()) if idm else [])
    opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-4)

    n = len(data["rows"]); idx = np.arange(n)
    tr = idx[[not data["rows"][k][3] for k in idx]]          # train: non-held-out drugs
    held_idx = idx[[data["rows"][k][3] for k in idx]]        # zero-shot: held-out drugs
    rng = np.random.default_rng(cfg["seed"]); rng.shuffle(tr)
    nval = len(tr) // 5; va, tr = tr[:nval], tr[nval:]
    ms = mean_shift(data, tr, device)

    def predict(zc, a):
        s = zc.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)      # [B,D,1,1,1]
        return pred(s, a.unsqueeze(-1))[:, :, 0, 0, 0]        # [B,D]

    def evaluate(eval_idx):
        pred.eval(); tot = dict(p=0., i=0., m=0., n=0)
        with torch.no_grad():
            for i in range(0, len(eval_idx), 2048):
                zc, zp, a, d = make_batch(data, eval_idx[i:i+2048], device)
                ph = predict(zc, a); b = zc.shape[0]
                tot["p"] += ((ph - zp) ** 2).mean().item() * b
                tot["i"] += ((zc - zp) ** 2).mean().item() * b
                tot["m"] += ((zc + ms[d] - zp) ** 2).mean().item() * b
                tot["n"] += b
        nn_ = max(1, tot["n"])
        return dict(skill_noeffect=tot["i"]/max(1e-9, tot["p"]),
                    skill_meanshift=tot["m"]/max(1e-9, tot["p"]))

    best = {"skill_meanshift": -1}
    for ep in range(cfg["epochs"]):
        pred.train(); rng.shuffle(tr)
        for i in range(0, len(tr), cfg.get("bs", 1024)):
            zc, zp, a, d = make_batch(data, tr[i:i+cfg.get("bs", 1024)], device)
            ph = predict(zc, a)
            loss = F.mse_loss(ph, zp)
            if cfg["sig"] > 0: loss = loss + cfg["sig"] * sig(ph - zc, d)
            if cfg["path"] > 0:
                P = (zp @ _modblob);  loss = loss + cfg["path"] * path(ph, P)
            if idm is not None: loss = loss + cfg["idm"] * F.mse_loss(idm(zc, zp), a)
            # eb_jepa OT: distribution-level match (per-drug stratum) instead of pure pairwise MSE
            if cfg.get("ot", 0) > 0: loss = loss + cfg["ot"] * grouped_sliced_wasserstein(ph, zp, d)
            opt.zero_grad(); loss.backward(); opt.step()
        m = evaluate(va)
        if m["skill_meanshift"] > best["skill_meanshift"]:
            best = {**m, "epoch": ep}
            best["state"] = {k: v.detach().cpu().clone() for k, v in pred.state_dict().items()}
    if len(held_idx):
        best["zeroshot"] = evaluate(held_idx)
    return best, pred, ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--fp", default="")
    ap.add_argument("--out", default="artifacts/tahoe/exp")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 1000, 10000])
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    blob = torch.load(args.cache, weights_only=False)
    fp_table = torch.load(args.fp, weights_only=False) if args.fp and os.path.exists(args.fp) else None

    global _modblob
    Xz = blob["X"].float(); Xz = (Xz - Xz.mean(0, keepdim=True)) / (Xz.std(0, keepdim=True) + 1e-6)
    mods = blob["modules"]; nmod = int(blob["n_modules"]); Dd = Xz.shape[1]
    oh = torch.zeros(Dd, nmod); oh[torch.arange(Dd), mods] = 1.0
    _modblob = (oh / oh.sum(0).clamp_min(1.0)).to(device)

    def data_for(action="onehot", max_cells=0, holdout=False, seed=0):
        return build_pairs(blob, fp_table, action, max_cells, holdout, seed)

    results = {}

    # ---- 1+2. ablation + IDM (3 seeds) ----
    configs = {
        "baseline":      dict(sig=0, path=0, idm=0),
        "+pathway":      dict(sig=0, path=1, idm=0),
        "+perturbsig":   dict(sig=1, path=0, idm=0),
        "full":          dict(sig=1, path=1, idm=0),
        "+ot":           dict(sig=0, path=0, idm=0, ot=0.5),   # eb_jepa sliced-Wasserstein OT alone
        "full+ot":       dict(sig=1, path=1, idm=0, ot=0.5),   # biology losses + OT distribution match
        "IDM(known)":    dict(sig=0, path=0, idm=1),
    }
    abl = {}
    for name, c in configs.items():
        sk = []
        for s in args.seeds:
            d = data_for(seed=s)
            best, _, _ = train_one(d, device, dict(**c, seed=s, epochs=args.epochs))
            sk.append(best["skill_meanshift"]); print(f"[{name} seed{s}] skill_meanshift={best['skill_meanshift']:.3f}")
        abl[name] = dict(mean=float(np.mean(sk)), std=float(np.std(sk)), seeds=sk)
    results["ablation"] = abl

    # ---- 3. scaling curve ----
    scaling = {}
    for nc in [10000, 30000, 80000, 0]:
        d = data_for(max_cells=nc, seed=1)
        best, _, _ = train_one(d, device, dict(sig=1, path=1, idm=0, seed=1, epochs=args.epochs))
        ncell = len(d["rows"]); scaling[ncell] = best["skill_meanshift"]
        print(f"[scaling n={ncell}] skill={best['skill_meanshift']:.3f}")
    results["scaling"] = scaling

    # ---- 4. zero-shot held-out drugs (fingerprint action) ----
    if fp_table is not None:
        d = data_for(action="fp", holdout=True, seed=1)
        best, _, _ = train_one(d, device, dict(sig=1, path=1, idm=0, seed=1, epochs=args.epochs))
        results["zeroshot"] = dict(seen=best["skill_meanshift"], unseen=best.get("zeroshot"))
        print("[zero-shot] seen:", best["skill_meanshift"], "unseen:", best.get("zeroshot"))

    # ---- 5. in-silico screening (rank drugs toward a target state) ----
    d = data_for(seed=1)
    best, pred, ms = train_one(d, device, dict(sig=1, path=1, idm=0, seed=1, epochs=args.epochs))
    pred.eval()
    with torch.no_grad():
        # target = mean treated state of a chosen "reference" drug; control = a random ctrl cell
        j, line, _, _ = d["rows"][0]; zc = d["ctrl_of"](line, d["rng"]).to(device)
        scores = []
        for drg in range(d["n_drugs"]):
            a = d["A"][drg].to(device).unsqueeze(0)
            s = zc.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            ph = pred(s, a.unsqueeze(-1))[:, :, 0, 0, 0][0]
            scores.append(float((ph - zc).norm()))   # magnitude of predicted perturbation
        order = np.argsort(scores)[::-1]
        results["screening_topdrugs"] = [(blob["drug_names"][int(i)], round(scores[int(i)], 3)) for i in order[:10]]

    json.dump(results, open(os.path.join(args.out, "results.json"), "w"), indent=2)
    print("saved", os.path.join(args.out, "results.json"))


if __name__ == "__main__":
    main()
