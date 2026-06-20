"""Linear probe of FROZEN MosaicFM embeddings (cell_line / drug) — comparable to the
SetTransformer probe in main.py. Reads the perturbation cache (MosaicFM X + labels)."""
import argparse, numpy as np, torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from sklearn.decomposition import PCA

ap = argparse.ArgumentParser()
ap.add_argument("--cache", required=True)
ap.add_argument("--n", type=int, default=20000)
a = ap.parse_args()

b = torch.load(a.cache, weights_only=False)
X = b["X"].float().numpy(); cl = b["cell_line"].numpy(); dr = b["drug"].numpy()
mu, sd = X.mean(0, keepdims=True), X.std(0, keepdims=True) + 1e-6
X = (X - mu) / sd
rng = np.random.default_rng(0); idx = rng.permutation(len(X))[: a.n]
X, cl, dr = X[idx], cl[idx], dr[idx]
n = len(X); k = int(n * 0.8); tr, va = np.arange(k), np.arange(k, n)
P = PCA(n_components=50).fit(X[tr]).transform(X)


def probe(name, feat, y):
    ytr, yva = y[tr], y[va]
    if len(np.unique(ytr)) < 2:
        return
    clf = LogisticRegression(max_iter=200).fit(feat[tr], ytr)
    pred = clf.predict(feat[va])
    f1 = f1_score(yva, pred, average="macro"); acc = accuracy_score(yva, pred)
    print("  %-16s F1=%.3f acc=%.3f" % (name, f1, acc))


print("MosaicFM probe | cells=%d lines=%d drugs=%d" % (n, len(set(cl)), len(set(dr))))
print("cell_line:"); probe("MosaicFM(2560)", X, cl); probe("PCA-50", P, cl)
print("drug:");      probe("MosaicFM(2560)", X, dr); probe("PCA-50", P, dr)
