"""Shared numeric helpers (numpy / scipy / sklearn only)."""
import numpy as np
from sklearn.neighbors import NearestNeighbors


def subsample(X, n, rng, *others):
    """Random row subset of X (and aligned arrays in `others`), without replacement."""
    if len(X) <= n:
        return (X, *others) if others else X
    idx = rng.choice(len(X), size=n, replace=False)
    if others:
        return (X[idx], *[o[idx] for o in others])
    return X[idx]


def class_stats(X, y, n_classes):
    """Per-class centers (K,D), RMS within-class radius (K,), counts (K,), global mean (D,)."""
    D = X.shape[1]
    centers = np.zeros((n_classes, D), dtype=np.float64)
    radius = np.zeros(n_classes, dtype=np.float64)
    counts = np.zeros(n_classes, dtype=np.int64)
    for c in range(n_classes):
        m = y == c
        counts[c] = m.sum()
        if counts[c] == 0:
            continue
        Xc = X[m].astype(np.float64)
        centers[c] = Xc.mean(0)
        radius[c] = np.sqrt(((Xc - centers[c]) ** 2).sum(1).mean())
    return centers, radius, counts, X.astype(np.float64).mean(0)


def pca_frame(X, center=None):
    """Eigen-decomposition of the covariance of X. Returns (mean, eigvals desc, eigvecs cols desc)."""
    mu = X.mean(0) if center is None else center
    Xc = X.astype(np.float64) - mu
    cov = Xc.T @ Xc / max(1, len(Xc) - 1)
    w, V = np.linalg.eigh(cov)
    order = np.argsort(w)[::-1]
    return mu, np.clip(w[order], 0, None), V[:, order]


def knn_radii(fit_X, query_X, k=10, exclude_self=False):
    """Distance to the k-th nearest neighbour of each query row within fit_X."""
    nn = NearestNeighbors(n_neighbors=k + (1 if exclude_self else 0)).fit(fit_X)
    d, _ = nn.kneighbors(query_X)
    return d[:, -1]


def linear_cka(X, Y):
    """Linear CKA between two (N, Dx), (N, Dy) representations of the same N inputs."""
    Xc = X - X.mean(0)
    Yc = Y - Y.mean(0)
    hsic = np.linalg.norm(Yc.T @ Xc, "fro") ** 2
    nx = np.linalg.norm(Xc.T @ Xc, "fro")
    ny = np.linalg.norm(Yc.T @ Yc, "fro")
    return float(hsic / (nx * ny + 1e-12))


def nearest_center(X, centers):
    d = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
    return d.argmin(1), d


def safe_corr(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.std() < 1e-12 or b.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])
