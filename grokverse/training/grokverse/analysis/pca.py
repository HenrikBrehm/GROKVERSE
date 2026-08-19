"""PCA projection of embeddings to 3D for the explorer (Phase 3 / Phase 5).

We fit a single PCA on the *final* (grokked) embeddings and apply that same
projection to every logged step. Earlier steps then show a blob collapsing onto
the final periodic structure — the visual story of grokking — under one
deterministic, consistent projection.
"""
from __future__ import annotations

import numpy as np


def fit_pca(X: np.ndarray, n_components: int = 3) -> dict:
    """Mean + top components via SVD (deterministic). X: [n, d]."""
    mean = X.mean(axis=0)
    U, S, Vt = np.linalg.svd(X - mean, full_matrices=False)
    comps = Vt[:n_components].copy()                 # [k, d]
    # sign convention: largest-|loading| entry positive (stable across runs)
    for i in range(comps.shape[0]):
        j = int(np.argmax(np.abs(comps[i])))
        if comps[i, j] < 0:
            comps[i] *= -1
    explained = (S[:n_components] ** 2) / (S ** 2).sum()
    return {"mean": mean, "components": comps, "explained_variance_ratio": explained}


def project(X: np.ndarray, pca: dict) -> np.ndarray:
    return (X - pca["mean"]) @ pca["components"].T   # [n, k]


def project_embeddings_over_time(embeds: np.ndarray, p: int, n_components: int = 3,
                                 fit_index: int = -1) -> dict:
    """embeds: [T, vocab, d]. Project number tokens [:p] to k-D using a PCA fit
    on the `fit_index` step. Returns coords [T, p, k] rescaled to unit RMS radius
    at the fit step (stable camera framing) + explained-variance ratios."""
    num = np.asarray(embeds)[:, :p, :]               # [T, p, d]
    pca = fit_pca(num[fit_index], n_components)
    coords = np.stack([project(num[t], pca) for t in range(num.shape[0])])  # [T, p, k]
    rms = float(np.sqrt((coords[fit_index] ** 2).sum(axis=1).mean()))
    if rms > 0:
        coords = coords / rms
    return {
        "coords": coords,
        "explained_variance_ratio": pca["explained_variance_ratio"].tolist(),
    }
