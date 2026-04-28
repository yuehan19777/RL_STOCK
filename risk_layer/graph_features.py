from __future__ import annotations

import numpy as np


def fuse_graph_scales(A_scales_t: np.ndarray, alpha: np.ndarray | None = None) -> np.ndarray:
    """
    融合多尺度小波图。

    A_scales_t:
        (S, N, N)

    alpha:
        (S,)
        如果 None，则等权融合。
    """
    A_scales_t = np.asarray(A_scales_t, dtype=np.float32)

    if A_scales_t.ndim != 3:
        raise ValueError(f"A_scales_t must be (S,N,N), got {A_scales_t.shape}")

    S = int(A_scales_t.shape[0])

    if alpha is None:
        alpha_arr = np.ones(S, dtype=np.float32) / float(S)
    else:
        alpha_arr = np.asarray(alpha, dtype=np.float32).reshape(-1)

        if alpha_arr.shape[0] != S:
            raise ValueError(f"alpha length {alpha_arr.shape[0]} != S={S}")

        s = float(alpha_arr.sum())
        if abs(s) < 1e-8:
            alpha_arr = np.ones(S, dtype=np.float32) / float(S)
        else:
            alpha_arr = alpha_arr / s

    weights = alpha_arr.reshape(S, 1, 1)

    A = np.sum(A_scales_t * weights, axis=0)

    A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
    A = 0.5 * (A + A.T)
    np.fill_diagonal(A, 0.0)

    return A.astype(np.float32)


def project_psd(A: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    把矩阵投影为半正定矩阵，避免二次型风险项数值不稳定。
    """
    A = np.asarray(A, dtype=np.float64)
    A = 0.5 * (A + A.T)

    vals, vecs = np.linalg.eigh(A)
    vals = np.maximum(vals, eps)

    A_psd = (vecs * vals) @ vecs.T
    A_psd = 0.5 * (A_psd + A_psd.T)

    return A_psd.astype(np.float32)

def graph_summary(A_scales: np.ndarray) -> np.ndarray:
    T, S, N, _ = A_scales.shape
    strength_s = A_scales.sum(axis=(2, 3)) / max(N * (N - 1), 1)
    mean_strength = strength_s.mean(axis=1, keepdims=True)
    dfro = np.zeros((T, S), dtype=np.float32)
    dedge = np.zeros((T, S), dtype=np.float32)
    for t in range(1, T):
        for s in range(S):
            A0 = A_scales[t - 1, s].copy(); A1 = A_scales[t, s].copy()
            dfro[t, s] = np.linalg.norm(A1 - A0, ord="fro") / (np.linalg.norm(A0, ord="fro") + 1e-8)
            E0 = A0 > 1e-12; E1 = A1 > 1e-12
            np.fill_diagonal(E0, False); np.fill_diagonal(E1, False)
            inter = np.logical_and(E0, E1).sum(); union = np.logical_or(E0, E1).sum()
            dedge[t, s] = 0.0 if union == 0 else 1.0 - inter / union
    feats = np.concatenate([strength_s, mean_strength, dfro.mean(axis=1, keepdims=True), dedge.mean(axis=1, keepdims=True)], axis=1)
    return np.nan_to_num(feats.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

