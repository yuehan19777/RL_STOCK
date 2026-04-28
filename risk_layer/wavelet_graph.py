from __future__ import annotations
from dataclasses import dataclass
from typing import List
import numpy as np

@dataclass
class WaveletGraphConfig:
    wavelet: str = "db2"
    levels: int = 3
    window: int = 252
    min_obs: int = 120
    top_m: int = 5
    residual_mode: str = "demean"

def residualize_returns(R: np.ndarray, mode: str = "demean") -> np.ndarray:
    R = np.nan_to_num(R.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if mode == "none":
        return R
    if mode == "demean":
        return R - R.mean(axis=1, keepdims=True)
    raise ValueError(f"Unknown residual_mode: {mode}")

def _swt_details(x: np.ndarray, wavelet: str, levels: int) -> List[np.ndarray]:
    import pywt
    x = np.asarray(x, dtype=np.float32)
    n = len(x)
    if n <= 2:
        return [np.zeros(n, dtype=np.float32) for _ in range(levels)]
    base = 2 ** levels
    target = int(np.ceil(n / base) * base)
    pad = target - n
    x_pad = np.pad(x, (0, pad), mode="edge") if pad > 0 else x
    coeffs = pywt.swt(x_pad, wavelet, level=levels, trim_approx=False)
    details = [cD.astype(np.float32) for (_, cD) in coeffs]
    details = list(reversed(details))
    return [d[:n].astype(np.float32) for d in details]

def _corrcoef_safe(M: np.ndarray) -> np.ndarray:
    if M.shape[0] < 2:
        return np.eye(M.shape[1], dtype=np.float32)
    C = np.corrcoef(M, rowvar=False)
    C = np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(C, 0.0)
    return np.maximum(C, 0.0).astype(np.float32)

def _topm_sparsify(A: np.ndarray, top_m: int) -> np.ndarray:
    N = A.shape[0]
    B = np.zeros_like(A, dtype=np.float32)
    if top_m <= 0 or top_m >= N:
        B = A.copy()
    else:
        for i in range(N):
            idx = np.argsort(A[i])[-top_m:]
            B[i, idx] = A[i, idx]
        B = np.maximum(B, B.T)
    np.fill_diagonal(B, 0.0)
    return B.astype(np.float32)

def build_wavelet_graphs_from_returns(R: np.ndarray, cfg: WaveletGraphConfig) -> np.ndarray:
    Rres = residualize_returns(R, cfg.residual_mode)
    T, N = Rres.shape
    S = int(cfg.levels)
    A_scales = np.zeros((T, S, N, N), dtype=np.float32)
    for t in range(T):
        start = max(0, t - cfg.window)
        X = Rres[start:t]
        if X.shape[0] < cfg.min_obs:
            continue
        details_by_stock = [_swt_details(X[:, j], cfg.wavelet, S) for j in range(N)]
        for s in range(S):
            M = np.stack([details_by_stock[j][s] for j in range(N)], axis=1)
            A_scales[t, s] = _topm_sparsify(_corrcoef_safe(M), cfg.top_m)
        if (t + 1) % 100 == 0 or t == T - 1:
            print(f"[WaveGraph] built {t+1}/{T}")
    return A_scales

def save_graph_cache(path: str, dates, instruments, A_scales: np.ndarray):
    import os
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez_compressed(path, dates=np.array(dates, dtype=object), instruments=np.array(instruments, dtype=object), A_scales=A_scales.astype(np.float32))

def load_graph_cache(path: str):
    d = np.load(path, allow_pickle=True)
    return d["A_scales"].astype(np.float32)
