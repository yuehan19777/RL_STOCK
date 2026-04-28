from __future__ import annotations
import numpy as np

def rolling_covariance(R: np.ndarray, t: int, window: int = 60, shrinkage: float = 0.3) -> np.ndarray:
    T, N = R.shape
    start = max(0, t - window)
    X = R[start:t].astype(np.float64)
    if X.shape[0] < 2:
        return np.eye(N, dtype=np.float32) * 1e-4
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    C = np.cov(X, rowvar=False)
    if C.ndim == 0:
        C = np.eye(N) * float(C)
    C = np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)
    diag = np.diag(np.diag(C))
    C = (1 - shrinkage) * C + shrinkage * diag + np.eye(N) * 1e-8
    return C.astype(np.float32)

def market_state_from_returns(R: np.ndarray, t: int, windows=(5, 20, 60)) -> np.ndarray:
    feats = []
    for w in windows:
        start = max(0, t - w)
        X = R[start:t]
        if len(X) == 0:
            feats += [0.0, 0.0]
        else:
            ew = X.mean(axis=1)
            feats += [float(np.prod(1 + ew) - 1), float(np.std(ew))]
    if t > 0:
        row = R[t - 1]
        feats += [float(np.mean(row)), float(np.std(row)), float(np.mean(row > 0))]
    else:
        feats += [0.0, 0.0, 0.0]
    return np.array(feats, dtype=np.float32)
