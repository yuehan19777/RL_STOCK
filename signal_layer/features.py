from __future__ import annotations
from typing import List, Tuple, Optional
import numpy as np

def _rolling_return(R: np.ndarray, t: int, w: int) -> np.ndarray:
    start = max(0, t - w)
    if start >= t:
        return np.zeros(R.shape[1], dtype=np.float32)
    gross = 1.0 + np.nan_to_num(R[start:t], nan=0.0)
    return (np.prod(gross, axis=0) - 1.0).astype(np.float32)

def _rolling_std(R: np.ndarray, t: int, w: int) -> np.ndarray:
    start = max(0, t - w)
    if start >= t:
        return np.zeros(R.shape[1], dtype=np.float32)
    return np.nanstd(R[start:t], axis=0).astype(np.float32)

def _rolling_downside(R: np.ndarray, t: int, w: int) -> np.ndarray:
    start = max(0, t - w)
    if start >= t:
        return np.zeros(R.shape[1], dtype=np.float32)
    x = np.minimum(R[start:t], 0.0)
    return np.sqrt(np.nanmean(x * x, axis=0)).astype(np.float32)

def _rolling_mdd(R: np.ndarray, t: int, w: int) -> np.ndarray:
    start = max(0, t - w)
    if start >= t:
        return np.zeros(R.shape[1], dtype=np.float32)
    eq = np.cumprod(1.0 + np.nan_to_num(R[start:t], nan=0.0), axis=0)
    peak = np.maximum.accumulate(eq, axis=0)
    dd = eq / (peak + 1e-12) - 1.0
    return np.nanmin(dd, axis=0).astype(np.float32)

def _cs_zscore(x: np.ndarray) -> np.ndarray:
    return ((x - np.nanmean(x)) / (np.nanstd(x) + 1e-8)).astype(np.float32)

def _cs_rank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(np.argsort(x))
    n = len(x)
    return ((order + 1 - (n + 1) / 2.0) / n).astype(np.float32)

def make_base_features(R: np.ndarray, windows: List[int]) -> Tuple[np.ndarray, List[str]]:
    T, N = R.shape
    feats, names = [], []
    for w in windows:
        arrays = {
            f"mom_{w}": np.stack([_rolling_return(R, t, w) for t in range(T)]),
            f"vol_{w}": np.stack([_rolling_std(R, t, w) for t in range(T)]),
            f"downvol_{w}": np.stack([_rolling_downside(R, t, w) for t in range(T)]),
            f"mdd_{w}": np.stack([_rolling_mdd(R, t, w) for t in range(T)]),
        }
        for name, arr in arrays.items():
            feats.append(arr[..., None]); names.append(name)
            feats.append(np.stack([_cs_zscore(arr[t]) for t in range(T)])[..., None]); names.append(name + "_csz")
            feats.append(np.stack([_cs_rank(arr[t]) for t in range(T)])[..., None]); names.append(name + "_rank")
    F = np.concatenate(feats, axis=-1).astype(np.float32)
    return np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0), names

def add_graph_node_features(F: np.ndarray, A_scales: Optional[np.ndarray], R: np.ndarray, windows: List[int]) -> Tuple[np.ndarray, List[str]]:
    if A_scales is None:
        return F, []
    T, S, N, _ = A_scales.shape
    A = np.nanmean(A_scales, axis=1).astype(np.float32)
    out, names = [], []
    centrality = A.sum(axis=-1)
    out.append(centrality[..., None]); names.append("graph_centrality")
    for w in windows:
        mom = np.stack([_rolling_return(R, t, w) for t in range(T)])
        vol = np.stack([_rolling_std(R, t, w) for t in range(T)])
        denom = A.sum(axis=-1) + 1e-8
        out.append((np.einsum("tij,tj->ti", A, mom) / denom)[..., None]); names.append(f"neighbor_mom_{w}")
        out.append((np.einsum("tij,tj->ti", A, vol) / denom)[..., None]); names.append(f"neighbor_vol_{w}")
    G = np.concatenate(out, axis=-1).astype(np.float32)
    return np.concatenate([F, np.nan_to_num(G)], axis=-1), names

def build_signal_features(R: np.ndarray, windows: List[int], A_scales: Optional[np.ndarray] = None, use_graph_features: bool = True):
    F, names = make_base_features(R, windows)
    if use_graph_features and A_scales is not None:
        F, gnames = add_graph_node_features(F, A_scales, R, windows=[windows[0], windows[-1]])
        names += gnames
    return F.astype(np.float32), names
