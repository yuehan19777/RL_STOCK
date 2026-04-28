from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import os
import numpy as np
import pandas as pd


# 读取 NPZ 数据和按日期划分 train/valid/test

def to_datestr_arr(dates_arr) -> np.ndarray:
    out = []
    for x in dates_arr:
        if isinstance(x, bytes):
            out.append(x.decode("utf-8")[:10])
        else:
            out.append(str(x)[:10])
    return np.array(out, dtype=object)

def slice_by_date(dates: np.ndarray, start: str, end: str) -> np.ndarray:
    d = to_datestr_arr(dates)
    m = (d >= str(start)[:10]) & (d <= str(end)[:10])
    idx = np.where(m)[0]
    if len(idx) == 0:
        raise ValueError(f"Empty slice for [{start}, {end}]")
    return idx

@dataclass
class MarketData:
    X: np.ndarray
    R: np.ndarray
    dates: np.ndarray
    instruments: np.ndarray
    feature_names: Optional[np.ndarray] = None

def load_npz_dataset(npz_path: str) -> MarketData:
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"dataset_npz not found: {npz_path}")
    d = np.load(npz_path, allow_pickle=True)
    for k in ["X", "R", "dates", "instruments"]:
        if k not in d:
            raise KeyError(f"NPZ missing key {k}. keys={list(d.keys())}")
    X = d["X"].astype(np.float32)
    R = d["R"].astype(np.float32)
    dates = to_datestr_arr(d["dates"])
    instruments = np.array([str(x) for x in d["instruments"].tolist()], dtype=object)
    feature_names = d["feature_names"] if "feature_names" in d else None
    if X.shape[:2] != R.shape:
        raise ValueError(f"X/R shape mismatch: X={X.shape}, R={R.shape}")
    return MarketData(X=X, R=R, dates=dates, instruments=instruments, feature_names=feature_names)

def split_indices_from_cfg(dates: np.ndarray, cfg: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    sp = cfg["data"]["split"]
    tr = slice_by_date(dates, sp["train"][0], sp["train"][1])
    va = slice_by_date(dates, sp["valid"][0], sp["valid"][1])
    te = slice_by_date(dates, sp["test"][0], sp["test"][1])
    return tr, va, te
