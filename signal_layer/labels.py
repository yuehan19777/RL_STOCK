from __future__ import annotations

from typing import Dict, List
import numpy as np


def future_cum_returns(R: np.ndarray, horizon: int) -> np.ndarray:
    """
    计算未来 horizon 日累计收益。

    R:
        (T, N)，其中 R[t, i] 表示在 t 时刻建仓后下一期可获得的收益。

    返回:
        y_raw: (T, N)
        y_raw[t, i] = R[t, i] + R[t+1, i] + ... + R[t+horizon-1, i]

    注意:
        最后 horizon-1 天无法构造完整标签，置为 nan。
    """
    R = np.asarray(R, dtype=np.float64)
    T, N = R.shape

    out = np.full((T, N), np.nan, dtype=np.float64)

    for t in range(T):
        end = t + horizon
        if end <= T:
            out[t] = np.nansum(R[t:end], axis=0)

    return out.astype(np.float32)


def make_cross_sectional_label(
    R: np.ndarray,
    horizon: int = 5,
    label_type: str = "zscore_excess",
) -> np.ndarray:
    """
    构造横截面标签。

    标签目标不是预测价格，而是预测股票未来相对表现。

    raw future return:
        fr_{i,t}^{(h)}

    excess:
        fr_{i,t}^{(h)} - mean_j fr_{j,t}^{(h)}

    zscore_excess:
        (fr_{i,t}^{(h)} - mean_j fr_{j,t}^{(h)}) / std_j(fr_{j,t}^{(h)})

    rank:
        当日横截面排序百分位。
    """
    fr = future_cum_returns(R, horizon=horizon).astype(np.float64)

    label = np.full_like(fr, np.nan, dtype=np.float64)

    T, N = fr.shape

    for t in range(T):
        x = fr[t]
        mask = np.isfinite(x)

        if int(mask.sum()) < 4:
            continue

        mean = float(np.nanmean(x[mask]))
        std = float(np.nanstd(x[mask]))

        if label_type == "raw":
            label[t, mask] = x[mask]

        elif label_type == "excess":
            label[t, mask] = x[mask] - mean

        elif label_type == "zscore_excess":
            label[t, mask] = (x[mask] - mean) / (std + 1e-8)

        elif label_type == "rank":
            order = np.argsort(x[mask])
            ranks = np.empty(int(mask.sum()), dtype=np.float64)
            ranks[order] = np.arange(int(mask.sum()), dtype=np.float64)

            if int(mask.sum()) > 1:
                ranks = ranks / float(int(mask.sum()) - 1)
            else:
                ranks[:] = 0.5

            label[t, mask] = ranks

        else:
            raise ValueError(f"Unknown label_type: {label_type}")

    return label.astype(np.float32)


def make_multi_horizon_labels(
    R: np.ndarray,
    horizons: List[int],
    label_type: str = "zscore_excess",
) -> Dict[int, np.ndarray]:
    """
    为多个 horizon 构造标签。

    返回:
        {
            5: label_5d,
            10: label_10d,
            20: label_20d
        }
    """
    labels: Dict[int, np.ndarray] = {}

    for h in horizons:
        labels[int(h)] = make_cross_sectional_label(
            R=R,
            horizon=int(h),
            label_type=label_type,
        )

    return labels