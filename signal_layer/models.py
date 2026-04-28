from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional
import numpy as np
import joblib

from sklearn.ensemble import HistGradientBoostingRegressor


@dataclass
class SignalArtifacts:
    """
    保存信号模型训练后的所有必要信息。

    model_type:
        ridge / hgb / xgb_reg / xgb_ranker / mlp

    model:
        训练好的模型对象。对于 ridge、hgb、xgb_reg、xgb_ranker，
        它们都有 predict 方法；对于 mlp，这里保存 torch 模型参数字典。

    mean, std:
        只用训练集计算出来的特征标准化参数。

    feature_names:
        特征名称，方便后续解释特征重要性。

    cfg:
        训练时使用的配置，预测时需要读取 winsorize_z 等参数。
    """
    model_type: str
    model: Any
    mean: np.ndarray
    std: np.ndarray
    feature_names: list[str]
    cfg: Dict[str, Any]


def _winsorize(X: np.ndarray, z: float = 5.0) -> np.ndarray:
    """
    截尾极端值，避免少数极端因子值影响模型。
    """
    if z is None or float(z) <= 0:
        return X
    return np.clip(X, -float(z), float(z))


def _standardize_with_train_stats(X: np.ndarray, mean: np.ndarray, std: np.ndarray, z: float) -> np.ndarray:
    """
    用训练集均值和标准差标准化，并进行 winsorize。
    """
    Xn = (X - mean) / (std + 1e-8)
    Xn = _winsorize(Xn, z)
    Xn = np.nan_to_num(Xn, nan=0.0, posinf=0.0, neginf=0.0)
    return Xn.astype(np.float32)


def flatten_xy(F: np.ndarray, y: np.ndarray, idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    把面板数据展开成普通监督学习格式。

    F: (T, N, K)
    y: (T, N)
    idx: 日期下标

    return:
        X:  (samples, K)
        yy: (samples,)
    """
    X = F[idx].reshape(-1, F.shape[-1])
    yy = y[idx].reshape(-1)

    mask = np.isfinite(yy) & np.all(np.isfinite(X), axis=1)

    return X[mask].astype(np.float32), yy[mask].astype(np.float32)


def _label_to_relevance(y_day: np.ndarray, n_bins: int = 5) -> np.ndarray:
    """
    把某一天的连续收益标签转成排序 relevance label。

    XGBRanker 更适合学习 0,1,2,3,4 这类等级标签。
    这里按横截面排序，把股票分成 n_bins 个等级：

    最差的一组 -> 0
    最好的一组 -> n_bins - 1
    """
    y_day = np.asarray(y_day, dtype=np.float64)
    n = len(y_day)

    if n == 0:
        return np.zeros(0, dtype=np.float32)

    order = np.argsort(y_day)
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(n, dtype=np.float64)

    if n <= 1:
        rel = np.zeros(n, dtype=np.float32)
    else:
        pct_rank = ranks / float(n - 1)
        rel = np.floor(pct_rank * float(n_bins)).astype(np.int64)
        rel = np.clip(rel, 0, n_bins - 1).astype(np.float32)

    return rel.astype(np.float32)


def flatten_xy_grouped_for_ranker(
    F: np.ndarray,
    y: np.ndarray,
    idx: np.ndarray,
    n_bins: int = 5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    为 XGBRanker 构造训练数据。

    排序模型不能简单把所有样本混在一起，因为它要知道：
    哪些股票属于同一天，应该在同一天内部排序。

    所以这里返回：
        X_all:      所有样本特征
        y_rel_all:  每个样本的 relevance label
        group:      每一天有效股票数量

    group 的含义：
        如果第 1 天有 20 只股票，第 2 天有 19 只股票，
        那么 group = [20, 19, ...]
    """
    X_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []
    group: list[int] = []

    for t_raw in idx:
        t = int(t_raw)

        X_day = np.asarray(F[t], dtype=np.float32)       # (N, K)
        y_day = np.asarray(y[t], dtype=np.float32)       # (N,)

        mask = np.isfinite(y_day) & np.all(np.isfinite(X_day), axis=1)

        if int(mask.sum()) < 4:
            continue

        X_valid = X_day[mask]
        y_valid = y_day[mask]

        y_rel = _label_to_relevance(y_valid, n_bins=n_bins)

        X_list.append(X_valid.astype(np.float32))
        y_list.append(y_rel.astype(np.float32))
        group.append(int(len(y_rel)))

    if len(X_list) == 0:
        raise ValueError("No valid grouped samples for XGBRanker. Please check labels/features.")

    X_all = np.concatenate(X_list, axis=0).astype(np.float32)
    y_all = np.concatenate(y_list, axis=0).astype(np.float32)
    group_arr = np.asarray(group, dtype=np.int32)

    return X_all, y_all, group_arr


def fit_signal_model(
    F: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    cfg: Dict[str, Any],
    feature_names: Optional[list[str]] = None,
) -> SignalArtifacts:
    """
    训练信号模型。

    支持：
        ridge:
            线性基准模型。

        hgb:
            sklearn 的 HistGradientBoostingRegressor。
            不需要安装 xgboost，适合先做树模型基准。

        xgb_reg:
            XGBoost 回归模型。
            预测未来横截面超额收益分数。

        xgb_ranker:
            XGBoost 排序模型。
            每一天作为一个 group，直接学习股票之间的相对排序。

        mlp:
            小型神经网络基准。
    """
    sig_cfg = cfg["signal"]
    model_type = str(sig_cfg.get("model_type", "ridge")).lower()
    winsorize_z = float(sig_cfg.get("winsorize_z", 5.0))
    seed = int(cfg.get("project", {}).get("seed", cfg.get("seed", 42)))

    # 对 ranker 和非 ranker 都先计算训练集标准化参数
    Xtr_raw, ytr_raw = flatten_xy(F, y, train_idx)

    mean = Xtr_raw.mean(axis=0).astype(np.float32)
    std = Xtr_raw.std(axis=0).astype(np.float32)
    std = np.where(std < 1e-8, 1.0, std).astype(np.float32)

    if model_type == "ridge":
        from sklearn.linear_model import Ridge

        Xtr = _standardize_with_train_stats(Xtr_raw, mean, std, winsorize_z)

        model = Ridge(
            alpha=float(sig_cfg.get("ridge_alpha", 10.0)),
            random_state=seed,
        )
        model.fit(Xtr, ytr_raw)

    elif model_type == "hgb":
        hgb_cfg = sig_cfg.get("hgb", {})

        Xtr = _standardize_with_train_stats(Xtr_raw, mean, std, winsorize_z)

        model = HistGradientBoostingRegressor(
            max_iter=int(hgb_cfg.get("max_iter", 500)),
            learning_rate=float(hgb_cfg.get("learning_rate", 0.03)),
            max_leaf_nodes=int(hgb_cfg.get("max_leaf_nodes", 15)),
            min_samples_leaf=int(hgb_cfg.get("min_samples_leaf", 30)),
            l2_regularization=float(hgb_cfg.get("l2_regularization", 1.0)),
            early_stopping=bool(hgb_cfg.get("early_stopping", True)),
            validation_fraction=float(hgb_cfg.get("validation_fraction", 0.1)),
            n_iter_no_change=int(hgb_cfg.get("n_iter_no_change", 30)),
            random_state=seed,
        )
        model.fit(Xtr, ytr_raw)

    elif model_type == "xgb_reg":
        try:
            from xgboost import XGBRegressor
        except ImportError as e:
            raise ImportError(
                "You selected signal.model_type='xgb_reg', but xgboost is not installed. "
                "Please run: pip install xgboost"
            ) from e

        xgb_cfg = sig_cfg.get("xgb_reg", {})

        Xtr = _standardize_with_train_stats(Xtr_raw, mean, std, winsorize_z)

        model = XGBRegressor(
            n_estimators=int(xgb_cfg.get("n_estimators", 600)),
            max_depth=int(xgb_cfg.get("max_depth", 3)),
            learning_rate=float(xgb_cfg.get("learning_rate", 0.03)),
            subsample=float(xgb_cfg.get("subsample", 0.8)),
            colsample_bytree=float(xgb_cfg.get("colsample_bytree", 0.8)),
            reg_alpha=float(xgb_cfg.get("reg_alpha", 0.1)),
            reg_lambda=float(xgb_cfg.get("reg_lambda", 2.0)),
            min_child_weight=float(xgb_cfg.get("min_child_weight", 5.0)),
            objective=str(xgb_cfg.get("objective", "reg:pseudohubererror")),
            tree_method=str(xgb_cfg.get("tree_method", "hist")),
            random_state=seed,
            n_jobs=int(xgb_cfg.get("n_jobs", -1)),
        )
        model.fit(Xtr, ytr_raw)

    elif model_type == "xgb_ranker":
        try:
            from xgboost import XGBRanker
        except ImportError as e:
            raise ImportError(
                "You selected signal.model_type='xgb_ranker', but xgboost is not installed. "
                "Please run: pip install xgboost"
            ) from e

        rank_cfg = sig_cfg.get("xgb_ranker", {})
        n_bins = int(rank_cfg.get("label_bins", 5))

        X_rank_raw, y_rel, group = flatten_xy_grouped_for_ranker(
            F=F,
            y=y,
            idx=train_idx,
            n_bins=n_bins,
        )

        X_rank = _standardize_with_train_stats(X_rank_raw, mean, std, winsorize_z)

        model = XGBRanker(
            n_estimators=int(rank_cfg.get("n_estimators", 500)),
            max_depth=int(rank_cfg.get("max_depth", 3)),
            learning_rate=float(rank_cfg.get("learning_rate", 0.03)),
            subsample=float(rank_cfg.get("subsample", 0.8)),
            colsample_bytree=float(rank_cfg.get("colsample_bytree", 0.8)),
            reg_alpha=float(rank_cfg.get("reg_alpha", 0.1)),
            reg_lambda=float(rank_cfg.get("reg_lambda", 2.0)),
            min_child_weight=float(rank_cfg.get("min_child_weight", 5.0)),
            objective=str(rank_cfg.get("objective", "rank:pairwise")),
            tree_method=str(rank_cfg.get("tree_method", "hist")),
            random_state=seed,
            n_jobs=int(rank_cfg.get("n_jobs", -1)),
        )

        model.fit(X_rank, y_rel, group=group)

    elif model_type == "mlp":
        Xtr = _standardize_with_train_stats(Xtr_raw, mean, std, winsorize_z)
        model = _fit_mlp(Xtr, ytr_raw, cfg)

    else:
        raise ValueError(
            f"Unknown signal.model_type: {model_type}. "
            "Available: ridge, hgb, xgb_reg, xgb_ranker, mlp"
        )

    return SignalArtifacts(
        model_type=model_type,
        model=model,
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
        feature_names=feature_names or [],
        cfg=cfg,
    )


def predict_signal(art: SignalArtifacts, F: np.ndarray) -> np.ndarray:
    """
    对每一天、每只股票预测 alpha 分数。

    输出：
        pred: (T, N)

    注意：
        最后会做每日横截面中心化：
            pred[t] = pred[t] - mean(pred[t])

        因为组合优化中更关心横截面相对强弱。
    """
    T, N, K = F.shape

    X = F.reshape(-1, K).astype(np.float32)

    winsorize_z = float(art.cfg["signal"].get("winsorize_z", 5.0))

    Xn = _standardize_with_train_stats(
        X=X,
        mean=art.mean,
        std=art.std,
        z=winsorize_z,
    )

    if art.model_type in ["ridge", "hgb", "xgb_reg", "xgb_ranker"]:
        predict_fn = getattr(art.model, "predict", None)

        if predict_fn is None:
            raise TypeError("The loaded signal model does not have a predict method.")

        pred = predict_fn(Xn)

    elif art.model_type == "mlp":
        pred = _predict_mlp(art.model, Xn)

    else:
        raise ValueError(art.model_type)

    pred_arr = np.asarray(pred, dtype=np.float32).reshape(T, N)

    # 每日横截面中心化，使预测值成为相对 alpha
    pred_arr = pred_arr - np.nanmean(pred_arr, axis=1, keepdims=True)

    pred_arr = np.nan_to_num(pred_arr, nan=0.0, posinf=0.0, neginf=0.0)

    return pred_arr.astype(np.float32)


def save_signal_model(art: SignalArtifacts, path: str) -> None:
    joblib.dump(art, path)


def load_signal_model(path: str) -> SignalArtifacts:
    return joblib.load(path)


def get_feature_importance(art: SignalArtifacts) -> Optional[np.ndarray]:
    """
    提取树模型特征重要性。

    对 xgb_reg / xgb_ranker：
        使用 feature_importances_

    对 hgb：
        HistGradientBoostingRegressor 默认没有稳定的 feature_importances_。
    """
    importance = getattr(art.model, "feature_importances_", None)

    if importance is None:
        return None

    return np.asarray(importance, dtype=np.float64)


def save_feature_importance(art: SignalArtifacts, path: str) -> None:
    """
    保存特征重要性，便于解释模型学到了什么。
    """
    imp = get_feature_importance(art)

    if imp is None:
        return

    import pandas as pd

    names = art.feature_names
    if not names or len(names) != len(imp):
        names = [f"f{i}" for i in range(len(imp))]

    df = pd.DataFrame(
        {
            "feature": names,
            "importance": imp,
        }
    ).sort_values("importance", ascending=False)

    df.to_csv(path, index=False, encoding="utf-8-sig")


def _fit_mlp(X: np.ndarray, y: np.ndarray, cfg: Dict[str, Any]) -> Dict[str, Any]:
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader

    train_cfg = cfg["signal"].get("train", {})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    hidden = int(train_cfg.get("hidden_dim", 64))
    dropout = float(train_cfg.get("dropout", 0.1))
    epochs = int(train_cfg.get("epochs", 20))
    batch_size = int(train_cfg.get("batch_size", 4096))
    lr = float(train_cfg.get("lr", 1e-3))
    weight_decay = float(train_cfg.get("weight_decay", 1e-5))

    net = nn.Sequential(
        nn.Linear(X.shape[1], hidden),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, 1),
    ).to(device)

    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.HuberLoss()

    ds = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y[:, None], dtype=torch.float32),
    )

    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )

    net.train()

    for ep in range(epochs):
        total = 0.0

        for xb, yb in dl:
            xb = xb.to(device)
            yb = yb.to(device)

            loss = loss_fn(net(xb), yb)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()

            total += float(loss.item()) * len(xb)

        print(f"[Signal-MLP] epoch={ep + 1}/{epochs} loss={total / len(ds):.6f}")

    return {
        "state_dict": net.cpu().state_dict(),
        "in_dim": int(X.shape[1]),
        "hidden": hidden,
        "dropout": dropout,
    }


def _predict_mlp(model_obj: Dict[str, Any], X: np.ndarray) -> np.ndarray:
    import torch
    import torch.nn as nn

    hidden = int(model_obj["hidden"])
    dropout = float(model_obj["dropout"])
    in_dim = int(model_obj["in_dim"])

    net = nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, 1),
    )

    net.load_state_dict(model_obj["state_dict"])
    net.eval()

    outs = []

    with torch.no_grad():
        for i in range(0, len(X), 65536):
            xb = torch.tensor(X[i:i + 65536], dtype=torch.float32)
            out = net(xb).squeeze(-1).numpy()
            outs.append(out)

    return np.concatenate(outs, axis=0).astype(np.float32)