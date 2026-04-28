from __future__ import annotations
import os, sys, argparse, json
from pathlib import Path
import numpy as np
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from environment_setting.io_config import load_config, resolve_path, ensure_dir
from environment_setting.data_loader import load_npz_dataset, split_indices_from_cfg
from risk_layer.wavelet_graph import load_graph_cache
from signal_layer.features import build_signal_features
from signal_layer.labels import make_cross_sectional_label
from signal_layer.models import fit_signal_model, predict_signal, save_signal_model
from utils.seed import set_seed

def rank_ic(pred: np.ndarray, label: np.ndarray, idx: np.ndarray) -> tuple[float, float]:
    """
    计算每日横截面 Rank IC。

    Rank IC 本质上是：
    对每一天 t，把所有股票的预测值 pred[t] 和真实标签 label[t] 做 Spearman 相关。
    这里不用 scipy.stats.spearmanr(...).correlation，是为了避免 Pylance 类型误报。
    """
    from scipy.stats import rankdata

    vals: list[float] = []

    for t_raw in idx:
        t = int(t_raw)

        y = np.asarray(label[t], dtype=np.float64)
        p = np.asarray(pred[t], dtype=np.float64)

        mask = np.isfinite(y) & np.isfinite(p)

        if int(mask.sum()) > 3:
            # Spearman 相关 = rank 后的 Pearson 相关
            p_rank = rankdata(p[mask]).astype(np.float64)
            y_rank = rankdata(y[mask]).astype(np.float64)

            corr_mat = np.corrcoef(p_rank, y_rank)
            corr = float(corr_mat[0, 1])

            if np.isfinite(corr):
                vals.append(corr)

    vals_arr = np.asarray(vals, dtype=np.float64)

    if vals_arr.size == 0:
        return 0.0, 0.0

    return float(vals_arr.mean()), float(vals_arr.std())
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="config.yaml"); args = ap.parse_args()
    cfg = load_config(args.config); set_seed(int(cfg["project"].get("seed", 42))); project_root = cfg["project"]["root_dir"]
    data = load_npz_dataset(resolve_path(cfg["data"]["dataset_npz"], project_root)); tr, va, te = split_indices_from_cfg(data.dates, cfg)
    A = None; graph_path = resolve_path(cfg["risk"]["graph_cache_npz"], project_root)
    if cfg["signal"].get("use_graph_features", True) and os.path.exists(graph_path): A = load_graph_cache(graph_path)
    F, names = build_signal_features(data.R, cfg["signal"].get("feature_windows", [5,10,20,60]), A, cfg["signal"].get("use_graph_features", True))
    y = make_cross_sectional_label(data.R, int(cfg["signal"].get("horizon", 5)), cfg["signal"].get("label_type", "zscore_excess"))
    art = fit_signal_model(F, y, tr, cfg, feature_names=names); pred = predict_signal(art, F)
    out_dir = ensure_dir(resolve_path("outputs", project_root)); model_path = os.path.join(out_dir, "signal_model.joblib"); save_signal_model(art, model_path)
    np.savez_compressed(os.path.join(out_dir, "signal_predictions.npz"), pred=pred, label=y, dates=data.dates, instruments=data.instruments)
    metrics = {}
    for name, idx in [("train", tr), ("valid", va), ("test", te)]:
        m, s = rank_ic(pred, y, idx); metrics[name] = {"rank_ic_mean": m, "rank_ic_std": s, "rank_ic_ir": m/(s+1e-8)}
    with open(os.path.join(out_dir, "signal_metrics.json"), "w", encoding="utf-8") as f: json.dump(metrics, f, ensure_ascii=False, indent=2)
    print("[OK] saved signal model:", model_path); print(json.dumps(metrics, ensure_ascii=False, indent=2))
if __name__ == "__main__": main()
