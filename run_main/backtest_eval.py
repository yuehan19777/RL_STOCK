from __future__ import annotations

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Any, Dict, Tuple, cast

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from environment_setting.io_config import load_config, resolve_path, ensure_dir
from environment_setting.data_loader import load_npz_dataset, split_indices_from_cfg

from signal_layer.models import load_signal_model, predict_signal
from signal_layer.features import build_signal_features

from risk_layer.wavelet_graph import load_graph_cache
from risk_layer.covariance import rolling_covariance
from risk_layer.graph_features import fuse_graph_scales, project_psd

from portfolio_layer.optimizer import PortfolioParams, solve_portfolio, apply_risk_overlay
from portfolio_layer.metrics import summarize_backtest, drawdown_series

from controller_ppo.networks import ActorCritic, ActorCriticConfig
from controller_ppo.meta_env import build_state, transform_raw_action, compute_controller_context


# ============================================================
# 1. 通用工具函数
# ============================================================

def _as_array_graph_cache(graph_obj: Any) -> np.ndarray:
    """
    兼容不同版本的 load_graph_cache 返回值。

    可能返回：
    1. A_scales ndarray
    2. dict-like object with A_scales
    3. dataclass/object with .A_scales
    """
    if isinstance(graph_obj, np.ndarray):
        return np.asarray(graph_obj, dtype=np.float32)

    if isinstance(graph_obj, dict):
        if "A_scales" not in graph_obj:
            raise KeyError("graph cache dict must contain key 'A_scales'.")
        return np.asarray(graph_obj["A_scales"], dtype=np.float32)

    A_scales_obj = getattr(graph_obj, "A_scales", None)

    if A_scales_obj is not None:
        return np.asarray(A_scales_obj, dtype=np.float32)

    raise TypeError(f"Unknown graph cache type: {type(graph_obj)}")

def _safe_json_dump(obj: Dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _safe_mean(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0
    return float(x.mean())


def _compute_benchmark_equity(bench_rets: list[float], init_equity: float = 1.0) -> list[float]:
    eq = [float(init_equity)]
    cur = float(init_equity)

    for r in bench_rets:
        cur *= float(1.0 + r)
        eq.append(cur)

    return eq


# ============================================================
# 2. 加载信号预测
# ============================================================

def load_or_build_signal_prediction(
    cfg: Dict[str, Any],
    data_R: np.ndarray,
    A_scales: np.ndarray,
    project_root: str,
) -> np.ndarray:
    """
    优先读取 outputs/signal_predictions.npz。
    如果不存在，则加载 signal_model.joblib 并重新构造特征预测。
    """
    sig_cfg = cfg.get("signal", {})

    pred_path = sig_cfg.get("predictions_npz", "outputs/signal_predictions.npz")
    pred_path = resolve_path(pred_path, project_root)

    if os.path.exists(pred_path):
        pred_npz = np.load(pred_path, allow_pickle=True)

        if "pred" in pred_npz:
            mu_pred = pred_npz["pred"].astype(np.float32)
            print(f"[OK] loaded signal predictions from {pred_path}")
            return mu_pred

        if "mu_pred" in pred_npz:
            mu_pred = pred_npz["mu_pred"].astype(np.float32)
            print(f"[OK] loaded signal predictions from {pred_path}")
            return mu_pred

        raise KeyError(f"{pred_path} must contain key 'pred' or 'mu_pred'.")

    model_path = resolve_path(sig_cfg.get("model_path", "outputs/signal_model.joblib"), project_root)

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Cannot find signal prediction file or signal model.\n"
            f"Missing prediction: {pred_path}\n"
            f"Missing model: {model_path}\n"
            f"Please run: python run_main/train_signal.py --config config.yaml"
        )

    print("[INFO] signal prediction file not found, rebuilding predictions from signal model...")

    model = load_signal_model(model_path)

    F, _ = build_signal_features(
        data_R,
        sig_cfg.get("feature_windows", [5, 10, 20, 60]),
        A_scales,
        sig_cfg.get("use_graph_features", True),
    )

    mu_pred = predict_signal(model, F).astype(np.float32)

    return mu_pred


# ============================================================
# 3. PPO 相关工具
# ============================================================

def _load_ppo_controller(
    ppo_path: str,
    cfg: Dict[str, Any],
    state_dim: int,
    action_dim: int,
    device: str = "cpu",
) -> ActorCritic:
    """
    兼容不同保存格式的 PPO controller。

    支持 checkpoint keys:
    - "state_dict"
    - "model"
    """
    import torch

    if not os.path.exists(ppo_path):
        raise FileNotFoundError(
            f"PPO controller not found: {ppo_path}\n"
            f"Please run: python run_main/train_ppo_controller.py --config config.yaml"
        )

    ckpt = torch.load(ppo_path, map_location=device)

    ckpt_cfg = ckpt.get("cfg", cfg)
    ppo_cfg = ckpt_cfg.get("ppo", {})
    net_cfg = ppo_cfg.get("network", {})

    state_dim_ckpt = int(ckpt.get("state_dim", state_dim))
    action_dim_ckpt = int(ckpt.get("action_dim", action_dim))

    model = ActorCritic(
        ActorCriticConfig(
            state_dim=state_dim_ckpt,
            action_dim=action_dim_ckpt,
            hidden_dim=int(net_cfg.get("hidden_dim", 64)),
            n_layers=int(net_cfg.get("n_layers", 2)),
            init_log_std=float(net_cfg.get("init_log_std", -0.5)),
        )
    )

    if "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"])
    elif "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    else:
        raise KeyError("PPO checkpoint must contain key 'state_dict' or 'model'.")

    model.to(device)
    model.eval()

    print(f"[OK] loaded PPO controller from {ppo_path}")

    return model


def _build_ppo_state_robust(
    t: int,
    R: np.ndarray,
    mu_pred: np.ndarray,
    A_scales: np.ndarray,
    w_prev_asset: np.ndarray,
    equity: float,
    equity_peak: float,
    cfg: Dict[str, Any],
) -> np.ndarray:
    """
    兼容不同版本的 controller_ppo.meta_env.build_state。

    推荐新版接口：
        build_state(
            R=R,
            mu_pred=mu_pred,
            A_scales=A_scales,
            t=t,
            w_prev=w_prev_asset,
            equity=equity,
            equity_peak=equity_peak,
        )

    旧版接口：
        build_state(t, R, mu_pred, A_scales, w_prev_asset, equity, cfg)
    """
    build_state_fn = cast(Any, build_state)

    try:
        state = build_state_fn(
            R=R,
            mu_pred=mu_pred,
            A_scales=A_scales,
            t=int(t),
            w_prev=w_prev_asset,
            equity=float(equity),
            equity_peak=float(equity_peak),
        )
        return np.asarray(state, dtype=np.float32)

    except TypeError:
        pass

    try:
        state = build_state_fn(
            int(t),
            R,
            mu_pred,
            A_scales,
            w_prev_asset,
            float(equity),
            cfg,
        )
        return np.asarray(state, dtype=np.float32)

    except TypeError as e:
        raise TypeError(
            "Your controller_ppo.meta_env.build_state interface is not compatible. "
            "Please check its function signature."
        ) from e


def _infer_ppo_raw_action(model: ActorCritic, state: np.ndarray, device: str = "cpu") -> np.ndarray:
    """
    回测阶段使用策略均值动作，不采样，保证回测可复现。
    """
    import torch

    st = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

    with torch.no_grad():
        if hasattr(model, "distribution"):
            dist = model.distribution(st)
            raw = dist.mean.squeeze(0).detach().cpu().numpy()
        else:
            out = model(st)
            raw = out[0].squeeze(0).detach().cpu().numpy()

    return np.asarray(raw, dtype=np.float32)


def _transform_raw_action_robust(
    raw_action: np.ndarray,
    cfg: Dict[str, Any],
    num_scales: int,
    context: dict | None = None,
):
    """
    兼容不同版本的 transform_raw_action。
    """
    transform_fn = cast(Any, transform_raw_action)

    try:
        return transform_fn(
            raw_action=raw_action,
            cfg=cfg,
            num_scales=num_scales,
            context=context,
        )

    except TypeError:
        try:
            return transform_fn(
                raw_action,
                cfg,
                num_scales=num_scales,
                context=context,
            )
        except TypeError:
            return transform_fn(
                raw_action,
                cfg,
                levels=num_scales,
            )

# ============================================================
# 4. 参数生成：fixed 或 PPO
# ============================================================

def _make_fixed_params(cfg: Dict[str, Any]) -> Tuple[PortfolioParams, None, Dict[str, float]]:
    fixed_cfg = cfg["portfolio"]["fixed_params"]

    params = PortfolioParams(
        float(fixed_cfg.get("lambda_risk", 5.0)),
        float(fixed_cfg.get("rho_graph", 1.0)),
        float(fixed_cfg.get("eta_turnover", 0.1)),
        float(fixed_cfg.get("exposure", 0.8)),
    )

    meta_info = {
        "lambda_risk": float(params.lambda_risk),
        "rho_graph": float(params.rho_graph),
        "eta_turnover": float(params.eta_turnover),
        "exposure": float(params.exposure),
    }

    return params, None, meta_info


def _make_ppo_params(
    model: ActorCritic,
    cfg: Dict[str, Any],
    t: int,
    R: np.ndarray,
    mu_pred: np.ndarray,
    A_scales: np.ndarray,
    w_prev_asset: np.ndarray,
    equity: float,
    equity_peak: float,
    device: str,
):
    state = _build_ppo_state_robust(
        t=t,
        R=R,
        mu_pred=mu_pred,
        A_scales=A_scales,
        w_prev_asset=w_prev_asset,
        equity=equity,
        equity_peak=equity_peak,
        cfg=cfg,
    )

    raw_action = _infer_ppo_raw_action(
        model=model,
        state=state,
        device=device,
    )
    
    context = compute_controller_context(
        R=R,
        mu_pred=mu_pred,
        A_scales=A_scales,
        t=t,
        equity=equity,
        equity_peak=equity_peak,
    )

    meta_action = _transform_raw_action_robust(
        raw_action=raw_action,
        cfg=cfg,
        num_scales=int(A_scales.shape[1]),
        context=context,
    )

    params = PortfolioParams(
        float(meta_action.lambda_risk),
        float(meta_action.rho_graph),
        float(meta_action.eta_turnover),
        float(meta_action.exposure),
    )

    alpha = getattr(meta_action, "alpha", None)

    meta_info = {
        "lambda_risk": float(params.lambda_risk),
        "rho_graph": float(params.rho_graph),
        "eta_turnover": float(params.eta_turnover),
        "exposure": float(params.exposure),
    }

    meta_info.update(
        {
            "trend_score": float(context["trend_score"]),
            "risk_score": float(context["risk_score"]),
            "signal_score": float(context["signal_score"]),
            "ret20": float(context["ret20"]),
            "ret60": float(context["ret60"]),
            "vol20": float(context["vol20"]),
            "up_ratio": float(context["up_ratio"]),
        }
    )

    if alpha is not None:
        for i, a in enumerate(alpha):
            meta_info[f"alpha_{i + 1}"] = float(a)

    return params, alpha, meta_info


# ============================================================
# 5. 风险覆盖层
# ============================================================

def _compute_train_vol_threshold(
    R: np.ndarray,
    train_idx: np.ndarray,
    cfg: Dict[str, Any],
) -> float:
    overlay_cfg = cfg.get("portfolio", {}).get("risk_overlay", {})
    if overlay_cfg is None:
        overlay_cfg = {}

    vol_window = int(overlay_cfg.get("vol_window", 20))
    high_vol_quantile = float(overlay_cfg.get("high_vol_quantile", 0.8))

    ew_train = np.nanmean(R[train_idx], axis=1)

    train_vols = (
        pd.Series(ew_train)
        .rolling(vol_window, min_periods=5)
        .std()
        .dropna()
        .values
    )

    train_vols = np.asarray(train_vols, dtype=np.float64)

    if train_vols.size == 0:
        return 1e9

    return float(np.quantile(train_vols, high_vol_quantile))


def _apply_overlay_if_needed(
    params: PortfolioParams,
    R: np.ndarray,
    t: int,
    equity_curve: list[float],
    vol_thr: float,
    cfg: Dict[str, Any],
) -> Tuple[PortfolioParams, Dict[str, float]]:
    overlay_cfg = cfg.get("portfolio", {}).get("risk_overlay", {})
    if overlay_cfg is None:
        overlay_cfg = {}

    vol_window = int(overlay_cfg.get("vol_window", 20))

    s = max(0, int(t) - vol_window)
    recent_ew = np.nanmean(R[s:int(t)], axis=1)

    if len(recent_ew) >= 5:
        recent_vol = float(np.nanstd(recent_ew))
        high_vol = bool(recent_vol > vol_thr)
    else:
        recent_vol = 0.0
        high_vol = False

    cur_equity = float(equity_curve[-1])
    peak_equity = float(max(equity_curve))
    drawdown_now = float(cur_equity / (peak_equity + 1e-12) - 1.0)

    try:
        new_params = apply_risk_overlay(
            params,
            {"drawdown": drawdown_now, "high_vol": high_vol},
            cfg,
        )
    except Exception:
        new_params = params

    overlay_info = {
        "recent_vol": recent_vol,
        "high_vol": float(high_vol),
        "drawdown_before_trade": drawdown_now,
    }

    return new_params, overlay_info


# ============================================================
# 6. 核心回测函数
# ============================================================

def run_backtest(cfg: Dict[str, Any], mode: str = "fixed") -> Dict[str, Any]:
    if mode not in ["fixed", "ppo"]:
        raise ValueError("mode must be 'fixed' or 'ppo'.")

    project_root = cfg["project"]["root_dir"]

    # ---------- 加载数据 ----------
    data = load_npz_dataset(
        resolve_path(cfg["data"]["dataset_npz"], project_root)
    )

    train_idx, valid_idx, test_idx = split_indices_from_cfg(data.dates, cfg)

    R = np.asarray(data.R, dtype=np.float32)
    dates_all = data.dates
    instruments = data.instruments

    T, N = R.shape

    # ---------- 加载小波图 ----------
    graph_obj = load_graph_cache(
        resolve_path(cfg["risk"]["graph_cache_npz"], project_root)
    )
    A_scales = _as_array_graph_cache(graph_obj)

    if A_scales.shape[0] != T:
        raise ValueError(f"A_scales time length {A_scales.shape[0]} != R time length {T}")

    # ---------- 加载信号预测 ----------
    mu_pred = load_or_build_signal_prediction(
        cfg=cfg,
        data_R=R,
        A_scales=A_scales,
        project_root=project_root,
    )

    if mu_pred.shape != R.shape:
        raise ValueError(f"mu_pred shape {mu_pred.shape} != R shape {R.shape}")

    # ---------- 组合设置 ----------
    port_cfg = cfg.get("portfolio", {})
    risk_cfg = cfg.get("risk", {})

    use_cash = bool(port_cfg.get("use_cash", port_cfg.get("allow_cash", True)))
    init_equity = float(port_cfg.get("init_equity", 1.0))
    cost_kappa = float(port_cfg.get("cost_kappa", 0.001))
    w_max = float(port_cfg.get("w_max", 0.2))
    turnover_limit = float(port_cfg.get("turnover_limit", 0.8))
    allow_short = bool(port_cfg.get("allow_short", False))

    cov_window = int(risk_cfg.get("cov_window", port_cfg.get("window_cov", 60)))
    cov_shrinkage = float(risk_cfg.get("cov_shrinkage", port_cfg.get("shrinkage", 0.3)))
    psd_project_graph = bool(risk_cfg.get("psd_project_graph", True))

    # ---------- 初始化权重 ----------
    w_prev_asset = np.ones(N, dtype=np.float64) / float(N)

    if use_cash:
        # 初始不持现金，也就是资产等权。
        w_prev_full = np.concatenate([w_prev_asset, np.array([0.0])])
    else:
        w_prev_full = w_prev_asset.copy()

    # ---------- 加载 PPO ----------
    ppo_model = None
    device = "cpu"

    if mode == "ppo":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"

        dummy_state = _build_ppo_state_robust(
            t=int(test_idx[0]),
            R=R,
            mu_pred=mu_pred,
            A_scales=A_scales,
            w_prev_asset=w_prev_asset,
            equity=init_equity,
            equity_peak=init_equity,
            cfg=cfg,
        )

        use_scale_alpha = bool(cfg.get("ppo", {}).get("use_scale_alpha", False))
        action_dim = 4 + int(A_scales.shape[1]) if use_scale_alpha else 4

        ppo_path = resolve_path(
            cfg.get("ppo", {}).get("model_path", "outputs/ppo_controller.pt"),
            project_root,
        )

        ppo_model = _load_ppo_controller(
            ppo_path=ppo_path,
            cfg=cfg,
            state_dim=len(dummy_state),
            action_dim=action_dim,
            device=device,
        )

    # ---------- 风险覆盖层阈值 ----------
    vol_thr = _compute_train_vol_threshold(R, train_idx, cfg)

    # ---------- 回测容器 ----------
    equity_curve = [init_equity]
    bench_equity_curve = [init_equity]

    out_dates = []
    port_rets = []
    bench_rets = []
    active_rets = []
    net_rets = []
    turnovers = []
    costs = []
    graph_risks = []
    cashes = []
    weights_rows = []
    params_rows = []

    # ========================================================
    # 回测循环
    # ========================================================
    for t_raw in test_idx:
        t = int(t_raw)
        date_str = str(dates_all[t])[:10]

        cur_equity = float(equity_curve[-1])
        cur_peak = float(max(equity_curve))

        # ---------- 1. 生成组合参数 ----------
        if mode == "fixed":
            params, alpha, meta_info = _make_fixed_params(cfg)
        else:
            assert ppo_model is not None

            params, alpha, meta_info = _make_ppo_params(
                model=ppo_model,
                cfg=cfg,
                t=t,
                R=R,
                mu_pred=mu_pred,
                A_scales=A_scales,
                w_prev_asset=w_prev_asset,
                equity=cur_equity,
                equity_peak=cur_peak,
                device=device,
            )

        # ---------- 2. 风险覆盖层 ----------
        params_before_overlay = params

        params, overlay_info = _apply_overlay_if_needed(
            params=params,
            R=R,
            t=t,
            equity_curve=equity_curve,
            vol_thr=vol_thr,
            cfg=cfg,
        )

        # ---------- 3. 构造风险矩阵 ----------
        A_t = fuse_graph_scales(A_scales[t], alpha=alpha)

        if psd_project_graph:
            A_t = project_psd(A_t)

        Sigma_t = rolling_covariance(
            R,
            int(t),
            cov_window,
            cov_shrinkage,
        )

        # ---------- 4. 当前 alpha 信号 ----------
        mu_t = np.asarray(mu_pred[t], dtype=np.float64)
        mu_t = np.nan_to_num(mu_t, nan=0.0, posinf=0.0, neginf=0.0)

        # ---------- 5. 求解组合权重 ----------
        w_full = solve_portfolio(
            mu=mu_t,
            Sigma=Sigma_t,
            A=A_t,
            w_prev_asset=w_prev_asset,
            params=params,
            w_max=w_max,
            allow_cash=use_cash,
        )

        w_full = np.asarray(w_full, dtype=np.float64)

        if use_cash:
            asset_w = w_full[:N]
            cash_w = float(w_full[-1])
        else:
            asset_w = w_full[:N]
            cash_w = 0.0

        # ---------- 6. 计算收益 ----------
        # 注意：这里假设 R[t] 是 t 日特征对应的 next-day return。
        # 也就是说，使用 mu_pred[t] 形成组合后，吃到 R[t] 的未来一期收益。
        r_t = np.asarray(R[t], dtype=np.float64)

        port_ret = float(asset_w @ r_t)
        bench_ret = float(np.nanmean(r_t))
        active_ret = port_ret - bench_ret

        turnover = float(np.sum(np.abs(asset_w - w_prev_asset)))
        cost = float(cost_kappa * turnover)

        net_ret = port_ret - cost

        new_equity = float(cur_equity * (1.0 + net_ret))
        new_bench_equity = float(bench_equity_curve[-1] * (1.0 + bench_ret))

        graph_risk = float(asset_w @ A_t @ asset_w)

        # ---------- 7. 记录 ----------
        out_dates.append(date_str)
        port_rets.append(port_ret)
        bench_rets.append(bench_ret)
        active_rets.append(active_ret)
        net_rets.append(net_ret)
        turnovers.append(turnover)
        costs.append(cost)
        graph_risks.append(graph_risk)
        cashes.append(cash_w)
        equity_curve.append(new_equity)
        bench_equity_curve.append(new_bench_equity)

        weights_rows.append(w_full.tolist())

        param_row = {
            "date": date_str,
            "lambda_risk_raw": float(params_before_overlay.lambda_risk),
            "rho_graph_raw": float(params_before_overlay.rho_graph),
            "eta_turnover_raw": float(params_before_overlay.eta_turnover),
            "exposure_raw": float(params_before_overlay.exposure),
            "lambda_risk": float(params.lambda_risk),
            "rho_graph": float(params.rho_graph),
            "eta_turnover": float(params.eta_turnover),
            "exposure": float(params.exposure),
            "cash": cash_w,
            "turnover": turnover,
            "cost": cost,
            "graph_risk": graph_risk,
            "equity": new_equity,
            **meta_info,
            **overlay_info,
        }

        params_rows.append(param_row)

        # ---------- 8. 更新上一期权重 ----------
        w_prev_asset = asset_w.astype(np.float64)
        w_prev_full = w_full.astype(np.float64)

    # ========================================================
    # 结果整理
    # ========================================================

    report = summarize_backtest(
        equity_curve,
        port_rets,
        bench_rets,
        turnovers,
        cashes,
    )

    # 额外补充一些更直观的指标
    report["final_benchmark_equity"] = float(bench_equity_curve[-1])
    report["active_total_return"] = float(equity_curve[-1] - bench_equity_curve[-1])
    report["avg_graph_risk"] = float(np.mean(graph_risks)) if graph_risks else 0.0
    report["avg_cost"] = float(np.mean(costs)) if costs else 0.0
    report["mode"] = mode

    dd = drawdown_series(equity_curve)[1:]
    bench_dd = drawdown_series(bench_equity_curve)[1:]

    curve_df = pd.DataFrame(
        {
            "date": out_dates,
            "equity": equity_curve[1:],
            "benchmark_equity": bench_equity_curve[1:],
            "port_ret": port_rets,
            "bench_ret": bench_rets,
            "active_ret": active_rets,
            "net_ret": net_rets,
            "turnover": turnovers,
            "cost": costs,
            "graph_risk": graph_risks,
            "cash": cashes,
            "drawdown": dd,
            "benchmark_drawdown": bench_dd,
        }
    )

    weight_cols = [str(x) for x in instruments]
    if use_cash:
        weight_cols += ["cash"]

    weights_df = pd.DataFrame(weights_rows, columns=weight_cols)
    weights_df.insert(0, "date", out_dates)

    params_df = pd.DataFrame(params_rows)

    return {
        "report": report,
        "curve": curve_df,
        "weights": weights_df,
        "params": params_df,
    }


# ============================================================
# 7. 保存结果与可视化
# ============================================================

def save_outputs(res: Dict[str, Any], cfg: Dict[str, Any], mode: str) -> None:
    project_root = cfg["project"]["root_dir"]

    out_base = cfg.get("backtest", {}).get("out_dir", "outputs/backtest")
    out_dir = ensure_dir(
        os.path.join(
            resolve_path(out_base, project_root),
            mode,
        )
    )

    report_path = os.path.join(out_dir, "report.json")
    curve_path = os.path.join(out_dir, "curves.csv")
    weights_path = os.path.join(out_dir, "weights.csv")
    params_path = os.path.join(out_dir, "controller_params.csv")

    _safe_json_dump(res["report"], report_path)

    res["curve"].to_csv(curve_path, index=False, encoding="utf-8-sig")
    res["weights"].to_csv(weights_path, index=False, encoding="utf-8-sig")
    res["params"].to_csv(params_path, index=False, encoding="utf-8-sig")

    curve = res["curve"].copy()
    x = pd.to_datetime(curve["date"])

    # 1. 净值曲线
    plt.figure(figsize=(10, 5))
    plt.plot(x, curve["equity"], label="Strategy")
    if "benchmark_equity" in curve.columns:
        plt.plot(x, curve["benchmark_equity"], label="Equal-weight Benchmark", linestyle="--")
    plt.title(f"Equity Curve ({mode})")
    plt.xlabel("Date")
    plt.ylabel("Equity")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "equity_curve.png"), dpi=160)
    plt.close()

    # 2. 回撤曲线
    plt.figure(figsize=(10, 5))
    plt.plot(x, curve["drawdown"], label="Strategy")
    if "benchmark_drawdown" in curve.columns:
        plt.plot(x, curve["benchmark_drawdown"], label="Benchmark", linestyle="--")
    plt.title(f"Drawdown ({mode})")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "drawdown_curve.png"), dpi=160)
    plt.close()

    # 3. Cash 权重
    plt.figure(figsize=(10, 5))
    plt.plot(x, curve["cash"])
    plt.title(f"Cash Weight ({mode})")
    plt.xlabel("Date")
    plt.ylabel("Cash")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "cash_weight.png"), dpi=160)
    plt.close()

    # 4. 图风险
    plt.figure(figsize=(10, 5))
    plt.plot(x, curve["graph_risk"])
    plt.title(f"Graph Risk w'A'w ({mode})")
    plt.xlabel("Date")
    plt.ylabel("Graph Risk")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "graph_risk.png"), dpi=160)
    plt.close()

    # 5. 参数曲线
    params = res["params"].copy()

    plot_cols = [
        c for c in [
            "lambda_risk",
            "rho_graph",
            "eta_turnover",
            "exposure",
        ]
        if c in params.columns
    ]

    if len(plot_cols) > 0:
        plt.figure(figsize=(10, 5))
        for c in plot_cols:
            plt.plot(x, params[c], label=c)
        plt.title(f"Controller Parameters ({mode})")
        plt.xlabel("Date")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "controller_params.png"), dpi=160)
        plt.close()

    print(f"[OK] saved backtest outputs to {out_dir}")
    print(json.dumps(res["report"], ensure_ascii=False, indent=2))


# ============================================================
# 8. 命令行入口
# ============================================================

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config.yaml")
    ap.add_argument("--mode", type=str, default=None, choices=["fixed", "ppo"])
    args = ap.parse_args()

    cfg = load_config(args.config)

    mode = args.mode or cfg.get("backtest", {}).get("mode", "fixed")

    result = run_backtest(cfg, mode=mode)

    save_outputs(result, cfg, mode=mode)


if __name__ == "__main__":
    main()