from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple

import numpy as np

from portfolio_layer.optimizer import PortfolioParams, solve_portfolio
from risk_layer.covariance import rolling_covariance
from risk_layer.graph_features import fuse_graph_scales


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = np.clip(x, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-x))

def compute_controller_context(
    R: np.ndarray,
    mu_pred: np.ndarray,
    A_scales: np.ndarray,
    t: int,
    equity: float,
    equity_peak: float,
) -> dict:
    """
    计算 PPO 控制器需要的风险/趋势/信号上下文。

    目标：
    - risk_score 高：说明市场风险高，应该降低 exposure；
    - trend_score 高：说明涨势较好，可以提高 exposure；
    - signal_score 高：说明信号区分度强，可以更积极。
    """
    t = int(t)
    R = np.asarray(R, dtype=np.float64)
    mu_pred = np.asarray(mu_pred, dtype=np.float64)

    # ---------- 市场收益：只能使用 t 之前的信息 ----------
    s20 = max(0, t - 20)
    s60 = max(0, t - 60)

    hist20 = R[s20:t]
    hist60 = R[s60:t]

    if hist20.shape[0] > 0:
        ew20 = np.nanmean(hist20, axis=1)
    else:
        ew20 = np.array([], dtype=np.float64)

    if hist60.shape[0] > 0:
        ew60 = np.nanmean(hist60, axis=1)
    else:
        ew60 = np.array([], dtype=np.float64)

    ret20 = float(np.nansum(ew20)) if ew20.size > 0 else 0.0
    ret60 = float(np.nansum(ew60)) if ew60.size > 0 else 0.0
    vol20 = float(np.nanstd(ew20)) if ew20.size > 1 else 0.0

    # t 时刻决策前，只能看到 t-1 的截面收益
    if t > 0:
        last_ret = R[t - 1]
        up_ratio = float(np.nanmean(last_ret > 0.0))
    else:
        up_ratio = 0.5

    # ---------- 当前回撤 ----------
    drawdown = float(equity / (equity_peak + 1e-12) - 1.0)

    # ---------- 信号强度 ----------
    mu_t = mu_pred[t]
    mu_t = mu_t[np.isfinite(mu_t)]

    if len(mu_t) >= 4:
        q80 = np.quantile(mu_t, 0.8)
        q20 = np.quantile(mu_t, 0.2)
        top_mean = float(np.mean(mu_t[mu_t >= q80]))
        bot_mean = float(np.mean(mu_t[mu_t <= q20]))
        signal_spread = top_mean - bot_mean
        signal_std = float(np.std(mu_t)) + 1e-8
        signal_raw = signal_spread / signal_std
    else:
        signal_spread = 0.0
        signal_raw = 0.0

    # ---------- 图强度 ----------
    A_t = np.asarray(A_scales[t], dtype=np.float64)
    A_mean = np.nanmean(A_t, axis=0)
    np.fill_diagonal(A_mean, 0.0)
    graph_strength = float(np.nanmean(A_mean))

    # ---------- 映射到 0~1 分数 ----------
    def sigmoid_scalar(x: float) -> float:
        x = float(np.clip(x, -30.0, 30.0))
        return float(1.0 / (1.0 + np.exp(-x)))

    # 趋势分数：
    # 20日收益、60日收益、上涨股票比例越高，趋势越好；
    # 波动越高，趋势分数略微下降。
    trend_raw = (
        12.0 * ret20
        + 6.0 * ret60
        + 2.0 * (up_ratio - 0.5)
        - 8.0 * vol20
    )
    trend_score = sigmoid_scalar(trend_raw)

    # 风险分数：
    # 波动越高、回撤越大、图强度越高，风险越高；
    # 近期收益越差，风险越高。
    risk_raw = (
        20.0 * vol20
        + 8.0 * max(0.0, -drawdown - 0.05)
        + 2.0 * graph_strength
        - 8.0 * ret20
    )
    risk_score = sigmoid_scalar(risk_raw)

    # 信号分数：
    # top-bottom spread 越明显，说明横截面机会越强。
    signal_score = sigmoid_scalar(2.0 * signal_raw)

    return {
        "ret20": ret20,
        "ret60": ret60,
        "vol20": vol20,
        "up_ratio": up_ratio,
        "drawdown": drawdown,
        "graph_strength": graph_strength,
        "signal_spread": signal_spread,
        "trend_score": trend_score,
        "risk_score": risk_score,
        "signal_score": signal_score,
    }


@dataclass
class MetaAction:
    lambda_risk: float
    rho_graph: float
    eta_turnover: float
    exposure: float
    alpha: Optional[np.ndarray] = None


def _map_to_range(z: float, lo: float, hi: float) -> float:
    s = float(sigmoid(np.array([z]))[0])
    return float(lo + (hi - lo) * s)


def transform_raw_action(
    raw_action: np.ndarray,
    cfg: Dict[str, Any],
    num_scales: int | None = None,
    levels: int | None = None,
    context: dict | None = None,
) -> MetaAction:
    """
    把 PPO 输出的 raw action 转换成组合优化参数。

    关键修改：
    1. 不再直接输出绝对参数；
    2. 围绕 fixed 参数输出倍率；
    3. 加入 rule prior：
       - risk_score 高时降低 exposure；
       - trend_score 和 signal_score 高时提高 exposure；
       - risk_score 高时提高 lambda/rho。
    """
    raw_action = np.asarray(raw_action, dtype=np.float64).reshape(-1)

    ppo_cfg = cfg.get("ppo", {})
    fixed = cfg["portfolio"]["fixed_params"]

    base_lambda = float(fixed.get("lambda_risk", 5.0))
    base_rho = float(fixed.get("rho_graph", 1.0))
    base_eta = float(fixed.get("eta_turnover", 0.1))
    base_exp = float(fixed.get("exposure", 0.8))

    # ---------- 1. 规则先验 ----------
    rule_cfg = ppo_cfg.get("rule_prior", {})
    rule_enabled = bool(rule_cfg.get("enabled", True))

    if context is None:
        context = {}

    trend_score = float(context.get("trend_score", 0.5))
    risk_score = float(context.get("risk_score", 0.5))
    signal_score = float(context.get("signal_score", 0.5))

    if rule_enabled:
        risk_cut = float(rule_cfg.get("risk_exposure_cut", 0.55))
        trend_boost = float(rule_cfg.get("trend_exposure_boost", 0.35))

        min_fac = float(rule_cfg.get("min_exposure_factor", 0.35))
        max_fac = float(rule_cfg.get("max_exposure_factor", 1.20))

        # 风险越高，exposure factor 越低；
        # 趋势越强、信号越强，exposure factor 越高。
        exposure_factor = (
            1.0
            - risk_cut * risk_score
            + trend_boost * trend_score * signal_score
        )
        exposure_factor = float(np.clip(exposure_factor, min_fac, max_fac))

        lambda_base_dyn = base_lambda * (
            1.0 + float(rule_cfg.get("lambda_risk_boost", 0.80)) * risk_score
        )

        rho_base_dyn = base_rho * (
            1.0 + float(rule_cfg.get("rho_graph_boost", 1.20)) * risk_score
        )

        eta_base_dyn = base_eta * (
            1.0 + float(rule_cfg.get("eta_weak_signal_boost", 0.80)) * (1.0 - signal_score)
        )

        exposure_base_dyn = base_exp * exposure_factor

    else:
        lambda_base_dyn = base_lambda
        rho_base_dyn = base_rho
        eta_base_dyn = base_eta
        exposure_base_dyn = base_exp

    # ---------- 2. PPO 输出倍率 ----------
    mult_cfg = ppo_cfg.get("action_multipliers", {})

    mult_bounds = {
        "lambda_risk": mult_cfg.get("lambda_risk", [0.7, 1.5]),
        "rho_graph": mult_cfg.get("rho_graph", [0.5, 2.0]),
        "eta_turnover": mult_cfg.get("eta_turnover", [0.7, 1.5]),
        "exposure": mult_cfg.get("exposure", [0.65, 1.25]),
    }

    control_params = list(ppo_cfg.get("control_params", ["exposure"]))

    def sigmoid_scalar(x: float) -> float:
        x = float(np.clip(x, -30.0, 30.0))
        return float(1.0 / (1.0 + np.exp(-x)))

    def map_multiplier(z: float, lo: float, hi: float) -> float:
        return float(lo + (hi - lo) * sigmoid_scalar(z))

    pos = 0

    lambda_risk = float(lambda_base_dyn)
    rho_graph = float(rho_base_dyn)
    eta_turnover = float(eta_base_dyn)
    exposure = float(exposure_base_dyn)

    if "lambda_risk" in control_params:
        lo, hi = mult_bounds["lambda_risk"]
        lambda_risk = lambda_base_dyn * map_multiplier(raw_action[pos], lo, hi)
        pos += 1

    if "rho_graph" in control_params:
        lo, hi = mult_bounds["rho_graph"]
        rho_graph = rho_base_dyn * map_multiplier(raw_action[pos], lo, hi)
        pos += 1

    if "eta_turnover" in control_params:
        lo, hi = mult_bounds["eta_turnover"]
        eta_turnover = eta_base_dyn * map_multiplier(raw_action[pos], lo, hi)
        pos += 1

    if "exposure" in control_params:
        lo, hi = mult_bounds["exposure"]
        exposure = exposure_base_dyn * map_multiplier(raw_action[pos], lo, hi)
        pos += 1

    exposure = float(np.clip(exposure, 0.05, 1.0))

    # ---------- 3. 可选：小波尺度权重 ----------
    alpha = None
    use_scale_alpha = bool(ppo_cfg.get("use_scale_alpha", False))

    S = num_scales if num_scales is not None else levels

    if use_scale_alpha and S is not None:
        S = int(S)
        z = raw_action[pos:pos + S]
        z = z - np.max(z)
        ez = np.exp(z)
        alpha = (ez / (ez.sum() + 1e-12)).astype(np.float32)

    return MetaAction(
        lambda_risk=float(lambda_risk),
        rho_graph=float(rho_graph),
        eta_turnover=float(eta_turnover),
        exposure=float(exposure),
        alpha=alpha,
    )

def _safe_stats(x: np.ndarray) -> Tuple[float, float, float, float]:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]

    if x.size == 0:
        return 0.0, 0.0, 0.0, 0.0

    return float(x.mean()), float(x.std()), float(x.min()), float(x.max())


def build_state(
    R: np.ndarray,
    mu_pred: np.ndarray,
    A_scales: np.ndarray,
    t: int,
    w_prev: np.ndarray,
    equity: float,
    equity_peak: float,
) -> np.ndarray:
    """
    构造 PPO 的状态向量。

    状态包括：
    1. 市场收益与波动
    2. 信号分布
    3. 图强度
    4. 当前组合状态
    """
    T, N = R.shape
    t = int(t)

    # ---------- 市场状态 ----------
    # 决策时点 t 只能使用 t 之前已经发生的收益 R[:t]
    market_feats = []

    for w in [5, 20, 60]:
        s = max(0, t - w)
        hist = R[s:t]   # 不包含 R[t]

        if hist.shape[0] > 0:
            ew_ret = np.nanmean(hist, axis=1)
            market_feats.append(float(np.nanmean(ew_ret)))
            market_feats.append(float(np.nanstd(ew_ret)))
        else:
            market_feats.append(0.0)
            market_feats.append(0.0)

    # 上一期横截面收益统计，不能用 R[t]
    if t > 0:
        last_ret = R[t - 1]
        market_feats.append(float(np.nanmean(last_ret)))
        market_feats.append(float(np.nanstd(last_ret)))
        market_feats.append(float(np.nanmean(last_ret > 0.0)))
    else:
        market_feats.append(0.0)
        market_feats.append(0.0)
        market_feats.append(0.0)

    # ---------- 信号状态 ----------
    mu_t = np.asarray(mu_pred[t], dtype=np.float64)
    mu_mean, mu_std, mu_min, mu_max = _safe_stats(mu_t)

    q80 = np.nanquantile(mu_t, 0.8)
    q20 = np.nanquantile(mu_t, 0.2)
    top_mean = float(np.nanmean(mu_t[mu_t >= q80])) if np.any(mu_t >= q80) else 0.0
    bot_mean = float(np.nanmean(mu_t[mu_t <= q20])) if np.any(mu_t <= q20) else 0.0
    spread = top_mean - bot_mean

    signal_feats = [
        mu_mean,
        mu_std,
        mu_min,
        mu_max,
        top_mean,
        bot_mean,
        spread,
    ]

    # ---------- 图状态 ----------
    A_t = np.asarray(A_scales[t], dtype=np.float64)  # (S,N,N)
    S = A_t.shape[0]

    graph_feats = []

    for s in range(S):
        A_s = A_t[s].copy()
        np.fill_diagonal(A_s, 0.0)
        graph_feats.append(float(np.nanmean(A_s)))
        graph_feats.append(float(np.nanstd(A_s)))

    A_fused = fuse_graph_scales(A_scales[t])
    graph_strength = float(np.nanmean(A_fused))
    graph_feats.append(graph_strength)

    if t > 0:
        A_prev = fuse_graph_scales(A_scales[t - 1])
        diff = np.linalg.norm(A_fused - A_prev, ord="fro")
        base = np.linalg.norm(A_prev, ord="fro") + 1e-8
        graph_change = float(diff / base)
    else:
        graph_change = 0.0

    graph_feats.append(graph_change)

    # ---------- 组合状态 ----------
    w_prev = np.asarray(w_prev, dtype=np.float64)
    cash = float(max(0.0, 1.0 - np.sum(w_prev)))

    concentration = float(np.sum(w_prev ** 2))

    graph_risk = float(w_prev @ A_fused @ w_prev)

    drawdown = float(equity / (equity_peak + 1e-12) - 1.0)

    portfolio_feats = [
        cash,
        concentration,
        graph_risk,
        drawdown,
    ]

    state = np.asarray(
        market_feats + signal_feats + graph_feats + portfolio_feats,
        dtype=np.float32,
    )

    state = np.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0)

    return state.astype(np.float32)


class MetaPortfolioEnv:
    """
    PPO 用的元环境。

    PPO 不直接输出股票权重，而是输出组合优化参数：
        lambda_risk, rho_graph, eta_turnover, exposure

    然后 solve_portfolio 生成最终权重。
    """

    def __init__(
        self,
        R: np.ndarray,
        mu_pred: np.ndarray,
        A_scales: np.ndarray,
        dates: np.ndarray,
        idx: np.ndarray,
        cfg: Dict[str, Any],
    ):
        self.R_all = np.asarray(R, dtype=np.float32)
        self.mu_all = np.asarray(mu_pred, dtype=np.float32)
        self.A_all = np.asarray(A_scales, dtype=np.float32)
        self.dates_all = dates
        self.idx = np.asarray(idx, dtype=np.int64)
        self.cfg = cfg

        self.port_cfg = cfg.get("portfolio", {})
        self.ppo_cfg = cfg.get("ppo", {})

        self.window_cov = int(self.port_cfg.get("window_cov", 60))
        self.shrinkage = float(self.port_cfg.get("shrinkage", 0.3))
        self.w_max = float(self.port_cfg.get("w_max", 0.25))

        self.reset()

    @property
    def num_scales(self) -> int:
        return int(self.A_all.shape[1])

    @property
    def action_dim(self) -> int:
        control_params = list(self.ppo_cfg.get("control_params", ["exposure"]))
        dim = len(control_params)

        if bool(self.ppo_cfg.get("use_scale_alpha", False)):
            dim += self.num_scales

        return int(dim)
    
    @property
    def state_dim(self) -> int:
        s = self.reset()
        return int(s.shape[0])

    def reset(self) -> np.ndarray:
        self.pos = 0
        self.equity = 1.0
        self.equity_peak = 1.0

        N = self.R_all.shape[1]
        self.w_prev = np.zeros(N, dtype=np.float32)

        t = int(self.idx[self.pos])
        return build_state(
            R=self.R_all,
            mu_pred=self.mu_all,
            A_scales=self.A_all,
            t=t,
            w_prev=self.w_prev,
            equity=self.equity,
            equity_peak=self.equity_peak,
        )

    def step(self, raw_action: np.ndarray):
        t = int(self.idx[self.pos])

        context = compute_controller_context(
            R=self.R_all,
            mu_pred=self.mu_all,
            A_scales=self.A_all,
            t=t,
            equity=self.equity,
            equity_peak=self.equity_peak,
        )

        meta = transform_raw_action(
            raw_action=raw_action,
            cfg=self.cfg,
            num_scales=self.num_scales,
            context=context,
        )

        mu_t = self.mu_all[t].astype(np.float64)
        Sigma_t = rolling_covariance(
            R=self.R_all,
            t=t,
            window=self.window_cov,
            shrinkage=self.shrinkage,
        )

        A_t = fuse_graph_scales(self.A_all[t], alpha=meta.alpha)

        params = PortfolioParams(
            lambda_risk=meta.lambda_risk,
            rho_graph=meta.rho_graph,
            eta_turnover=meta.eta_turnover,
            exposure=meta.exposure,
        )

        w_full = solve_portfolio(
            mu=mu_t,
            Sigma=Sigma_t,
            A=A_t,
            w_prev_asset=self.w_prev,
            params=params,
            w_max=self.w_max,
            allow_cash=True,
        )

        w_asset = np.asarray(w_full[:-1], dtype=np.float64)
        w_cash = float(w_full[-1])

        # 收益使用下一期 return，避免当前动作吃到当前已知收益
        # 当前动作在 t 时点形成组合，吃到 R[t]
        # 因为 state[t] 已经改成只使用 R[:t]，所以这里使用 R[t] 不再泄露
        r_t = self.R_all[t].astype(np.float64)

        port_ret = float(np.dot(w_asset, r_t))
        bench_ret = float(np.nanmean(r_t))

        turnover = float(np.sum(np.abs(w_asset - self.w_prev)))
        cost_kappa = float(self.port_cfg.get("cost_kappa", 0.001))
        cost = cost_kappa * turnover

        net_ret = port_ret - cost

        self.equity *= float(1.0 + net_ret)
        self.equity_peak = max(self.equity_peak, self.equity)
        drawdown = float(self.equity / (self.equity_peak + 1e-12) - 1.0)

        graph_risk = float(w_asset @ A_t @ w_asset)

        rew_cfg = self.ppo_cfg.get("reward", {})
        active_weight = float(rew_cfg.get("active_weight", 1.0))
        cost_weight = float(rew_cfg.get("cost_weight", 1.0))
        graph_weight = float(rew_cfg.get("graph_weight", 0.01))
        drawdown_weight = float(rew_cfg.get("drawdown_weight", 0.10))
        dd_tol = float(rew_cfg.get("drawdown_tolerance", 0.05))

        risk_exposure_penalty = float(rew_cfg.get("risk_exposure_penalty", 0.05))
        trend_exposure_bonus = float(rew_cfg.get("trend_exposure_bonus", 0.03))

        active_ret = port_ret - bench_ret

        # reward = (
        #     active_weight * active_ret
        #     - cost_weight * cost
        #     - graph_weight * graph_risk
        #     - drawdown_weight * max(0.0, -drawdown - dd_tol)
        #     - risk_exposure_penalty * context["risk_score"] * meta.exposure
        #     + trend_exposure_bonus * context["trend_score"] * context["signal_score"] * meta.exposure
        # )
        reward = (
            active_weight * active_ret
            - cost_weight * cost
            - drawdown_weight * max(0.0, -drawdown - dd_tol)
        )

        # 防止极端值让 critic 崩掉
        reward_clip = float(rew_cfg.get("reward_clip", 0.05))
        reward = float(np.clip(reward, -reward_clip, reward_clip))

        self.w_prev = w_asset.astype(np.float32)

        self.pos += 1
        done = self.pos >= len(self.idx) - 1

        if not done:
            t2 = int(self.idx[self.pos])
            next_state = build_state(
                R=self.R_all,
                mu_pred=self.mu_all,
                A_scales=self.A_all,
                t=t2,
                w_prev=self.w_prev,
                equity=self.equity,
                equity_peak=self.equity_peak,
            )
        else:
            next_state = build_state(
                R=self.R_all,
                mu_pred=self.mu_all,
                A_scales=self.A_all,
                t=t,
                w_prev=self.w_prev,
                equity=self.equity,
                equity_peak=self.equity_peak,
            )

        info = {
            "date": str(self.dates_all[t])[:10],
            "lambda_risk": meta.lambda_risk,
            "rho_graph": meta.rho_graph,
            "eta_turnover": meta.eta_turnover,
            "exposure": meta.exposure,
            "cash": w_cash,
            "port_ret": port_ret,
            "bench_ret": bench_ret,
            "net_ret": net_ret,
            "turnover": turnover,
            "cost": cost,
            "graph_risk": graph_risk,
            "drawdown": drawdown,
            "equity": self.equity,

            "trend_score": float(context["trend_score"]),
            "risk_score": float(context["risk_score"]),
            "signal_score": float(context["signal_score"]),
            "ret20": float(context["ret20"]),
            "ret60": float(context["ret60"]),
            "vol20": float(context["vol20"]),
            "up_ratio": float(context["up_ratio"]),
        }

        if meta.alpha is not None:
            for i, a in enumerate(meta.alpha):
                info[f"alpha_{i + 1}"] = float(a)

        return next_state, float(reward), done, info