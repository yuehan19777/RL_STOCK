from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize


@dataclass
class PortfolioParams:
    lambda_risk: float = 5.0
    rho_graph: float = 1.0
    eta_turnover: float = 0.10
    exposure: float = 0.80


def _fallback_topk(mu: np.ndarray, exposure: float, w_max: float, allow_cash: bool = True) -> np.ndarray:
    mu = np.asarray(mu, dtype=np.float64)
    N = len(mu)

    w = np.zeros(N, dtype=np.float64)

    order = np.argsort(-mu)
    remain = float(exposure)

    for i in order:
        if remain <= 1e-12:
            break
        wi = min(float(w_max), remain)
        w[i] = wi
        remain -= wi

    if allow_cash:
        cash = max(0.0, 1.0 - float(w.sum()))
        return np.concatenate([w, np.array([cash], dtype=np.float64)])

    s = w.sum()
    if s <= 1e-12:
        w[:] = 1.0 / N
    else:
        w = w / s
    return w


def solve_portfolio(
    mu: np.ndarray,
    Sigma: np.ndarray,
    A: np.ndarray,
    w_prev_asset: np.ndarray,
    params: PortfolioParams,
    w_max: float = 0.25,
    allow_cash: bool = True,
) -> np.ndarray:
    """
    解组合优化：

    max_w:
        mu'w - lambda*w'Sigma*w - rho*w'A*w - eta*|w-w_prev|_1

    s.t.
        0 <= w_i <= w_max
        sum(w_i) <= exposure
        cash = 1 - sum(w_i)
    """
    mu = np.asarray(mu, dtype=np.float64)
    Sigma = np.asarray(Sigma, dtype=np.float64)
    A = np.asarray(A, dtype=np.float64)
    w_prev_asset = np.asarray(w_prev_asset, dtype=np.float64)

    N = len(mu)

    if w_prev_asset.shape[0] != N:
        w_prev_asset = np.zeros(N, dtype=np.float64)

    Sigma = np.nan_to_num(Sigma, nan=0.0, posinf=0.0, neginf=0.0)
    A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
    mu = np.nan_to_num(mu, nan=0.0, posinf=0.0, neginf=0.0)

    exposure = float(np.clip(params.exposure, 0.0, 1.0))

    def obj(w):
        ret_term = -float(np.dot(mu, w))
        risk_term = float(params.lambda_risk) * float(w @ Sigma @ w)
        graph_term = float(params.rho_graph) * float(w @ A @ w)
        turnover_term = float(params.eta_turnover) * float(np.sum(np.abs(w - w_prev_asset)))
        return ret_term + risk_term + graph_term + turnover_term

    bounds = [(0.0, float(w_max)) for _ in range(N)]

    cons = [
        {
            "type": "ineq",
            "fun": lambda w: exposure - float(np.sum(w)),
        }
    ]

    # 初始点
    x0 = np.minimum(np.maximum(w_prev_asset, 0.0), float(w_max))
    sx = float(x0.sum())

    if sx <= 1e-12:
        x0[:] = min(exposure / max(N, 1), float(w_max))
    elif sx > exposure:
        x0 = x0 / sx * exposure

    try:
        res = minimize(
            obj,
            x0=x0,
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
            options={"maxiter": 200, "ftol": 1e-8, "disp": False},
        )

        if not res.success or res.x is None:
            return _fallback_topk(mu, exposure, w_max, allow_cash=allow_cash)

        w_asset = np.asarray(res.x, dtype=np.float64)
        w_asset = np.clip(w_asset, 0.0, float(w_max))

        if w_asset.sum() > exposure + 1e-8:
            w_asset = w_asset / w_asset.sum() * exposure

    except Exception:
        return _fallback_topk(mu, exposure, w_max, allow_cash=allow_cash)

    if allow_cash:
        cash = max(0.0, 1.0 - float(w_asset.sum()))
        return np.concatenate([w_asset, np.array([cash], dtype=np.float64)])

    s = float(w_asset.sum())
    if s <= 1e-12:
        w_asset[:] = 1.0 / N
    else:
        w_asset = w_asset / s

    return w_asset

def apply_risk_overlay(params: PortfolioParams, market_state: dict, cfg: dict) -> PortfolioParams:
    overlay = cfg["portfolio"].get("risk_overlay", {})
    if not overlay.get("enabled", False):
        return params
    exposure = params.exposure
    if market_state.get("drawdown", 0.0) < float(overlay.get("drawdown_threshold", -0.10)):
        exposure = min(exposure, float(overlay.get("min_exposure", 0.4)))
    if market_state.get("high_vol", False):
        exposure = min(exposure, max(float(overlay.get("min_exposure", 0.4)), 0.6))
    return PortfolioParams(params.lambda_risk, params.rho_graph, params.eta_turnover, exposure)
