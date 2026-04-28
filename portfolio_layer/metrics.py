from __future__ import annotations
import numpy as np

def annualized_sharpe(rets: np.ndarray, periods: int = 252) -> float:
    rets = np.asarray(rets, dtype=np.float64)
    sd = np.std(rets)
    return 0.0 if sd < 1e-12 else float(np.sqrt(periods) * np.mean(rets) / sd)

def max_drawdown(equity: np.ndarray) -> float:
    e = np.asarray(equity, dtype=np.float64)
    peak = np.maximum.accumulate(e)
    return float(np.min(e / (peak + 1e-12) - 1.0))

def calmar(equity: np.ndarray, periods: int = 252) -> float:
    e = np.asarray(equity, dtype=np.float64)
    if len(e) < 2:
        return 0.0
    total = e[-1] / e[0] - 1
    years = (len(e)-1) / periods
    ann = (1 + total) ** (1 / max(years, 1e-8)) - 1
    return float(ann / (abs(max_drawdown(e)) + 1e-12))

def summarize_backtest(equity, port_ret, benchmark_ret=None, turnover=None, cash=None) -> dict:
    eq = np.asarray(equity, dtype=np.float64); r = np.asarray(port_ret, dtype=np.float64)
    out = {"final_equity": float(eq[-1]), "total_return": float(eq[-1] / eq[0] - 1.0), "ann_sharpe": annualized_sharpe(r), "mdd": max_drawdown(eq), "calmar": calmar(eq), "ann_vol": float(np.std(r) * np.sqrt(252)), "avg_daily_ret": float(np.mean(r)), "worst_day": float(np.min(r)) if len(r) else 0.0}
    if benchmark_ret is not None:
        br = np.asarray(benchmark_ret, dtype=np.float64); active = r[:len(br)] - br[:len(r)]
        out["active_ann_sharpe"] = annualized_sharpe(active); out["avg_active_daily_ret"] = float(np.mean(active))
    if turnover is not None: out["avg_turnover"] = float(np.mean(turnover))
    if cash is not None: out["avg_cash"] = float(np.mean(cash)); out["max_cash"] = float(np.max(cash))
    return out

def drawdown_series(equity):
    e = np.asarray(equity, dtype=np.float64); peak = np.maximum.accumulate(e)
    return e / (peak + 1e-12) - 1.0
