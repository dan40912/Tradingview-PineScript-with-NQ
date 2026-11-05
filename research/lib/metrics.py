"""Performance metrics. Always report per-year and per-split, and the number of
variants you tried (for multiple-testing awareness)."""
from __future__ import annotations
import numpy as np
import pandas as pd


def max_drawdown(pnl_seq) -> float:
    eq = np.cumsum(np.asarray(pnl_seq, float))
    if len(eq) == 0:
        return 0.0
    peak = np.maximum.accumulate(np.concatenate([[0.0], eq]))[1:]
    return float(np.min(eq - peak))


def summary(trades: pd.DataFrame, n_days: int | None = None, pnl_col="pnl") -> dict:
    """n_days = number of tradable days in the period (for daily Sharpe incl. flat days)."""
    if trades is None or len(trades) == 0:
        return {"n": 0}
    p = trades[pnl_col].values.astype(float)
    daily = trades.groupby("date")[pnl_col].sum()
    nd = n_days or len(daily)
    dvec = np.concatenate([daily.values, np.zeros(max(nd - len(daily), 0))])
    sd = dvec.std(ddof=1) if len(dvec) > 1 else np.nan
    gains, losses = p[p > 0].sum(), -p[p < 0].sum()
    t = p.mean() / (p.std(ddof=1) / np.sqrt(len(p))) if len(p) > 1 and p.std() > 0 else np.nan
    mdd = max_drawdown(daily.sort_index().values)
    return {
        "n": len(p), "win%": round(100 * (p > 0).mean(), 1), "avg": round(p.mean(), 1), "total": round(p.sum(), 0),
        "pf": round(gains / losses, 2) if losses > 0 else np.inf, "maxdd": round(mdd, 0),
        "ret/dd": round(p.sum() / -mdd, 2) if mdd < 0 else np.inf,
        "sharpe_ann": round(dvec.mean() / sd * np.sqrt(252), 2) if sd and sd > 0 else np.nan,
        "t": round(t, 2) if t == t else np.nan,
    }


def by_period(trades: pd.DataFrame, col="year", pnl_col="pnl") -> pd.DataFrame:
    rows = {}
    for k, grp in trades.groupby(col):
        rows[k] = summary(grp, pnl_col=pnl_col)
    rows["ALL"] = summary(trades, pnl_col=pnl_col)
    return pd.DataFrame(rows).T


def bootstrap_ci(p, n=5000, alpha=0.05, seed=0):
    rng = np.random.default_rng(seed)
    p = np.asarray(p, float)
    if len(p) < 2:
        return (np.nan, np.nan)
    m = rng.choice(p, size=(n, len(p)), replace=True).mean(1)
    return tuple(np.quantile(m, [alpha / 2, 1 - alpha / 2]).round(1))
