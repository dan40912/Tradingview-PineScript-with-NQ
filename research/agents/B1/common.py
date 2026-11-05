"""Shared helpers for B1 scripts (loading, metrics with flat-day Sharpe, permutation test)."""
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.join(ROOT, "research")); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
from lib import data, features, sim, metrics, strategies
import indicators as ind
RTH_OFF = data.RTH_OFF

_cache = {}
def load(symbol="NQ"):
    if symbol not in _cache:
        g = data.load_grid(symbol); lv = features.daily_levels(g)
        _cache[symbol] = (g, lv)
    return _cache[symbol]

def ndays(g, split):
    return int((g.tradable & (g.split == split)).sum())

def summ(g, tr, split):
    t = tr[tr.split == split] if len(tr) else tr
    s = metrics.summary(t, n_days=ndays(g, split)) if len(t) else {"n": 0}
    return s

def table(g, tr, splits=("discovery", "validation")):
    return pd.DataFrame({sp: summ(g, tr, sp) for sp in splits}).T

def perm_p(pnl, keep_mask, nperm=5000, seed=1):
    """P(random keep of same count has mean >= observed kept mean)."""
    pnl = np.asarray(pnl, float); k = int(keep_mask.sum()); n = len(pnl)
    if k == 0 or k == n:
        return np.nan
    obs = pnl[keep_mask].mean()
    rng = np.random.default_rng(seed)
    idx = np.argsort(rng.random((nperm, n)), axis=1)[:, :k]
    sims = pnl[idx].mean(1)
    return float((sims >= obs - 1e-9).mean())
