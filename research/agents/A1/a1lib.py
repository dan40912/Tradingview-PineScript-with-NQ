"""A1 helpers: day table, metrics with flat days, I2 overlap, variant counter."""
from __future__ import annotations
import os, sys, json
import numpy as np
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))  # research/
from lib import data, features, sim, metrics, strategies  # noqa
from lib.data import RTH_OFF

_cache = {}


def load(sym="NQ"):
    if sym not in _cache:
        g = data.load_grid(sym)  # NEVER include_holdout
        lv = features.daily_levels(g)
        _cache[sym] = (g, lv)
    return _cache[sym]


def daytable(g, lv):
    """Per-day pre-computed RTH features (only info up to the time each is used)."""
    n = len(g)
    rO, rH, rL, rC = g.rth(g.O), g.rth(g.H), g.rth(g.L), g.rth(g.C)
    d = lv[["i", "date", "split", "year", "has_prev", "tradable", "open_0930", "pdh", "pdl", "pdc", "pdo",
            "onh", "onl", "val", "vah", "poc", "i2_dir"]].copy()
    # prior RTH day ranges -> ATR-like (mean of prior 14 RTH day ranges, same contract not required)
    rng = np.nanmax(rH, 1) - np.nanmin(rL, 1)
    rng[~g.has_rth] = np.nan
    s = pd.Series(rng)
    d["rng"] = rng
    # ATR of previous 14 RTH days (shifted: uses only completed days)
    d["atr14"] = s.where(pd.Series(g.has_rth)).shift(1).rolling(20, min_periods=10).apply(lambda x: np.nanmean(x[-14:]), raw=True).values
    last_close = np.array([r[np.where(~np.isnan(r))[0][-1]] if np.isfinite(r).any() else np.nan for r in rC])
    d["rth_close"] = last_close
    return d


def summ(tr, g, split=None):
    """metrics.summary with n_days = tradable days in the split (Sharpe incl. flat days)."""
    if split is not None:
        tr = tr[tr.split == split] if len(tr) else tr
        nd = int((g.tradable & (g.split == split)).sum())
    else:
        nd = int(g.tradable.sum())
    if tr is None or len(tr) == 0:
        return {"n": 0}
    return metrics.summary(tr, n_days=nd)


def table(tr, g, splits=("discovery", "validation")):
    return pd.DataFrame({s: summ(tr, g, s) for s in splits}).T


def i2_days(g, lv):
    tr = sim.run_signals(g, strategies.i2_v1_signals(g, lv))
    return set(tr.date)


def overlap(tr, g, lv, split=None):
    days = i2_days(g, lv)
    t = tr if split is None else tr[tr.split == split]
    a = t[t.date.isin(days)]; b = t[~t.date.isin(days)]
    return {"on_I2_days": metrics.summary(a) if len(a) else {"n": 0},
            "off_I2_days": metrics.summary(b) if len(b) else {"n": 0}}


LOG = os.path.join(HERE, "variants_log.csv")


def log_variants(family, rows):
    """rows: list of dicts (variant name + discovery metrics). Appends to variants_log.csv."""
    df = pd.DataFrame(rows); df.insert(0, "family", family)
    hdr = not os.path.exists(LOG)
    df.to_csv(LOG, mode="a", header=hdr, index=False)


def halves(tr):
    """Discovery sub-period stability: per half-year avg/total, and excl. 2025-04."""
    t = tr.copy()
    t["h"] = t.date.str[:4] + np.where(t.date.str[5:7].astype(int) <= 6, "H1", "H2")
    out = t.groupby("h").pnl.agg(["count", "mean", "sum"]).round(0)
    ex = t[~t.date.str.startswith("2025-04")]
    out.loc["ex2025-04"] = [len(ex), round(ex.pnl.mean(), 0), round(ex.pnl.sum(), 0)]
    return out
