"""A2 helpers: loading, evaluation (daily Sharpe incl. flat tradable days), I2 overlap,
and a variant log so every tested variant is counted.  Uses only the shared engine for P&L.
Never loads the 2026 holdout."""
from __future__ import annotations
import os, sys, json
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RESEARCH = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, RESEARCH)
from lib import data, features, sim, metrics, strategies  # noqa: E402

LOG = os.path.join(HERE, "variant_log.csv")
_CACHE = {}


def load(sym="NQ"):
    if sym not in _CACHE:
        g = data.load_grid(sym)            # holdout hidden by default
        assert all(d < "2026-01-01" for d in g.dates)
        lv = features.daily_levels(g)
        _CACHE[sym] = (g, lv)
    return _CACHE[sym]


def n_tradable(g, split):
    return int(((g.split == split) & g.tradable).sum())


def summ(g, tr, split):
    if tr is None or len(tr) == 0:
        return {"n": 0}
    t = tr[tr.split == split]
    if len(t) == 0:
        return {"n": 0}
    return metrics.summary(t, n_days=n_tradable(g, split))


def i2_days(g, lv):
    tr = sim.run_signals(g, strategies.i2_v1_signals(g, lv))
    return set(tr.date), tr


def evaluate(g, tr, name="", family="", log=True, extra=None, quiet=False):
    """Return dict with disc/val summaries. Appends to variant log."""
    d = summ(g, tr, "discovery")
    v = summ(g, tr, "validation")
    row = {"family": family, "name": name, "sym": g.symbol}
    for k, s in (("d", d), ("v", v)):
        for f in ["n", "win%", "avg", "total", "maxdd", "ret/dd", "sharpe_ann"]:
            row[f"{k}_{f}"] = s.get(f, np.nan)
    if extra:
        row.update(extra)
    if log:
        pd.DataFrame([row]).to_csv(LOG, mode="a", header=not os.path.exists(LOG), index=False)
    if not quiet:
        print(f"{name:45s} D n={row['d_n']!s:>4} avg={row['d_avg']!s:>7} tot={row['d_total']!s:>8} dd={row['d_maxdd']!s:>7} "
              f"sh={row['d_sharpe_ann']!s:>5} | V n={row['v_n']!s:>3} avg={row['v_avg']!s:>7} sh={row['v_sharpe_ann']!s:>5}")
    return row


def disc_only(g, tr, name="", family="", log=True, quiet=False):
    """Discovery-only screening (validation not looked at)."""
    d = summ(g, tr, "discovery")
    row = {"family": family, "name": name, "sym": g.symbol}
    for f in ["n", "win%", "avg", "total", "maxdd", "ret/dd", "sharpe_ann"]:
        row[f"d_{f}"] = d.get(f, np.nan)
    if log:
        pd.DataFrame([row]).to_csv(LOG, mode="a", header=not os.path.exists(LOG), index=False)
    if not quiet:
        print(f"{name:50s} n={row['d_n']!s:>4} win={row['d_win%']!s:>5} avg={row['d_avg']!s:>7} tot={row['d_total']!s:>8} "
              f"dd={row['d_maxdd']!s:>7} r/dd={row['d_ret/dd']!s:>5} sh={row['d_sharpe_ann']!s:>5}")
    return row


def overlap(g, tr, i2set, split):
    t = tr[tr.split == split]
    a = t[t.date.isin(i2set)]
    b = t[~t.date.isin(i2set)]
    f = lambda x: {"n": len(x), "avg": round(x.pnl.mean(), 1) if len(x) else np.nan, "total": round(x.pnl.sum(), 0)}
    return {"on_I2_days": f(a), "off_I2_days": f(b)}


def combo(g, trs, split):
    """Daily-sum combination of several trade lists (1 contract each)."""
    t = pd.concat([x[x.split == split][["date", "pnl", "split"]] for x in trs])
    return metrics.summary(t, n_days=n_tradable(g, split))
