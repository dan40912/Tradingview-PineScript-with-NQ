"""Red-team audit helpers (Phase E prep). NEVER loads the 2026 holdout.
Loads every agent's CANDIDATES the same way research/run_holdout.py does, without touching agent files."""
from __future__ import annotations
import os, sys, importlib.util
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.dirname(HERE)
if RES not in sys.path:
    sys.path.insert(0, RES)
from lib import data, features, sim, metrics, strategies  # noqa: E402

AGENTS = ["A1", "A2", "A3", "B1", "B2", "B3", "C1", "C2", "C3"]
ES_ONLY = {"C3/C3_ES_I2_40_15"}
# candidates whose exit_slot embeds the A3 'no progress' rule (N=30 min, X=20 pts, base 12:29)
NOPROG = {"A3/A3_C3_POC_SL60_NP30", "C1/C1_I2_A3X_TS", "C1/C1_ORB15_POC_A3X_TS", "C3/C3_PORT_A3C3_ES"}
POC_TARGET = NOPROG  # same set uses target = max(dir*(POC-fill), 20)

_G = {}


def load(sym="NQ"):
    if sym not in _G:
        g = data.load_grid(sym)                     # holdout hidden (include_holdout=False)
        assert not (g.split == "holdout").any()
        _G[sym] = (g, features.daily_levels(g))
    return _G[sym]


def load_agent(agent):
    p = os.path.join(RES, "agents", agent, "candidates.py")
    d = os.path.dirname(p)
    sys.path.insert(0, d)
    try:
        spec = importlib.util.spec_from_file_location(f"audit_cand_{agent}", p)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
    finally:
        sys.path.remove(d)
    return m


def all_candidates():
    out = {}
    for a in AGENTS:
        m = load_agent(a)
        for k, fn in getattr(m, "CANDIDATES", {}).items():
            out[f"{a}/{k}"] = fn
    return out


def ndays(g, split=None):
    m = g.tradable.copy()
    if split:
        m &= g.split == split
    return int(m.sum())


def summ(g, tr, split=None):
    if tr is None or len(tr) == 0:
        return {"n": 0}
    t = tr if split is None else tr[tr.split == split]
    if len(t) == 0:
        return {"n": 0}
    return metrics.summary(t, n_days=ndays(g, split))


def run(g, sigs, **kw):
    tr = sim.run_signals(g, sigs, **kw)
    if len(tr):
        cmap = {}
        for s in sigs:
            cmap[(s["i"])] = float(s.get("contracts", 1.0))
        tr["contracts"] = tr.i.map(cmap).astype(float)
    return tr


def daily_vec(g, tr, split=None):
    m = g.tradable.copy()
    if split:
        m &= g.split == split
    idx = g.dates[m]
    s = pd.Series(0.0, index=idx)
    if tr is not None and len(tr):
        x = tr.groupby("date").pnl.sum()
        x = x[x.index.isin(idx)]
        s.loc[x.index] += x.values
    return s


def legs(name, C, fn=None):
    """[(sym, G, lv, signals)] — C3 portfolios as real NQ+ES two-leg books; ES-only candidate on the ES grid."""
    g, lv = load("NQ")
    es, lves = load("ES")
    es_fn = C["C3/C3_ES_I2_40_15"]
    if name == "C3/C3_ES_I2_40_15":
        return [("ES", es, lves, es_fn(es, lves))]
    if name in ("C3/C3_PORT_A3C3_ES", "C3/C3_PORT_I2v1_ES"):
        return [("NQ", g, lv, C[name](g, lv)), ("ES", es, lves, es_fn(es, lves, contracts=0.8))]
    f = fn if fn is not None else C[name]
    return [("NQ", g, lv, f(g, lv))]


def trades(name, C, fn=None):
    out = []
    for sym, G, L, s in legs(name, C, fn):
        tr = run(G, s)
        if len(tr):
            tr["sym"] = sym
            out.append(tr)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()
