"""A3 helpers: thin wrappers around the shared engine (research/lib). No new simulator.

- load(sym)           -> g, lv (lv gets extra pre-entry columns: atr14, on_range, rth_rng_prev)
- i2_entries(g, lv)   -> frozen I2 v1 signal dicts (entry/dir only matter; exits overridden later)
- with_exit(sigs, ...) -> copy of signals with exit parameters replaced (pts or per-signal callables)
- stats(tr, g, split) -> metrics.summary with daily Sharpe incl. flat (tradable) days
- noprog(...)         -> "no progress after N minutes" exit expressed as a causal exit_slot
All quantities used to set exits are known at (or before) the entry fill.
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from lib import data, features, sim, metrics, strategies  # noqa: E402

RTH = data.RTH_OFF
EXIT_1229 = RTH + 179


def add_vol_cols(g, lv):
    n = len(g)
    rH, rL = g.rth(g.H), g.rth(g.L)
    rng = np.full(n, np.nan)
    for i in range(n):
        if g.has_rth[i] and np.isfinite(rH[i]).any():
            rng[i] = np.nanmax(rH[i]) - np.nanmin(rL[i])
    atr = np.full(n, np.nan)
    prev_rng = np.full(n, np.nan)
    hist = []
    for i in range(n):
        if len(hist) >= 1:
            prev_rng[i] = hist[-1]
        if len(hist) >= 14:
            atr[i] = np.mean(hist[-14:])
        if g.has_rth[i] and not g.equity_holiday[i] and np.isfinite(rng[i]):
            hist.append(rng[i])   # full-session days only, appended AFTER day i is used
    lv = lv.copy()
    lv["atr14"] = atr
    lv["rng_prev"] = prev_rng
    lv["on_range"] = lv.onh - lv.onl
    return lv


_CACHE = {}


def load(sym="NQ"):
    if sym not in _CACHE:
        g = data.load_grid(sym)
        assert not (g.split == "holdout").any()
        lv = add_vol_cols(g, features.daily_levels(g))
        _CACHE[sym] = (g, lv)
    return _CACHE[sym]


def ndays(g, split):
    if split == "ALL":
        return int(g.tradable.sum())
    return int((g.tradable & (g.split == split)).sum())


def stats(tr, g, split="discovery"):
    if tr is None or len(tr) == 0:
        return {"n": 0}
    t = tr if split == "ALL" else tr[tr.split == split]
    if len(t) == 0:
        return {"n": 0}
    s = metrics.summary(t, n_days=ndays(g, split))
    return s


def i2_entries(g, lv, h1=False):
    return strategies.i2_v1_signals(g, lv, h1=h1)


def entry_px(g, s):
    e = s["entry_slot"]
    while np.isnan(g.O[s["i"], e]):
        e += 1
    return g.O[s["i"], e]


def with_exit(g, lv, sigs, tp=55.0, sl=145.0, exit_slot=EXIT_1229, contracts=None, **kw):
    """tp/sl/exit_slot/contracts may be numbers or callables f(g, lvrow, sig, px) -> value.
    Returns new signal list. Signals whose callable returns None are dropped."""
    out = []
    lvi = lv.set_index("i")
    for s in sigs:
        r = lvi.loc[s["i"]]
        px = entry_px(g, s)
        d = dict(s)
        for key, v in (("tp_pts", tp), ("sl_pts", sl), ("exit_slot", exit_slot), ("contracts", contracts)):
            if v is None and key == "contracts":
                continue
            val = v(g, r, s, px) if callable(v) else v
            if val is None or (isinstance(val, float) and np.isnan(val)):
                d = None
                break
            d[key] = val
        if d is None:
            continue
        d.update(kw)
        out.append(d)
    return out


def run(g, sigs):
    return sim.run_signals(g, sigs)


def noprog_exit_slot(g, s, px, N, X, base_exit=EXIT_1229):
    """Causal 'no progress' rule: if by the CLOSE of minute e+N-1 the best favourable excursion
    (from H/L of minutes e..e+N-1) is < X points, exit at the close of minute e+N-1.
    (If TP/SL hit earlier the simulator exits first anyway.) Returns exit_slot."""
    i, d = s["i"], s["dir"]
    e = s["entry_slot"]
    while np.isnan(g.O[i, e]):
        e += 1
    last = e + N - 1
    if last >= base_exit:
        return base_exit
    seg_h, seg_l = g.H[i, e:last + 1], g.L[i, e:last + 1]
    fav = np.nanmax(seg_h - px) if d == 1 else np.nanmax(px - seg_l)
    return last if fav < X else base_exit


def fmt(s, keys=("n", "win%", "avg", "total", "maxdd", "ret/dd", "sharpe_ann")):
    return " ".join(f"{k}={s.get(k)}" for k in keys)


def row(name, tr, g, splits=("discovery",)):
    d = {"variant": name}
    for sp in splits:
        s = stats(tr, g, sp)
        for k in ("n", "win%", "avg", "total", "maxdd", "ret/dd", "sharpe_ann"):
            d[f"{sp[:3]}_{k}"] = s.get(k)
    return d


def overlap_split(tr, base_dates):
    """Split trades into days on which I2 v1 trades vs not."""
    m = tr.date.isin(base_dates)
    return tr[m], tr[~m]
