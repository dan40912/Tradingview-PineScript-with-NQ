"""C3 helpers (thin wrappers around the shared engine research/lib; no new simulator).

- levels(g, bin_)      : features.daily_levels(g) with the prior-day value area recomputed using the
                         FROZEN value_area algorithm but a different bin size (e.g. ES 1.25 / 2.5 / 5 pts).
- add_atr(g, lv)       : ATR14 of prior full RTH sessions (known before the open).
- i2_exit_sigs(...)    : frozen I2 v1 entries + alternative exits (fixed TP/SL, POC target w/ floor,
                         no-progress exit) -- all decided from information available at/after the fill.
- daily(tr, g, splits) : daily P&L vector over ALL tradable days (flat days = 0) for Sharpe / corr.
- stats(...)           : summary with daily Sharpe incl. flat days.
Holdout is never loaded unless include_holdout=True is passed explicitly (Phase E only).
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
VARIANTS = []          # running log of every variant evaluated (name, period) for multiple-testing count


def log_variant(name):
    VARIANTS.append(name)


def levels(g, bin_=5.0):
    lv = features.daily_levels(g)
    if bin_ == 5.0:
        return lv
    rH, rL, rV = g.rth(g.H), g.rth(g.L), g.rth(g.V)
    cache = {}
    for r in lv.itertuples():
        if not r.has_prev:
            continue
        j = g.prev_rth_day(r.i)
        if j not in cache:
            cache[j] = features.value_area(rH[j], rL[j], rV[j], bin_=bin_)
        lv.loc[r.Index, ["val", "poc", "vah"]] = cache[j]
    lv["va_width"] = lv.vah - lv.val
    lv["i2_dir"] = np.where(lv.open_0930 > lv.vah, -1, np.where(lv.open_0930 < lv.val, 1, 0))
    lv.loc[~lv.has_prev, "i2_dir"] = 0
    return lv


def add_atr(g, lv):
    n = len(g)
    rH, rL = g.rth(g.H), g.rth(g.L)
    atr = np.full(n, np.nan)
    hist = []
    for i in range(n):
        if len(hist) >= 14:
            atr[i] = np.mean(hist[-14:])
        if g.has_rth[i] and not g.equity_holiday[i] and np.isfinite(rH[i]).any():
            hist.append(np.nanmax(rH[i]) - np.nanmin(rL[i]))
    lv = lv.copy()
    lv["atr14"] = atr
    return lv


def fill(g, s, last=EXIT_1229):
    e = s["entry_slot"]
    while e <= last and np.isnan(g.O[s["i"], e]):
        e += 1
    return e, (g.O[s["i"], e] if e <= last else np.nan)


def noprog_slot(g, s, e, px, N, X, last=EXIT_1229):
    """Causal 'no progress' exit: at the close of minute e+N-1, if best favourable excursion < X -> exit."""
    end = e + N - 1
    if end >= last:
        return last
    i, d = s["i"], s["dir"]
    fav = np.nanmax(g.H[i, e:end + 1] - px) if d == 1 else np.nanmax(px - g.L[i, e:end + 1])
    return end if fav < X else last


def i2_exit_sigs(g, lv, tp=None, sl=None, poc_floor=None, np_n=None, np_x=None, contracts=None,
                 sigs=None, tag="I2"):
    """tp: fixed target pts; poc_floor: if set, target = max(dir*(POC-fill), poc_floor) (overrides tp)."""
    poc = dict(zip(lv.i, lv.poc))
    base = sigs if sigs is not None else strategies.i2_v1_signals(g, lv)
    out = []
    for s in base:
        e, px = fill(g, s)
        if np.isnan(px):
            continue
        t = tp
        if poc_floor is not None:
            t = float(max(s["dir"] * (poc[s["i"]] - px), poc_floor))
        x = EXIT_1229 if np_n is None else noprog_slot(g, s, e, px, np_n, np_x)
        d = dict(s, tp_pts=t, sl_pts=sl, exit_slot=x, tag=tag)
        if contracts is not None:
            d["contracts"] = contracts
        out.append(d)
    return out


def run(g, sigs):
    tr = sim.run_signals(g, sigs)
    if len(tr) == 0:
        tr = pd.DataFrame(columns=["date", "dir", "pnl", "split", "year", "i"])
    return tr


def half(date):
    d = str(date)
    return d[:4] + ("H1" if d[5:7] <= "06" else "H2")


def tradable_days(g, split=None):
    m = g.tradable.copy()
    if split is not None:
        m &= (g.split == split)
    return list(g.dates[m])


def daily(tr, days):
    """Daily P&L Series indexed by the given day list (flat days = 0)."""
    s = pd.Series(0.0, index=pd.Index(days, name="date"))
    if tr is not None and len(tr):
        dd = tr.groupby("date")["pnl"].sum()
        dd = dd[dd.index.isin(s.index)]
        s.loc[dd.index] += dd.values
    return s


def stats(tr, days, label=None):
    """tr restricted to days; daily Sharpe incl flat days; maxDD on daily equity."""
    if tr is not None and len(tr):
        tr = tr[tr.date.isin(days)]
    dv = daily(tr, days)
    n = 0 if tr is None else len(tr)
    out = {"n": n}
    if n:
        p = tr.pnl.values.astype(float)
        out.update({"win%": round(100 * (p > 0).mean(), 1), "avg": round(p.mean(), 1), "total": round(p.sum(), 0)})
    else:
        out.update({"win%": np.nan, "avg": np.nan, "total": 0.0})
    mdd = metrics.max_drawdown(dv.values)
    out["maxdd"] = round(mdd, 0)
    out["ret/dd"] = round(dv.sum() / -mdd, 2) if mdd < 0 else np.nan
    sd = dv.std(ddof=1)
    out["sharpe"] = round(dv.mean() / sd * np.sqrt(252), 2) if sd > 0 else np.nan
    out["worst_day"] = round(dv.min(), 0)
    return out


def half_avgs(tr, halves=("2024H1", "2024H2", "2025H1")):
    if tr is None or len(tr) == 0:
        return {h: np.nan for h in halves}
    h = tr.date.map(half)
    return {k: round(tr.pnl[h == k].mean(), 1) if (h == k).any() else np.nan for k in halves}
