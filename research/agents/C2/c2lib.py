"""C2 helpers: loading (never 2026), per-day context, evaluation with half-year breakdown,
risk-normalized sizing, I2 overlap/combination, and a variant log (every tested variant counted).
All P&L via the shared engine sim.run_signals."""
from __future__ import annotations
import os, sys, warnings
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RESEARCH = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, RESEARCH)
from lib import data, features, sim, metrics, strategies  # noqa: E402

RTH = data.RTH_OFF
LOG = os.path.join(HERE, "variant_log.csv")
_C = {}


def load():
    if "g" not in _C:
        g = data.load_grid("NQ")                       # holdout hidden by default
        assert all(d < "2026-01-01" for d in g.dates)
        lv = features.daily_levels(g)
        lv["adr10"] = adr(g, 10)
        i2 = sim.run_signals(g, strategies.i2_v1_signals(g, lv))
        _C.update(g=g, lv=lv, i2=i2)
    return _C["g"], _C["lv"], _C["i2"]


def adr(g, n=10):
    """Mean RTH high-low of the previous n RTH sessions (strictly prior) -- same as A2."""
    rH, rL = g.rth(g.H), g.rth(g.L)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rng = np.nanmax(rH, 1) - np.nanmin(rL, 1)
    a = np.full(len(g), np.nan)
    idxs = np.where(g.has_rth)[0]
    for k, i in enumerate(idxs):
        if k >= n:
            a[i] = np.mean(rng[idxs[k - n:k]])
    return a


def half(date):
    d = str(date)
    return d[:4] + ("H1" if d[5:7] <= "06" else "H2")


def n_days(g, split=None, hal=None):
    m = g.tradable.copy()
    if split:
        m &= g.split == split
    if hal:
        m &= np.array([half(d) == hal for d in g.dates])
    return int(m.sum())


def summ(g, tr, split=None, hal=None, col="pnl"):
    if tr is None or len(tr) == 0:
        return {"n": 0}
    t = tr
    if split:
        t = t[t.split == split]
    if hal:
        t = t[t.date.map(half) == hal]
    if len(t) == 0:
        return {"n": 0}
    return metrics.summary(t, n_days=n_days(g, split, hal), pnl_col=col)


HALVES = ["2024H1", "2024H2", "2025H1"]


def size_risk(sigs, risk_usd, point=20.0, step=0.1, cap=3.0):
    """contracts = risk_usd / (sl_pts*point), floored to 0.1 (MNQ granularity), capped."""
    out = []
    for s in sigs:
        s = dict(s)
        sl = s.get("sl_pts")
        if sl is None or not np.isfinite(sl) or sl <= 0:
            continue
        c = np.floor(risk_usd / (sl * point) / step + 1e-9) * step
        c = min(c, cap)
        if c < step:
            continue
        s["contracts"] = round(c, 1)
        s["ct"] = round(c, 1)
        out.append(s)
    return out


def disc(g, tr, name, family="", log=True, quiet=False, extra=None):
    """Discovery-only screen: overall + 3 half-years (validation NOT looked at)."""
    d = summ(g, tr, "discovery")
    row = {"family": family, "name": name}
    for f in ["n", "win%", "avg", "total", "maxdd", "ret/dd", "sharpe_ann"]:
        row["d_" + f] = d.get(f, np.nan)
    hs = []
    for h in HALVES:
        s = summ(g, tr, hal=h)
        row[h] = s.get("avg", np.nan)
        row[h + "_n"] = s.get("n", 0)
        hs.append(s.get("avg", np.nan))
    row["all_halves_pos"] = bool(all(np.isfinite(x) and x > 0 for x in hs))
    if extra:
        row.update(extra)
    if log:
        pd.DataFrame([row]).to_csv(LOG, mode="a", header=not os.path.exists(LOG), index=False)
    if not quiet:
        print(f"{name:48s} n={row['d_n']!s:>4} win={row['d_win%']!s:>5} avg={row['d_avg']!s:>7} tot={row['d_total']!s:>8} "
              f"dd={row['d_maxdd']!s:>7} sh={row['d_sharpe_ann']!s:>5} | " +
              " ".join(f"{h[2:]}:{row[h]:>6.0f}({row[h+'_n']})" if np.isfinite(row[h]) else f"{h[2:]}:   nan" for h in HALVES)
              + ("  *" if row["all_halves_pos"] else ""))
    return row


def daily(g, tr, col="pnl"):
    """Daily P&L vector over ALL tradable days (flat days = 0), indexed by date."""
    idx = g.dates[g.tradable]
    s = pd.Series(0.0, index=idx)
    if tr is not None and len(tr):
        x = tr.groupby("date")[col].sum()
        s.loc[x.index.intersection(idx)] += x.loc[x.index.intersection(idx)]
    return s


def corr_i2(g, tr, i2, split=None):
    a, b = daily(g, tr), daily(g, i2)
    if split:
        m = np.array([data.split_of(d) == split for d in a.index])
        a, b = a[m], b[m]
    return round(float(np.corrcoef(a, b)[0, 1]), 3)


def overlap(tr, i2, split):
    s = set(i2.date)
    t = tr[tr.split == split]
    a, b = t[t.date.isin(s)], t[~t.date.isin(s)]
    f = lambda x: (len(x), round(x.pnl.mean(), 0) if len(x) else np.nan)
    return {"on_I2": f(a), "off_I2": f(b)}


def combo(g, trs, split=None, hal=None):
    t = pd.concat([x[["date", "pnl", "split"]] for x in trs if x is not None and len(x)])
    return summ(g, t, split, hal)


def rth(g, i):
    return (g.O[i, RTH:RTH + 390], g.H[i, RTH:RTH + 390], g.L[i, RTH:RTH + 390], g.C[i, RTH:RTH + 390])
