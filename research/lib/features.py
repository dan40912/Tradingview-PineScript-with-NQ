"""Daily features computed strictly from information available before / at the time used.

daily_levels(grid) -> DataFrame indexed by day i with prior-day RTH levels
(value area exactly as the frozen I2 Pine script), overnight levels, 09:30 open.
NaN when the previous trading day uses a different contract (roll day) -> no trade.
"""
from __future__ import annotations
import math
import numpy as np
import pandas as pd
from .data import RTH_OFF, Grid

BIN = 5.0
VA_FRAC = 0.70


def value_area(h, l, v, bin_=BIN, frac=VA_FRAC):
    """Frozen I2 algorithm: spread each 1m bar's volume evenly across 5-pt bins
    floor(low/bin)..floor(high/bin); POC = max bin (lowest on ties); expand toward the
    larger neighbour (down on ties) until >= frac of volume. Returns (val, poc, vah)."""
    m = ~np.isnan(v)
    h, l, v = h[m], l[m], v[m]
    if len(v) == 0:
        return (np.nan,) * 3
    a = np.floor(l / bin_).astype(int)
    b = np.floor(h / bin_).astype(int)
    kmin, kmax = a.min(), b.max()
    prof = np.zeros(kmax - kmin + 1)
    for ai, bi, vi in zip(a, b, v):
        prof[ai - kmin: bi - kmin + 1] += vi / (bi - ai + 1)
    total = prof.sum()
    p = int(np.argmax(prof))  # first max = lowest price on ties
    lo = hi = p
    acc = prof[p]
    n = len(prof)
    while acc < frac * total and (lo > 0 or hi < n - 1):
        lv = prof[lo - 1] if lo > 0 else -1.0
        hv = prof[hi + 1] if hi < n - 1 else -1.0
        if hv > lv:
            hi += 1
            acc += max(0.0, hv)
        else:
            lo -= 1
            acc += max(0.0, lv)
    return (lo + kmin) * bin_, (p + kmin) * bin_, (hi + kmin + 1) * bin_


def daily_levels(g: Grid) -> pd.DataFrame:
    n = len(g)
    rows = []
    rH, rL, rC, rV, rO = g.rth(g.H), g.rth(g.L), g.rth(g.C), g.rth(g.V), g.rth(g.O)
    va_cache = [value_area(rH[i], rL[i], rV[i]) if g.has_rth[i] else (np.nan,) * 3 for i in range(n)]
    for i in range(n):
        d = {"i": i, "date": g.dates[i], "year": g.year[i], "split": g.split[i],
             "contract": g.contract[i], "quality": g.quality[i]}
        d["open_0930"] = rO[i, 0]
        ok = bool(g.tradable[i]) and g.prev_same_contract(i)
        d["tradable"] = bool(g.tradable[i])
        d["has_prev"] = ok
        if ok:
            j = g.prev_rth_day(i)
            d["val"], d["poc"], d["vah"] = va_cache[j]
            d["pdh"] = np.nanmax(rH[j]); d["pdl"] = np.nanmin(rL[j])
            last = np.where(~np.isnan(rC[j]))[0]
            d["pdc"] = rC[j, last[-1]] if len(last) else np.nan
            d["pdo"] = rO[j, 0]
            d["prev_rth_vol"] = np.nansum(rV[j])
        else:
            for k in ["val", "poc", "vah", "pdh", "pdl", "pdc", "pdo", "prev_rth_vol"]:
                d[k] = np.nan
        # overnight = slots 0..929 (18:00 prev .. 09:29)
        onH, onL, onV = g.H[i, :RTH_OFF], g.L[i, :RTH_OFF], g.V[i, :RTH_OFF]
        if np.isfinite(onH).any():
            d["onh"] = np.nanmax(onH); d["onl"] = np.nanmin(onL); d["on_vol"] = np.nansum(onV)
            tp = (g.H[i, :RTH_OFF] + g.L[i, :RTH_OFF] + g.C[i, :RTH_OFF]) / 3
            mm = np.isfinite(tp)
            d["on_vwap"] = np.sum(tp[mm] * onV[mm]) / max(np.sum(onV[mm]), 1)
        else:
            d["onh"] = d["onl"] = d["on_vol"] = d["on_vwap"] = np.nan
        rows.append(d)
    df = pd.DataFrame(rows)
    df["va_width"] = df.vah - df.val
    df["i2_dir"] = np.where(df.open_0930 > df.vah, -1, np.where(df.open_0930 < df.val, 1, 0))
    df.loc[~df.has_prev, "i2_dir"] = 0
    return df
