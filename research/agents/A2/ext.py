"""Shared helpers for extension signals (ADR10 strictly from prior days)."""
import numpy as np
from lib.data import RTH_OFF
_ADR = {}
def adr10(g, n=10):
    key = (g.symbol, n, len(g))
    if key in _ADR: return _ADR[key]
    rH, rL = g.rth(g.H), g.rth(g.L)
    with np.errstate(all="ignore"):
        import warnings; warnings.simplefilter("ignore")
        rng = np.nanmax(rH, 1) - np.nanmin(rL, 1)
    a = np.full(len(g), np.nan)
    idxs = np.where(g.has_rth)[0]
    for k, i in enumerate(idxs):
        if k >= n: a[i] = np.mean(rng[idxs[k-n:k]])
    _ADR[key] = a
    return a

def ext_follow(g, lv, k=0.75, s=0.35, R=None, start=30, end=325, exit_idx=389, fade=False, n=10, sides=(1, -1), ref="open"):
    """First 5m close in [09:30+start, 09:30+end) with |close - ref| > k*ADR -> trade in direction of move
    (or fade). ref: 'open' (09:30 open) or 'pdc' (prior RTH close)."""
    adr = adr10(g, n)
    out = []
    for r in lv.itertuples():
        i = r.i
        if not r.has_prev or not np.isfinite(adr[i]) or not np.isfinite(r.open_0930): continue
        base = r.open_0930 if ref == "open" else r.pdc
        C = g.C[i, RTH_OFF:RTH_OFF+390]
        for idx in range(start + 4, end, 5):
            c = C[idx]
            if not np.isfinite(c): continue
            m = c - base
            if abs(m) > k * adr[i]:
                d = int(np.sign(m)) * (-1 if fade else 1)
                if d not in sides: break
                if 28 <= idx + 1 <= 32: break
                out.append({"i": i, "entry_slot": RTH_OFF+idx+1, "dir": d, "sl_pts": s*adr[i],
                            "tp_pts": (R*s*adr[i] if R else None), "exit_slot": RTH_OFF+exit_idx, "tag": "extF"})
                break
    return out
