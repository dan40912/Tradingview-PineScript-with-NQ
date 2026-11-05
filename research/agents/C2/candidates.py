"""C2 frozen candidates (parameters fixed on discovery 2024-01-02..2025-06-30, BEFORE validation was run).

Usage:
    import sys; sys.path.insert(0, "research"); sys.path.insert(0, "research/agents/C2")
    from lib import data, features, sim
    from candidates import CANDIDATES
    g = data.load_grid("NQ"); lv = features.daily_levels(g)
    tr = sim.run_signals(g, CANDIDATES["C2_PM_PULLBACK"](g, lv))

Every signal carries `contracts` (risk-normalised size, MNQ granularity 0.1 NQ contract) so that the dollar
risk to the stop is ~RISK_USD per trade; pnl rows from sim.run_signals are therefore already sized.
The 1-contract version is available via fn(g, lv, sized=False).

No look-ahead: ADR10 uses the 10 previous RTH sessions only; value area / POC are the prior RTH day's
(lv.vah/val, same as I2 v1); the 09:30 open is known at 09:30; every trigger uses closes of minutes <= entry_slot-1;
morning VWAP uses minutes 09:30..12:29 for a 12:30 entry.

1) C2_TE_OUTSIDE  -- trend-day continuation, ONLY on days that open OUTSIDE the prior value area
   (A2_TREND_EXT_FOLLOW with a value-area day filter; inside-open days are skipped because breakouts fail there).
   From 10:04 to 14:59, first 5-min close (minute idx%5==4) with |close - 09:30 open| > 0.75*ADR10 -> trade in
   the direction of the move at the next minute's open (skip entries 09:58-10:02). Stop 0.5*ADR10, no target,
   exit at the 15:29 close. One trade per day.
2) C2_PM_PULLBACK -- afternoon "value migrated, pulled back into old value" continuation (any day type).
   At 12:29 close: morning VWAP (09:30..12:29, typical price x volume) is ABOVE prior VAH (or BELOW prior VAL)
   = value migrated up (down), AND the 12:29 close is back INSIDE prior VAL..VAH -> trade in the migration
   direction (long if VWAP>VAH, short if VWAP<VAL) at the 12:30 open. Stop 0.35*ADR10, no target, exit 15:29 close.
   (Discovered as the mirror image of the losing rule "fade the migration back to POC", see REPORT.)
3) C2_BOOK -- both of the above (they can both trade on the same day); to be run next to I2 v1 (1 contract).
"""
from __future__ import annotations
import warnings
import numpy as np

RTH_OFF = 930
RISK_USD = 1000.0      # dollar risk to the stop per trade (I2 v1 risks 145 pts = $2,900)
STEP, CAP = 0.1, 3.0   # MNQ granularity (0.1 NQ), max 3 NQ-equivalent


def _adr(g, n=10):
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


def _size(sigs, g, sized):
    if not sized:
        return sigs
    out = []
    for s in sigs:
        c = np.floor(RISK_USD / (s["sl_pts"] * g.point) / STEP + 1e-9) * STEP
        c = round(min(c, CAP), 1)
        if c >= STEP:
            out.append(dict(s, contracts=c, ct=c))
    return out


def te_outside(g, lv, sized=True):
    k, s, start, end, exit_idx = 0.75, 0.5, 30, 325, 359
    adr = _adr(g, 10)
    sigs = []
    for r in lv.itertuples():
        i = r.i
        if not r.has_prev or r.i2_dir == 0 or not np.isfinite(adr[i]) or not np.isfinite(r.open_0930):
            continue
        C = g.C[i, RTH_OFF:RTH_OFF + 390]
        for idx in range(start + 4, end, 5):
            c = C[idx]
            if not np.isfinite(c):
                continue
            m = c - r.open_0930
            if abs(m) > k * adr[i]:
                if 28 <= idx + 1 <= 32:
                    break
                sigs.append({"i": i, "entry_slot": RTH_OFF + idx + 1, "dir": int(np.sign(m)), "sl_pts": s * adr[i],
                             "tp_pts": None, "exit_slot": RTH_OFF + exit_idx, "tag": "C2_TE_OUTSIDE"})
                break
    return _size(sigs, g, sized)


def pm_pullback(g, lv, sized=True):
    T, s, exit_idx = 180, 0.35, 359          # entry 12:30, stop 0.35*ADR10, exit 15:29
    adr = _adr(g, 10)
    sigs = []
    for r in lv.itertuples():
        i = r.i
        if not r.has_prev or not np.isfinite(adr[i]) or not np.isfinite(r.open_0930):
            continue
        a, b = RTH_OFF, RTH_OFF + T
        tp_ = (g.H[i, a:b] + g.L[i, a:b] + g.C[i, a:b]) / 3
        v = g.V[i, a:b]
        m = np.isfinite(tp_) & np.isfinite(v)
        if m.sum() < 30 or v[m].sum() <= 0:
            continue
        vw = np.sum(tp_[m] * v[m]) / np.sum(v[m])
        c = g.C[i, b - 1]
        if not np.isfinite(c):
            continue
        side = 1 if vw > r.vah else (-1 if vw < r.val else 0)
        if side == 0 or not (r.val <= c <= r.vah):
            continue
        sigs.append({"i": i, "entry_slot": b, "dir": side, "sl_pts": s * adr[i], "tp_pts": None,
                     "exit_slot": RTH_OFF + exit_idx, "tag": "C2_PM_PULLBACK"})
    return _size(sigs, g, sized)


def book(g, lv, sized=True):
    return te_outside(g, lv, sized) + pm_pullback(g, lv, sized)


CANDIDATES = {
    "C2_TE_OUTSIDE": te_outside,
    "C2_PM_PULLBACK": pm_pullback,
    "C2_BOOK": book,
}
