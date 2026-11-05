"""Trade simulator on the 1-minute grid (conservative: if SL and TP are both touched in
the same minute, SL is assumed first — same as the frozen rule).

A signal = (day i, entry_slot, direction ±1, tp_pts, sl_pts, exit_slot).
Entry at the OPEN of entry_slot (i.e. signal decided on the close of entry_slot-1).
TP/SL are checked from the entry minute onward (including the entry minute's high/low).
If neither hits, exit at the CLOSE of exit_slot (or last available bar <= exit_slot).
tp_pts / sl_pts may be np.inf / None to disable.
Optional: be_trigger (move stop to entry+be_offset after favourable excursion >= be_trigger),
trail_pts (trailing stop distance, activated after trail_trigger points of profit).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from .data import RTH_OFF, Grid


def slot_of(hhmm: str) -> int:
    """'09:30' -> 930 (slot in 18:00-based grid). Accepts times 18:00..17:59 ET."""
    h, m = map(int, hhmm.split(":"))
    mins = h * 60 + m
    return (mins - 18 * 60) % 1440


def hhmm_of(slot: int) -> str:
    t = (slot + 18 * 60) % 1440
    return f"{t // 60:02d}:{t % 60:02d}"


def simulate_one(g: Grid, i, entry_slot, direction, tp_pts=55.0, sl_pts=145.0, exit_slot=None,
                 be_trigger=None, be_offset=0.0, trail_pts=None, trail_trigger=0.0, contracts=1.0):
    if exit_slot is None:
        exit_slot = RTH_OFF + 179
    O, H, L, C = g.O[i], g.H[i], g.L[i], g.C[i]
    e = int(entry_slot)
    # entry price = open of entry slot; if missing use next available open
    while e <= exit_slot and np.isnan(O[e]):
        e += 1
    if e > exit_slot:
        return None
    px = O[e]
    d = direction
    tp = px + d * tp_pts if tp_pts is not None and np.isfinite(tp_pts) else None
    sl = px - d * sl_pts if sl_pts is not None and np.isfinite(sl_pts) else None
    best = 0.0
    reason, xpx, xs = "time", np.nan, exit_slot
    for s in range(e, exit_slot + 1):
        h, l = H[s], L[s]
        if np.isnan(h):
            continue
        if sl is not None and ((d == 1 and l <= sl) or (d == -1 and h >= sl)):
            reason, xpx, xs = ("sl" if abs((sl - px)) > 1e-9 and (sl - px) * d < 0 else "stop"), sl, s
            break
        if tp is not None and ((d == 1 and h >= tp) or (d == -1 and l <= tp)):
            reason, xpx, xs = "tp", tp, s
            break
        # update excursion AFTER checks on this bar (conservative)
        fav = (h - px) if d == 1 else (px - l)
        best = max(best, fav)
        if be_trigger is not None and best >= be_trigger:
            nsl = px + d * be_offset
            sl = nsl if sl is None else (max(sl, nsl) if d == 1 else min(sl, nsl))
        if trail_pts is not None and best >= trail_trigger:
            nsl = (px + d * (best - trail_pts))
            sl = nsl if sl is None else (max(sl, nsl) if d == 1 else min(sl, nsl))
    if reason == "time":
        cc = C[e:exit_slot + 1]
        k = np.where(~np.isnan(cc))[0]
        xs = e + k[-1]
        xpx = cc[k[-1]]
    pts = d * (xpx - px)
    pnl = (pts * g.point - g.cost) * contracts
    return {"date": g.dates[i], "i": i, "dir": d, "entry_slot": e, "entry_time": hhmm_of(e), "entry": px,
            "exit_slot": xs, "exit_time": hhmm_of(xs), "exit": xpx, "reason": reason, "pts": pts, "pnl": pnl,
            "year": g.year[i], "split": g.split[i]}


def run_signals(g: Grid, signals, **default_kw) -> pd.DataFrame:
    """signals: iterable of dicts with keys i, entry_slot, dir and optional tp, sl, exit_slot, tag, ...
    Extra keys are copied into the trade row."""
    out = []
    for s in signals:
        s = dict(s)
        kw = dict(default_kw)
        for k in ["tp_pts", "sl_pts", "exit_slot", "be_trigger", "be_offset", "trail_pts", "trail_trigger", "contracts"]:
            if k in s:
                kw[k] = s.pop(k)
        i, es, d = s.pop("i"), s.pop("entry_slot"), s.pop("dir")
        r = simulate_one(g, i, es, d, **kw)
        if r is not None:
            r.update(s)
            out.append(r)
    return pd.DataFrame(out)
