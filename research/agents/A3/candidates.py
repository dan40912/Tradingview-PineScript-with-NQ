"""A3 frozen candidates: I2 v1 entries (unchanged) + new exit structure. Parameters are FROZEN
(chosen on discovery 2024-01-02..2025-06-30 only, before looking at validation).

Usage:
    import sys; sys.path.insert(0, "research"); sys.path.insert(0, "research/agents/A3")
    from lib import data, features, sim
    from candidates import CANDIDATES
    g = data.load_grid("NQ"); lv = features.daily_levels(g)
    tr = sim.run_signals(g, CANDIDATES["A3_C2_TP100_SL60_NP30"](g, lv))

No look-ahead:
  * Entries are exactly strategies.i2_v1_signals (decided on 5-min closes before entry_slot).
  * C3's target uses prior-day POC and the fill price (open of the entry minute). A bracket order
    is placed right after the fill, so using the fill price is legitimate.
  * The 'no progress' rule (C2, C3) is a time exit decided at the CLOSE of minute e+29 (e = entry
    minute) using only the highs/lows of minutes e..e+29: if the best favourable excursion so far is
    < 20 points, the trade is closed at that close. It is expressed as an exit_slot so the shared
    simulator handles it; if TP/SL is hit earlier the simulator exits there first.
"""
from __future__ import annotations
import os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from lib import strategies  # noqa: E402
from lib.data import RTH_OFF  # noqa: E402

EXIT_1229 = RTH_OFF + 179
SL_PTS = 60.0
TP_PTS = 100.0
NP_MIN, NP_PTS = 30, 20.0
POC_FLOOR = 20.0


def _fill(g, s):
    e = s["entry_slot"]
    while e <= EXIT_1229 and np.isnan(g.O[s["i"], e]):
        e += 1
    return e, (g.O[s["i"], e] if e <= EXIT_1229 else np.nan)


def _noprog_exit(g, s, e, px, N=NP_MIN, X=NP_PTS):
    last = e + N - 1
    if last >= EXIT_1229:
        return EXIT_1229
    i, d = s["i"], s["dir"]
    fav = np.nanmax(g.H[i, e:last + 1] - px) if d == 1 else np.nanmax(px - g.L[i, e:last + 1])
    return last if fav < X else EXIT_1229


def _base(g, lv, h1=False):
    return strategies.i2_v1_signals(g, lv, h1=h1)


def c1_tp100_sl60(g, lv):
    """I2 v1 entries, TP +100 / SL -60 points, time exit 12:29 ET."""
    out = []
    for s in _base(g, lv):
        s = dict(s, tp_pts=TP_PTS, sl_pts=SL_PTS, exit_slot=EXIT_1229, tag="A3_C1")
        out.append(s)
    return out


def c2_tp100_sl60_np30(g, lv):
    """C1 + 'no progress' exit: after 30 minutes, if the trade never went +20 points, exit."""
    out = []
    for s in _base(g, lv):
        e, px = _fill(g, s)
        if np.isnan(px):
            continue
        out.append(dict(s, tp_pts=TP_PTS, sl_pts=SL_PTS, exit_slot=_noprog_exit(g, s, e, px), tag="A3_C2"))
    return out


def c3_poc_sl60_np30(g, lv):
    """Target = prior-day POC (at least +20 points from the fill), SL -60, no-progress 30 min / 20 pts."""
    poc = dict(zip(lv.i, lv.poc))
    out = []
    for s in _base(g, lv):
        e, px = _fill(g, s)
        if np.isnan(px):
            continue
        tp = max(s["dir"] * (poc[s["i"]] - px), POC_FLOOR)
        out.append(dict(s, tp_pts=float(tp), sl_pts=SL_PTS, exit_slot=_noprog_exit(g, s, e, px), tag="A3_C3"))
    return out


# Frozen before validation: C1, C2, C3. C1 and C2 FAILED validation (avg/trade <= 0) -> moved to REJECTED.
CANDIDATES = {
    "A3_C3_POC_SL60_NP30": c3_poc_sl60_np30,
}
REJECTED = {
    "A3_C1_TP100_SL60": c1_tp100_sl60,          # validation -$8/trade
    "A3_C2_TP100_SL60_NP30": c2_tp100_sl60_np30,  # validation -$33/trade
}
ALL_PREREGISTERED = {**REJECTED, **CANDIDATES}
