"""B1 frozen candidates (parameters fixed 2026-11-05 after discovery, before validation).

    import sys; sys.path.insert(0, "research"); sys.path.insert(0, "research/agents/B1")
    from lib import data, features, sim
    from candidates import CANDIDATES
    g = data.load_grid("NQ"); lv = features.daily_levels(g)
    tr = sim.run_signals(g, CANDIDATES["B1_I2_volpoor"](g, lv))

All three keep the I2 v1 trade structure: TP +55, SL -145, exit 12:29 close, 1 trade/day, entry at the
open of the minute after the 2nd consecutive 5-min close inside the reference value area, entries
09:34..11:30 except 09:58-10:02.  Only bars that closed before the entry minute are used.
"""
from __future__ import annotations
import os, sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import indicators as ind  # noqa: E402

RTH_OFF = 930
TP, SL, EXIT_IDX, LAST_IDX = 55.0, 145.0, 179, 120
VOLPOOR_TH = 0.25      # C1: extreme-bin volume / mean-bin volume threshold (frozen)
COMPOSITE_K = 3        # C2: number of prior RTH sessions in composite profile (frozen)


def _i2_like(g, i, val, vah, open_, tag, m=2):
    """I2 v1 entry engine on an arbitrary reference value area."""
    d = -1 if open_ > vah else (1 if open_ < val else 0)
    if d == 0:
        return None
    C = g.C[i]
    cnt = 0
    for idx in range(4, 120, 5):
        c = C[RTH_OFF + idx]
        if np.isnan(c):
            continue
        cnt = cnt + 1 if (val <= c <= vah) else 0
        if cnt >= m:
            ei = idx + 1
            if (28 <= ei <= 32) or ei > LAST_IDX:
                return None
            return {"i": i, "entry_slot": RTH_OFF + ei, "dir": d, "tp_pts": TP, "sl_pts": SL,
                    "exit_slot": RTH_OFF + EXIT_IDX, "tag": tag}
    return None


def _extreme_bin_rel(g, j, side):
    """Volume in the prior session's highest (side=+1) or lowest (side=-1) 5-pt bin / mean bin volume."""
    rH, rL, rV = g.H[j, RTH_OFF:RTH_OFF + 390], g.L[j, RTH_OFF:RTH_OFF + 390], g.V[j, RTH_OFF:RTH_OFF + 390]
    k, pr = ind.profile(rH, rL, rV)
    if pr is None:
        return np.nan
    return (pr[-1] if side == 1 else pr[0]) / pr.mean()


def b1_i2_volpoor(g, lv):
    """C1: frozen I2 v1, but SKIP the day if the prior RTH session's extreme on the open side was
    'poor' by volume (top bin for a short / bottom bin for a long holds >= 0.25x the mean bin volume:
    an unfinished auction that price tends to revisit, i.e. continue away from value)."""
    out = []
    for r in lv.itertuples():
        if not r.has_prev or r.i2_dir == 0 or np.isnan(r.open_0930):
            continue
        j = g.prev_rth_day(r.i)
        rel = _extreme_bin_rel(g, j, +1 if r.i2_dir == -1 else -1)
        if not (rel < VOLPOOR_TH):
            continue
        s = _i2_like(g, r.i, r.val, r.vah, r.open_0930, "B1_I2_volpoor")
        if s:
            out.append(s)
    return out


def b1_composite3_i2(g, lv):
    """C2: I2 engine on the 3-session composite value area (sum of the last 3 same-contract RTH
    volume profiles, frozen 5-pt / 70% algorithm)."""
    out = []
    profs = {}
    for r in lv.itertuples():
        if not r.has_prev or np.isnan(r.open_0930):
            continue
        prev = ind._prev_rth_list(g, r.i, COMPOSITE_K)
        if len(prev) < COMPOSITE_K:
            continue
        parts = []
        for j in prev:
            if j not in profs:
                profs[j] = ind.profile(g.H[j, RTH_OFF:RTH_OFF + 390], g.L[j, RTH_OFF:RTH_OFF + 390], g.V[j, RTH_OFF:RTH_OFF + 390])
            parts.append(profs[j])
        val, _, vah = ind.va_from_profile(*ind.add_profiles(parts))
        s = _i2_like(g, r.i, val, vah, r.open_0930, "B1_composite3_I2")
        if s:
            out.append(s)
    return out


def b1_onva_i2(g, lv):
    """C3: I2 engine on today's OVERNIGHT (18:00-09:29 ET) value area. Requires a prior RTH session
    on the same contract (lv.has_prev) so that tradable days match I2's universe."""
    out = []
    for r in lv.itertuples():
        if not r.has_prev or np.isnan(r.open_0930):
            continue
        i = r.i
        val, _, vah = ind.va_from_profile(*ind.profile(g.H[i, :RTH_OFF], g.L[i, :RTH_OFF], g.V[i, :RTH_OFF]))
        if not np.isfinite(val):
            continue
        s = _i2_like(g, i, val, vah, r.open_0930, "B1_ONVA_I2")
        if s:
            out.append(s)
    return out


def b1_onva_and_i2(g, lv):
    """LEAD (post-hoc, NOT a candidate): 'double confirmation'.  Day must have an I2 direction (open outside
    the prior RTH VA) AND the open outside today's overnight VA on the same side.  Enter only after BOTH
    re-entry conditions have happened (2 consecutive 5-min closes inside the prior RTH VA, and 2 inside
    the overnight VA): entry = the later of the two I2-engine entry minutes (no look-ahead).
    Idea came from the protocol's I2-day / non-I2-day split of C3; frozen after validation was seen,
    so it needs holdout / forward data before it means anything."""
    i2 = {s["i"]: s for s in _i2_signals_plain(g, lv)}
    out = []
    for s in b1_onva_i2(g, lv):
        k = i2.get(s["i"])
        if k is None or k["dir"] != s["dir"]:
            continue
        e = dict(s); e["entry_slot"] = max(s["entry_slot"], k["entry_slot"]); e["tag"] = "B1_ONVA_and_I2"
        out.append(e)
    return out


def _i2_signals_plain(g, lv):
    out = []
    for r in lv.itertuples():
        if not r.has_prev or r.i2_dir == 0 or np.isnan(r.open_0930):
            continue
        s = _i2_like(g, r.i, r.val, r.vah, r.open_0930, "I2v1")
        if s:
            out.append(s)
    return out


# Frozen candidates offered by B1 (C2 failed validation and is kept only for reference).
CANDIDATES = {
    "B1_I2_volpoor": b1_i2_volpoor,
    "B1_ONVA_I2": b1_onva_i2,
}
REJECTED = {"B1_composite3_I2": b1_composite3_i2}   # discovery +$335/trade, validation -$37/trade
LEADS = {"B1_ONVA_and_I2": b1_onva_and_i2}          # post-hoc; holdout / forward test only

