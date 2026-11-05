"""Quick generic baselines used only to study regime/event interactions (B2).
All return signal lists for sim.run_signals. Signals use only bars closed before entry_slot."""
from __future__ import annotations
import numpy as np
from lib.data import RTH_OFF

EXIT_1229 = RTH_OFF + 179


def orb(g, lv, or_min=30, last_idx=120, tp_mult=1.0, sl_mult=1.0, exit_slot=EXIT_1229, min_rng=20.0):
    """Opening-range breakout: OR = 09:30..09:30+or_min-1. First 1-min CLOSE beyond OR (after OR ends,
    up to idx last_idx) -> enter next minute open. SL = sl_mult*OR width, TP = tp_mult*OR width."""
    sigs = []
    for r in lv.itertuples():
        if not r.tradable or np.isnan(r.open_0930):
            continue
        i = r.i
        H, L, C = g.H[i], g.L[i], g.C[i]
        oh = np.nanmax(H[RTH_OFF:RTH_OFF + or_min]); ol = np.nanmin(L[RTH_OFF:RTH_OFF + or_min])
        w = oh - ol
        if not np.isfinite(w) or w < min_rng:
            continue
        for idx in range(or_min, last_idx):
            c = C[RTH_OFF + idx]
            if np.isnan(c):
                continue
            d = 1 if c > oh else (-1 if c < ol else 0)
            if d:
                sigs.append({"i": i, "entry_slot": RTH_OFF + idx + 1, "dir": d, "tp_pts": tp_mult * w,
                             "sl_pts": sl_mult * w, "exit_slot": exit_slot, "tag": "ORB", "or_w": w})
                break
    return sigs


def gap_fade(g, lv, min_gap=0.0, max_gap_atr=None, atr=None, sl_mult=1.0, entry_idx=1, exit_slot=EXIT_1229):
    """Fade the RTH opening gap toward prior RTH close. Enter at 09:30+entry_idx open.
    TP = gap fill (pdc), SL = sl_mult*|gap|."""
    sigs = []
    for r in lv.itertuples():
        if not r.has_prev or np.isnan(r.open_0930) or np.isnan(r.pdc):
            continue
        gp = r.open_0930 - r.pdc
        if abs(gp) < max(min_gap, 10.0):
            continue
        if max_gap_atr is not None and atr is not None and abs(gp) > max_gap_atr * atr[r.i]:
            continue
        sigs.append({"i": r.i, "entry_slot": RTH_OFF + entry_idx, "dir": -int(np.sign(gp)), "tp_pts": abs(gp),
                     "sl_pts": sl_mult * abs(gp), "exit_slot": exit_slot, "tag": "GAPF"})
    return sigs


def fixed_time(g, lv, idx=5, direction=1, tp=55.0, sl=145.0, exit_slot=EXIT_1229):
    """Control: enter every tradable day at a fixed time in a fixed direction."""
    return [{"i": r.i, "entry_slot": RTH_OFF + idx, "dir": direction, "tp_pts": tp, "sl_pts": sl,
             "exit_slot": exit_slot, "tag": "FIX"} for r in lv.itertuples() if r.tradable and r.has_prev]
