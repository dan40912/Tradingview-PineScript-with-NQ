"""A2 frozen candidates (parameters fixed on discovery 2024-01-02..2025-06-30; do not tune).
Each fn(g, lv) -> list of signal dicts for sim.run_signals(g, fn(g, lv)).
Point-based parameters are in NQ points; for ES they are scaled by ES_SCALE (only used in robustness checks).
No look-ahead: every decision uses closes up to entry_slot-1 and levels from prior days / the overnight session.
"""
from __future__ import annotations
import numpy as np

RTH_OFF = 930          # 09:30 ET slot
ES_SCALE = 0.275       # ES/NQ price ratio used to scale point parameters for the ES robustness check


def _scale(g):
    return 1.0 if g.symbol == "NQ" else ES_SCALE


def _adr(g, n=10):
    """Mean RTH high-low range of the previous n RTH sessions (strictly prior days)."""
    import warnings
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


# ---------------------------------------------------------------------------------------------
# 1) ON_SWEEP_2R: overnight-range stop run that fails during RTH
#    09:30 open must be strictly inside the overnight (18:00-09:29) range.  After 09:30, once a 1m bar
#    trades >= 2 pts beyond ONH (ONL), the first 5m close (minute idx%5==4) back inside the range
#    -> fade (short after ONH sweep / long after ONL sweep) at the next minute's open.
#    Decision window: entry idx <= 120 (11:30), skip entries 09:58-10:02, one decision per day.
#    Stop = sweep extreme (RTH high/low so far) + 5 pts (min 10, skip day if > 80).  Target = 2R.
#    Time exit 15:59.
# ---------------------------------------------------------------------------------------------
def on_sweep_2r(g, lv):
    f = _scale(g)
    buf, stop_buf, min_risk, max_risk, R, last_idx, exit_idx = 2.0 * f, 5.0 * f, 10.0 * f, 80.0 * f, 2.0, 120, 389
    sigs = []
    for r in lv.itertuples():
        if not r.has_prev or not np.isfinite(r.open_0930):
            continue
        hi, lo, o, i = r.onh, r.onl, r.open_0930, r.i
        if not (np.isfinite(hi) and np.isfinite(lo) and lo < o < hi):
            continue
        H = g.H[i, RTH_OFF:RTH_OFF + 390]; L = g.L[i, RTH_OFF:RTH_OFF + 390]; C = g.C[i, RTH_OFF:RTH_OFF + 390]
        sh = sl_ = False
        xh, xl = -np.inf, np.inf
        for idx in range(0, last_idx + 1):
            if np.isfinite(H[idx]):
                xh = max(xh, H[idx]); xl = min(xl, L[idx])
                sh |= H[idx] >= hi + buf
                sl_ |= L[idx] <= lo - buf
            if idx % 5 != 4 or not np.isfinite(C[idx]):
                continue
            c = C[idx]
            d = -1 if (sh and c < hi) else (1 if (sl_ and c > lo) else 0)
            if d == 0:
                continue
            if idx + 1 > last_idx or 28 <= idx + 1 <= 32:
                break
            risk = max(abs((xh if d == -1 else xl) - c) + stop_buf, min_risk)
            if risk > max_risk:
                break
            sigs.append({"i": i, "entry_slot": RTH_OFF + idx + 1, "dir": d, "tp_pts": R * risk, "sl_pts": risk,
                         "exit_slot": RTH_OFF + exit_idx, "tag": "A2_ON_SWEEP_2R"})
            break
    return sigs


# ---------------------------------------------------------------------------------------------
# 2) TREND_EXT_FOLLOW: big-extension trend-day continuation, flat before the last 30 minutes
#    First 5m close between 10:04 and 14:59 where |close - 09:30 open| > 0.75 * ADR10
#    -> trade in the direction of the move at the next minute's open (skip 09:58-10:02).
#    Stop = 0.5 * ADR10, no target, time exit at the 15:29 close.
# ---------------------------------------------------------------------------------------------
def trend_ext_follow(g, lv):
    k, s, start, end, exit_idx = 0.75, 0.5, 30, 325, 359
    adr = _adr(g, 10)
    sigs = []
    for r in lv.itertuples():
        i = r.i
        if not r.has_prev or not np.isfinite(adr[i]) or not np.isfinite(r.open_0930):
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
                             "tp_pts": None, "exit_slot": RTH_OFF + exit_idx, "tag": "A2_TREND_EXT"})
                break
    return sigs


# ---------------------------------------------------------------------------------------------
# 3) CLOSE_REVERSAL_1530: last-30-minute reversal of the daily move
#    At the 15:29 close compare with the prior RTH close (lv.pdc); trade AGAINST that sign at the 15:30 open.
#    Stop 30 pts, no target, exit at the 15:59 close.  Skips days without 15:29 data (early closes).
# ---------------------------------------------------------------------------------------------
def close_reversal_1530(g, lv):
    sl = 30.0 * _scale(g)
    sig_slot, exit_slot = 1289, 1319     # 15:29, 15:59
    sigs = []
    for r in lv.itertuples():
        if not r.has_prev:
            continue
        c, b = g.C[r.i, sig_slot], r.pdc
        if not (np.isfinite(c) and np.isfinite(b)) or c == b:
            continue
        sigs.append({"i": r.i, "entry_slot": sig_slot + 1, "dir": -int(np.sign(c - b)), "tp_pts": None, "sl_pts": sl,
                     "exit_slot": exit_slot, "tag": "A2_CLOSE_REV"})
    return sigs


# Only the candidate that passed the one-shot validation (both periods avg > 0) is delivered.
CANDIDATES = {
    "A2_TREND_EXT_FOLLOW": trend_ext_follow,
}

# Frozen before validation but FAILED it (validation avg < 0). Kept only for reproducibility; do not use.
FAILED_VALIDATION = {
    "A2_ON_SWEEP_2R": on_sweep_2r,              # discovery +$148/trade, validation -$146/trade
    "A2_CLOSE_REVERSAL_1530": close_reversal_1530,  # discovery +$121/trade, validation -$15/trade
}
ALL_FROZEN = {**CANDIDATES, **FAILED_VALIDATION}
