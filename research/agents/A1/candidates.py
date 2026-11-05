"""A1 frozen candidates (parameters hard-coded). Usage:
    import sys; sys.path.insert(0, "research"); sys.path.insert(0, "research/agents/A1")
    from lib import data, features, sim
    from candidates import CANDIDATES
    g = data.load_grid("NQ"); lv = features.daily_levels(g)
    tr = sim.run_signals(g, CANDIDATES["OOV_ORB15_EOD"](g, lv))

Idea ("out-of-value opening-range breakout"):
  * Regime (known at 09:30): today's 09:30 open is OUTSIDE yesterday's RTH value area (same VA as I2 v1;
    `lv.vah/lv.val`, needs lv.has_prev). Open inside value -> no trade (ORB loses badly on those days).
  * Opening range = high/low of the first N minutes of RTH (09:30 .. 09:30+N-1).
  * Signal: first 1-minute CLOSE above OR high (long) or below OR low (short), between the end of the OR and
    the cutoff; entry at the OPEN of the next minute.  One trade per day.
  * Exit: stop = stop_mult x OR width beyond the opposite OR side measured from entry (stop_mult=1 -> stop at the
    opposite OR side); optional target = k x OR width from entry; otherwise exit at the 15:59 close.
No future data: OR uses bars <= 09:30+N-1, the trigger uses closes <= entry_slot-1, the regime uses the 09:30 open.
"""
import numpy as np

RTH_OFF = 930


def _slot(hhmm):
    h, m = map(int, hhmm.split(":"))
    return (h * 60 + m - 18 * 60) % 1440


def oov_orb(g, lv, N=15, cutoff="12:00", stop_mult=1.0, k=None, exit_time="15:59", regime="outVA",
            risk_usd=None, tag="OOV_ORB"):
    xs, cs = _slot(exit_time), _slot(cutoff)
    sigs = []
    for r in lv.itertuples():
        i = r.i
        if not (r.tradable and r.has_prev) or not np.isfinite(r.open_0930):
            continue
        o = r.open_0930
        if regime == "outVA":
            if r.val <= o <= r.vah:
                continue
        elif regime == "outPR":
            if r.pdl <= o <= r.pdh:
                continue
        h = np.nanmax(g.H[i, RTH_OFF:RTH_OFF + N]); l = np.nanmin(g.L[i, RTH_OFF:RTH_OFF + N])
        if not (np.isfinite(h) and np.isfinite(l)) or h <= l:
            continue
        rng = h - l
        C = g.C[i, RTH_OFF + N: cs]          # closes of minutes after the OR, up to cutoff-1
        up = np.where(C > h)[0]; dn = np.where(C < l)[0]
        ku = up[0] if len(up) else 10 ** 9; kd = dn[0] if len(dn) else 10 ** 9
        if ku == kd:                          # no break before cutoff
            continue
        d = 1 if ku < kd else -1
        es = RTH_OFF + N + min(ku, kd) + 1
        px = g.O[i, es]
        if not np.isfinite(px):
            continue
        opp = l if d == 1 else h
        sl = max(abs(px - opp) + (stop_mult - 1.0) * rng, 1.0)
        s = {"i": i, "entry_slot": es, "dir": d, "sl_pts": sl, "tp_pts": (k * rng if k else None),
             "exit_slot": xs, "tag": tag, "or_width": rng}
        if risk_usd is not None:            # fixed-$ risk sizing in micro (0.1) steps, max 1 contract
            c = np.floor(risk_usd / (sl * g.point) * 10) / 10
            if c < 0.1:
                continue
            s["contracts"] = float(min(c, 1.0))
        sigs.append(s)
    return sigs


def OOV_ORB15_EOD(g, lv):
    return oov_orb(g, lv, N=15, cutoff="12:00", stop_mult=1.0, k=None, exit_time="15:59", tag="OOV_ORB15_EOD")


def OOV_ORB30_1R(g, lv):
    return oov_orb(g, lv, N=30, cutoff="12:00", stop_mult=1.0, k=1.0, exit_time="15:59", tag="OOV_ORB30_1R")


def OOV_ORB30_1R_RISK1500(g, lv):
    """Same signals as OOV_ORB30_1R, but position size = $1,500 / (stop pts x $20), floored to 0.1-contract
    (MNQ) steps, capped at 1 NQ. Cost in the engine scales with size ($48 x contracts)."""
    return oov_orb(g, lv, N=30, cutoff="12:00", stop_mult=1.0, k=1.0, exit_time="15:59", risk_usd=1500.0,
                   tag="OOV_ORB30_1R_RISK1500")


CANDIDATES = {
    "OOV_ORB15_EOD": OOV_ORB15_EOD,
    "OOV_ORB30_1R": OOV_ORB30_1R,
    "OOV_ORB30_1R_RISK1500": OOV_ORB30_1R_RISK1500,
}
