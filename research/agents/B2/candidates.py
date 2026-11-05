"""B2 frozen candidates (parameters fixed on DISCOVERY 2024-01..2025-06; do not tune).

Usage (repo root):
    import sys; sys.path.insert(0, "research"); sys.path.insert(0, "research/agents/B2")
    from lib import data, features, sim
    from candidates import CANDIDATES
    g = data.load_grid("NQ"); lv = features.daily_levels(g)
    tr = sim.run_signals(g, CANDIDATES["B2_I2_TrendSize"](g, lv))

All signals use only information available before entry (calendar dates are public in advance;
ATR20 / 20-day return use completed prior sessions; the gap uses the 09:30 open, entry 09:31).
"""
from __future__ import annotations
import os
import sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_RES = os.path.dirname(os.path.dirname(_HERE))
for _p in (_RES, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from lib.data import RTH_OFF  # noqa: E402
from lib import strategies  # noqa: E402
import regime  # noqa: E402

EXIT_1229 = RTH_OFF + 179


def b2_orb30_830(g, lv):
    """Opening-range breakout ONLY on CPI / NFP (08:30 ET tier-1 release) days.
    OR = 09:30-09:59 high/low (width >= 20 pts). First 1-min close above/below OR between 10:00 and 11:29
    -> enter at next minute open in the breakout direction. TP = SL = OR width. Exit 12:29 close. Max 1/day."""
    ev = regime.event_flags(g.dates)
    rel = ev.rel830.values
    sigs = []
    for r in lv.itertuples():
        i = r.i
        if not r.tradable or not rel[i] or np.isnan(r.open_0930):
            continue
        H, L, C = g.H[i], g.L[i], g.C[i]
        oh = np.nanmax(H[RTH_OFF:RTH_OFF + 30]); ol = np.nanmin(L[RTH_OFF:RTH_OFF + 30])
        w = oh - ol
        if not np.isfinite(w) or w < 20.0:
            continue
        for idx in range(30, 120):
            c = C[RTH_OFF + idx]
            if np.isnan(c):
                continue
            d = 1 if c > oh else (-1 if c < ol else 0)
            if d:
                sigs.append({"i": i, "entry_slot": RTH_OFF + idx + 1, "dir": d, "tp_pts": w, "sl_pts": w,
                             "exit_slot": EXIT_1229, "tag": "B2_ORB30_830"})
                break
    return sigs


def b2_gapfade_nonews(g, lv):
    """Fade the RTH opening gap (09:30 open vs prior RTH close) on days WITHOUT CPI / NFP / FOMC.
    Requires |gap| >= 0.1 x ATR20 (and >= 10 pts). Enter 09:31 open against the gap.
    TP = gap fully filled (prior close), SL = 0.75 x |gap|. Exit 12:29 close."""
    ev = regime.event_flags(g.dates)
    t1 = ev.any_tier1.values
    rf = regime.regime_features(g, lv)
    atr = rf.atr20.values
    sigs = []
    for r in lv.itertuples():
        i = r.i
        if not r.has_prev or t1[i] or np.isnan(r.open_0930) or np.isnan(r.pdc) or not np.isfinite(atr[i]):
            continue
        gp = r.open_0930 - r.pdc
        if abs(gp) < max(10.0, 0.1 * atr[i]):
            continue
        sigs.append({"i": i, "entry_slot": RTH_OFF + 1, "dir": -int(np.sign(gp)), "tp_pts": abs(gp),
                     "sl_pts": 0.75 * abs(gp), "exit_slot": EXIT_1229, "tag": "B2_GAPF_NoNews"})
    return sigs


def b2_i2_trendsize(g, lv):
    """Frozen I2 v1 signals, sized by trend agreement: 1 contract when the I2 direction agrees with the sign
    of the prior 20-session return (same-contract chained RTH closes), 0.5 contract (e.g. 5 MNQ) when it
    disagrees. Undefined trend -> 1 contract."""
    rf = regime.regime_features(g, lv)
    r20 = rf.ret20.values
    out = []
    for s in strategies.i2_v1_signals(g, lv):
        x = r20[s["i"]]
        against = np.isfinite(x) and np.sign(x) == -s["dir"]
        out.append(dict(s, contracts=0.5 if against else 1.0, tag="B2_I2_TrendSize"))
    return out


# Passed the PROTOCOL gates (discovery >= +$50/trade, discovery AND validation > 0, neighbours mostly positive).
CANDIDATES = {
    "B2_I2_TrendSize": b2_i2_trendsize,
}

# Frozen before validation but FAILED validation (avg/trade < 0 in 2025H2). Kept only for reference /
# later forward-testing decisions by the coordinator; do NOT treat as candidates.
REJECTED = {
    "B2_ORB30_830": b2_orb30_830,       # validation 10 trades, -$1,276/trade
    "B2_GAPF_NoNews": b2_gapfade_nonews,  # validation 88 trades, -$19/trade
}
