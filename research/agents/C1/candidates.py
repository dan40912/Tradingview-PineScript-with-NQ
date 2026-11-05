"""C1 frozen candidates (parameters fixed on DISCOVERY 2024-01..2025-06, before running validation).

Usage (repo root):
    import sys; sys.path.insert(0, "research"); sys.path.insert(0, "research/agents/C1")
    from lib import data, features, sim
    from candidates import CANDIDATES
    g = data.load_grid("NQ"); lv = features.daily_levels(g)
    tr = sim.run_signals(g, CANDIDATES["C1_ORB15_POC_A3X_TS"](g, lv))

Common to all three
  * Regime: prior-day RTH value area exists (lv.has_prev), day tradable, 09:30 open OUTSIDE the value area.
    Direction is always TOWARD value (= lv.i2_dir: open above VAH -> short only, below VAL -> long only).
  * One trade per day. Entry at the OPEN of the minute after the trigger bar closes. Entry idx <= 120 (11:30);
    a trigger whose entry minute falls in 09:58-10:02 cancels the day (same as frozen I2 v1).
  * Risk overlay (B2_I2_TrendSize, frozen by B2): 0.5 contract when the trade direction is against the sign of the
    prior 20-session return (B2 regime.ret20, uses only completed sessions), else 1 contract.
  * 'A3X' exit = A3_C3 exit (frozen by A3): target = prior-day POC measured from the fill (at least 20 pts),
    stop 60 pts, no-progress exit at the close of minute e+29 if the best favourable excursion over minutes
    e..e+29 is < 20 pts, time exit 12:29 close. The no-progress rule is passed to the shared engine as a
    pre-computed exit_slot: it only uses bars up to the decision minute (if TP/SL hits first the engine exits
    there first, and if it fires the TP was by construction not reached).

Candidates
  C1_I2_A3X_TS        : frozen I2 v1 entry (2 consecutive 5-min closes inside VA) + A3X exit + trend sizing.
  C1_ORB15_POC_A3X_TS : earlier entry -- first 1-min close beyond the 09:30-09:44 opening range on the side
                        TOWARD value (breaks away from value are ignored) + A3X exit + trend sizing.
  C1_ORB15_TP110_TS   : same ORB15 entry, TP +110 / SL -145, hold to the 15:59 close + trend sizing.
"""
from __future__ import annotations
import os, sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import c1lib as c  # noqa: E402

# ---- frozen parameters -------------------------------------------------------------------
ORB_N = 15
POC_FLOOR, SL_A3X, NP_MIN, NP_PTS = 20.0, 60.0, 30, 20.0
TP_HOLD, SL_HOLD = 110.0, 145.0
A3X = dict(tp="poc", poc_floor=POC_FLOOR, sl=SL_A3X, np=(NP_MIN, NP_PTS), exit="1229")
HOLD = dict(tp=TP_HOLD, sl=SL_HOLD, exit="1559")


def _lv_plus(g, lv):
    """Add ret20 (B2) if missing. Only causal columns are added."""
    if "ret20" in lv:
        return lv
    lv = lv.copy()
    lv["ret20"] = c.B2REG.regime_features(g, lv).ret20.values
    if "volpoor_rel" not in lv:
        lv["volpoor_rel"] = np.nan
    return lv


def _finish(g, lv, entries, exit_spec, tag):
    sigs = c.make_signals(g, lv, entries, exit_spec, tag=tag)
    return c.overlay(sigs, lv, volpoor=None, trend=True)


def c1_i2_a3x_ts(g, lv):
    lv = _lv_plus(g, lv)
    return _finish(g, lv, c.ent_i2(g, lv), A3X, "C1_I2_A3X_TS")


def c1_orb15_poc_a3x_ts(g, lv):
    lv = _lv_plus(g, lv)
    return _finish(g, lv, c.ent_orb(g, lv, ORB_N, first=False), A3X, "C1_ORB15_POC_A3X_TS")


def c1_orb15_tp110_ts(g, lv):
    lv = _lv_plus(g, lv)
    return _finish(g, lv, c.ent_orb(g, lv, ORB_N, first=False), HOLD, "C1_ORB15_TP110_TS")


CANDIDATES = {
    "C1_I2_A3X_TS": c1_i2_a3x_ts,
    "C1_ORB15_POC_A3X_TS": c1_orb15_poc_a3x_ts,
    "C1_ORB15_TP110_TS": c1_orb15_tp110_ts,
}
