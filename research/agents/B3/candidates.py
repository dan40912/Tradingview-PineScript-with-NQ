"""B3 frozen candidates (rule-based; no model fitting, so nothing is trained on any period).

All parameters are frozen.  Each fn(g, lv) -> list of signal dicts for lib.sim.run_signals.
Signals use only closes up to entry_slot-1 (entry at the OPEN of entry_slot).

B3_NQES_RS_0945_T120   : at 09:45 ET go in the direction of NQ's 60-minute return minus ES's
                         60-minute return (relative strength), hold 120 min (exit 11:44 close),
                         no TP/SL.  Pre-selected by the discovery-only IC screen (the only feature
                         outside the 10:00 cluster with same sign and |t|>=2 in 2024 and 2025H1).
(rejected, kept only for reproducibility, NOT in CANDIDATES)
B3_MOM_OPEN_1000_X80   : at 10:00 ET go in the direction of (last close - 09:30 open), +-80 bracket.
                         Chosen after seeing validation of sibling families, and only 7/18
                         neighbours are positive in validation -> rejected.
"""
from __future__ import annotations
import numpy as np
from lib import data, sim

RTH_OFF = 930


def _last(arr, upto):
    seg = arr[: upto + 1]
    k = np.where(np.isfinite(seg))[0]
    return seg[k[-1]] if len(k) else np.nan


def _es_for(g):
    es = data.load_grid("ES", include_holdout=bool(g.dates[-1] >= data.HOLDOUT_START))
    return es


def nqes_rs_signals(g, lv, t="09:45", lookback=60, hold=120, es=None, tag="B3_NQES_RS"):
    es = es if es is not None else _es_for(g)
    e = sim.slot_of(t)
    sigs = []
    for i in range(len(g)):
        if not g.tradable[i]:
            continue
        k = es.dindex.get(g.dates[i])
        if k is None:
            continue
        P, P0 = _last(g.C[i], e - 1), _last(g.C[i], e - 1 - lookback)
        E, E0 = _last(es.C[k], e - 1), _last(es.C[k], e - 1 - lookback)
        if not np.all(np.isfinite([P, P0, E, E0])):
            continue
        rel = (P / P0 - 1) - (E / E0 - 1)
        if rel == 0:
            continue
        sigs.append({"i": i, "entry_slot": e, "dir": int(np.sign(rel)), "tp_pts": None, "sl_pts": None,
                     "exit_slot": e + hold - 1, "tag": tag})
    return sigs


def mom_open_signals(g, lv, t="10:00", x=80.0, exit_hhmm="15:59", tag="B3_MOM_OPEN"):
    e = sim.slot_of(t)
    xs = sim.slot_of(exit_hhmm)
    sigs = []
    for i in range(len(g)):
        if not g.tradable[i]:
            continue
        op = g.O[i, RTH_OFF]
        P = _last(g.C[i], e - 1)
        if not (np.isfinite(op) and np.isfinite(P)) or P == op:
            continue
        sigs.append({"i": i, "entry_slot": e, "dir": 1 if P > op else -1, "tp_pts": x, "sl_pts": x,
                     "exit_slot": xs, "tag": tag})
    return sigs


def B3_NQES_RS_0945_T120(g, lv):
    return nqes_rs_signals(g, lv, t="09:45", lookback=60, hold=120, tag="B3_NQES_RS_0945_T120")


def B3_MOM_OPEN_1000_X80(g, lv):
    return mom_open_signals(g, lv, t="10:00", x=80.0, exit_hhmm="15:59", tag="B3_MOM_OPEN_1000_X80")


CANDIDATES = {
    "B3_NQES_RS_0945_T120": B3_NQES_RS_0945_T120,
}

REJECTED = {"B3_MOM_OPEN_1000_X80": B3_MOM_OPEN_1000_X80}
