"""C3 frozen candidates (parameters chosen on DISCOVERY 2024-01-02..2025-06-30 only, frozen before validation).

  C3_ES_I2_40_15      ES-only: frozen I2 v1 entry logic applied to ES with ES's own prior-day value area
                      (frozen algorithm, 5-pt bins, 70%), TP +40 / SL -15 ES points, time exit 12:29 ET.
                      fn(g, lv) with g = data.load_grid("ES"), lv = features.daily_levels(g).
                      NOTE: bin 5 / the I2 entry on ES was already looked at by earlier work on data through
                      2026-10 (ES shadow study 22/58) -> the ES entry side is NOT clean out-of-sample.
  C3_PORT_A3C3_ES     Portfolio: 0.9 NQ (=9 MNQ) on A3_C3 exits  +  0.8 ES (=8 MES) on C3_ES_I2_40_15.
  C3_PORT_I2v1_ES     Portfolio: 0.6 NQ (=6 MNQ) on frozen I2 v1  +  0.8 ES (=8 MES) on C3_ES_I2_40_15.
                      Sizes: risk parity (w ∝ 1/discovery daily sd per contract), scaled so the portfolio's
                      discovery daily sd = that of 1 NQ I2 v1, rounded to 0.1 contract (micro).
                      Micro cost assumption: $4.80 per MNQ RT, $10.80 per MES RT (= full cost x contracts).

Portfolio candidates are evaluated with run_portfolio(name, include_holdout) -> trades DataFrame
(date, symbol, sleeve, contracts, pnl, split, year). CANDIDATES maps every name to a callable:
  * C3_ES_I2_40_15: fn(g, lv) -> signals (sim.run_signals-compatible), for an ES grid.
  * portfolio names: fn(g, lv) returns only the NQ-leg signals (with 'contracts') for an NQ grid, so the
    standard NQ evaluator still works; the full two-market result needs run_portfolio().
No look-ahead: entries are strategies.i2_v1_signals (5-min closes before the entry minute); TP/SL/POC are
set from prior-day levels and the fill price; the A3 no-progress exit uses only bars up to its decision minute.
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", ".."))
from lib import data, features, sim, strategies  # noqa: E402
from lib.data import RTH_OFF  # noqa: E402

EXIT_1229 = RTH_OFF + 179
ES_TP, ES_SL = 40.0, 15.0
# A3_C3 exit (copied, frozen by A3): POC target floor 20, SL 60, no-progress 30 min / 20 pts
A3_POC_FLOOR, A3_SL, A3_NP_N, A3_NP_X = 20.0, 60.0, 30, 20.0
WEIGHTS = {
    "C3_PORT_A3C3_ES": {"NQ": ("A3C3", 0.9), "ES": ("ES_40_15", 0.8)},
    "C3_PORT_I2v1_ES": {"NQ": ("I2v1", 0.6), "ES": ("ES_40_15", 0.8)},
}


def _fill(g, s):
    e = s["entry_slot"]
    while e <= EXIT_1229 and np.isnan(g.O[s["i"], e]):
        e += 1
    return e, (g.O[s["i"], e] if e <= EXIT_1229 else np.nan)


def _a3c3(g, lv, contracts=1.0):
    poc = dict(zip(lv.i, lv.poc))
    out = []
    for s in strategies.i2_v1_signals(g, lv):
        e, px = _fill(g, s)
        if np.isnan(px):
            continue
        tp = max(s["dir"] * (poc[s["i"]] - px), A3_POC_FLOOR)
        last = e + A3_NP_N - 1
        x = EXIT_1229
        if last < EXIT_1229:
            i, d = s["i"], s["dir"]
            fav = np.nanmax(g.H[i, e:last + 1] - px) if d == 1 else np.nanmax(px - g.L[i, e:last + 1])
            if fav < A3_NP_X:
                x = last
        out.append(dict(s, tp_pts=float(tp), sl_pts=A3_SL, exit_slot=x, contracts=contracts, tag="A3C3"))
    return out


def _i2v1(g, lv, contracts=1.0):
    return [dict(s, contracts=contracts) for s in strategies.i2_v1_signals(g, lv)]


def es_i2_40_15(g, lv, contracts=1.0):
    """ES I2: frozen I2 v1 entries on ES's own value area (5-pt bins) with TP 40 / SL 15 ES points, 12:29."""
    assert g.symbol == "ES", "C3_ES_I2_40_15 is defined for the ES grid"
    return [dict(s, tp_pts=ES_TP, sl_pts=ES_SL, exit_slot=EXIT_1229, contracts=contracts, tag="C3_ES_40_15")
            for s in strategies.i2_v1_signals(g, lv)]


_NQ_LEG = {"A3C3": _a3c3, "I2v1": _i2v1}


def _nq_leg_fn(name):
    def fn(g, lv):
        assert g.symbol == "NQ"
        leg, w = WEIGHTS[name]["NQ"]
        return _NQ_LEG[leg](g, lv, contracts=w)
    fn.__name__ = name
    return fn


def run_portfolio(name: str = "C3_PORT_A3C3_ES", include_holdout: bool = False) -> pd.DataFrame:
    """Full two-market portfolio trades. include_holdout is passed to data.load_grid (Phase E only)."""
    out = []
    for sym in ["NQ", "ES"]:
        g = data.load_grid(sym, include_holdout=include_holdout)
        lv = features.daily_levels(g)
        leg, w = WEIGHTS[name][sym]
        sig = _NQ_LEG[leg](g, lv, contracts=w) if sym == "NQ" else es_i2_40_15(g, lv, contracts=w)
        tr = sim.run_signals(g, sig)
        tr["symbol"], tr["sleeve"], tr["contracts"] = sym, leg, w
        out.append(tr)
    tr = pd.concat(out, ignore_index=True).sort_values(["date", "entry_slot"]).reset_index(drop=True)
    return tr[["date", "symbol", "sleeve", "contracts", "dir", "entry_time", "exit_time", "reason", "pts", "pnl",
               "split", "year"]]


CANDIDATES = {
    "C3_ES_I2_40_15": es_i2_40_15,
    "C3_PORT_A3C3_ES": _nq_leg_fn("C3_PORT_A3C3_ES"),
    "C3_PORT_I2v1_ES": _nq_leg_fn("C3_PORT_I2v1_ES"),
}
PORTFOLIOS = {"C3_PORT_A3C3_ES", "C3_PORT_I2v1_ES"}
