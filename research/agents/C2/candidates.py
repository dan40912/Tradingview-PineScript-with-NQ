"""C2 candidates.  Functions are imported unchanged from `frozen_prevalidation_candidates.py`
(sha256 9182580373b4...; frozen on discovery BEFORE the one-shot validation, see FROZEN_candidates.sha256).

Usage:
    import sys; sys.path.insert(0, "research"); sys.path.insert(0, "research/agents/C2")
    from lib import data, features, sim
    from candidates import CANDIDATES
    g = data.load_grid("NQ"); lv = features.daily_levels(g)
    tr = sim.run_signals(g, CANDIDATES["C2_TE_OUTSIDE"](g, lv))       # risk-sized ($1,000 to the stop)
    tr1 = sim.run_signals(g, CANDIDATES["C2_TE_OUTSIDE"](g, lv, sized=False))   # 1 contract

Status after validation (2025H2):
  C2_TE_OUTSIDE       frozen before validation, PASSED (val +$303/trade 1ct, +$39 sized). Partially contaminated:
                      its parent A2_TREND_EXT_FOLLOW's validation result was already published by A2.
  C2_TE_OUTSIDE_NOI2  POST-HOC / CONTAMINATED. Same rule, but skip the trade if I2 v1 has already entered that
                      day before the signal. It was in the discovery screen before validation (05_screen_trend.py,
                      'TE day=out_noI2'), but it is singled out only now, after seeing the validation split by I2
                      status. Only the 2026 holdout / forward test can judge it.
  REJECTED: C2_PM_PULLBACK (val -$482/trade 1ct, 10 trades) and C2_BOOK (val sized -$35/trade).
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from frozen_prevalidation_candidates import te_outside, pm_pullback, book, _size, RTH_OFF  # noqa: E402
from lib import strategies  # noqa: E402


def te_outside_noi2(g, lv, sized=True):
    """C2_TE_OUTSIDE, but only if I2 v1 has NOT entered earlier that day (I2's signal is known in real time:
    it is decided on a 5-min close before its own entry minute). Same parameters, same sizing."""
    i2e = {s["i"]: s["entry_slot"] for s in strategies.i2_v1_signals(g, lv)}
    sigs = [s for s in te_outside(g, lv, sized=False) if not (s["i"] in i2e and i2e[s["i"]] < s["entry_slot"])]
    return _size(sigs, g, sized)


CANDIDATES = {
    "C2_TE_OUTSIDE": te_outside,              # passed validation (partially contaminated, see above)
    "C2_TE_OUTSIDE_NOI2": te_outside_noi2,    # POST-HOC, contaminated -> holdout/forward test only
}
REJECTED = {
    "C2_PM_PULLBACK": pm_pullback,
    "C2_BOOK": book,
}
