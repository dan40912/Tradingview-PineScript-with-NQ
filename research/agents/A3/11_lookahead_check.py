"""Look-ahead check for A3_C3: mask every minute after the decision minute (entry+29) and confirm the
signal dict (entry, tp_pts, exit_slot) is unchanged."""
import os, sys, copy
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import data, features
from candidates import CANDIDATES
g = data.load_grid("NQ"); lv = features.daily_levels(g)
fn = CANDIDATES["A3_C3_POC_SL60_NP30"]
ref = fn(g, lv)
bad = 0
for s in ref:
    g2 = copy.copy(g)
    cut = s["entry_slot"] + 30
    for nm in "OHLCV":
        a = getattr(g, nm).copy(); a[s["i"], cut:] = np.nan; setattr(g2, nm, a)
    s2 = [x for x in fn(g2, lv) if x["i"] == s["i"]][0]
    if (s2["entry_slot"], s2["tp_pts"], s2["exit_slot"]) != (s["entry_slot"], s["tp_pts"], s["exit_slot"]):
        bad += 1
print("signals", len(ref), "changed after masking future minutes:", bad)
